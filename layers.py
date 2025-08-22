import torch
from torch import nn

class DeepseekV3RMSNorm(nn.Module):
    def __init__(self,
                 hidden_size,
                 eps=1e-6):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self,hidden_size):
        input_dtype = hidden_size.dtype
        hidden_states = hidden_states.to(torch.float32)

        variance = hidden_states.pow(2).mean(-1,keepdim=True)
        hidden_states = hidden_states + torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)
    
class DeepseekV3RotartEmbedding(nn.Module):
    def __init__(self,
                 config,
                 device=None):
        super().__init__()

        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling , dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"

        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config,device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self,
                x,
                position_ids):
        inv_freq_expanded = self.inv_freq[None,:,None].float().expand(position_ids.shape[0],-1,1).to(x.device)
        position_ids_expanded = position_ids[:,None,:].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"

        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).float().transponse(1,2)
            emb = torch.cat((freqs,freqs),dim=-1)

            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    
class DeepseekV3MLP(nn.Module):
    def __init__(self,
                 config,
                 hidden_size=None,
                 intermediate_size=None):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size,bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self,x):
        x = self.down_proj(self.act_fn(self.gate_proj(x) * self.up_proj(x)))
        return x
    
class DeepseekV3TopkRouter(nn.Module):
    def __init__(self,
                 config):
        super().__init__()

        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_router_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob

        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
        self.register_buffer("e_score_correction_bias", torch.zeros(self.n_routed_experts))

    def get_topk_indices(self,
                         scores):
        scores_for_choice = scores.view(-1, self.n_routed_experts) + self.e_score_correction_bias
        group_scores = (
            scores_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sun(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)

        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1,self.n_group,self.n_routed_experts//self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(),0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        return topk_indices
    
    def forward(self,x):
        x = x.view(-1, self.config.hidden_size)
        router_logits = F.linear(x.type(torch.float32), self.weight.type(torch.float32))
        scores = router_logits.sigmoid()
        topk_indices = self.get_topk_indices(scores)

        topk_weights = scores.gather(1, topk_indices)

        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_indices /- denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights
    

class DeepseekV3MoE(nn.Module):
    def __init__(self,
                 config):
        super().__init__()

        self.config = config
        self.experts = nn.ModuleList(
            [
                DeepseekV3MLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(config.n_routed_experts)
            ]
        )

        self.gate = DeepseekV3TopkRouter(config)
        self.shared_experts = DeepseekV3MLP(
            config=config, 
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts
        )

    def moe(self,
            hidden_states,
            topk_indices,
            topk_weights):
        
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        expert_mask = torch.nn.functional.one_hot(topk_indices, num_classes = len(self.experts))
        expert_mask = expert_mask.permute(2,0,1)

        for expert_idx in range(len(self.experts)):
            expert = self.experts[expert_idx]
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = torch.where(mask)

            if token_indices.numel() > 0:
                expert_weights = topk_weights[token_indices, weight_indices]
                expert_input = hidden_states[token_indices]
                expert_output = expert(expert_input)

                weighted_output = expert_output * expert_weights.unsqueeze(-1)

                final_hidden_states.index_add_(0, token_indices, weighted_output)
        return final_hidden_states.type(hidden_states.dtype)
    

    def forward(self,x):
        residual = x
        orig_shape = x.shape
        topk_indices, topk_weights = self.gate(x)
        x = x.view(-1, x.shape[-1])
        x = self.moe(x, topk_indices, topk_weights).view(*orig_shape)
        x = x + self.shared_experts(residual)
        return x
    
def rotate_half(x):
    x1 = x[...:x.shape[-1]//2]
    x2 = x[...,x.shape//2:]
    return torch.cat((-x2,x1),dim=1)

def apply_rotary_pos_emb(
        q,
        k,
        cos,
        sin,
        position_ids=None,
        unsqueeze_dim=1
):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def eager_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask,
    scaling,
    dropout= 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def apply_rotary_pos_emb_interleave(q,k,cos,sin,position_ids=None,unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    b,h,s,d = q.shape
    q = q.view(b,h,s,d//2,2).transpose(4,3).reshape(b,h,s,d)

    b,h,s,d, = k.shape
    k = k.view(b,h,s,d//2,2).transpose(4,3).reshape(b,h,s,d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def get_yarn_mscale(scale=1,mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) * 1.0

class DeepseekV3Attention(nn.Module):
    def __init__(self,
                 config,
                 layer_ids):
        super().__init__()

        self.config = config
        self.layer_ids = layer_ids
        self.num_key_value_groups = config.num_attention_heads // self.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.num_head = config.num_attention_heads
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = config.qk_head_dim

        self.is_causal = True
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(config.hidden_size,self.num_heads * self.qk_head_dim,bias=False)
        else:
            self.q_a_proj = nn.Linear(config.hidden_size, self.q_lora_rank,bias=config.attention_bias)
            self.q_a_layernorm = DeepseekV3RMSNorm(config.q_lora_rank)
            self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)
        
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias
        )

        self.kv_a_layernorm = DeepseekV3RMSNorm(self.kv_lora_rank)

        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_head * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False
        )

        self.o_proj = nn.Linear(
            self.num_head * self.v_head_dim,
            config.hidden_size,
            bias=config.attention_bias
        )

        self.scaling = self.qk_head_dim ** (-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.scaling = self.scaling * mscale * mscale
        