import torch
from torch import nn
import torch.nn.functional as F
import math
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from attn_implementation import eager_paged_attention_forward
from transformers.modeling_outputs import BaseModelOutputWithPast


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

        self.act_fn = nn.GELU(approximate='tanh')

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

def yarn_get_mscale(scale=1,mscale=1):
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
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
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
    
    def forward(self,
                hidden_states,
                position_embeddings,
                attention_mask=None,
                past_key_values=None,
                cache_position=None,
                **kwargs
                ):
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        if self.q_lora_rank is None:
            q_states = self.q_proj(hidden_states)
        else:
            q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        
        q_states = q_states.view(query_shape).transpose(1,2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim],dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim],dim=-1)

        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass).view(key_shape).transpose(1,2))
        k_pass , value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim],dim=-1)

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        cos, sin = position_embeddings

        if self.config.rope_interleave:
            q_rot , k_rot = apply_rotary_pos_emb_interleave(q_rot,k_rot,cos,sin)
        else:
            q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)

        k_rot = k_rot.expand(*k_pass.shape[:-1],-1)

        query_states = torch.cat((q_pass, q_rot),dim=-1)
        key_states = torch.cat((k_pass, k_rot),dim=-1)

        if past_key_values is not None:
            cache_kwargs = {"sin":sin, "cos":cos, "cache_position":cache_position}
            key_states , value_states = past_key_values.update(key_states, value_states, self.layer_ids, cache_kwargs)
        
        if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0,self.qk_head_dim-self.v_head_dim])

        attention_inference = eager_paged_attention_forward
        
        attn_output, attn_weights = attention_inference(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs
        )
        
        if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
            attn_output = attn_output[:,:,:,:self.v_head_dim]
        
        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
    

class Deepseekv3DecoderLayer(nn.Module):
    def __init__(self,
                 config,
                 layer_idx):
        super().__init__()

        self.hidden_size = config.hidden_size
        
        self.self_attn = DeepseekV3Attention(config=config, layer_ids=layer_idx)

        if layer_idx >= config.first_k_dense_replace:
            self.mlp = DeepseekV3MoE(config)
        else:
            self.mlp = DeepseekV3MLP(config)

        self.input_layernorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


    def forward(self,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                use_cache,
                cahce_position,
                position_embeddings,
                **kwargs):
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states,
            attnetion_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cahce_position=cahce_position,
            position_embeddings=position_embeddings
        )

        hidden_states = residual + hidden_states

        residual  = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
    

class DeepseekV3Model(nn.Module):
    def __init__(self,
                 config):
        super().__init__()

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        self.layers = nn.ModuleList(
            [Deepseekv3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = DeepseekV3RMSNorm(config.hidden_size , eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV3RotartEmbedding(config=config)
        self.gradient_checkpointing = False

    def forward(self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                input_embeds=None,
                cache_position=None,
                use_cache=None,
                **kwargs
                ):
        
        if input_embeds is None:
            input_embeds = self.embed_tokens(input_ids)
        
        # if use_cache and past_key_values is None:
        #     past_key_values = DynamicCache(config=self.config)
        
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + input_embeds.shape[1], device=input_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=input_embeds, 
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids
        )

        hidden_states = input_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[:self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_states=hidden_states,
            past_key_values=past_key_values
        )
    

class DeepseekV3ForCausalLM(nn.Module):
    def __init__(self,
                 config):
        super().__init__()

        self.model = DeepseekV3Model(config)
        self.vocab_size = config.vocab_size

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
    
    def forward(self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                labels=None,
                use_cache=None,
                cache_position=None,
                logits_to_keep=0,
                **kwargs
                ):
        outputs = self.model(
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            inputs_embeds,
            use_cache,
            cache_position,
            **kwargs
        )

        hidden_states = outputs.last_hidden_states

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

        logits = self.lm_head(hidden_states[:,slice_indices,:])

        return logits