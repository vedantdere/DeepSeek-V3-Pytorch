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
    
