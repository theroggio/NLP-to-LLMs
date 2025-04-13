# all utils and small code components

# Tokenizer
import tiktoken
def get_tokenizer(tokenizer_name, model_name = None):
    if model_name == None:
        enc = tiktoken.get_encoding(tokenizer_name)
    else:
        enc = tiktoken.encoding_for_model(model_name)
    assert enc.decode(enc.encode("Bella fra")) == "Bella fra"
    return enc

from typing import Optional
import torch
import torch.nn as nn 
class RotaryPosEmb(nn.Module):
    def __init__(self, dim, max_len, base):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.base = base
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / (self.base ** torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache()

    def build_rope_cache(self):
        # Create position indexes `[0, 1, ..., max_len - 1]`
        seq_idx = torch.arange(
            self.max_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x, input_pos: Optional[torch.Tensor] = None):
        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)

        # extract the values based on whether input_pos is set or not
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)
