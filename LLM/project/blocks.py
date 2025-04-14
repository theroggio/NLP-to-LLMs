# block composing the model

import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    def __init__(self, inner, head_size, block_size, dropout=0.1, decode=False):
        super().__init__()
        self.key = nn.Linear(inner, head_size, bias=False)
        self.query = nn.Linear(inner, head_size, bias=False)
        self.value = nn.Linear(inner, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.decode = decode

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        ww = q @ k.transpose(-2,-1) * k.shape[-1]**(-0.5)
        if self.decode:
            ww = ww.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        ww = F.softmax(ww,dim=-1)
        ww = self.dropout(ww)
        return ww @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, inner, num_heads, block_size, dropout, decode=False):
        super().__init__()
        self.heads = nn.ModuleList([Head(inner, inner//num_heads, block_size, dropout, decode) for _ in range(num_heads)])
        self.proj = nn.Linear(inner, inner)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class SwiGLU(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.w1 = nn.Linear(inner, inner)
        self.w2 = nn.Linear(inner, inner)
        self.w3 = nn.Linear(inner, inner)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)

class MoE(nn.Module):
    def __init__(self, n_experts, n_embd, dropout):
        super().__init__()
        self.n_experts = n_experts
        self.router = nn.Sequential(
            nn.Linear(n_embd, 2 * n_embd).cuda(),
            SwiGLU(2 * n_embd).cuda(),
            nn.Linear(2 * n_embd, n_experts).cuda(),
        )
        self.experts = [ FeedForward(n_embd, dropout).cuda() for _ in range(n_experts) ]

    def forward(self, x):
        router_out = self.router(x)
        prob_expert = F.softmax(router_out, dim=-1)
        # select half of the experts
        expert_id = prob_expert.topk( int(self.n_experts/2),dim=-1)[1]
        out = torch.vstack([self.experts[ el  ](x).unsqueeze(0) for el in range(int(self.n_experts)) ])
        masks = torch.full(out.shape, 0, device=x.device)
        for _id in range(masks.shape[0]):
            idxs = torch.where(expert_id == _id)
            masks[_id][ idxs[0], idxs[1],: ] = 1.
        out *= masks
        out = out.mean(0)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            SwiGLU(4 * n_embd),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, inner, num_heads, block_size, dropout, decode=False):
        super().__init__()
        self.mh_attention = MultiHeadAttention(inner, num_heads, block_size, dropout, decode)
        self.ff = FeedForward(inner, dropout)
        self.ln1 = nn.modules.normalization.RMSNorm(inner)
        self.ln2 = nn.modules.normalization.RMSNorm(inner)

    def forward(self, x):
        x = x + self.mh_attention(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        inner = cfg["embedding_size"]
        num_heads = cfg["num_heads"]
        block_size = cfg["block_size"]
        dropout = cfg["dropout"]
        self.blocks = nn.Sequential(*[Block(inner, num_heads, block_size, dropout) for _ in range(6)])

    def forward(self, x):
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        return x


class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        inner = cfg["embedding_size"]
        num_heads = cfg["num_heads"]
        block_size = cfg["block_size"]
        dropout = cfg["dropout"]
        self.blocks = nn.Sequential(*[Block(inner, num_heads, block_size, dropout, True) for _ in range(6)])
        self.proj = nn.Linear(inner, cfg["out"])

    def forward(self,x): 
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        return self.proj(x)

class MoEBlock(nn.Module):
    def __init__(self, inner, num_heads, block_size, dropout, decode=False):
        super().__init__()
        self.mh_attention = MultiHeadAttention(inner, num_heads, block_size, dropout, decode)
        self.moe = MoE(4, inner, dropout)
        self.ln1 = nn.modules.normalization.RMSNorm(inner)
        self.ln2 = nn.modules.normalization.RMSNorm(inner)

    def forward(self, x):
        x = x + self.mh_attention(self.ln1(x))
        x = x + self.moe(self.ln2(x))
        return x

class MoEDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        inner = cfg["embedding_size"]
        num_heads = cfg["num_heads"]
        block_size = cfg["block_size"]
        dropout = cfg["dropout"]
        self.blocks = nn.Sequential(*[MoEBlock(inner, num_heads, block_size, dropout, True) for _ in range(6)])
        self.proj = nn.Linear(inner, cfg["out"])

    def forward(self,x): 
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        return self.proj(x)


