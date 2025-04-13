# block composing the model

import torch
import torch.nn as nn

class HeadAttention(nn.Module):
    def __init__(self, inner, head_size, block_size, dropout=0.1, decode=False):
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
    def __init__(self, inner, num_heads, block_size, dropout, decode=false):
        self.heads = nn.ModuleList(*[Head(inner, inner//num_heads, block_size, dropout, decode) for _ in range(num_heads)])
        self.proj = nn.Linear(inner, inner)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class SwiGLU(nn.Module):
    def __init__(self, inner):
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
        self.router = nn.Sequential(
            nn.Linear(n_embd, 2 * n_embd),
            SwiGLU(2 * n_embd),
            nn.Linear(2 * n_embd, n_experts),
        )
        self.experts = [ FeedForward(n_embd, dropout) for _ in range(n_experts) ]

    def forward(self, x):
        router_out = self.router(x)
        prob_expert = F.Softmax(router_out, dim=-1)
        expert_id = torch.argmax(prob_expert)
        out = self.experts[ expert_id ](x) * prob_expert[ expert_id] 
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
        inner = cfg["embedding_size"]
        num_heads = cfg["n_heads"]
        block_size = cfg["block_size"]
        dropout = cfg["dropout"]
        self.blocks = nn.ModuleList(*[Block(inner, num_heads, block_size, dropout) for _ in range(6)])

    def forward(self, x):
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        return x


class Decoder(nn.Module):
    def __init__(self, cfg):
        inner = cfg["embedding_size"]
        num_heads = cfg["n_heads"]
        block_size = cfg["block_size"]
        dropout = cfg["dropout"]
        self.blocks = nn.ModuleList(*[Block(inner, num_heads, block_size, dropout, True) for _ in range(6)])

   def forward(self,x): 
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        return x


