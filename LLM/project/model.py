import torch
import torch.nn as nn
from torch.nn import Functional as F
from blocks import Encoder, Decoder
from utils import get_tokenizer, RotaryPosEmb

class LLM(nn.Module):
    def __init__(self, cfg):
        self.cfg = cfg
        self.token_emb = get_tokenizer(cfg["tokenizer_name"], cfg["model_name"])
        self.pose_emb = RotaryPosEmb(cfg["embedding_size"]//cfg["num_heads"], 1000, 10000)
        self.encoder = Encoder(cfg).to(device)
        self.decoder = Decoder(cfg).to(device)
        self.normalizer = nn.modules.normalization.RMSNorm(cfg["embedding_size"])
        self.device = cfg["device"]

    def forward(self, x):
        B, T = x.shape

        tokens = self.token_emb(x)
        pos = self.pose_emb(torch.arange(T), device=device)
        x = tokens + pos
        out = self.decoder(self.encoder(x))
        return self.normalizer(out)


