# all utils and small code components

# Tokenizer
import tiktoken
def get_tokenizer(tokenizer_name, model_name):
    if model_name == "":
        enc = tiktoken.get_encoding(tokenizer_name)
    else:
        enc = tiktoken.encoding_for_model(model_name)
    assert enc.decode(enc.encode("Bella fra")) == "Bella fra"
    return enc

import os
def get_data(folder_name):
    files = os.listdir(folder_name)
    files.sort()
    shakespear = []
    modern = []
    # files have modern and traditional, so we have half the len of list
    for title in files[::2]:
        with open(os.path.join(folder_name, title),"r") as f:
            shakespear.append(f.read())
        with open(os.path.join(folder_name, title.replace("original","modern")),"r") as f:
            modern.append(f.read())
    return shakespear, modern


from typing import Optional
import torch
import numpy as np
import torch.nn as nn 
class RotaryPosEmb(nn.Module):
    def __init__(self, dim, max_len, device):
        super().__init__()
        self.d_model = dim
        self.max_len = max_len
        self.device = device
        p , i = np.meshgrid(np.arange(float(max_len)), np.arange(self.d_model/2)*2)
        theta = (p/1e4**(i/self.d_model)).T
        self.pos_emb = np.stack([np.sin(theta), np.cos(theta)], axis=-1)
        self.pos_emb = self.pos_emb.reshape((self.max_len,self.d_model))[None]
        self.get_freqs()
    
    def sinusoidal_embeddings(self):
        return self.pos_emb # (1, maxlen, dim)
    
    def get_freqs(self):
        self.sin_freqs = torch.tensor(np.repeat(self.pos_emb[..., None, ::2], repeats=2, axis=-1)).to(self.device)
        self.cos_freqs = torch.tensor(np.repeat(self.pos_emb[..., None, 1::2], repeats=2, axis=-1)).to(self.device)
    
    def forward(self, q): #, k):
        T = q.shape[0]
  
        minus_swap_alternate = lambda x: torch.stack([-x[..., 1::2], x[..., ::2]], axis=-1).reshape(x.shape)

        q = q * self.cos_freqs[:, :T, :, :] + minus_swap_alternate(q) * self.sin_freqs[:, :T, :, :] # (B, T, h, dq)*(1, T, 1, dq) + (B, T, h, dq)*(1, T, 1, dq)
        #k = k * self.cos_freqs[:, :T, :, :] + minus_swap_alternate(k) * self.sin_freqs[:, :T, :, :] # (B, T, h, dq)*(1, T, 1, dq) + (B, T, h, dq)*(1, T, 1, dq)
        return q #, k # (B, T, h, dq), (B, T, h, dq)

