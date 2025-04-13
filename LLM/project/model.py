import torch
import torch.nn as nn
from torch.nn import Functional as F
from blocks import Encoder, Decoder
from utils import get_tokenizer, RotaryPosEmb

class LLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = self.cfg["device"]
        self.token_emb = get_tokenizer(cfg["tokenizer_name"], cfg["model_name"]).to(self.device)
        self.pose_emb = RotaryPosEmb(cfg["embedding_size"]//cfg["num_heads"], 1000, 10000).to(self.device)
        self.encoder = Encoder(cfg).to(self.device)
        self.decoder = Decoder(cfg).to(self.device)
        self.normalizer = nn.modules.normalization.RMSNorm(cfg["embedding_size"]).to(self.device)
        self.optimizer = torch.optim.AdamW(self.params(), lr=1e-3)

    def getLoss(self, x, y):
        return F.cross_entropy(x,y)

    def forward(self, x, y = None):
        B, T = x.shape

        tokens = self.token_emb.encode(x)
        pos = self.pose_emb(torch.arange(T), device=device)
        x = tokens + pos
        out = self.decoder(self.encoder(x))
        out = self.normalizer(out)
        
        if y is None:
            return out , None

        B, T, C = out.shape
        logits = out.view(B*T,C)
        targets = y.view(B*T)

        loss = self.getLoss(logits,targets)
        return out, loss 

    def get_batch(self, data):
        block_size = self.cfg["block_size"]
        batch_size = self.cfg["batch_size"]
        # get a random value between 0 and len - block_size (index of last possible sequence of the right dimension) of size (batch_size,)
        idx = torch.randint(len(data) - block_size, (batch_size,))
        # populate x and y with the data
        x = torch.stack([data[i:i+block_size] for i in idx]).to(self.device)
        y = torch.stack([data[i+1:i+block_size+1] for i in idx]).to(self.device)
        return x, y

    def train(self, data):
        total_loss = 0.0
        for epoch in range(self.cfg["epochs"]):
            for _ in range(1000): # num steps for epoch
                x, y = self.get_batch(data)
                self.optimizer.zero_grad()
                out, loss = self.forward(x,y)
                loss += total_loss
                loss.backward()
                optimizer.step()
            total_loss /= 1000
            print(f"At epoch {epoch} mean loss is: {total_loss}.")
            total_loss *= 0.0
            if epoch % 10:
                self.generate()
    
    def generate(self):
        with torch.no_grad():
            x = torch.zeros((1,1), dtype=torch.long).to(self.device)
            x[0][0] = torch.randint(low=0, high=200) # random starting token
            print(f"Starting from token of meaning: {self.token_emb.decode(x)}.")
            for _ in range(100):
                # get predictions
                logits, loss = self(x[:, -self.cfg["block_size"]:]) # need to be sure we use only block_size elements or the pos embed breaks
                # get last step
                logits = logits[:,-1,:]
                # get probabilities
                probs = F.softmax(logits, dim=-1)
                # sample from probability distribution
                idx_next = torch.multinomial(probs, num_samples=1)
                # append as previous context to proceed
                x = torch.cat((x, idx_next), dim=1)
        print(f"Final sentence is: {self.token_emb.decode(x)}.")
        return x




