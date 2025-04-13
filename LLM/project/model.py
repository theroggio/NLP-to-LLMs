import torch
import torch.nn as nn
from torch.nn import functional as F
from blocks import Encoder, Decoder
from utils import get_tokenizer, RotaryPosEmb
import yaml

class LLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = self.cfg["device"]
        self.tokenizer = get_tokenizer(cfg["tokenizer_name"], cfg["model_name"])
        self.token_emb = nn.Embedding(self.tokenizer.max_token_value, cfg["embedding_size"]).to(self.device)
        self.pose_emb = RotaryPosEmb( torch.tensor(cfg["embedding_size"]), torch.tensor(1000), device=self.device)
        self.encoder = Encoder(cfg).to(self.device)
        cfg["out"] = self.tokenizer.max_token_value
        self.decoder = Decoder(cfg).to(self.device)
        self.normalizer = nn.modules.normalization.RMSNorm(self.tokenizer.max_token_value).to(self.device)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)

    def getLoss(self, x, y):
        return F.cross_entropy(x.float(),y.long())

    def forward(self, x, y = None):
        B, T = x.shape
        
        tokens = self.token_emb(x)
        pos = self.pose_emb(torch.arange(T).unsqueeze(0).unsqueeze(0).repeat(B,T,1).to(self.device))[0]
        x = tokens + pos
        out = self.decoder(self.encoder(x.float()))
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
        idx = torch.randint(len(data) - block_size, (batch_size,)).to(self.device)
        # populate x and y with the data
        x = torch.stack([data[i:i+block_size] for i in idx])
        y = torch.stack([data[i+1:i+block_size+1] for i in idx])
        return x, y

    def train(self, data):
        total_loss = 0.0
        steps = 200
        data = torch.tensor(self.tokenizer.encode(data), dtype=torch.long).to(self.device)
        for epoch in range(self.cfg["epochs"]):
            for _ in range(steps): # num steps for epoch
                x, y = self.get_batch(data)
                self.optimizer.zero_grad()
                out, loss = self.forward(x,y)
                total_loss += loss
                loss.backward()
                self.optimizer.step()
            total_loss /= steps
            print(f"At epoch {epoch} mean loss is: {total_loss}.")
            total_loss *= 0.0
            if epoch % 10 == 0:
                self.generate()
    
    def generate(self):
        with torch.no_grad():
            x = torch.tensor(self.tokenizer.encode("Two households, both alike in dignity,\nIn fgair Verona, where we lay our scene")).unsqueeze(0)[:,:16].to(self.device)
            for _ in range(150):
                # get predictions
                logits, loss = self(x[:, -16:]) # need to be sure we use only block_size elements or the pos embed breaks
                # get last step
                logits = logits[:,-1,:]
                # get probabilities
                probs = F.softmax(logits, dim=-1)
                # sample from probability distribution
                idx_next = torch.multinomial(probs, num_samples=1)
                # append as previous context to proceed
                x = torch.cat((x, idx_next), dim=1)
        print(f"Final sentence is: {self.tokenizer.decode([xx.item() for xx in x[0]])}.")
        return x

if __name__ == "__main__":
    cfg = yaml.safe_load(open("config.yml"))
    model = LLM(cfg)
    data_name = '../input.txt'
    with open(data_name, 'r', encoding='utf-8') as f:
        text = f.read()
    model.train(text[:int(len(text)*0.9)])


