import torch
import torch.nn as nn
from torch.nn import functional as F
from blocks import Encoder, Decoder, MoEDecoder
from utils import get_tokenizer, RotaryPosEmb, get_data
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
        self.decoder = MoEDecoder(cfg).to(self.device)
        self.normalizer = nn.modules.normalization.RMSNorm(self.tokenizer.max_token_value).to(self.device)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=5e-3)

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

    def get_batch(self, data, label):
        block_size = self.cfg["block_size"]
        batch_size = self.cfg["batch_size"]
        # get a random value between 0 and len - block_size (index of last possible sequence of the right dimension) of size (batch_size,)
        play_id = torch.randint(len(data), (self.cfg["batch_size"],)).to(self.device)
        idx = torch.vstack( [torch.randint( min(len(data[xx]),len(label[xx])) - block_size, (1,)) for xx in play_id] ).to(self.device)
        x = torch.stack([data[_id][i:i+block_size] for i, _id in zip(idx, play_id)])
        y = torch.stack([label[_id][i:i+block_size] for i, _id in zip(idx,play_id)])
        return x, y

    def fine_tune(self, original, translate):
        total_loss = 0.0
        min_loss = 200
        steps = 500
        data_train = [torch.tensor(self.tokenizer.encode(data), dtype=torch.long).to(self.device) for data in original]
        data_label = [torch.tensor(self.tokenizer.encode(data), dtype=torch.long).to(self.device) for data in translate]
        for epoch in range(self.cfg["epochs"]):
            for _ in range(steps): # num steps for epoch
                x, y = self.get_batch(data_train, data_label)
                self.optimizer.zero_grad()
                out, loss = self.forward(x,y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.detach()
            total_loss /= steps
            if total_loss < min_loss:
                min_loss = total_loss.detach()
                self.save_checkpoint()
            print(f"At epoch {epoch} mean loss is: {total_loss}.")
            total_loss *= 0.0
            if epoch % 10 == 0:
                self.generate()
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), "best_ft.ckpt")

    def generate(self):
        with torch.no_grad():
            x = torch.tensor(self.tokenizer.encode("Two households, both alike in dignity,\nIn fair Verona, where we lay our scene")).unsqueeze(0)[:,:self.cfg["block_size"]].to(self.device)
            if x.shape[1] < self.cfg["block_size"]:
                #padding to block_size
                x = F.pad(x, (self.cfg["block_size"] - x.shape[1] - 1, 1), "constant", 0 )
            for _ in range(150):
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
        print(f"Final sentence is: {self.tokenizer.decode([xx.item() for xx in x[0]])}.")
        return x

if __name__ == "__main__":
    cfg = yaml.safe_load(open("config.yml"))
    model = LLM(cfg)
    model.load_state_dict(torch.load("best.ckpt", weights_only=True))
    data_folder = "data"
    original, translate = get_data(data_folder)
    model.fine_tune(original, translate)


