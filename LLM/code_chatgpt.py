# From Let's build ChatGPT: from scratch, in code, spell out by Andrej Karpathy
import torch

# load your text file for training
data_name = 'input.txt'
with open(data_name, 'r', encoding='utf-8') as f:
    text = f.read()

# extract single unique chars as our vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

# tokenization per single char
# real chatpgt use TIKTOKEN library
class Tokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.stoi = {ch : i for i, ch in enumerate(self.vocab)}
        self.itos = {i : ch for i, ch in enumerate(self.vocab)}

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, embedding):
        return ''.join([self.itos[i] for i in embedding])

tokenizer = Tokenizer(chars)
print("Working exampke of the sentence: my hands are full.")
print(tokenizer.encode("my hands are full"))
print(tokenizer.decode(tokenizer.encode("my hands are full")))

# we now encode the whole text corpus
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

# We now split the dataset in train and val (90% and 10%)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# define a block size, so how big of a sequence we want to use
# we cannot just take one sentence because the length changes and it may be too long
block_size = 8

# examples from a block of size block_size
x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    # goven each context sequence
    context = x[:t+1]
    # we can say what is going to be next
    target = y[t]
    print(f"When input is {context} the target is {target}.")

# we also need to define batch_size, so independent sequences to process together
batch_size = 4
torch.manual_seed(1337)

# how to get a batch
def get_batch(split):
    # select from which data to get it
    data = train_data if split == "train" else val_data
    # get a random value between 0 and len - block_size (index of last possible sequence of the right dimension) of size (batch_size,)
    idx = torch.randint(len(data) - block_size, (batch_size,))
    # populate x and y with the data 
    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx])
    return x, y

xb, yb = get_batch("train")

# simplest neural network: bigram language model
import torch.nn as nn
from torch.nn import functional as F

# this model is only based on single token
# i am token X, probably after me it is gonna be token Y 
class BigramLM(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
 
        if targets is None:
            return logits , None
        
        # fix shapes for cross entropy
        B, T, C = logits.shape
        logits = logits.view(B*T,C)
        targets = targets.view(B*T)
        
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # get predictions
            logits, loss = self(idx)
            # get last step
            logits = logits[:,-1,:]
            # get probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append as previous context to proceed
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramLM(vocab_size)
logits, loss = model(xb, yb)
print(loss)

idx = torch.zeros((1,1), dtype=torch.long)
print(tokenizer.decode(model.generate(idx, max_new_tokens=100)[0].tolist()))

# create optimizer, needed to train
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

batch_size = 32
for steps in range(10000):

    xb, yb = get_batch("train")

    logits, loss = model(xb,yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("At the end of training the loss is now: {loss.item()}")
print("We try to generate again starting from 0,0 and we get something less crazy:")
print(tokenizer.decode(model.generate(idx, max_new_tokens=100)[0].tolist()))

# We need a better model, since BigramLM only look at the last char, it is hard to create words that make sense 

# toy example to understand trasnformers
B,T,C = 4,8,2
x = torch.randn(B,T,C)
# lossy but intuitive way to include previous informatio in my
# current time step data 
xbow = torch.zeros((B,T,C))
for b in range(B):
    for t in range(T):
        xprev = x[b,:t+1]
        xbow[b,t] = xprev.mean(0)

# the trick is doing it with matrix multiplication
a = torch.tril(torch.ones(3,3)) # select only me and previous elements
a /= torch.sum(a, 1, keepdim=True) # if we normalize, we are doing the average over me and prevs
b = torch.randint(0,10,(3,2)).float()
c = a @ b 

# lets apply this to our example
weights = torch.tril(torch.ones(T,T))
weights /= weights.sum(1, keepdim=True)
xbow_v2 = weights @ x # (T,T) @ (B,T,C) ----> (ghost B=1, T,T) @ (B,T,C) ----> (B,T,C) 

# this is way more efficient than the double for loop! And result is the same:
print(torch.allclose(xbow, xbow_v2))

# we have another trick 
tril = torch.tril(torch.ones(T,T))
weights = torch.zeros((T,T))
weights = weights.masked_fill(tril == 0, float('-inf'))
weights = F.softmax(weights, dim=-1)
xbow_v3 = weights @ x

# we still get the same result, but we are using a softmax to normalize and the tril as mask
# it is a bit better because we can optimize the weights matrix and then force that we do not see the future
# and end up with probabilities 
print(torch.allclose(xbow_v2, xbow_v3))

# this is not sufficient because we are still averaging
# we want the network to learn which previous tokens to look at 
# which one are really relevant for the prediction of the new one
head_size = 16
key = nn.Linear(C, head_size, bias=False) # this linear layer has weights that we can learn!
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
# self attention: SELF because K,Q,V all comes from same X
k = key(x) # B, T, 16
q = query(x) # B, T, 16
v = value(x)
weights = q @ k.transpose(-2,-1) # B T 16 x B 16 T ---> B T T 

# now we fo the same, remove the future, softmax 
tril = torch.tril(torch.ones(T,T))
weights = weights.masked_fill(tril == 0, float('-inf')) # this masking is not relevant for non-generative tasks
                                                        # for example for text analysis, you have access to everything
                                                        # and you can just learn connections between past and future tokens
weights = F.softmax(weights, dim=-1)
out = weights @ v

# scaled attention has an extra normalization technique
# we change this line:  weights = q @ k.transpose(-2,-1) # B T 16 x B 16 T ---> B T T 
weights = q @ k.transpose(-2,-1) * head_size**(-0.5)
# this scaling is important before softmax which often creates very squewed distributions if values are large

# here we put it together to use it more easily
class Head(nn.Module):
    def __init__(self, emb_size,  head_size):
        super().__init__()
        self.key = nn.Linear(emb_size, head_size, bias=False).cuda()
        self.query = nn.Linear(emb_size, head_size, bias=False).cuda()
        self.value = nn.Linear(emb_size, head_size, bias=False).cuda()
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        w = q @ k.transpose(-2,-1) * C**(-0.5)
        w = w.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        w = F.softmax(w,dim=-1)
        return w @ v

# multi head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(embed_dim, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

# feed forward layer add non linearity after attention
class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # this 4 multiplier comes from paper
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# common block to use !!
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # the block starts to be deep, we need residuals!
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# lets now make the model
class AttentionNet(nn.Module):

    def __init__(self, vocab_size, block_size, embed_dim):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim).cuda()
        self.position_embedding_table = nn.Embedding(block_size, embed_dim).cuda()
        # add the self attention 
        # self.sa_head = Head(embed_dim, embed_dim).cuda()
        # self.sa_head = MultiHeadAttention(embed_dim, 4, int(embed_dim/4) )
        self.lm_head = nn.Linear(embed_dim, vocab_size).cuda()
        # self.ffw = FeedFoward(embed_dim, dropout=0.1)
        self.blocks = nn.Sequential(*[Block(embed_dim, 4) for _ in range(4)])
        self.ln_f = nn.LayerNorm(embed_dim)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos_emb = self.position_embedding_table(torch.arange(T).cuda())
        token_emb = self.token_embedding_table(idx)
        x = token_emb + pos_emb
        # apply self attention
        #x = self.sa_head(x)
        #x = self.ffw(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits , None
        
        # fix shapes for cross entropy
        B, T, C = logits.shape
        logits = logits.view(B*T,C)
        targets = targets.view(B*T)
        
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # get predictions
            logits, loss = self(idx[:, -block_size:]) # need to be sure we use only block_size elements or the pos embed breaks
            # get last step
            logits = logits[:,-1,:]
            # get probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append as previous context to proceed
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = AttentionNet(vocab_size, block_size=8, embed_dim=32).cuda()
# create optimizer, needed to train
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

batch_size = 32
for steps in range(5000):

    xb, yb = get_batch("train")

    logits, loss = model(xb.cuda(),yb.cuda())

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if steps in [500, 1000, 1500, 3000, 4000, 4999]: # small validation to check how it is going
        idx = torch.zeros((1,1), dtype=torch.long).cuda()
        print(tokenizer.decode(model.generate(idx, max_new_tokens=100)[0].tolist()))

