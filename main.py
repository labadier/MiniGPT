#%%
import random; random.seed(0)
import numpy as np; np.random.seed(0)
import torch; torch.manual_seed(0)

import tiktoken
import einops
from tqdm import tqdm
from torch.nn import functional as F

from settings import settings


class tokenizer:

    def __init__(self):
        with open('dataset.txt', 'r') as f:
            z = list(set(f.read()))
        self.token_to_id = {c: i for i, c in enumerate(z)}
        self.id_to_token = {i: c for i, c in enumerate(z)}

        self.n_vocab = len(z)
    
    def encode(self, text: str) -> list:
        return [self.token_to_id[c] for c in text]
    
    def decode(self, tensor: list) -> str:
        return ''.join([self.id_to_token[i] for i in tensor])
    
# enc = tokenizer()
enc = tiktoken.get_encoding('gpt2')

def encode(text: str) -> torch.Tensor:
    return torch.tensor(enc.encode(text), dtype=torch.long)

def decode(tensor: torch.Tensor) -> str:
    return enc.decode(tensor.tolist())

def get_batch(source):
    
    chunks = torch.randint(0, len(source) - settings.block_size, (settings.batch_size,))
    x = torch.stack([source[chunk:chunk + settings.block_size] for chunk in chunks])
    y = torch.stack([source[chunk + 1:chunk + settings.block_size + 1] for chunk in chunks])
    return x, y

class Head(torch.nn.Module):

    def __init__(self, dim: int):
        super(Head, self).__init__()
        self.to_qkv = torch.nn.Linear(dim, 3*settings.head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(settings.block_size, settings.block_size)))

        self.dropout = torch.nn.Dropout(settings.dropout)

    def forward(self, x: torch.tensor) -> torch.tensor:
        qkv = self.to_qkv(x)

        q = qkv[:, :, :settings.head_size]
        k = qkv[:, :, settings.head_size:2*settings.head_size]
        v = qkv[:, :, 2*settings.head_size:] # B, T, H
        
        score = q @ k.transpose(-2, -1) / np.sqrt(settings.head_size)
        score = score.masked_fill(self.tril[:x.size(1), : x.size(1)] == 0, float('-inf'))
        score = F.softmax(score , dim=-1) # B, T, T
        score = self.dropout(score)

        return score @ v
    
class MultiHeadAttention(torch.nn.Module):

    def __init__(self, dim: int):
        super(MultiHeadAttention, self).__init__()
        self.heads = torch.nn.ModuleList([Head(dim) for _ in range(settings.n_heads)])
        self.proj = torch.nn.Linear(settings.n_heads*settings.head_size, settings.n_heads*settings.head_size)
        self.dropout = torch.nn.Dropout(settings.dropout)
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)

        return out

class FeedForward(torch.nn.Module):

    def __init__(self, dim: int):
        super(FeedForward, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim*4),
            torch.nn.ReLU(),
            torch.nn.Linear(dim*4, dim),
            torch.nn.Dropout(settings.dropout)
        )
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.mlp(x)
    
class DecoderBlock(torch.nn.Module):

    def __init__(self, dim: int):
        super(DecoderBlock, self).__init__()
        self.attention = MultiHeadAttention(dim)
        self.mlp = FeedForward(dim)

        self.norm1 = torch.nn.LayerNorm(dim)
        self.norm2 = torch.nn.LayerNorm(dim)
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        x  = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm1(x))
        return x
class GPT2(torch.nn.Module):
    
    def __init__(self, vocab_size: int):
        super(GPT2, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, settings.embedding_size)
        self.positional_embedding = torch.nn.Embedding(settings.block_size, settings.embedding_size)
        
        self.decoder = torch.nn.Sequential(*[DecoderBlock(settings.embedding_size) for _ in range(settings.n_decoder_blocks)],
                                           torch.nn.LayerNorm(settings.embedding_size))
        self.lm_head = torch.nn.Linear(settings.embedding_size, vocab_size)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, idx: torch.tensor, tagets: torch.tensor = None) -> torch.tensor:
        
        token_emb =  self.embedding(idx.to(self.device)) # B, T, D
        pos_emb = self.positional_embedding(torch.arange(idx.size(1)).to(self.device)) # T, D
        
        x = self.decoder(token_emb + pos_emb) # B, T, V
        logits = self.lm_head(x) # B, T, V

        if tagets is None:
            loss = None
        else:
            logits = einops.rearrange(logits, 'b t v -> (b t) v')
            tagets = einops.rearrange(tagets, 'b t -> (b t)')
            loss = self.criterion(logits, tagets.to(self.device))

        return logits, loss
    
    def generate(self, idx: torch.tensor,
                 max_tokens: int = settings.max_new_tokens) -> torch.tensor:
        
        with torch.no_grad():
            for _ in range(max_tokens):

                idx_cond = idx[:, -settings.block_size:]
                logits, _ = self.forward(idx_cond)
                logits = logits[:,-1,:]
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                idx = torch.cat([idx, next_token.cpu()], dim=1)
        return idx
    
    def make_optimizer(self, lr: float = 0.001) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=lr)

#load dataset
with open('dataset.txt', 'r') as f:
    data = f.read()

data = encode(data)
train = data[:int(settings.train_size*len(data))]
test = data[int(settings.train_size*len(data)):]
settings.block_size = 8

m = GPT2(enc.n_vocab)
print(m)

optimizer = m.make_optimizer(lr=settings.lr)

@torch.no_grad()
def eval_loss(model, data):

    out = {'train': [], 'test':[]}
    model.eval()
    for split in ['train', 'test']:
        for k in range(1000):
            x, y = get_batch(data[split])
            _, loss = model(x, y)
            out[split] += [loss.item()]
    
    out = {k: np.mean(v) for k, v in out.items()}
    model.train()

    return out

for i in tqdm(range(5001)):

    x, y = get_batch(train)
    
    optimizer.zero_grad()
    logits, loss = m(x, y)
    loss.backward()
    optimizer.step()

    if i % 1000 == 0:
        print(eval_loss(m, {'train': train, 'test': test}))
# %%

decode(m.generate(torch.tensor([[0]]))[0])
# %%
