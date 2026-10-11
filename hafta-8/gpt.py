import torch, time
import torch.nn as nn
from torch.nn import functional as F

MODE = 'big'          # 'ablation' or 'big'
data_file = 'turkce_ham.txt'
batch_size = 64
block_size = 128
n_embd = 256
n_head = 4
n_layer = 4
dropout = 0.2
learning_rate = 3e-4
max_iters = 5000
eval_interval = 500
eval_iters = 200
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

text = open(data_file, encoding='utf-8').read()
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
print(f"{len(text):,} chars, vocab_size={vocab_size}")

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k, q = self.key(x), self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = self.dropout(F.softmax(wei, dim=-1))
        return wei @ self.value(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, proj=True):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd) if proj else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, wide=True):
        super().__init__()
        if wide:
            self.net = nn.Sequential(nn.Linear(n_embd, 4*n_embd), nn.ReLU(),
                                     nn.Linear(4*n_embd, n_embd), nn.Dropout(dropout))
        else:
            self.net = nn.Sequential(nn.Linear(n_embd, n_embd), nn.ReLU())

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_head, ffwd=True, res=True, ln=True):
        super().__init__()
        self.res = res
        self.sa = MultiHeadAttention(n_head, n_embd // n_head, proj=res)
        self.ffwd = FeedForward(wide=res) if ffwd else None
        self.ln1 = nn.LayerNorm(n_embd) if ln else nn.Identity()
        self.ln2 = nn.LayerNorm(n_embd) if ln else nn.Identity()

    def forward(self, x):
        if self.res:
            x = x + self.sa(self.ln1(x))
            if self.ffwd: x = x + self.ffwd(self.ln2(x))
        else:
            x = self.sa(self.ln1(x))
            if self.ffwd: x = self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, n_head, n_layer, ffwd=True, res=True, ln=True):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_head, ffwd, res, ln) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) if ln else nn.Identity()
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx) + self.position_embedding_table(torch.arange(T, device=device))
        logits = self.lm_head(self.ln_f(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B*T, -1), targets.view(B*T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -block_size:])
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
        return idx

def train(model, lr, iters):
    model = model.to(device)
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    t0 = time.time()
    for it in range(iters):
        if it % eval_interval == 0 or it == iters - 1:
            l = estimate_loss(model)
            print(f"step {it}: train {l['train']:.4f}, val {l['val']:.4f} ({time.time()-t0:.0f}s)")
        xb, yb = get_batch('train')
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model

if MODE == 'ablation':
    block_size, batch_size, n_embd, dropout = 8, 32, 32, 0.0
    for name, kw in [
        ("single head",   dict(n_head=1, n_layer=1, ffwd=False, res=False, ln=False)),
        ("multi-head",    dict(n_head=4, n_layer=1, ffwd=False, res=False, ln=False)),
        ("+ feedforward", dict(n_head=4, n_layer=1, ffwd=True,  res=False, ln=False)),
        ("3 blocks",      dict(n_head=4, n_layer=3, ffwd=True,  res=False, ln=False)),
        ("+ residual",    dict(n_head=4, n_layer=3, ffwd=True,  res=True,  ln=False)),
        ("+ LayerNorm",   dict(n_head=4, n_layer=3, ffwd=True,  res=True,  ln=True)),
    ]:
        torch.manual_seed(1337)
        print("\n==", name)
        train(GPT(**kw), 1e-3, 5000)
else:
    torch.manual_seed(1337)
    model = train(GPT(n_head, n_layer), learning_rate, max_iters)
    torch.save(model.state_dict(), 'model.pt')
    ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(ctx, 1000)[0].tolist()))