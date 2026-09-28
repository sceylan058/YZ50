import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
 
from dataset import get_splits
from model import build_model, set_training
 
BLOCK_SIZE = 8
MAX_STEPS = 200000
BATCH_SIZE = 32
 
(Xtr, Ytr), (Xdev, Ydev), (Xte, Yte), stoi, itos = get_splits(block_size=BLOCK_SIZE)
vocab_size = len(itos)
print(Xtr.shape, Ytr.shape)
print(Xdev.shape, Ydev.shape)
 
model = build_model(vocab_size)
parameters = model.parameters()
print(sum(p.nelement() for p in parameters))
 
 
torch.manual_seed(0)
lossi = []
for i in range(MAX_STEPS):
  ix = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,))
  Xb, Yb = Xtr[ix], Ytr[ix]
 
  logits = model(Xb)
  loss = F.cross_entropy(logits, Yb)
 
  for p in parameters:
    p.grad = None
  loss.backward()
 
  lr = 0.1 if i < 150000 else 0.01
  for p in parameters:
    p.data += -lr * p.grad
 
  if i % 10000 == 0:
    print(f'{i:7d}/{MAX_STEPS:7d}: {loss.item():.4f}')
  lossi.append(loss.log10().item())
 
plt.plot(torch.tensor(lossi).view(-1, 1000).mean(1))
plt.savefig('loss.png')
 
set_training(model, False)
 
 
@torch.no_grad()
def split_loss(split):
  x, y = {'train': (Xtr, Ytr), 'val': (Xdev, Ydev), 'test': (Xte, Yte)}[split]
  logits = model(x)
  loss = F.cross_entropy(logits, y)
  print(split, loss.item())
 
 
split_loss('train')
split_loss('val')
 
torch.save(model, 'model.pt')