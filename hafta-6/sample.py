
import torch
import torch.nn.functional as F
 
from dataset import get_splits
from model import set_training
 
BLOCK_SIZE = 8
 
_, _, _, stoi, itos = get_splits(block_size=BLOCK_SIZE)
model = torch.load('model.pt', weights_only=False)
set_training(model, False)
 
for _ in range(20):
  out = []
  context = [0] * BLOCK_SIZE
  while True:
    logits = model(torch.tensor([context]))
    probs = F.softmax(logits, dim=1)
    ix = torch.multinomial(probs, num_samples=1).item()
    context = context[1:] + [ix]
    out.append(ix)
    if ix == 0:
      break
  print(''.join(itos[i] for i in out))