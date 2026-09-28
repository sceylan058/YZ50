import torch
from layers import Embedding, FlattenConsecutive, Linear, BatchNorm1d, Tanh, Sequential
 
 
# def build_model(vocab_size, block_size =8, n_embd=10, n_hidden=300, seed=42):
#   torch.manual_seed(seed)
#   model = Sequential([
#     Embedding(vocab_size, n_embd),
#     FlattenConsecutive(block_size), Linear(n_embd * block_size, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
#     Linear(n_hidden, vocab_size),
#   ])
#   with torch.no_grad():
#     model.layers[-1].weight *= 0.1
#   for p in model.parameters():
#     p.requires_grad = True
#   return model
# 32 80


 
def build_model(vocab_size, n_embd=10, n_hidden=128, seed=42):
  torch.manual_seed(seed)
  model = Sequential([
    Embedding(vocab_size, n_embd),
    FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(), 
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size),
  ])
  with torch.no_grad():
    model.layers[-1].weight *= 0.1
  for p in model.parameters():
    p.requires_grad = True
  return model
 
 
def set_training(model, training):
  for layer in model.layers:
    layer.training = training
    
    
    
if __name__ == '__main__':
  model = build_model(27, n_embd=10, n_hidden=128)
  x = torch.randint(0, 27, (32, 8))
  for layer in model.layers:
    x = layer(x)
    print(layer.__class__.__name__, x.shape)