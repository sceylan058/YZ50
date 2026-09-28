import random
import torch
 
 
def load_words(path='names.txt'):
  return open(path, 'r').read().splitlines()
 
 
def build_vocab(words):
  chars = sorted(list(set(''.join(words))))
  stoi = {s: i + 1 for i, s in enumerate(chars)}
  stoi['.'] = 0
  itos = {i: s for s, i in stoi.items()}
  return stoi, itos
 
 
def build_dataset(words, stoi, block_size):
  X, Y = [], []
  for w in words:
    context = [0] * block_size
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      context = context[1:] + [ix]
  X = torch.tensor(X)
  Y = torch.tensor(Y)
  return X, Y
 
 
def get_splits(path='names.txt', block_size=8, seed=42):
  words = load_words(path)
  stoi, itos = build_vocab(words)
  random.seed(seed)
  random.shuffle(words)
  n1 = int(0.8 * len(words))
  n2 = int(0.9 * len(words))
  Xtr, Ytr = build_dataset(words[:n1], stoi, block_size)
  Xdev, Ydev = build_dataset(words[n1:n2], stoi, block_size)
  Xte, Yte = build_dataset(words[n2:], stoi, block_size)
  return (Xtr, Ytr), (Xdev, Ydev), (Xte, Yte), stoi, itos