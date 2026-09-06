from bigram_counting import N, stoi, itos
import torch

"""p = N[0].float()
p /= p.sum()
print(p)"""

"""g= torch.Generator().manual_seed(2147483647)
p= torch.rand(3,generator=g)
p= p/p.sum()
print(p)"""

P = N.float()
P /= P.sum(dim=1, keepdim=True)

g= torch.Generator().manual_seed(216748647 +3)

for i in range(10):
    ix= 0
    out= []
    while True:
        p= P[ix]
        #p=N[ix].float()
        #p= p / p.sum()
        
        ix= torch.multinomial(p, num_samples=1, replacement=True,generator=g).item()
        out.append(itos[ix])
        if ix ==0:
            break
    print("".join(out))
    