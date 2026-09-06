import torch
import matplotlib.pyplot as plt



names = open("names.txt", "r").read().splitlines()
#print(len(names))
#print(names[:5])

b = {}
for name in names[:2]:
    chs = ["."] + list(name) + ["."]
    for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
        bigram = (ch1, ch2, ch3)
        b[bigram] = b.get(bigram, 0) + 1

print(sorted(b.items(), key = lambda kv: -kv[1]))



chars = sorted(list(set("".join(names))))


stoi = {}
for i, s in enumerate(chars):
    stoi[s] = i + 1
stoi["."] = 0

N = torch.zeros((27, 27,27), dtype=torch.int32)

for name in names:
    chs =["."] + list(name)+ ["."]
    for ch1, ch2,ch3 in zip(chs ,chs[1:],chs[2:]):
        ix1= stoi[ch1]
        ix2= stoi[ch2] 
        ix3= stoi[ch3] 
        N[ix1,ix2,ix3] += 1
        
itos = {i:s for s,i in stoi.items()}
print(N.shape)
print(len(chars))


N = N + 1
P = N.float()
P /= P.sum(dim=2, keepdim=True)

g= torch.Generator().manual_seed(216748647 +2)

for i in range(10):
    ix1 , ix2 = 0 , 0
    out= []
    while True:
        p= P[ix1,ix2]
        #p=N[ix].float()
        #p= p / p.sum()
        
        ix3= torch.multinomial(p, num_samples=1, replacement=True,generator=g).item()
        out.append(itos[ix3])
        if ix3 ==0:
            break
        ix1, ix2 = ix2, ix3
    print("".join(out))
    
    
    
n=0
loglikelihood = 0.0

for name in names:
    chs =["."] + list(name)+ ["."]
    for ch1, ch2,ch3 in zip(chs ,chs[1:],chs[2:]):
        ix1= stoi[ch1]
        ix2= stoi[ch2] 
        ix3= stoi[ch3] 
        prob = P[ix1,ix2,ix3]
        logprob= torch.log(prob)
        loglikelihood += logprob
        n += 1
        """print(f"{ch1}{ch2}: {prob:.4f}: {logprob:.4f}")"""
        
print(loglikelihood)
nll = - loglikelihood
print(-loglikelihood/n)






""""xs =[]
ys=[]
for name in names:
    chs =["."] + list(name)+ ["."]
    for ch1, ch2,ch3 in zip(chs ,chs[1:],chs[2:]):
        ix1= stoi[ch1]
        ix2= stoi[ch2] 
        ix3= stoi[ch3]
        xs.append((ix1,ix2))
        ys.append(ix3)
        
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()

g=torch.Generator().manual_seed(2167483647)
w= torch.randn((27,27,27), generator=g, requires_grad=True)


num = len(xs)
xenc = torch.zeros((num, 27, 27))
for idx, (ix1, ix2) in enumerate(xs):
    xenc[idx, ix1, ix2] = 1


for k in range(100): 
    num = len(xs)
    xenc = torch.zeros((num, 27, 27))
    for idx, (ix1, ix2) in enumerate(xs):
        xenc[idx, ix1, ix2] = 1
    logits = torch.einsum('bik,ikj->bj', xenc, w)

    counts= logits.exp()
    probs = counts /counts.sum(1,keepdims=True)

    loss = - probs[torch.arange(num), ys].log().mean()

    w.grad= None 
    loss.backward()

    learning_rate = 10
    w.data += - learning_rate * w.grad
    
    print(loss.item())"""


#print(logits)
#print(xenc.shape)
#print(logits.shape)