import torch
from bigram_counting import N, stoi, itos
import torch.nn.functional as F

names = open("names.txt", "r").read().splitlines()

xs =[]
ys=[]
for name in names:
    chs =["."] + list(name)+ ["."]
    for ch1, ch2 in zip(chs ,chs[1:]):
        ix1= stoi[ch1]
        ix2= stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)
        
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()


g=torch.Generator().manual_seed(2167483647)
w= torch.randn((27,27), generator=g, requires_grad=True)


for k in range(1): 
    xenc = F.one_hot(xs,num_classes=27).float()
    logits = xenc @ w

    counts= logits.exp()
    probs = counts /counts.sum(1,keepdims=True)

    loss = - probs[torch.arange(num), ys].log().mean()

    w.grad= None 
    loss.backward()

    learning_rate = 10
    w.data += - learning_rate * w.grad
    
#print(loss.item())


#print(logits)
#print(xenc.shape)
#print(logits.shape)

        
        