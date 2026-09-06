import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F



raw_data = open("turkish_names.txt", "r").read().splitlines()

tr_names = []
for line in raw_data:
    row=line.split()
    lowered = [name.replace("İ", "i").replace("I", "ı").lower() for name in row]
    tr_names.extend(lowered)

tr_names = list(set(tr_names))
    
f = open("turkish_names.txt", "w")
for name in tr_names:
    f.write(name + "\n")
f.close()

#print(len(tr_names))
#print(names[:5])

names = open("turkish_names.txt", "r").read().splitlines()


N =torch.zeros((30,30),dtype = torch.int32)

turkish_alphabet = "abcçdefgğhıijklmnoöprsştuüvyz"
chars = sorted(set("".join(names)), key=lambda c: turkish_alphabet.index(c))

#chars = sorted(list(set("".join(names))))

stoi = {}
for i, s in enumerate(chars):
    stoi[s] = i + 1
stoi["."] = 0


for name in names:
    chs =["."] + list(name)+ ["."]
    for ch1, ch2 in zip(chs ,chs[1:]):
        ix1= stoi[ch1]
        ix2= stoi[ch2] 
        N[ix1,ix2] += 1
        
itos = {i:s for s,i in stoi.items()}

#print(chars)

#print(len(chars))
    
#print(len(names))
#print(len(set(names)))

#print(N.sum().item())

"""plt.figure(figsize=(12,12))
plt.imshow(N, cmap='Blues')
for i in range(30):
    for j in range(30):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')

plt.axis('off');
plt.show()"""







P = N.float()
P /= P.sum(dim=1, keepdim=True)

g= torch.Generator().manual_seed(216748647)

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








n=0
loglikelihood = 0.0

for name in names[:2]:
    chs =["."] + list(name)+ ["."]
    for ch1, ch2 in zip(chs ,chs[1:]):
        ix1= stoi[ch1]
        ix2= stoi[ch2]
        prob = P[ix1,ix2]
        logprob= torch.log(prob)
        loglikelihood += logprob
        n += 1
        print(f"{ch1}{ch2}: {prob:.4f}: {logprob:.4f}")
        
print(loglikelihood)
nll = - loglikelihood
print(nll/n)













"""xs =[]
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
w= torch.randn((30,30), generator=g, requires_grad=True)
for k in range(100): 
    xenc = F.one_hot(xs,num_classes=30).float()
    logits = xenc @ w

    counts= logits.exp()
    probs = counts /counts.sum(1,keepdims=True)

    loss = - probs[torch.arange(num), ys].log().mean()

    w.grad= None 
    loss.backward()

    learning_rate = 50
    w.data += - learning_rate * w.grad
    
    print(loss.item())


print(logits)
print(xenc.shape)
print(logits.shape)"""


        