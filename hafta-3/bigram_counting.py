import torch
import matplotlib.pyplot as plt



names = open("names.txt", "r").read().splitlines()
#print(len(names))
#print(names[:5])

b = {}
for name in names:
    chs = ["."] + list(name) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1

#print(sorted(b.items(), key = lambda kv: -kv[1]))

N =torch.zeros((27,27),dtype = torch.int32)

chars = sorted(list(set("".join(names))))

print(chars)

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

#print(len(chars))
#print(chars)
#print(N)

        


"""plt.figure(figsize=(12,12))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')

plt.axis('off');
plt.show()"""