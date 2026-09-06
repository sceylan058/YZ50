import torch
from bigram_counting import N, stoi, itos
from probability_sampling import P
names = open("names.txt", "r").read().splitlines()


n=0
loglikelihood = 0.0

for name in names:
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
print(-loglikelihood/n)