

f = open("turkce_ham.txt", "r", encoding="utf-8")
text = f.read()
f.close()

#text =  "Kim ne derse desin aşk için önce hoş sonra boş gelir Her seferinde canım yanar Aşk bana yalan gelir"

tokens = text.encode("utf-8")
tokens = list(map(int, tokens))

#print(tokens)

#for c in text:
    #print(c, ord(c), list(c.encode("utf-8")))
    
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1 
    return counts 
    
stat_trak = get_stats(tokens)
#print(stat_trak)
#print(sorted(((v,k) for k,v in stat_trak.items()), reverse=True))
print()

top_pair = max(stat_trak, key=stat_trak.get)

def merge(ids, pair, idx):
    newids=[]
    i=0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i+=1
    return newids
    
vocab_size = 276
num_merges = vocab_size - 256      
ids = list(tokens)                  
merges = {}     
    
for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)
    idx = 256 + i
    print(f"merging {pair} into a new token {idx}")
    ids = merge(ids, pair, idx)
    merges[pair] = idx
    
print("before:", len(tokens))
print("after:", len(ids))
print("rate:", len(tokens) / len(ids))


vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]
    
for (p0, p1), idx in merges.items():
    print(idx, repr(vocab[idx].decode("utf-8", errors="replace")))
    
def decode(ids):
    tokens = b"".join(vocab[idx] for idx in ids)
    return tokens.decode("utf-8", errors="replace")
    
# print(vocab[101])              
# print(vocab[256])            
# print(decode([104, 105]))      
# print(decode([159]))           


def encode(text):
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens
    
    
ornek = "Kim ne derse desin aşk için önce hoş sonra boş gelir Her seferinde canım yanar Aşk bana yalan gelir"
print(decode(encode(ornek)) == ornek)



own_tokens = encode(ornek)
print(len(ornek.encode("utf-8")), len(own_tokens), len(own_tokens) / len(ornek.split()))