import regex as re

split_pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
vocab_size = 276
num_merges = vocab_size - 256

f = open("turkce_ham.txt", "r", encoding="utf-8")
text = f.read()
f.close()

chunks = re.findall(split_pattern, text)
print(chunks[:10])

def get_stats(ids, counts=None):
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

def train(chunk_list, num_merges):
    chunk_list = [list(c) for c in chunk_list]
    merges = {}
    for i in range(num_merges):
        stats = {}
        for c in chunk_list:
            get_stats(c, stats)
        pair = max(stats, key=stats.get)
        idx = 256 + i
        chunk_list = [merge(c, pair, idx) for c in chunk_list]
        merges[pair] = idx
    return merges

def build_vocab(merges):
    vocab = {i: bytes([i]) for i in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
    return vocab

merges_plain = train([text.encode("utf-8")], num_merges)
chunks = re.findall(split_pattern, text)
merges_regex = train([c.encode("utf-8") for c in chunks], num_merges)

vocab_plain = build_vocab(merges_plain)
vocab_regex = build_vocab(merges_regex)

for idx in range(256, vocab_size):
    a = vocab_plain[idx].decode("utf-8", errors="replace")
    b = vocab_regex[idx].decode("utf-8", errors="replace")
    print(idx, repr(a).ljust(14), repr(b))

print(re.findall(split_pattern, "Ankara'dan"))
print(re.findall(split_pattern, "İstanbul'da"))