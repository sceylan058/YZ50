import torch
import tiktoken
from torch.nn import functional as F
from main import GPT, get_device

num_return_sequences = 5
max_new_tokens = 30
top_k = 50

en = "The old lighthouse keeper opened his notebook and began to write"
tr = "Yaşlı deniz feneri bekçisi defterini açtı ve yazmaya başladı"

device = get_device()
model = GPT.from_pretrained("gpt2")
model.eval()
model.to(device)
enc = tiktoken.get_encoding("gpt2")


def generate(prompt):
    tokens = enc.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).repeat(num_return_sequences, 1).to(device)
    torch.manual_seed(42)
    while x.size(1) < len(tokens) + max_new_tokens:
        with torch.no_grad():
            logits, _ = model(x)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)
            x = torch.cat((x, xcol), dim=1)
    for i in range(num_return_sequences):
        print(">", enc.decode(x[i].tolist()))


generate(en)
print()
generate(tr)
print()

enc4 = tiktoken.get_encoding("cl100k_base")
for s in (en, tr):
    ids = enc.encode(s)
    print(f"gpt2: {len(ids)} tokens, cl100k: {len(enc4.encode(s))} tokens, {len(s)} chars, {len(s) / len(ids):.2f} chars/token")
    print([enc.decode([t]) for t in ids])