

import torch
import tiktoken
from main import GPT, get_device



pairs = [
    ("The baker forgot the bread in the oven again", "Fırıncı ekmeği yine fırında unuttu"),
    ("My grandmother grows tomatoes on the balcony", "Babaannem balkonda domates yetiştiriyor"),
    ("The train to the mountains leaves at seven", "Dağlara giden tren yedide kalkıyor"),
]

device = get_device()
model = GPT.from_pretrained("gpt2")
model.eval()
model.to(device)
enc = tiktoken.get_encoding("gpt2")

for en, tr in pairs:
    for s in (en, tr):
        ids = torch.tensor(enc.encode(s), dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            _, loss = model(ids[:, :-1], ids[:, 1:])
        print(f"loss {loss.item():.3f} | {ids.size(1)} tokens | {s}")
    print()