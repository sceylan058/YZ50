
import math
import torch
import tiktoken
from main import GPT, GPTConfig, get_device

B, T = 4, 32
max_steps = 300
learning_rate = 3e-4

device = get_device()
torch.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")
with open("input.txt", "r") as f:
    text = f.read()
tokens = enc.encode(text[:1000])
buf = torch.tensor(tokens[:B * T + 1]).to(device)
x = buf[:-1].view(B, T)
y = buf[1:].view(B, T)

model = GPT(GPTConfig())
model.to(device)

logits, loss = model(x, y)
print(f"initial loss: {loss.item():.4f}, expected -ln(1/50257): {math.log(50257):.4f}")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for i in range(max_steps):
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    if i % 10 == 0 or i == max_steps - 1:
        print(f"step {i}, loss: {loss.item():.4f}")