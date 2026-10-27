import torch
import tiktoken
from transformers import GPT2LMHeadModel
from main import GPT

model = GPT.from_pretrained("gpt2")
model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()
model_hf.eval()

enc = tiktoken.get_encoding("gpt2")
x = torch.tensor(enc.encode("A small robot was painting the moon on a quiet Tuesday"), dtype=torch.long).unsqueeze(0)

with torch.no_grad():
    logits, _ = model(x)
    logits_hf = model_hf(x).logits

print(logits.shape, logits_hf.shape)
print("max abs diff:", (logits - logits_hf).abs().max().item())
print("allclose:", torch.allclose(logits, logits_hf, atol=1e-4))