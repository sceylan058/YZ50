import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel

model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
sd_hf = model_hf.state_dict()

for k, v in sd_hf.items():
    print(k, tuple(v.shape))

n_params = sum(p.numel() for p in model_hf.parameters())
print(f"total parameters: {n_params:,}")

wpe = sd_hf["transformer.wpe.weight"]

plt.figure()
plt.imshow(wpe, cmap="gray")
plt.title("wpe (1024 x 768)")

plt.figure()
plt.plot(wpe[:, 150])
plt.plot(wpe[:, 200])
plt.plot(wpe[:, 250])
plt.title("wpe columns 150, 200, 250")

plt.show()