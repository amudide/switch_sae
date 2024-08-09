# %%
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from dictionary_learning.trainers.switch import SwitchAutoEncoder
import einops
import matplotlib.pyplot as plt
import torch
import numpy as np

torch.set_grad_enabled(False)



k = 128
num_experts = 16
device = "cuda:0"

ae = SwitchAutoEncoder.from_pretrained(f"../dictionaries/fixed-width/{num_experts}_experts/k{k}/ae.pt", k=k, experts=num_experts, device=device)

print(ae.encoder.weight.shape)
vectors_per_expert = len(ae.encoder.weight) // num_experts
experts = einops.rearrange(ae.encoder.weight, "(experts n) d -> experts n d", experts=num_experts)
# %%

normalized_experts = experts / experts.norm(dim=-1, keepdim=True)
# %%

im = []
for i, expert_i in enumerate(normalized_experts):
    row = []
    for j, expert_j in enumerate(normalized_experts):
        if i == j:
            row.append(np.nan)
            continue

        max_i_to_j = (expert_i @ expert_j.T).max(dim=-1).values
        average_max = max_i_to_j.mean()
        row.append(average_max.cpu().numpy())
    im.append(row)

plt.imshow(im)
plt.colorbar()
# %%

fig, axs = plt.subplots(len(normalized_experts) // 4, 4, figsize=(20, 20))
axs = axs.flatten()

for i, expert_i in enumerate(normalized_experts):
    maxes = None
    for j, expert_j in enumerate(normalized_experts):
        if i == j:
            row.append(np.nan)
            continue

        max_i_to_j = (expert_i @ expert_j.T).max(dim=-1).values
        if maxes is None:
            maxes = max_i_to_j
        else:
            maxes = torch.max(maxes, max_i_to_j)
    axs[i].hist(maxes.cpu().numpy(), bins=np.arange(0, 1, 0.025))
    axs[i].set_title(f"Expert {i}")

fig.suptitle("Max cosine similarity of expert dictionary elements to any other expert's dictionary element")

plt.tight_layout()

plt.show()


# %%

fig, axs = plt.subplots(num_experts // 4, 4, figsize=(20, 20))
axs = axs.flatten()

for i, expert_i in enumerate(normalized_experts):
    maxes = None
    for j, expert_j in enumerate(normalized_experts):
        if i == j:
            row.append(np.nan)
            continue

        max_i_to_j = (expert_i @ expert_j.T).max(dim=-1).values
        if maxes is None:
            maxes = max_i_to_j
        else:
            maxes = torch.min(maxes, max_i_to_j)
    axs[i].hist(maxes.cpu().numpy(), bins=np.arange(0.5, 1, 0.025))
    axs[i].set_title(f"Expert {i}")

fig.suptitle("Max cosine similarity of expert dictionary elements to any other expert's dictionary element")

plt.tight_layout()

plt.show()

# %%

num_experts_limit = 6

fig, axs = plt.subplots(num_experts_limit, num_experts_limit, figsize=(20, 20))

for i, expert_i in enumerate(normalized_experts[:num_experts_limit]):
    for j, expert_j in enumerate(normalized_experts[:num_experts_limit]):
        if i == j:
            continue
        max_i_to_j = (expert_i @ expert_j.T).max(dim=-1).values
        axs[i, j].hist(max_i_to_j.cpu().numpy(), bins=100)

plt.show()
# %%


