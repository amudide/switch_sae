# %%

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from dictionary_learning.trainers.switch import SwitchAutoEncoder
from dictionary_learning.trainers.top_k import AutoEncoderTopK
import einops
import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
import os
import pandas as pd


os.makedirs("plots/compare_geometry", exist_ok=True)

torch.set_grad_enabled(False)


experts = [1, 16, 32, 64, 128]
ks = [8, 16, 32, 48, 64, 96, 128, 192]

device = "cuda:1"
# device = "cpu"

threshold = 0.9


intra_sae_max_sims = {}

fig, axs = plt.subplots(len(experts), len(ks), figsize=(20, 10))

data = []

for i, num_experts in enumerate(experts):

    for j, k in enumerate(tqdm(ks)):
        ax = axs[i, j]

        if num_experts == 1:
            ae = AutoEncoderTopK.from_pretrained(f"../dictionaries/topk/k{k}/ae.pt", k=k, device=device)
        else:
            ae = SwitchAutoEncoder.from_pretrained(f"../dictionaries/fixed-width/{num_experts}_experts/k{k}/ae.pt", k=k, experts=num_experts, device=device)

        normalized_weights = ae.decoder.data / ae.decoder.data.norm(dim=-1, keepdim=True)

        sims = normalized_weights @ normalized_weights.T

        sims.fill_diagonal_(0)

        num_dupes = (sims > threshold).sum(dim=-1).cpu().numpy()

        ax.hist([i for i in num_dupes if i != 0], bins=100)

        ax.set_title(f"{num_experts} experts, k={k}")

        for feature_index, num_dupes in enumerate(num_dupes):
            if num_dupes == 0:
                continue
            data.append((num_experts, k, feature_index, num_dupes))

fig.suptitle(f"Number of duplicates per feature, threshold={threshold}")

fig.supxlabel("Number of duplicates")

fig.supylabel("Number of features")

plt.tight_layout()

plt.savefig("plots/compare_geometry/num_duplicates_per_feature_fixed_width.pdf")

df = pd.DataFrame(data, columns=["num_experts", "k", "feature_index", "num_dupes"])

# Sort by num_experts, then k, then num_dupes in descending order
df = df.sort_values(by=["num_experts", "k", "num_dupes"], ascending=[True, True, False])

os.makedirs("../data", exist_ok=True)
df.to_csv("../data/duplicates.csv", index=False)
