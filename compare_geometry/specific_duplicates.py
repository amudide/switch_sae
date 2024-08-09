# %%

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from dictionary_learning.trainers.switch import SwitchAutoEncoder
import einops
import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
import os
from sae_lens import SAE
import urllib.parse
import json
import webbrowser


os.makedirs("plots/compare_geometry", exist_ok=True)

torch.set_grad_enabled(False)

# %%

experts = [16, 32, 64, 128]
ks = [8, 16, 32, 48, 64, 96, 128, 192]

device = "cuda:1"
# device = "cpu"

threshold = 0.9

intra_sae_max_sims = {}

fig, axs = plt.subplots(len(experts), len(ks), figsize=(20, 10))

for i, num_experts in enumerate(experts):

    for j, k in enumerate(tqdm(ks)):
        ax = axs[i, j]

        ae = SwitchAutoEncoder.from_pretrained(f"../dictionaries/fixed-width/{num_experts}_experts/k{k}/ae.pt", k=k, experts=num_experts, device=device)

        normalized_weights = ae.encoder.weight / ae.encoder.weight.norm(dim=-1, keepdim=True)

        sims = normalized_weights @ normalized_weights.T

        sims.fill_diagonal_(0)

        num_dupes = (sims > threshold).sum(dim=-1).cpu().numpy()

        ax.hist([i for i in num_dupes if i != 0], bins=100)

        ax.set_title(f"{num_experts} experts, k={k}")


fig.suptitle(f"Number of duplicates per feature, threshold={threshold}")

fig.supxlabel("Number of duplicates")

fig.supylabel("Number of features")

plt.tight_layout()

plt.savefig("plots/compare_geometry/num_duplicates_per_feature_fixed_width.png")


# %%


def get_jb_sae(layer, device):
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
        sae_id=f"blocks.{layer}.hook_resid_pre",  # won't always be a hook point
        device=device,
    )
    return sae

def open_neuronpedia(features: list[int], layer: int, name: str = "temporary_list"):
    url = "https://neuronpedia.org/quick-list/"
    name = urllib.parse.quote(name)
    url = url + "?name=" + name
    list_feature = [
        {"modelId": "gpt2-small", "layer": f"{layer}-res-jb", "index": str(feature)}
        for feature in features
    ]
    url = url + "&features=" + urllib.parse.quote(json.dumps(list_feature))
    webbrowser.open(url)


jb_sae = get_jb_sae(layer=8, device=device)
normalized_jb_weights = jb_sae.W_dec.data / jb_sae.W_dec.data.norm(dim=-1, keepdim=True)



k = 32
num_experts = 64

ae = SwitchAutoEncoder.from_pretrained(f"../dictionaries/fixed-width/{num_experts}_experts/k{k}/ae.pt", k=k, experts=num_experts, device=device)

normalized_weights = ae.encoder.weight / ae.encoder.weight.norm(dim=-1, keepdim=True)

sims = normalized_weights @ normalized_weights.T

sims.fill_diagonal_(0)

num_dupes = (sims > threshold).sum(dim=-1).cpu().numpy()

plt.hist([i for i in num_dupes if i != 0], bins=100)
plt.show()

# Sort by number of duplicates
sorted_indices = np.argsort(num_dupes)[::-1]

cutoff = num_experts // 2

most_similar_to_jb = (normalized_weights @ normalized_jb_weights.T).max(dim=0).values
plt.hist(most_similar_to_jb.cpu().numpy(), bins=100)
plt.show()

average_sim = most_similar_to_jb.mean().item()
print(f"Average cosine similarity to JB SAE: {average_sim}")


for i in sorted_indices:
    if num_dupes[i] < cutoff:
        break

    print(f"Feature {i} has {num_dupes[i]} duplicates")

    jb_feature_sims = (normalized_weights[i] @ normalized_jb_weights.T)

    top_k = 10

    top_k_result = torch.topk(jb_feature_sims, top_k)

    top_k_indices = top_k_result.indices.cpu().numpy()
    top_k_values = top_k_result.values.cpu().numpy()

    open_neuronpedia(top_k_indices, layer=8, name=f"feature {i} ({num_dupes[i]} dupes) most similar: {top_k_values}")

    # Pause for user input, if q or esq then quit
    if input("Press enter to continue, or q to quit") in ["q", "esq"]:
        break
