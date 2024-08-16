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

torch.set_grad_enabled(False)

device = "cuda:0"

# %%

def save_fig(ks, num_experts):

    lower_lim = 0.2

    experts_to_upper_lim = {
        16: 0.5,
        32: 0.55,
        64: 0.6,
        128: 0.6
    }
    upper_lim = experts_to_upper_lim[num_experts]

    experts_to_line_width = {
        16: 2,
        32: 1.5,
        64: 0.5
    }
    line_width = experts_to_line_width[num_experts] if num_experts in experts_to_line_width else 0

    tick_jump = num_experts // 16

    # Create two imshow columns as subplots, one with normalized_switch_experts and one with normalized_normal_experts
    fig, axs = plt.subplots(len(ks), 2, figsize=(10, 5 * len(ks)))

    # Left plots
    for i, k in enumerate(ks):

        ax = axs[i]

        ae_switch = SwitchAutoEncoder.from_pretrained(f"../dictionaries/fixed-width/{num_experts}_experts/k{k}/ae.pt", k=k, experts=num_experts, device=device)

        ae_normal = AutoEncoderTopK.from_pretrained(f"../dictionaries/topk/k{k}/ae.pt", device=device)

        switch_experts = einops.rearrange(ae_switch.decoder.data, "(experts n) d -> experts n d", experts=num_experts)
        normal_experts = einops.rearrange(ae_normal.decoder.data, "(experts n) d -> experts n d", experts=num_experts)

        normalized_switch_experts = switch_experts / switch_experts.norm(dim=-1, keepdim=True)
        normalized_normal_experts = normal_experts / normal_experts.norm(dim=-1, keepdim=True)


        im = []
        for i, expert_i in enumerate(normalized_switch_experts):
            row = []
            for j, expert_j in enumerate(normalized_switch_experts):
                if i == j:
                    row.append(np.nan)
                    continue

                max_i_to_j = (expert_i @ expert_j.T).max(dim=-1).values
                average_max = max_i_to_j.mean()
                row.append(average_max.cpu().numpy())
            im.append(row)
        ax[0].imshow(im, vmin=lower_lim, vmax=upper_lim)
        ax[0].set_title(f"Switch SAE, k={k}")
        ax[0].set_xlabel("Expert")
        ax[0].set_ylabel("Expert")
        ax[0].set_xticks(np.arange(0, num_experts, tick_jump))
        ax[0].set_yticks(np.arange(0, num_experts, tick_jump))
        ax[0].set_xticklabels(np.arange(0, num_experts, tick_jump))
        ax[0].set_yticklabels(np.arange(0, num_experts, tick_jump))

        ax[0].set_xticks(np.arange(-0.5, num_experts), minor=True)
        ax[0].set_yticks(np.arange(-0.5, num_experts), minor=True)
        ax[0].grid(color='w', linestyle='-', linewidth=line_width, which='minor')

        # Right plot
        im = []
        for i, expert_i in enumerate(normalized_normal_experts):
            row = []
            for j, expert_j in enumerate(normalized_normal_experts):
                if i == j:
                    row.append(np.nan)
                    continue

                max_i_to_j = (expert_i @ expert_j.T).max(dim=-1).values
                average_max = max_i_to_j.mean()
                row.append(average_max.cpu().numpy())
            im.append(row)
        im = np.array(im)
        ax[1].imshow(im, vmin=lower_lim, vmax=upper_lim)
        ax[1].set_title(f"Normal SAE, arbitrarily broken into experts, k={k}")
        ax[1].set_xlabel("Expert")
        ax[1].set_ylabel("Expert")
        ax[1].set_xticks(np.arange(0, num_experts, tick_jump))
        ax[1].set_yticks(np.arange(0, num_experts, tick_jump))
        ax[1].set_xticklabels(np.arange(0, num_experts, tick_jump))
        ax[1].set_yticklabels(np.arange(0, num_experts, tick_jump))

        ax[1].set_xticks(np.arange(-0.5, num_experts), minor=True)
        ax[1].set_yticks(np.arange(-0.5, num_experts), minor=True)
        ax[1].grid(color='w', linestyle='-', linewidth=line_width, which='minor')

    plt.suptitle(f"Average max cosine similarity between experts, {num_experts} experts", fontsize=16, y=1)
    plt.tight_layout()

    # Add vertical color bar with adjustable width
    fig.colorbar(axs[0][0].imshow(im, vmin=lower_lim, vmax=upper_lim), ax=axs, orientation='vertical', fraction=0.05)

    # plt.savefig(f"plots/compare_geometry/inter_expert_sims_{num_experts}.png", bbox_inches='tight')
    # plt.close()

    plt.show()

# experts = [16, 32, 64, 128]
experts = [16]
ks = [8, 16, 32, 48, 64, 96, 128, 192]

for num_experts in experts:
    save_fig(ks, num_experts)

# %%

# We don't save any of these plots below for now, they show histograms instead of averages
exit(0)

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
        axs[i, j].set_title(f"Expert {i} to Expert {j}")

plt.tight_layout()
plt.title("Max cos sims between experts")
plt.show()

# %%


