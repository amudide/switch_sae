# %%
import os
import sys
from tqdm import tqdm
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from dictionary_learning.trainers.switch import SwitchAutoEncoder
from dictionary_learning.trainers.top_k import AutoEncoderTopK
import einops
import matplotlib.pyplot as plt
import torch
import numpy as np

torch.set_grad_enabled(False)

# device = "cuda:0"
device = "cpu"

# %%

def save_fig(ks, num_experts):

    global image_0
    assert len(ks) == 8

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
    fig, axs = plt.subplots(4, 4, figsize=(15, 15))

    # Left plots
    for i, k in enumerate(ks):

        if i < 4:
            ax0 = axs[0, i]
            ax1 = axs[2, i]
        elif i < 8:
            ax0 = axs[1, i - 4]
            ax1 = axs[3, i - 4]

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
        ax0.imshow(im, vmin=lower_lim, vmax=upper_lim)
        ax0.set_title(f"Switch SAE, k={k}")
        ax0.set_xlabel("Expert")
        ax0.set_ylabel("Expert")
        ax0.set_xticks(np.arange(0, num_experts, tick_jump))
        ax0.set_yticks(np.arange(0, num_experts, tick_jump))
        ax0.set_xticklabels(np.arange(0, num_experts, tick_jump))
        ax0.set_yticklabels(np.arange(0, num_experts, tick_jump))

        ax0.set_xticks(np.arange(-0.5, num_experts), minor=True)
        ax0.set_yticks(np.arange(-0.5, num_experts), minor=True)
        ax0.grid(color='w', linestyle='-', linewidth=line_width, which='minor')

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
        ax1.imshow(im, vmin=lower_lim, vmax=upper_lim)
        ax1.set_title(f"Normal SAE (arbitrary experts), k={k}")
        ax1.set_xlabel("Expert")
        ax1.set_ylabel("Expert")
        ax1.set_xticks(np.arange(0, num_experts, tick_jump))
        ax1.set_yticks(np.arange(0, num_experts, tick_jump))
        ax1.set_xticklabels(np.arange(0, num_experts, tick_jump))
        ax1.set_yticklabels(np.arange(0, num_experts, tick_jump))

        ax1.set_xticks(np.arange(-0.5, num_experts), minor=True)
        ax1.set_yticks(np.arange(-0.5, num_experts), minor=True)
        ax1.grid(color='w', linestyle='-', linewidth=line_width, which='minor')

    plt.suptitle(f"Average max cosine similarity between experts, {num_experts} experts", fontsize=16, y=1)
    plt.tight_layout()

    # Add vertical color bar with adjustable width
    fig.colorbar(axs[0][0].imshow(im, vmin=lower_lim, vmax=upper_lim), ax=axs, orientation='vertical', fraction=0.03)

    # Add a bold line between the first two rows and the last two rows
    line = plt.Line2D((0.05, 0.9), (0.49, 0.49), color='black', linewidth=5)
    fig.add_artist(line)

    plt.savefig(f"plots/compare_geometry/inter_expert_sims_{num_experts}.pdf", bbox_inches='tight')
    plt.close()

    # Plot just k = 64 and switch experts for the paper
    k = 64
    ae_switch = SwitchAutoEncoder.from_pretrained(f"../dictionaries/fixed-width/{num_experts}_experts/k{k}/ae.pt", k=k, experts=num_experts, device=device)
    
    switch_experts = einops.rearrange(ae_switch.decoder.data, "(experts n) d -> experts n d", experts=num_experts)
    normalized_switch_experts = switch_experts / switch_experts.norm(dim=-1, keepdim=True)
    
    
    image_0 = []
    for i, expert_i in enumerate(normalized_switch_experts):
        row = []
        for j, expert_j in enumerate(normalized_switch_experts):
            if i == j:
                row.append(np.nan)
                continue
            
            max_i_to_j = (expert_i @ expert_j.T).max(dim=-1).values
            average_max = max_i_to_j.mean()
            row.append(average_max.cpu().numpy())
        image_0.append(row)

experts = [16]
# experts = [16, 32, 64, 128]
ks = [8, 16, 32, 48, 64, 96, 128, 192]


for num_experts in experts:
    save_fig(ks, num_experts)


# %%


duplicate_thresholds = [0.9]

take_max = True
# for use_flop_matched in [False, True]:
for use_flop_matched in [False]:

    if use_flop_matched:
        experts = [1, 2, 4, 8]
        ks = [16, 32, 64, 128, 192]
    else:
        experts = [1, 16, 32, 64, 128]
        ks = [8, 16, 32, 48, 64, 96, 128, 192]

    im = {duplicate_threshold: [] for duplicate_threshold in duplicate_thresholds}

    intra_sae_max_sims = {}

    for num_experts in experts:
        for duplicate_threshold in duplicate_thresholds:
            im[duplicate_threshold].append([])

        for k in tqdm(ks):
            if num_experts == 1:
                ae = AutoEncoderTopK.from_pretrained(f"../dictionaries/topk/k{k}/ae.pt", k=k, device=device)
            elif use_flop_matched:
                ae = SwitchAutoEncoder.from_pretrained(f"../dictionaries/flop-matched/{num_experts}_experts/k{k}/ae.pt", k=k, experts=num_experts, device=device)
            else:
                ae = SwitchAutoEncoder.from_pretrained(f"../dictionaries/fixed-width/{num_experts}_experts/k{k}/ae.pt", k=k, experts=num_experts, device=device)


            normalized_weights = ae.decoder.data / ae.decoder.data.norm(dim=-1, keepdim=True)

            batch_size = 10000

            intra_sae_max_sims[(num_experts, k)] = []

            for duplicate_threshold in duplicate_thresholds:
                im[duplicate_threshold][-1].append(0)

            for i in range(0, len(normalized_weights), batch_size):
                batch = normalized_weights[i:i + batch_size]
                sims = normalized_weights @ batch.T
            
                # Zero what would be the diagonal in the full sims matrix
                for j in range(len(batch)):
                    sims[i + j, j] = 0 

                intra_sae_max_sims[(num_experts, k)].append(sims.max(dim=0).values)

                for duplicate_threshold in duplicate_thresholds:
                    if take_max:
                        num_duplicates = (sims > duplicate_threshold).max(dim = -1).values.sum()
                    else:
                        num_duplicates = (sims > duplicate_threshold).sum()
                    im[duplicate_threshold][-1][-1] += num_duplicates.item()

            intra_sae_max_sims[(num_experts, k)] = torch.cat(intra_sae_max_sims[(num_experts, k)])

    description_str = "flop_matched" if use_flop_matched else "fixed_width"

    # Plot average max sim 
    image_1 = []
    for i, num_experts in enumerate(experts):
        row = []
        for j, k in enumerate(ks):
            row.append(intra_sae_max_sims[(num_experts, k)].mean().item())
        image_1.append(row)


    # Plot single duplicate threshold 0.9 as single plot
    if take_max:
        to_display = np.array(im[0.9]) / 24576
        if not take_max:
            to_display = to_display / 24576
        
        if use_flop_matched:
            to_display /= np.array(experts)[:, None]
            if take_max:
                to_display /= np.array(experts)[:, None]
        
        image_2 = to_display

# %%


lower_lim = 0.2
upper_lim = 0.5
tick_jump = 1
num_experts = 16

# Add a small gap between cells (e.g., 5% of cell size)
gap = 0.05

# Create coordinates for pcolormesh
x = np.arange(0, num_experts + 1)
y = np.arange(0, num_experts + 1)
X, Y = np.meshgrid(x, y)

# Create a masked array to introduce gaps
masked_data = np.ma.masked_array(image_0, mask=False)

fig, ax = plt.subplots(figsize=(5.5/3, 2))
cax = ax.pcolormesh(X, Y, masked_data, vmin=lower_lim, vmax=upper_lim, edgecolors='w', linewidth=gap)

ax.set_xlabel("Expert", fontsize=8)
ax.set_ylabel("Expert", fontsize=8)

ax.set_xticks(np.arange(0.5, num_experts, tick_jump))
ax.set_yticks(np.arange(0.5, num_experts, tick_jump))
ax.set_xticklabels(np.arange(0, num_experts, tick_jump), fontsize=4, rotation=0)
ax.set_yticklabels(np.arange(0, num_experts, tick_jump), fontsize=4, rotation=0)

# Ensure the aspect ratio is equal and the plot is not distorted
ax.set_aspect('equal')

# Set limits to show full cells
ax.set_xlim(0, num_experts)
ax.set_ylim(0, num_experts)

bar = plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
# bar.set_label("Average max similarity", size=8)
bar.ax.tick_params(labelsize=6)

plt.savefig("plots/compare_geometry/inter_expert_sims_16.pdf", bbox_inches='tight', pad_inches=0.02)

# %%

experts = [1, 16, 32, 64, 128]

# Common parameters
figsize = (5.5/3, 2)
font_size_small = 6
font_size_medium = 6.5
font_size_large = 8
gap = 0.05

# Function to create a heatmap with gaps
def create_heatmap(ax, data, x_labels, y_labels, xlabel, ylabel, vmin=None, vmax=None):
    data_flipped = np.flipud(data)


    num_y, num_x = len(data_flipped), len(data_flipped[0])
    x = np.arange(num_x + 1)  # One more than data width
    y = np.arange(num_y + 1)  # One more than data height
    
    cax = ax.pcolormesh(x, y, data_flipped, vmin=vmin, vmax=vmax, edgecolors='w', linewidth=gap)
    
    ax.set_xlabel(xlabel, fontsize=font_size_large)
    ax.set_ylabel(ylabel, fontsize=font_size_large)
    
    ax.set_xticks(np.arange(0.5, num_x, 1))
    ax.set_yticks(np.arange(0.5, num_y, 1))
    ax.set_xticklabels(x_labels, fontsize=font_size_small, rotation=0)
    ax.set_yticklabels(y_labels[::-1], fontsize=font_size_small, rotation=0)
    
    ax.set_aspect('auto')
    ax.set_xlim(0, num_x)
    ax.set_ylim(0, num_y)
    
    return cax

# First plot
fig1, ax1 = plt.subplots(figsize=figsize)
cax1 = create_heatmap(ax1, image_1, ks, experts, "Sparsity (L0)", "# Experts")
bar1 = plt.colorbar(cax1, ax=ax1, fraction=0.046, pad=0.04)
bar1.ax.tick_params(labelsize=font_size_medium)
plt.savefig("plots/compare_geometry/averaeg_inter_SAE_sim.pdf", bbox_inches='tight', pad_inches=0.02)

# Second plot
fig2, ax2 = plt.subplots(figsize=figsize)
cax2 = create_heatmap(ax2, image_2, ks, experts, "Sparsity (L0)", "# Experts")
bar2 = plt.colorbar(cax2, ax=ax2, fraction=0.046, pad=0.04)
bar2.ax.tick_params(labelsize=font_size_medium)
plt.savefig("plots/compare_geometry/frac_nns_greater_than_0.9.pdf", bbox_inches='tight', pad_inches=0.02)

# %%
