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

os.makedirs("plots/compare_geometry", exist_ok=True)

torch.set_grad_enabled(False)

duplicate_thresholds = [0.7, 0.8, 0.9, 0.95, 0.97, 0.99]

for take_max in [False, True]:
    for use_flop_matched in [False, True]:

        if use_flop_matched:
            experts = [1, 2, 4, 8]
            ks = [16, 32, 64, 128, 192]
        else:
            experts = [1, 16, 32, 64, 128]
            ks = [8, 16, 32, 48, 64, 96, 128, 192]

        device = "cuda:0"
        # device = "cpu"

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

        # Plot hists
        fig, axs = plt.subplots(len(experts), len(ks), figsize=(10, 10), sharex=True, sharey=True)
        for i, num_experts in enumerate(experts):
            for j, k in enumerate(ks):
                ax = axs[i, j]
                ax.hist(intra_sae_max_sims[(num_experts, k)].cpu(), bins=100)
                ax.set_title(f"{num_experts} experts, k={k}")

        plt.tight_layout()
        plt.savefig(f"plots/compare_geometry/intra_sae_max_sims_hist_{description_str}.png")
        plt.close()


        # Plot average max sim 
        image = []
        for i, num_experts in enumerate(experts):
            row = []
            for j, k in enumerate(ks):
                row.append(intra_sae_max_sims[(num_experts, k)].mean().item())
            image.append(row)
        plt.imshow(image)
        plt.colorbar()
        plt.xticks(np.arange(len(ks)), ks)
        plt.yticks(np.arange(len(experts)), experts)
        plt.xlabel("k")
        plt.ylabel("# experts")
        plt.title(f"Average intra-SAE max cosine similarity")
        plt.savefig(f"plots/compare_geometry/intra_sae_max_sims_{description_str}.png")
        plt.close()

        # Plots dupes
        if use_flop_matched:
            fig, axs = plt.subplots(len(duplicate_thresholds) // 2, 2, figsize=(10, 10), sharex=True, sharey=True)
        else:
            fig, axs = plt.subplots(len(duplicate_thresholds) // 2, 2, figsize=(10, 10), sharex=True, sharey=True)
        axs = axs.flatten()


        for i, duplicate_threshold in enumerate(duplicate_thresholds):
            to_display = np.array(im[duplicate_threshold]) / 24576   
            ax = axs[i]
            if take_max:
                ax.set_title(f"Frac features with nearest neighbor cos sim > {duplicate_threshold}")
            else:
                ax.set_title(f"Frac pairs with cos sim > {duplicate_threshold}")
                to_display = to_display / 24576

            if use_flop_matched:
                to_display /= np.array(experts)[:, None]
                if take_max:
                    to_display /= np.array(experts)[:, None]
                    
            image = ax.imshow(to_display, aspect="auto")
            plt.colorbar(image, ax=ax)


        plt.xticks(np.arange(len(ks)), ks)
        plt.yticks(np.arange(len(experts)), experts)
        fig.supxlabel("k")
        fig.supylabel("# experts")
        if use_flop_matched:
            fig.suptitle(f"Flop matched")
        else:
            fig.suptitle(f"Fixed width")
        plt.tight_layout()
        plt.savefig(f"plots/compare_geometry/feature_dupes_{description_str}_{'feature-wise' if take_max else 'feature-pair-wise'}.png")
        plt.close()

# %%


