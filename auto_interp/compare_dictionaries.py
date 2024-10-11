# %%

import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import FixedLocator, FixedFormatter

flop_matched_color_dict = {
    1: "#9467bd",
    2: "#ff7f0e",
    4: "#ff6600",
    8: "#ff3300",
}

fixed_width_color_dict = {
    1: "#9467bd",
    16: "#ff7f0e",
    32: "#ff6600",
    64: "#ff4d00",
    128: "#ff3300",
}

dictionaries = [
    "dictionaries/topk/k64",
    "dictionaries/fixed-width/16_experts/k64",
    "dictionaries/fixed-width/32_experts/k64",
    "dictionaries/fixed-width/64_experts/k64",
    "dictionaries/fixed-width/128_experts/k64",
    "dictionaries/flop-matched/2_experts/k64",
    "dictionaries/flop-matched/4_experts/k64",
    "dictionaries/flop-matched/8_experts/k64",
]
results = pickle.load(open("results.pkl", "rb"))

def confidence_interval(successes, total, z=1.96):
    p = successes / total
    se = np.sqrt(p * (1 - p) / total)
    return z * se

def process_results(results, dictionary):
    total_per_quantile = [0 for _ in range(11)]
    total_correct_per_quantile = [0 for _ in range(11)]
    for quantile_positives, quantile_totals in results[0]:
        for i in range(11):
            total_per_quantile[i] += quantile_totals[i]
            total_correct_per_quantile[i] += quantile_positives[i]

    total_negative = 0
    total_negative_correct = 0
    for negative_positives, negative_totals in results[1]:
        total_negative += negative_totals
        total_negative_correct += negative_positives

    average_per_quantile = [
        total_correct_per_quantile[i] / total_per_quantile[i] for i in range(1, 11)
    ]
    average_negative = total_negative_correct / total_negative

    ci_per_quantile = [
        confidence_interval(total_correct_per_quantile[i], total_per_quantile[i])
        for i in range(1, 11)
    ]
    ci_negative = confidence_interval(total_negative_correct, total_negative)

    x = ["Not"] + ["Q" + str(i) for i in range(1, 11)]
    y = [1 - average_negative] + average_per_quantile[::-1]
    ci = [ci_negative] + ci_per_quantile[::-1]

    return x, y, ci, dictionary

# Create two subplots with the new figure size
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.5))

topk_data = None

for results, dictionary in zip(results, dictionaries):
    x, y, ci, dict_name = process_results(results, dictionary)
    
    for i in [2, 4, 8, 16, 32, 64, 128]:
        if f"/{i}_experts" in dict_name:
            label_name = f"Switch: {i}e"
            break
    if "topk" in dict_name:
        label_name = "Topk"

    if "topk" in dict_name:
        topk_data = (x, y, ci, dict_name)
        color = flop_matched_color_dict[1]  # Use color for 1 expert
        ax1.errorbar(x, y, yerr=ci, fmt='-o', capsize=5, label=label_name, color=color, markersize=5)
        ax2.errorbar(x, y, yerr=ci, fmt='-o', capsize=5, label=label_name, color=color, markersize=5)
    elif "flop-matched" in dict_name:
        ax = ax1
        color_dict = flop_matched_color_dict
        expert_count = int(dict_name.split("/")[2].split("_")[0])
        color = color_dict.get(expert_count, "#000000")  # Default to black if not found
        ax.errorbar(x, y, yerr=ci, fmt='-o', capsize=5, label=label_name, color=color, markersize=5)
    else:
        ax = ax2
        color_dict = fixed_width_color_dict
        expert_count = int(dict_name.split("/")[2].split("_")[0])
        color = color_dict.get(expert_count, "#000000")  # Default to black if not found
        ax.errorbar(x, y, yerr=ci, fmt='-o', capsize=5, label=label_name, color=color, markersize=5)

for ax in (ax1, ax2):
    ax.set_xlabel("Quantiles", fontsize=8)
    ax.set_ylabel('Accuracy', fontsize=8)
    ax.grid(True, which="both", ls="--", linewidth=0.5)
    ax.legend(loc='upper center', prop={'size': 5.5})
    
    # Set y-axis limits and ticks
    ax.set_ylim(0.2, 1.0)  # Adjust these values as needed
    
    # Adjust tick label size
    ax.tick_params(axis='both', labelsize=6.5)
    
    # Remove minor ticks
    ax.minorticks_off()

    # Tilt all x-axis labels
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_va('top')
    

# Turn off ticks on y axis of ax2
ax2.yaxis.set_ticks_position('none')
ax2.yaxis.set_tick_params(size=0)
ax2.yaxis.set_ticklabels([])
ax2.set_ylabel('')

ax1.set_title("FLOP-Matched", fontsize=9)
ax2.set_title("Width-Matched", fontsize=9)

os.makedirs("plots", exist_ok=True)
plt.savefig("plots/detection_split.pdf", bbox_inches='tight')
plt.show()
# %%
