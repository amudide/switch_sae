# %%

import pickle
import matplotlib.pyplot as plt
import numpy as np

dictionaries = ["dictionaries/topk/k64", "dictionaries/fixed-width/32_experts/k64"]
paths = [f"/media/jengels/sda/switch/{dictionary}" for dictionary in dictionaries]
results = [pickle.load(open(f"{path}/results.pkl", "rb")) for path in paths]

# %%

def confidence_interval(successes, total, z=1.96):
    p = successes / total
    se = np.sqrt(p * (1 - p) / total)
    return z * se

for (positive_scores, negative_scores, explanations, feature_ids), dictionary in zip(results, dictionaries):
    total_per_quantile = [0 for _ in range(11)]
    total_correct_per_quantile = [0 for _ in range(11)]
    for (quantile_positives, quantile_totals) in positive_scores:
        for i in range(11):
            total_per_quantile[i] += quantile_totals[i]
            total_correct_per_quantile[i] += quantile_positives[i]
    
    total_negative = 0
    total_negative_correct = 0
    for negative_positives, negative_totals in negative_scores:
        total_negative += negative_totals
        total_negative_correct += negative_positives

    average_per_quantile = [total_correct_per_quantile[i] / total_per_quantile[i] for i in range(1, 11)]
    average_negative = total_negative_correct / total_negative

    # Calculate confidence intervals
    ci_per_quantile = [confidence_interval(total_correct_per_quantile[i], total_per_quantile[i]) for i in range(1, 11)]
    ci_negative = confidence_interval(total_negative_correct, total_negative)

    x = ["Negative"] + ["Q" + str(i) for i in range(1, 11)]
    y = [1 - average_negative] + average_per_quantile[::-1]
    ci = [ci_negative] + ci_per_quantile[::-1]

    plt.errorbar(x, y, yerr=ci, fmt='-o', capsize=5, label=dictionary)

plt.legend()
plt.xlabel('Quantiles')
plt.ylabel('Accuracy')
plt.title('Detection')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()