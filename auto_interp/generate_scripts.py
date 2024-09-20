# %%
import glob

# Get all paths in dictionaries
dictionaries = glob.glob("dictionaries/**/*.pt", recursive=True)
parent_dirs = set(["/".join(d.split("/")[:-1]) for d in dictionaries])

# Write out run_all_feature_eval_generate.sh
with open("run_all_feature_eval_generate.sh", "w") as f:
    for parent_dir in parent_dirs:
        f.write(f"python sae_feature_evals.py --sae_path {parent_dir} --to_do generate\n")

# Write out run_all_feature_eval_eval.sh
with open("run_all_feature_eval_eval.sh", "w") as f:
    for parent_dir in parent_dirs:
        f.write(f"python sae_feature_evals.py --sae_path {parent_dir} --to_do eval\n")

# Write out run_all_feature_eval_generate_small.sh
# Only k = 64, and (fixed-width or topk)
with open("run_all_feature_eval_generate_small.sh", "w") as f:
    for parent_dir in parent_dirs:
        if "k64" in parent_dir and ("fixed-width" in parent_dir or "topk" in parent_dir):
            f.write(f"python sae_feature_evals.py --sae_path {parent_dir} --to_do generate\n")

# Write out run_all_feature_eval_eval_small.sh
# Only k = 64, and (fixed-width or topk)
with open("run_all_feature_eval_eval_small.sh", "w") as f:
    for parent_dir in parent_dirs:
        if "k64" in parent_dir and ("fixed-width" in parent_dir or "topk" in parent_dir):
            f.write(f"python sae_feature_evals.py --sae_path {parent_dir} --to_do eval\n")