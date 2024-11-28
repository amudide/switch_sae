# %%
import os
import sys
import pandas as pd

# Add the parent directory to the sys path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import itertools
import os
from dictionary_learning.trainers.top_k import AutoEncoderTopK, TrainerTopK
from nnsight import LanguageModel
import torch as t
from dictionary_learning import ActivationBuffer
from dictionary_learning.training import trainSAE
from dictionary_learning.utils import hf_dataset_to_generator, cfg_filename, str2bool
from dictionary_learning.trainers.switch import SwitchAutoEncoder, SwitchTrainer
from modified_eval import evaluate
import wandb
import argparse
import itertools
from config import hf

base_path = "/home/jengels/switch_sae/other_dictionaries"

sae_types = ['switch', 'topk']

# %%

def get_model_metrics(model_name, layers_to_eval, layer_types, sae_types, info_to_filename, d_submodule, device="cuda:0"):
    model = LanguageModel(model_name, dispatch=True, device_map=device)
    info_to_metrics = {}
    
    for layer in layers_to_eval:
        for layer_type in layer_types:
            values = []

            if model_name == "google/gemma-2b":
                submodule = model.model.layers[layer]
            else:
                submodule = model.transformer.h[layer]

            if layer_type == "resid":
                submodule = submodule
            elif layer_type == "mlp":
                submodule = submodule.mlp
            elif layer_type == "attn":
                submodule = submodule.attn

            for sae_type in sae_types:

                data = hf_dataset_to_generator(hf, split="train")
                buffer = ActivationBuffer(
                    data,
                    model,
                    submodule,
                    d_submodule=d_submodule,
                    n_ctxs=1e2,
                    device="cpu",
                    out_batch_size=8192,
                    refresh_batch_size=1 if model_name == "google/gemma-2b" else 128,
                )

                filename = info_to_filename[(layer, sae_type, layer_type)]

                if model_name == "google/gemma-2b":
                    model_path = f"{filename}/checkpoints/ae_{gemma_checkpoint_id}.pt"
                else:
                    model_path = f"{filename}/ae.pt"

                if sae_type == "switch":
                    ae = SwitchAutoEncoder.from_pretrained(
                        model_path,
                        k=64,
                        experts=8,
                        heaviside=False,
                        device=device,
                    )
                    metrics = evaluate(ae, 
                                       buffer, 
                                       device=device, 
                                       batch_size=1 if model_name == "google/gemma-2b" else 128,
                                       num_batches=32 if model_name == "google/gemma-2b" else 2)
                else:
                    ae = AutoEncoderTopK.from_pretrained(
                        model_path,
                        k=64,
                        device=device,
                    )
                    metrics = evaluate(ae, 
                                       buffer, 
                                       device=device, 
                                       batch_size=1 if model_name == "google/gemma-2b" else 128,
                                       num_batches=32 if model_name == "google/gemma-2b" else 2)

                values.append(metrics)
                print(metrics)

            key = (layer, layer_type)
            info_to_metrics[key] = values
            
    del model
    return info_to_metrics

# %%


# Generate all combinations of parameters
layers = [2, 4, 8, 10]
types = ['resid', 'attn', 'mlp']
devices = [f'cuda:{i}' for i in range(8)]

# First set of commands for GPT-2
gpt_info_to_filename = {}
for layer, sae_type, layer_type in itertools.product(layers, sae_types, types):

    device_num = len(gpt_info_to_filename) % 8
    if sae_type == 'switch':
        filename = f"dict_class:.SwitchAutoEncoder'>_activation_dim:768_dict_size:24576_auxk_alpha:0.03125_decay_start:16000_steps:20000_seed:0_device:cuda:{device_num}_layer:{layer}_lm_name:openai-communitygpt2_wandb_name:SwitchAutoEncoder_k:64_experts:8_heaviside:False_lb_alpha:3"
    else:
        filename = f"dict_class:_k.AutoEncoderTopK'>_activation_dim:768_dict_size:24576_auxk_alpha:0.03125_decay_start:16000_steps:20000_seed:0_device:cuda:{device_num}_layer:{layer}_lm_name:openai-communitygpt2_wandb_name:AutoEncoderTopK_k:64"

    filename = os.path.join(base_path, filename)


    if not os.path.exists(filename):
        print(layer, sae_type, layer_type)
        continue

    gpt_info_to_filename[(layer, sae_type,layer_type)] = filename


# Get GPT-2 metrics
gpt_info_to_metrics = get_model_metrics(
    model_name="openai-community/gpt2",
    layers_to_eval=layers,
    layer_types=types,
    sae_types=sae_types,
    info_to_filename=gpt_info_to_filename,
    d_submodule=768,
    device="cuda:1"
)


# %%
# Get Gemma metrics
gemma_info_to_metrics = {}
gemma_checkpoint_id = 2000

# First create filename mapping for Gemma
gemma_info_to_filename = {}
for i, sae_type in enumerate(sae_types):
    layer = 12
    device = f"cuda:{i}"
    if sae_type == "switch":
        filename = f"{base_path}/dict_class:.SwitchAutoEncoder'>_activation_dim:2048_dict_size:65536_auxk_alpha:0.03125_decay_start:1600000_steps:2000000_seed:0_device:cuda:{i}_layer:12_lm_name:googlegemma-2b_wandb_name:SwitchAutoEncoder_k:64_experts:8_heaviside:False_lb_alpha:3"
    else:
        filename = f"{base_path}/dict_class:_k.AutoEncoderTopK'>_activation_dim:2048_dict_size:65536_auxk_alpha:0.03125_decay_start:1600000_steps:2000000_seed:0_device:cuda:{i}_layer:12_lm_name:googlegemma-2b_wandb_name:AutoEncoderTopK_k:64"
    
    gemma_info_to_filename[(layer, sae_type, "resid")] = filename

gemma_info_to_metrics = get_model_metrics(
    model_name="google/gemma-2b",
    layers_to_eval=[12],
    layer_types=["resid"],
    sae_types=sae_types,
    info_to_filename=gemma_info_to_filename,
    d_submodule=2048,
    device="cuda:1"
)





# %%


# %%
rows = []
# Add GPT-2 results
for (layer, layer_type), values in gpt_info_to_metrics.items():
    row = {
        'Model': 'GPT-2',
        'Layer': layer,
        'Type': layer_type,
        'TopK FVE': f"{values[1]['frac_variance_explained']:.3f}",
        'Switch FVE': f"{values[0]['frac_variance_explained']:.3f}", 
        'TopK FR': f"{values[1]['frac_recovered']:.3f}",
        'Switch FR': f"{values[0]['frac_recovered']:.3f}"
    }
    rows.append(row)

# Add Gemma results
for (layer, layer_type), values in gemma_info_to_metrics.items():
    row = {
        'Model': 'Gemma',
        'Layer': layer,
        'Type': layer_type,
        'TopK FVE': f"{values[1]['frac_variance_explained']:.3f}",
        'Switch FVE': f"{values[0]['frac_variance_explained']:.3f}",
        'TopK FR': f"{values[1]['frac_recovered']:.3f}", 
        'Switch FR': f"{values[0]['frac_recovered']:.3f}"
    }
    rows.append(row)

df = pd.DataFrame(rows)
print("\nResults Table:")
print(df.to_markdown(index=False))

# %%

df.to_csv("results.csv", index=False)

# %%

# Convert to LaTeX table
latex_table = """\\begin{table}[h]
\\centering
\\begin{tabular}{lcccccccc}
\\toprule
Model & Layer & Type & TopK FVE & Switch FVE & TopK FR & Switch FR \\\\
\\midrule"""

for _, row in df.iterrows():
    latex_table += f"\n{row['Model']} & {row['Layer']} & {row['Type']} & {row['TopK FVE']} & {row['Switch FVE']} & {row['TopK FR']} & {row['Switch FR']} \\\\"

latex_table += """
\\bottomrule
\\end{tabular}
\\caption{Comparison of TopK and Switch autoencoders across different models, layers and component types. FVE = Fraction of Variance Explained, FR = Fraction Recovered.}
\\label{tab:model_comparison}
\\end{table}"""

print("\nLaTeX Table:")
print(latex_table)
