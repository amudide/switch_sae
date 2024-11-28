# %%
import os
import sys

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
from dictionary_learning.evaluation import evaluate
import wandb
import argparse
import itertools
from config import hf

base_path = "/home/jengels/switch_sae/other_dictionaries"

# Generate all combinations of parameters
layers = [2, 4, 8, 10]
sae_types = ['switch', 'topk']
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

    gpt_info_to_filename[(layer, sae_type,layer_type)] = filename

    if not os.path.exists(filename):
        print(layer, sae_type, layer_type)

# %%

device = "cuda:0"
lm = "openai-community/gpt2"
model = LanguageModel(lm, dispatch=True, device_map=device)
info_to_metrics = {}
for layer, sae_type, layer_type in itertools.product(layers, sae_types, types):
    if layer_type == "resid":
        submodule = model.transformer.h[layer]
    elif layer_type == "mlp":
        submodule = model.transformer.h[layer].mlp
    elif layer_type == "attn":
        submodule = model.transformer.h[layer].attn

    data = hf_dataset_to_generator(hf, split="train")
    buffer = ActivationBuffer(
        data,
        model,
        submodule,
        d_submodule=768,
        n_ctxs=1e3,
        device="cpu",
        out_batch_size=8192,
        refresh_batch_size=512,
    )

    filename = gpt_info_to_filename[(layer, sae_type, layer_type)]

    if sae_type == "switch":
        ae = SwitchAutoEncoder.from_pretrained(
            f"{filename}/ae.pt",
            k=64,
            experts=8,
            heaviside=False,
            device=device,
        )
        metrics = evaluate(ae, buffer, device=device)
    else:
        ae = AutoEncoderTopK.from_pretrained(
            f"{filename}/ae.pt",
            k=64,
            device=device,
        )
        metrics = evaluate(ae, buffer, device=device)

    info_to_metrics[(layer, sae_type, layer_type)] = metrics


    print(metrics)
# %%
