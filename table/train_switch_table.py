# %%
import os
import sys
# Add the parent directory to the sys path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


# %%
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:1', help='Device to run on')
parser.add_argument('--lm', type=str, help='Language model to use')
parser.add_argument('--layer', type=int, default=12, help='Layer to extract activations from')
parser.add_argument("--type", type=str, default="resid", choices=["resid", "mlp", "attn"])
parser.add_argument("--sae_type", type=str, default="switch", choices=["switch", "topk"])
parser.add_argument('--ks', nargs='+', type=int, default=[64], help='List of k values')
parser.add_argument('--activation_dim', type=int, default=2048, help='Dimension of activations')
parser.add_argument('--dict_ratio', type=int, default=32, help='Dictionary size ratio')
parser.add_argument('--num_experts', nargs='+', type=int, default=[8], help='List of number of experts')
parser.add_argument("--steps", type=int, default=10000, help="Number of steps to train for")

args = parser.parse_args()

device = args.device
layer = args.layer
lm = args.lm
print(lm)
ks = args.ks
activation_dim = args.activation_dim
dict_ratio = args.dict_ratio
num_experts = args.num_experts
lb_alphas = [3]
heavisides = [False]
n_ctxs = 3e4
batch_size = 8192
steps = args.steps

# %%

model = LanguageModel(lm, dispatch=True, device_map=device)

# %%

if lm == "openai-community/gpt2":

    if args.type == "resid":
        submodule = model.transformer.h[layer]
    elif args.type == "mlp":
        submodule = model.transformer.h[layer].mlp
    elif args.type == "attn":
        submodule = model.transformer.h[layer].attn

else:

    if args.type == "resid":
        submodule = model.model.layers[layer]
    elif args.type == "mlp":
        submodule = model.model.layers[layer].mlp
    elif args.type == "attn":
        submodule = model.model.layers[layer].self_attn

# %%


data = hf_dataset_to_generator(hf)
buffer = ActivationBuffer(
    data,
    model,
    submodule,
    d_submodule=activation_dim,
    n_ctxs=n_ctxs,
    device=device,
    out_batch_size=batch_size,
    refresh_batch_size=512 if lm == "openai-community/gpt2" else 64
)

if args.sae_type == "switch":

    base_trainer_config = {
        "trainer": SwitchTrainer,
        "dict_class": SwitchAutoEncoder,
        "activation_dim": activation_dim,
        "dict_size": dict_ratio * activation_dim,
        "auxk_alpha": 1 / 32,
        "decay_start": int(steps * 0.8),
        "steps": steps,
        "seed": 0,
        "device": device,
        "layer": layer,
        "lm_name": lm,
        "wandb_name": "SwitchAutoEncoder",
    }

    trainer_configs = [
        (
            base_trainer_config
            | {
                "k": combo[0],
                "experts": combo[1],
                "heaviside": combo[2],
                "lb_alpha": combo[3],
            }
        )
        for combo in itertools.product(ks, num_experts, heavisides, lb_alphas)
    ]
else:
    base_trainer_config = {
        'trainer' : TrainerTopK,
        'dict_class' : AutoEncoderTopK,
        'activation_dim' : activation_dim,
        'dict_size' : args.dict_ratio * activation_dim,
        'auxk_alpha' : 1/32,
        'decay_start' : int(steps * 0.8),
        'steps' : steps,
        'seed' : 0,
        'device' : device,
        'layer' : layer,
        'lm_name' : lm,
        'wandb_name' : 'AutoEncoderTopK'
    }

    trainer_configs = [(base_trainer_config | {'k': k}) for k in args.ks]


wandb.init(
    entity="josh_engels",
    project="Switch",
    config={
        f'{trainer_config["wandb_name"]}-{i}': trainer_config
        for i, trainer_config in enumerate(trainer_configs)
    },
)

trainSAE(
    buffer,
    trainer_configs=trainer_configs,
    save_dir="dictionaries",
    log_steps=1,
    steps=steps,
)

print("Training finished. Evaluating SAE...", flush=True)
for i, trainer_config in enumerate(trainer_configs):
    ae = SwitchAutoEncoder.from_pretrained(
        f"dictionaries/{cfg_filename(trainer_config)}/ae.pt",
        k=trainer_config["k"],
        experts=trainer_config["experts"],
        heaviside=trainer_config["heaviside"],
        device=device,
    )
    metrics = evaluate(ae, buffer, device=device)
    log = {}
    log.update(
        {f'{trainer_config["wandb_name"]}-{i}/{k}': v for k, v in metrics.items()}
    )
    wandb.log(log, step=steps + 1)
wandb.finish()
