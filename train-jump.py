from nnsight import LanguageModel
import torch as t
from dictionary_learning import ActivationBuffer
from dictionary_learning.training import trainSAE
from dictionary_learning.utils import hf_dataset_to_generator, cfg_filename
from dictionary_learning.dictionary import JumpReluAutoEncoder
from dictionary_learning.trainers.jump_relu import JumpReluTrainer
from dictionary_learning.evaluation import evaluate
import wandb
import argparse
from config import lm, activation_dim, layer, hf, steps, n_ctxs

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", required=True)
parser.add_argument('--lr', type=float, default=7e-5)
parser.add_argument('--dict_ratio', type=int, default=32)
parser.add_argument("--l0_penalties", nargs="+", type=float, required=True)
args = parser.parse_args()

device = f'cuda:{args.gpu}'
model = LanguageModel(lm, dispatch=True, device_map=device)
submodule = model.transformer.h[layer]
data = hf_dataset_to_generator(hf)
buffer = ActivationBuffer(data, model, submodule, d_submodule=activation_dim, n_ctxs=n_ctxs, device=device)

base_trainer_config = {
    'trainer' : JumpReluTrainer,
    'dict_class' : JumpReluAutoEncoder,
    'activation_dim' : activation_dim,
    'dict_size' : args.dict_ratio * activation_dim,
    'lr' : args.lr,
    'warmup_steps' : 1000,
    'seed' : 0,
    'device' : device,
    'layer' : layer,
    'lm_name' : lm,
    'wandb_name' : 'JumpReluTrainer'
}

trainer_configs = [(base_trainer_config | {'l0_penalty': l0_penalty}) for l0_penalty in args.l0_penalties]

wandb.init(entity="amudide", project="Jump", config={f'{trainer_config["wandb_name"]}-{i}' : trainer_config for i, trainer_config in enumerate(trainer_configs)})

trainSAE(buffer, trainer_configs=trainer_configs, save_dir='dictionaries', log_steps=1, steps=steps)

print("Training finished. Evaluating SAE...", flush=True)
for i, trainer_config in enumerate(trainer_configs):
    ae = JumpReluAutoEncoder.from_pretrained(f'dictionaries/{cfg_filename(trainer_config)}/ae.pt', device=device)
    metrics = evaluate(ae, buffer, device=device)
    log = {}
    log.update({f'{trainer_config["wandb_name"]}-{i}/{k}' : v for k, v in metrics.items()})
    wandb.log(log, step=steps+1)
wandb.finish()