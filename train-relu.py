from nnsight import LanguageModel
import torch as t
from dictionary_learning import ActivationBuffer
from dictionary_learning.training import trainSAE
from dictionary_learning.utils import hf_dataset_to_generator, cfg_filename
from dictionary_learning.dictionary import AutoEncoderNew
from dictionary_learning.trainers.standard_new import StandardTrainerNew
from dictionary_learning.evaluation import evaluate
import wandb
import argparse
from config import lm, activation_dim, layer, hf, steps

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", required=True)
parser.add_argument('--lr', type=float, default=5e-5) ## 3e-4
parser.add_argument('--dict_ratio', type=int, default=32)
parser.add_argument("--l1_penalties", nargs="+", type=float, required=True)
args = parser.parse_args()

device = f'cuda:{args.gpu}'
model = LanguageModel(lm, device_map=device)
submodule = model.transformer.h[layer]
data = hf_dataset_to_generator(hf)
buffer = ActivationBuffer(data, model, submodule, d_submodule=activation_dim, device=device)

base_trainer_config = {
    'trainer' : StandardTrainerNew,
    'dict_class' : AutoEncoderNew,
    'activation_dim' : activation_dim,
    'dict_size' : args.dict_ratio * activation_dim,
    'lr' : args.lr,
    'lambda_warm_steps' : int(steps * 0.05),
    'decay_start' : int(steps * 0.8),
    'steps' : steps,
    'seed' : 0,
    'device' : device,
    'wandb_name' : 'StandardTrainerNew_Anthropic'
}

trainer_configs = [(base_trainer_config | {'l1_penalty': l1_penalty}) for l1_penalty in args.l1_penalties]

wandb.init(entity="amudide", project="ReLU", config={f'{trainer_config["wandb_name"]}-{i}' : trainer_config for i, trainer_config in enumerate(trainer_configs)})

trainSAE(buffer, trainer_configs=trainer_configs, save_dir='dictionaries', log_steps=1000, steps=steps)

print("Training finished. Evaluating SAE...", flush=True)
for i, trainer_config in enumerate(trainer_configs):
    ae = AutoEncoderNew.from_pretrained(f'dictionaries/{cfg_filename(trainer_config)}/ae.pt')
    metrics = evaluate(ae, buffer)
    log = {}
    log.update({f'{trainer_config["wandb_name"]}-{i}/{k}' : v for k, v in metrics.items()})
    wandb.log(log, step=steps+1)
wandb.finish()