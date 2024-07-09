from nnsight import LanguageModel
import torch as t
from dictionary_learning import ActivationBuffer
from dictionary_learning.training import trainSAE
from dictionary_learning.utils import hf_dataset_to_generator, cfg_filename
from dictionary_learning.dictionary import AutoEncoder
from dictionary_learning.trainers.standard import StandardTrainer
from dictionary_learning.evaluation import evaluate
import wandb
import argparse
from config import lm, activation_dim, layer, hf, lr, dict_ratio

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", required=True)
parser.add_argument("-l", "--l1_penalties", nargs="+", type=float, required=True)
args = parser.parse_args()

device = f'cuda:{args.gpu}'
model = LanguageModel(lm, device_map=device)
submodule = model.transformer.h[layer] ## change?
data = hf_dataset_to_generator(hf)
buffer = ActivationBuffer(data, model, submodule, d_submodule=activation_dim, device=device)

base_trainer_config = {
    'trainer' : StandardTrainer,
    'dict_class' : AutoEncoder,
    'activation_dim' : activation_dim,
    'dict_size' : dict_ratio * activation_dim,
    'lr' : lr,
    'warmup_steps' : 1000,
    'resample_steps' : None,
    'seed' : 0,
    'device' : device,
    'layer' : layer,
    'lm_name' : lm,
    'wandb_name' : 'StandardTrainer',
}

trainer_configs = [(base_trainer_config | {'l1_penalty': l1_penalty}) for l1_penalty in args.l1_penalties]

wandb.init(entity="amudide", project="ReLU", config={f'{trainer_config["wandb_name"]}-{i}' : trainer_config for i, trainer_config in enumerate(trainer_configs)})

trainSAE(buffer, trainer_configs=trainer_configs, save_dir='dictionaries', log_steps=1000)

for i, trainer_config in enumerate(trainer_configs):
    ae = AutoEncoder.from_pretrained(f'dictionaries/{cfg_filename(trainer_config)}/ae.pt')
    metrics = evaluate(ae, buffer)
    log = {}
    log.update({f'{trainer_config["wandb_name"]}-{i}/{k}' : v for k, v in metrics.items()})
    wandb.log(log)
wandb.finish()