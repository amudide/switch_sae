from nnsight import LanguageModel
import torch as t
from dictionary_learning import ActivationBuffer
from dictionary_learning.training import trainSAE
from dictionary_learning.utils import hf_dataset_to_generator, cfg_filename
from dictionary_learning.trainers.top_k import AutoEncoderTopK, TopKTrainer
from dictionary_learning.evaluation import evaluate
import wandb
import argparse
from config import lm, activation_dim, layer, hf

steps = 100

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", required=True)
parser.add_argument('--dict_ratio', type=int, default=32)
parser.add_argument("--ks", nargs="+", type=int, required=True)
args = parser.parse_args()

device = f'cuda:{args.gpu}'
model = LanguageModel(lm, device_map=device)
submodule = model.transformer.h[layer]
data = hf_dataset_to_generator(hf)
buffer = ActivationBuffer(data, model, submodule, d_submodule=activation_dim, n_ctxs=5e3, device=device)

base_trainer_config = {
    'trainer' : TopKTrainer,
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

wandb.init(entity="amudide", project="TopK-Small", config={f'{trainer_config["wandb_name"]}-{i}' : trainer_config for i, trainer_config in enumerate(trainer_configs)})

trainSAE(buffer, trainer_configs=trainer_configs, save_dir='dictionaries', log_steps=10, steps=steps)

print("Training finished. Evaluating SAE...", flush=True)
for i, trainer_config in enumerate(trainer_configs):
    ae = AutoEncoderTopK.from_pretrained(f'dictionaries/{cfg_filename(trainer_config)}/ae.pt', k = trainer_config['k'], device=device)
    metrics = evaluate(ae, buffer, device=device)
    log = {}
    log.update({f'{trainer_config["wandb_name"]}-{i}/{k}' : v for k, v in metrics.items()})
    wandb.log(log)
wandb.finish()