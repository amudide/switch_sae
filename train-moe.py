from nnsight import LanguageModel
import torch as t
from dictionary_learning import ActivationBuffer
from dictionary_learning.training import trainSAE
from dictionary_learning.utils import hf_dataset_to_generator, cfg_filename, str2bool
from dictionary_learning.trainers.moe import MoEAutoEncoder, MoETrainer
from dictionary_learning.evaluation import evaluate
import wandb
import argparse
import itertools
from config import lm, activation_dim, layer, hf, steps, n_ctxs

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", required=True)
parser.add_argument('--dict_ratio', type=int, default=32)
parser.add_argument("--ks", nargs="+", type=int, required=True)
parser.add_argument("--num_experts", nargs="+", type=int, required=True)
parser.add_argument("--es", nargs="+", type=int, required=True)
parser.add_argument("--heavisides", nargs="+", type=str2bool, required=True)
args = parser.parse_args()

device = f'cuda:{args.gpu}'
model = LanguageModel(lm, dispatch=True, device_map=device)
submodule = model.transformer.h[layer]
data = hf_dataset_to_generator(hf)
buffer = ActivationBuffer(data, model, submodule, d_submodule=activation_dim, n_ctxs=n_ctxs, device=device)

base_trainer_config = {
    'trainer' : MoETrainer,
    'dict_class' : MoEAutoEncoder,
    'activation_dim' : activation_dim,
    'dict_size' : args.dict_ratio * activation_dim,
    'auxk_alpha' : 1/32,
    'decay_start' : int(steps * 0.8),
    'steps' : steps,
    'seed' : 0,
    'device' : device,
    'layer' : layer,
    'lm_name' : lm,
    'wandb_name' : 'MoEAutoEncoder'
}

trainer_configs = [(base_trainer_config | {'k': combo[0], 'experts': combo[1], 'e': combo[2], 'heaviside': combo[3]}) for combo in itertools.product(args.ks, args.num_experts, args.es, args.heavisides)]

wandb.init(entity="amudide", project="MoE", config={f'{trainer_config["wandb_name"]}-{i}' : trainer_config for i, trainer_config in enumerate(trainer_configs)})

trainSAE(buffer, trainer_configs=trainer_configs, save_dir='dictionaries', log_steps=1, steps=steps)

print("Training finished. Evaluating SAE...", flush=True)
for i, trainer_config in enumerate(trainer_configs):
    ae = MoEAutoEncoder.from_pretrained(f'dictionaries/{cfg_filename(trainer_config)}/ae.pt',
                                        k = trainer_config['k'], experts = trainer_config['experts'], e = trainer_config['e'], heaviside = trainer_config['heaviside'], device=device)
    metrics = evaluate(ae, buffer, device=device)
    log = {}
    log.update({f'{trainer_config["wandb_name"]}-{i}/{k}' : v for k, v in metrics.items()})
    wandb.log(log, step=steps+1)
wandb.finish()