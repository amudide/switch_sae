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
from config import lm, activation_dim, layer, hf, steps, n_ctxs

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", required=True)
parser.add_argument('--dict_ratio', type=int, default=32)
parser.add_argument("--ks", nargs="+", type=int, required=True)
parser.add_argument("--num_experts", nargs="+", type=int, required=True)
parser.add_argument("--lb_alphas", nargs="+", type=float, default=[3.0])
parser.add_argument("--heavisides", nargs="+", type=str2bool, default=[False])
args = parser.parse_args()

device = f'cuda:{args.gpu}'
model = LanguageModel(lm, dispatch=True, device_map=device)
submodule = model.transformer.h[layer]
data = hf_dataset_to_generator(hf)
buffer = ActivationBuffer(data, model, submodule, d_submodule=activation_dim, n_ctxs=n_ctxs, device=device)

base_trainer_config = {
    'trainer' : SwitchTrainer,
    'dict_class' : SwitchAutoEncoder,
    'activation_dim' : activation_dim,
    'auxk_alpha' : 1/32,
    'decay_start' : int(steps * 0.8),
    'steps' : steps,
    'seed' : 0,
    'device' : device,
    'layer' : layer,
    'lm_name' : lm,
    'wandb_name' : 'SwitchAutoEncoder'
}

trainer_configs = [(base_trainer_config | {'k': combo[0], 'experts': combo[1], 'dict_size' : args.dict_ratio * activation_dim * combo[1], 'heaviside': combo[2], 'lb_alpha': combo[3]}) for combo in itertools.product(args.ks, args.num_experts, args.heavisides, args.lb_alphas)]

wandb.init(entity="amudide", project="Switch (FLOP Matched, LB)", config={f'{trainer_config["wandb_name"]}-{i}' : trainer_config for i, trainer_config in enumerate(trainer_configs)})

trainSAE(buffer, trainer_configs=trainer_configs, save_dir='dictionaries', log_steps=1, steps=steps)

print("Training finished. Evaluating SAE...", flush=True)
for i, trainer_config in enumerate(trainer_configs):
    ae = SwitchAutoEncoder.from_pretrained(f'dictionaries/{cfg_filename(trainer_config)}/ae.pt',
                                        k = trainer_config['k'], experts = trainer_config['experts'], heaviside = trainer_config['heaviside'], device=device)
    metrics = evaluate(ae, buffer, device=device)
    log = {}
    log.update({f'{trainer_config["wandb_name"]}-{i}/{k}' : v for k, v in metrics.items()})
    wandb.log(log, step=steps+1)
wandb.finish()