""" 
Trains dense TopK SAEs of varying scale.
"""

import os
import sys
sys.path.append( # the switch_sae directory
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
os.environ['HF_HOME'] = '/om/user/ericjm/.cache/huggingface'


from nnsight import LanguageModel
import torch as t
from dictionary_learning import ActivationBuffer
from dictionary_learning.training import trainSAE
from dictionary_learning.utils import hf_dataset_to_generator, cfg_filename
from dictionary_learning.trainers.top_k import AutoEncoderTopK, TrainerTopK
from dictionary_learning.evaluation import evaluate
import wandb


lm = 'openai-community/gpt2'
activation_dim = 768
layer = 8
hf = 'Skylion007/openwebtext'
steps = 100_000 # test run!
n_ctxs = int(1e4)

dict_sizes = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
k = 32

save_dir = "topk_dense"

device = f'cuda:0'
model = LanguageModel(lm, dispatch=True, device_map=device)
submodule = model.transformer.h[layer]
data = hf_dataset_to_generator(hf)
buffer = ActivationBuffer(data, model, submodule, d_submodule=activation_dim, n_ctxs=n_ctxs, device=device)


base_trainer_config = {
    'trainer' : TrainerTopK,
    'dict_class' : AutoEncoderTopK,
    'activation_dim' : activation_dim,
    'k' : k,
    'auxk_alpha' : 1/32,
    'decay_start' : int(steps * 0.8),
    'steps' : steps,
    'seed' : 0,
    'device' : device,
    'layer' : layer,
    'lm_name' : lm,
    'wandb_name' : 'AutoEncoderTopK'
}

trainer_configs = [(base_trainer_config | {'dict_size': ds}) for ds in dict_sizes]

wandb.init(entity="ericjmichaud_", project="switch_saes_scaling_laws_attempt_0", config={f"{trainer_config['wandb_name']}-{i}" : trainer_config for i, trainer_config in enumerate(trainer_configs)})

trainSAE(buffer, trainer_configs=trainer_configs, save_dir=save_dir, log_steps=1, steps=steps)

print("Training finished. Evaluating SAE...", flush=True)
for i, trainer_config in enumerate(trainer_configs):
    ae = AutoEncoderTopK.from_pretrained(
        os.path.join(save_dir, f'{cfg_filename(trainer_config)}/ae.pt'), 
        k=trainer_config['k'], 
        device=device
    )
    metrics = evaluate(ae, buffer, device=device)
    log = {}
    log.update({f'{trainer_config["wandb_name"]}-{i}/{k}' : v for k, v in metrics.items()})
    wandb.log(log, step=steps+1)
wandb.finish()
