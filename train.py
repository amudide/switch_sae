from nnsight import LanguageModel
import torch as t
from dictionary_learning import ActivationBuffer
from dictionary_learning.training import trainSAE
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.dictionary import AutoEncoder
from dictionary_learning.trainers.standard import StandardTrainer
from dictionary_learning.evaluation import evaluate
import wandb

layer = 8
activation_dim = 768
dict_ratio = 32 ## make wider
lm = 'openai-community/gpt2' # alternatively, 'EleutherAI/pythia-70m-deduped'
hf = 'Skylion007/openwebtext'

model = LanguageModel(lm, device_map='cuda:4')
submodule = model.transformer.h[layer] ## change?
data = hf_dataset_to_generator(hf)
buffer = ActivationBuffer(data, model, submodule, d_submodule=activation_dim, n_ctxs=3e4, device='cuda:4')

trainer_configs = [
    {
        'trainer' : StandardTrainer,
        'dict_class' : AutoEncoder,
        'activation_dim' : activation_dim,
        'dict_size' : dict_ratio * activation_dim,
        'lr' : 1e-3,
        'l1_penalty' : 1e-1,
        'warmup_steps' : 1000,
        'resample_steps' : None,
        'seed' : 0,
        'device' : 'cuda:4',
        'layer' : layer,
        'lm_name' : lm,
        'wandb_name' : 'StandardTrainer',
    },
    {
        'trainer' : StandardTrainer,
        'dict_class' : AutoEncoder,
        'activation_dim' : activation_dim,
        'dict_size' : dict_ratio * activation_dim,
        'lr' : 1e-3,
        'l1_penalty' : 2e-1,
        'warmup_steps' : 1000,
        'resample_steps' : None,
        'seed' : 0,
        'device' : 'cuda:4',
        'layer' : layer,
        'lm_name' : lm,
        'wandb_name' : 'StandardTrainer',
    }
]

wandb.init(entity="amudide", project="ReLU", config={f'{trainer_config["wandb_name"]}-{i}' : trainer_config for i, trainer_config in enumerate(trainer_configs)})

trainSAE(buffer, trainer_configs=trainer_configs, save_dir='dictionaries-standard', log_steps=5, steps=400)

for i, trainer in enumerate(trainer_configs):
    ae = AutoEncoder.from_pretrained(f'dictionaries-standard/trainer{i}/ae.pt')
    metrics = evaluate(ae, buffer)
    log = {}
    log.update({f'{trainer["wandb_name"]}-{i}/{k}' : v for k, v in metrics.items()})
    wandb.log(log)
wandb.finish()