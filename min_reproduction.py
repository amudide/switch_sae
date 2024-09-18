# %%

from functools import partial
from nnsight import LanguageModel

import torch

from functools import partial

import torch
from nnsight import LanguageModel

from sae_auto_interp.autoencoders.wrapper import AutoencoderLatents
from sae_auto_interp.autoencoders.OpenAI import Autoencoder

# %%


try:
    from IPython import get_ipython  # type: ignore

    ipython = get_ipython()
    assert ipython is not None
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

    is_notebook = True
except:
    is_notebook = False

# %%


path = "/home/jengels/switch_sae/8.pt" # Change this line to your weights location.
state_dict = torch.load(path)
ae = Autoencoder.from_state_dict(state_dict=state_dict)
ae.to("cuda:0")

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)

# %%

def _forward(ae, x):
    latents, _ = ae.encode(x)
    return latents

# We can simply add the new module as an attribute to an existing
# submodule on GPT-2's module tree.
submodule = model.transformer.h[8]
submodule.ae = AutoencoderLatents(
    ae, 
    partial(_forward, ae),
    width=131_072
)

# %%
with model.edit(" ", inplace=True):
    acts = submodule.output[0]
    submodule.ae(acts, hook=True)

# %%

with model.trace("hello, my name is"):
    latents = submodule.ae.output.save()
# %%

from sae_auto_interp.features import FeatureCache
from sae_auto_interp.utils import load_tokenized_data

CTX_LEN = 64
BATCH_SIZE = 32
N_TOKENS = 500_000
N_SPLITS = 2

tokens = load_tokenized_data(
    CTX_LEN,
    model.tokenizer,
    "kh4dien/fineweb-100m-sample",
    "train[:15%]",
)

# %%

module_path = submodule.path

submodule_dict = {module_path : submodule}
module_filter = {module_path : torch.arange(100).to("cuda:0")}

cache = FeatureCache(
    model, 
    submodule_dict, 
    batch_size=BATCH_SIZE, 
    filters=module_filter
)

cache.run(N_TOKENS, tokens)
# %%

raw_dir = "/media/jengels/sda/switch/gpt2" # Change this line to your save location.
cache.save_splits(
    n_splits=N_SPLITS,
    save_dir=raw_dir,
)
# %%

from sae_auto_interp.features import FeatureDataset, pool_max_activation_windows, random_activation_windows, sample
from sae_auto_interp.config import FeatureConfig, ExperimentConfig

cfg = FeatureConfig(
    width = 131_072,
    min_examples = 100,
    max_examples = 10_000,
    example_ctx_len = CTX_LEN,
    n_splits = 2,
)

sample_cfg = ExperimentConfig(n_random=20)

# This is a hack because this isn't currently defined in the repo
sample_cfg.chosen_quantile = 0

dataset = FeatureDataset(
    raw_dir=raw_dir,
    cfg=cfg,
)
# %%

# Define this here so we don't need to edit the constructor function in the git submodule
def default_constructor(
    record,
    tokens,
    buffer_output,
    n_random: int,
    cfg: FeatureConfig
):
    pool_max_activation_windows(
        record,
        tokens=tokens,
        buffer_output=buffer_output,
        cfg=cfg
    )

    random_activation_windows(
        record,
        tokens=tokens,
        buffer_output=buffer_output,
        n_random=n_random,
        ctx_len=cfg.example_ctx_len,
    )


constructor=partial(
    default_constructor,
    n_random=sample_cfg.n_random,
    tokens=tokens,
    cfg=cfg
)

sampler = partial(
    sample,
    cfg=sample_cfg
)
# %%

all_records = []
for records in dataset.load(constructor=constructor, sampler=sampler):
    print(len(records))
    all_records.extend(records)

# %%

from sae_auto_interp.utils import display

display(all_records[0], model.tokenizer, n=5)
# %%

from sae_auto_interp.clients import OpenRouter

client = OpenRouter("meta-llama/llama-3.1-70b-instruct", api_key="INSERT_API_KEY_HERE")
# %%

from sae_auto_interp.explainers import SimpleExplainer

record = all_records[2]
# %%

explainer = SimpleExplainer(
    client,
    model.tokenizer,
    max_new_tokens=50,
    temperature=0.0
)

explainer_result = await explainer(record)

# %%
print(explainer_result.explanation)
record.explanation = explainer_result.explanation

# %%

from sae_auto_interp.scorers import RecallScorer

scorer = RecallScorer(
    client,
    model.tokenizer,
    max_tokens=25,
    temperature=0.0,
    batch_size=10,
)


score = await scorer(record)

# %%

quantile_positives = [0 for _ in range(11)]
quantile_totals = [0 for _ in range(11)]
negative_positives = 0
negative_totals = 0
for score_instance in score.score:
    quantile = score_instance.distance
    if quantile != -1 and score_instance.prediction != -1:
        quantile_totals[quantile] += 1
        if score_instance.prediction == 1:
            quantile_positives[quantile] += 1
    if quantile == -1 and score_instance.prediction != -1:
        negative_totals += 1
        if score_instance.prediction == 1:
            negative_positives += 1
            
print(quantile_positives, quantile_totals)
print(negative_positives / negative_totals)
print(sum(quantile_positives) / sum(quantile_totals))   

# %%
