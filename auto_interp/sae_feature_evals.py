# %%
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from functools import partial
from nnsight import LanguageModel
import asyncio

from dictionary_learning.trainers.switch import SwitchAutoEncoder
from dictionary_learning.trainers.top_k import AutoEncoderTopK

import torch
import einops
import os

from functools import partial

import torch
from nnsight import LanguageModel

from sae_auto_interp.autoencoders.wrapper import AutoencoderLatents
from sae_auto_interp.autoencoders.OpenAI import Autoencoder


from sae_auto_interp.features import FeatureCache
from sae_auto_interp.features.features import FeatureRecord
from sae_auto_interp.utils import load_tokenized_data


from sae_auto_interp.features import FeatureDataset, pool_max_activation_windows, random_activation_windows, sample
from sae_auto_interp.config import FeatureConfig, ExperimentConfig


from sae_auto_interp.explainers import SimpleExplainer
from sae_auto_interp.scorers import RecallScorer
from tqdm import tqdm
import pickle


from sae_auto_interp.clients import OpenRouter, Local
from sae_auto_interp.utils import display
import argparse

# %%

# Change dir to folder one level up from this file
this_dir = os.path.dirname(os.path.abspath(__file__))
one_level_up = os.path.dirname(this_dir)
os.chdir(one_level_up)

# %%

CTX_LEN = 128
BATCH_SIZE = 128
N_TOKENS = 10_000_000
N_SPLITS = 2
NUM_FEATURES_TO_TEST = 1000

device = "cuda:1"

# Set torch seed
torch.manual_seed(0)

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

if not is_notebook:
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--sae_path", type=str, required=True)
    argparser.add_argument("--to_do", type=str, required=True, choices=["generate", "eval", "both"])

    args = argparser.parse_args()
    SAE_PATH = args.sae_path
    to_do = args.to_do
else:
    SAE_PATH = "dictionaries/topk/k64"
    to_do = "both"
    # SAE_PATH = "dictionaries/fixed-width/16_experts/k64"

RAW_ACTIVATIONS_DIR = f"/media/jengels/sda/switch/{SAE_PATH}"
SAVE_FILE = f"/media/jengels/sda/switch/{SAE_PATH}/results.pkl"
FINAL_SAVE_FILE = f"/media/jengels/sda/switch/{SAE_PATH}/final_results.pkl"

# %%


if "topk" in SAE_PATH:
    k = int(SAE_PATH.split("/")[-1][1:])
    ae = AutoEncoderTopK.from_pretrained(f"{SAE_PATH}/ae.pt", k=k, device=device)
else:
    num_experts = int(SAE_PATH.split("/")[-2].split("_")[0])
    k = int(SAE_PATH.split("/")[-1][1:])
    ae = SwitchAutoEncoder.from_pretrained(f"{SAE_PATH}/ae.pt", k=k, experts=num_experts, device=device)

ae.to(device)

model = LanguageModel("openai-community/gpt2", device_map=device, dispatch=True)

# TODO: Ideally use openwebtext
tokens = load_tokenized_data(
    CTX_LEN,
    model.tokenizer,
    "kh4dien/fineweb-100m-sample",
    "train[:15%]",
)

# %%


WIDTH = ae.dict_size
# Get NUM_FEATURES_TO_TEST random features to test without replacement
random_features = torch.randperm(WIDTH)[:NUM_FEATURES_TO_TEST]

# %%

generate = to_do in ["generate", "both"]
if generate:

    def _forward(ae, x):
        _, _, top_acts, top_indices = ae.forward(x, output_features="all")

        expanded = torch.zeros(top_acts.shape[0], WIDTH, device=device)
        expanded.scatter_(1, top_indices, top_acts)

        expanded = einops.rearrange(expanded, "(b c) w -> b c w", b=x.shape[0], c=x.shape[1])
        return expanded

    # We can simply add the new module as an attribute to an existing
    # submodule on GPT-2's module tree.
    submodule = model.transformer.h[8]
    submodule.ae = AutoencoderLatents(
        ae, 
        partial(_forward, ae),
        width=ae.dict_size
    )

    with model.edit(" ", inplace=True):
        acts = submodule.output[0]
        submodule.ae(acts, hook=True)

    with model.trace("hello, my name is"):
        latents = submodule.ae.output.save()

    module_path = submodule.path

    submodule_dict = {module_path : submodule}
    module_filter = {module_path : random_features.to(device)}

    cache = FeatureCache(
        model, 
        submodule_dict, 
        batch_size=BATCH_SIZE, 
        filters=module_filter
    )

    cache.run(N_TOKENS, tokens)

    cache.save_splits(
        n_splits=N_SPLITS,
        save_dir=RAW_ACTIVATIONS_DIR,
    )

# %%


cfg = FeatureConfig(
    width = WIDTH,
    min_examples = 200,
    max_examples = 2_000,
    example_ctx_len = CTX_LEN,
    n_splits = 2,
)

sample_cfg = ExperimentConfig(n_random=50)

# This is a hack because this isn't currently defined in the repo
sample_cfg.chosen_quantile = 0

dataset = FeatureDataset(
    raw_dir=RAW_ACTIVATIONS_DIR,
    cfg=cfg,
)
# %%

if to_do == "generate":
    exit()

# %%

# Define these functions here so we don't need to edit the functions in the git submodule
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


def load(
    dataset,
    constructor,
    sampler,
    transform = None
):
    def _process(buffer_output):
        record = FeatureRecord(buffer_output.feature)
        if constructor is not None:
            constructor(record=record, buffer_output=buffer_output)

        if sampler is not None:
            sampler(record)

        if transform is not None:
            transform(record)

        return record

    for buffer in dataset.buffers:
        for data in buffer:
            if data is not None:
                yield _process(data)
# %%

record_iterator = load(constructor=constructor, sampler=sampler, dataset=dataset, transform=None)

# next_record = next(record_iterator)

# display(next_record, model.tokenizer, n=5)
# %%

# Command to run: vllm serve hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 --max_model_len 10000 --tensor-parallel-size 2
client = Local("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4")

# %%

async def run_async():

    global record_iterator


    positive_scores = []
    negative_scores = []
    explanations = []
    feature_ids = []
    total_positive_score = 0
    total_negative_score = 0
    total_evaluated = 0

    bar = tqdm(record_iterator, total=NUM_FEATURES_TO_TEST)



    for record in bar:

        explainer = SimpleExplainer(
            client,
            model.tokenizer,
            # max_new_tokens=50,
            max_tokens=50,
            temperature=0.0
        )

        explainer_result = await explainer(record)
        # explainer_result = asyncio.run(explainer(record))

        # print(explainer_result.explanation)
        record.explanation = explainer_result.explanation


        scorer = RecallScorer(
            client,
            model.tokenizer,
            max_tokens=25,
            temperature=0.0,
            batch_size=4,
        )


        score = await scorer(record)

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

        positive_scores.append((quantile_positives, quantile_totals))
        negative_scores.append((negative_positives, negative_totals))

        if (sum(quantile_totals) == 0) or (negative_totals == 0):
            continue

        total_positive_score += sum(quantile_positives) / sum(quantile_totals)
        total_negative_score += negative_positives / negative_totals
        total_evaluated += 1
        
        bar.set_description(f"Positive Recall: {total_positive_score / total_evaluated}, Negative Recall: {total_negative_score / total_evaluated}")

        print(quantile_positives, quantile_totals)

        explanations.append(record.explanation)

        feature_ids.append(record.feature.feature_index) 
        

        with open(SAVE_FILE, "wb") as f:
            pickle.dump((positive_scores, negative_scores, explanations, feature_ids), f)

    with open(FINAL_SAVE_FILE, "wb") as f:
        pickle.dump((positive_scores, negative_scores, explanations, feature_ids), f)


# Switch comment when running in notebook/command line
# await run_async()
asyncio.run(run_async())

# %%
