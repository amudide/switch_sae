# %%

import asyncio
from functools import partial

import orjson
import torch
from simple_parsing import ArgumentParser

from sae_auto_interp.clients import Local
from sae_auto_interp.config import FeatureConfig, ExperimentConfig
from sae_auto_interp.explainers import SimpleExplainer
from sae_auto_interp.features import FeatureDataset, sample, pool_max_activation_windows
from sae_auto_interp.pipeline import Pipe, Pipeline, process_wrapper
from sae_auto_interp.scorers import FuzzingScorer
from sae_auto_interp.utils import (
    load_tokenized_data,
    load_tokenizer,
)

### Set directories ###

raw_features = "/media/jengels/sda/switch/gpt2"
explanation_dir = "/media/jengels/sda/switch/gpt2_explanations"
fuzz_dir = "/media/jengels/sda/switch/gpt2_fuzz"

# %%

# TODO: Change to correct dataset
tokenizer = load_tokenizer("gpt2")
tokens = load_tokenized_data(
    128,
    tokenizer,
    "kh4dien/fineweb-100m-sample",
    "train[:15%]",
)
# %%

# TODO: Make sure this is correct layer
modules = [f".transformer.h.8"]

# %%

# TODO: Add all features
features = {mod: torch.arange(50) for mod in modules}

# %%

feature_config = FeatureConfig(
    width=768 * 16,
    example_ctx_len=128
)

dataset = FeatureDataset(
    raw_dir=raw_features,
    cfg=feature_config,
    modules=modules,
    features=features,
)
# %%

experiment_config = ExperimentConfig()

loader = partial(
    dataset.load,
    constructor=partial(
        pool_max_activation_windows, tokens=tokens, cfg=feature_config
    ),
    sampler=partial(sample, cfg=experiment_config)
)

# %%

client = Local("meta-llama/Meta-Llama-3.1-70B-Instruct")

# %%

def preprocess(record):
    test = []
    extra_examples = []

    for examples in record.test:
        test.append(examples[:5])
        extra_examples.extend(examples[5:])

    record.test = test
    record.extra_examples = extra_examples

    return record

def explainer_postprocess(result):
    
    with open(f"{explanation_dir}/{result.record.feature}.txt", "wb") as f:
        f.write(orjson.dumps(result.explanation))

    return result

explainer_pipe = process_wrapper(
    SimpleExplainer(
        client,
        tokenizer=tokenizer,
        activations=True,
        max_tokens=500,
        temperature=0.0,
    ),
    preprocess=preprocess,
    postprocess=explainer_postprocess,
)

### Build Scorer pipe ###

def scorer_preprocess(result):
    record = result.record
    record.explanation = result.explanation
    return record

def scorer_postprocess(result, score_dir):
    with open(f"{score_dir}/{result.record.feature}.txt", "wb") as f:
        f.write(orjson.dumps(result.score))

scorer_pipe = Pipe(
    process_wrapper(
        FuzzingScorer(
            client,
            tokenizer=tokenizer,
            verbose=True,
            max_tokens=50,
            temperature=0.0,
            batch_size=10,
        ),
        preprocess=scorer_preprocess,
        postprocess=partial(scorer_postprocess, score_dir=fuzz_dir),
    ),
)

### Build the pipeline ###

pipeline = Pipeline(
    loader,
    explainer_pipe,
    scorer_pipe,
)

asyncio.run(pipeline.run(max_processes=5))