# %%

from functools import partial
from nnsight import LanguageModel
import asyncio

import torch

from functools import partial

import torch
from nnsight import LanguageModel

from sae_auto_interp.autoencoders.wrapper import AutoencoderLatents
from sae_auto_interp.autoencoders.OpenAI import Autoencoder


from sae_auto_interp.features import FeatureCache
from sae_auto_interp.utils import load_tokenized_data


from sae_auto_interp.features import FeatureDataset, pool_max_activation_windows, random_activation_windows, sample
from sae_auto_interp.config import FeatureConfig, ExperimentConfig


from sae_auto_interp.explainers import SimpleExplainer
from sae_auto_interp.scorers import RecallScorer
from tqdm import tqdm
import pickle


from sae_auto_interp.clients import OpenRouter, Local
from sae_auto_interp.utils import display

# %%

CTX_LEN = 128
BATCH_SIZE = 32
N_TOKENS = 10_000_000
N_SPLITS = 2
RAW_ACTIVATIONS_DIR = "/media/jengels/sda/switch/gpt2"
NUM_FEATURES_TO_TEST = 1000

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

device = "cuda:1"

path = "/home/jengels/switch_sae/8.pt" # Change this line to your weights location.
state_dict = torch.load(path)
ae = Autoencoder.from_state_dict(state_dict=state_dict)
ae.to(device)

model = LanguageModel("openai-community/gpt2", device_map=device, dispatch=True)


tokens = load_tokenized_data(
    CTX_LEN,
    model.tokenizer,
    "Skylion007/openwebtext",
    "train[:15%]",
)

# %%

generate = False

if generate:

    def _forward(ae, x):
        latents, _ = ae.encode(x)
        return latents

    # We can simply add the new module as an attribute to an existing
    # submodule on GPT-2's module tree.
    submodule = model.transformer.h[8]
    submodule.ae = AutoencoderLatents(
        ae, 
        partial(_forward, ae),
        width=WIDTH
    )

    with model.edit(" ", inplace=True):
        acts = submodule.output[0]
        submodule.ae(acts, hook=True)

    with model.trace("hello, my name is"):
        latents = submodule.ae.output.save()

    module_path = submodule.path

    submodule_dict = {module_path : submodule}
    module_filter = {module_path : torch.arange(WIDTH // 10).to(device)}

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
    width = 131_072,
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

record_iterator = dataset.load(constructor=constructor, sampler=sampler)

# %%


next_record = next(record_iterator)

display(next_record, model.tokenizer, n=5)
# %%

# Command to run: vllm serve hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 --max_model_len 10000 --tensor-parallel-size 2
client = Local("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4")

# client = OpenRouter("meta-llama/llama-3.1-70b-instruct", api_key="INSERT_API_KEY_HERE")

# %%

async def run_async():

    global record_iterator

    save_file = "gpt2_layer_8_openai_feature_scores.pkl"

    positive_scores = []
    negative_scores = []
    explanations = []
    feature_ids = []
    total_positive_score = 0
    total_negative_score = 0
    total_evaluated = 0

    bar = tqdm(record_iterator)



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
        # score = asyncio.run(scorer(record))

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
                    
        # print(quantile_positives, quantile_totals)
        # print(negative_positives / negative_totals)
        # print(sum(quantile_positives) / sum(quantile_totals))   

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
        

        with open(save_file, "wb") as f:
            pickle.dump((positive_scores, negative_scores, explanations, feature_ids), f)

if is_notebook:
    run_async()
else:
    asyncio.run(run_async())

# %%
