# %%
# Imports
import torch as t
from nnsight import LanguageModel
import argparse
import itertools
import gc

from dictionary_learning.training import trainSAE
from dictionary_learning.trainers.standard import StandardTrainer
from dictionary_learning.trainers.top_k import TopKTrainer, AutoEncoderTopK
from dictionary_learning.utils import zst_to_generator
from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.dictionary import AutoEncoder

# %%
DEVICE = "cuda:0"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True, help="where to store sweep")
    parser.add_argument("--no_wandb_logging", action="store_true", help="omit wandb logging")
    parser.add_argument("--dry_run", action="store_true", help="dry run sweep")
    parser.add_argument("--layer", type=int, required=True, help="layer to train SAE on")
    args = parser.parse_args()
    return args


def run_sae_training(
    layer: int,
    save_dir: str,
    device: str,
    dry_run: bool = False,
    no_wandb_logging: bool = False,
):

    # model and data parameters
    model_name = "google/gemma-2b"
    dataset_name = '/share/data/datasets/pile/the-eye.eu/public/AI/pile/train/00.jsonl.zst'
    context_length = 64

    buffer_size = int(1e3)
    llm_batch_size = 128  # 256 for A100 GPU, 64 for 1080ti
    sae_batch_size = 2048
    num_tokens = 200_000_000

    # sae training parameters
    # random_seeds = t.arange(10).tolist()
    random_seeds = [0]
    initial_sparsity_penalties = [0.005, 0.01, 0.05]
    ks = [20, 100, 200]
    ks = {p: ks[i] for i, p in enumerate(initial_sparsity_penalties)}
    expansion_factors = [8, 16]

    steps = int(num_tokens / sae_batch_size)  # Total number of batches to train
    save_steps = None
    warmup_steps = 1000  # Warmup period at start of training and after each resample
    resample_steps = None

    # standard sae training parameters
    learning_rate = 0.0001

    # topk sae training parameters
    decay_start = 24000
    auxk_alpha = 1 / 32

    log_steps = 5  # Log the training on wandb
    if no_wandb_logging:
        log_steps = None

    model = LanguageModel(model_name, dispatch=True, device_map=DEVICE)
    submodule = model.model.layers[layer]
    submodule_name = f"resid_post_layer_{layer}"
    io = "out"
    activation_dim = model.config.hidden_size

    generator = zst_to_generator(dataset_name)

    activation_buffer = ActivationBuffer(
        generator,
        model,
        submodule,
        n_ctxs=buffer_size,
        ctx_len=context_length,
        refresh_batch_size=llm_batch_size,
        out_batch_size=sae_batch_size,
        io=io,
        d_submodule=activation_dim,
        device=device,
    )

    # create the list of configs
    trainer_configs = []
    for seed, initial_sparsity_penalty, expansion_factor in itertools.product(
        random_seeds, initial_sparsity_penalties, expansion_factors
    ):
        trainer_configs.extend([
            {
                "trainer": StandardTrainer,
                "dict_class": AutoEncoder,
                "activation_dim": activation_dim,
                "dict_size": expansion_factor * activation_dim,
                "lr": learning_rate,
                "l1_penalty": initial_sparsity_penalty,
                "warmup_steps": warmup_steps,
                "resample_steps": resample_steps,
                "seed": seed,
                "wandb_name": f"StandardTrainer-{model_name}-{submodule_name}",
                "layer": layer,
                "lm_name": model_name,
                "device": device,
            },
            {
                "trainer": TopKTrainer,
                "dict_class": AutoEncoderTopK,
                "activation_dim": activation_dim,
                "dict_size": expansion_factor * activation_dim,
                "k": ks[initial_sparsity_penalty],
                "auxk_alpha": auxk_alpha,  # see Appendix A.2
                "decay_start": decay_start,  # when does the lr decay start
                "steps": steps,  # when when does training end
                "seed": seed,
                "device": DEVICE,
                "layer": layer,
                "lm_name": model_name,
                "wandb_name": f"TopKTrainer-{model_name}-{submodule_name}",
            },

        ])

    print(f"len trainer configs: {len(trainer_configs)}")
    save_dir = f"{save_dir}/{submodule_name}"

    if not dry_run:
        # actually run the sweep
        trainSAE(
            data=activation_buffer,
            trainer_configs=trainer_configs,
            steps=steps,
            save_steps=save_steps,
            save_dir=save_dir,
            log_steps=log_steps,
        )


if __name__ == "__main__":
    args = get_args()
    run_sae_training(
        layer=args.layer,
        save_dir=args.save_dir,
        device="cuda:0",
        dry_run=args.dry_run,
        no_wandb_logging=args.no_wandb_logging,
    )
