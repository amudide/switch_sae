#%%

from nnsight import LanguageModel
from dictionary_learning.utils import hf_dataset_to_generator
from config import lm, activation_dim, layer, hf, n_ctxs
import torch as t
import einops
from tqdm import tqdm
import os

t.set_grad_enabled(False)

# %%


device = f'cuda:1'
model = LanguageModel(lm, dispatch=True, device_map=device)
submodule = model.transformer.h[layer]
data = hf_dataset_to_generator(hf)

# %%
batch_size = 256
num_batches = 128
ctx_len = 128

total_tokens = batch_size * num_batches * ctx_len
total_memory = total_tokens * activation_dim * 4 
print(f"Total contexts: {batch_size * num_batches / 1e3:.2f}K")
print(f"Total tokens: {total_tokens / 1e6:.2f}M")
print(f"Total memory: {total_memory / 1e9:.2f}GB")

# %%

# These functions copied from buffer.py

def text_batch():
    return [
        next(data) for _ in range(batch_size)
    ]

def tokenized_batch():
    texts = text_batch()
    return model.tokenizer(
        texts,
        return_tensors='pt',
        max_length=ctx_len,
        padding=True,
        truncation=True
    )

def get_activations(input):
    with t.no_grad():
        with model.trace(input):
            hidden_states = submodule.output.save()
        hidden_states = hidden_states.value
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        hidden_states = hidden_states[input['attention_mask'] != 0]
    return hidden_states
            
        

# %%

all_activations = []
all_tokens = []

for _ in tqdm(range(num_batches)):
    batch = tokenized_batch()
    all_tokens.append(batch['input_ids'].cpu())
    activations = get_activations(batch)
    activations = einops.rearrange(activations, "(b c) d -> b c d", b=batch_size)
    all_activations.append(activations.cpu())
    
# %%

concatenated_activations = t.cat(all_activations)
concatenated_tokens = t.cat(all_tokens)
print(concatenated_activations.shape, concatenated_tokens.shape)

# %%

# save activations 
os.makedirs('data', exist_ok=True)
t.save(concatenated_activations, f'data/gpt2_activations_layer{layer}.pt')
t.save(concatenated_tokens, f'data/tokens.pt')