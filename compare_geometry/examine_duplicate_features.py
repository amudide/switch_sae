# %%
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import pandas as pd
import torch
from dictionary_learning.trainers.switch import SwitchAutoEncoder
from dictionary_learning.trainers.top_k import AutoEncoderTopK
import einops
from tqdm import tqdm
from transformers import GPT2Tokenizer

torch.set_grad_enabled(False)

# %%

# device = "cuda:0"
device = "cpu"
duplicate_features = pd.read_csv('../data/duplicates.csv')
data = torch.load("../data/gpt2_activations_layer8.pt", map_location=device)
tokens = torch.load("../data/gpt2_tokens.pt", map_location=device)
save_location = "../data/top_activating_for_dupes_layer8.pt"
ctx_len = 128

# %%

batch_size = 256
store_topk_activating = 100

unique_num_experts = duplicate_features['num_experts'].unique()
unique_topks = duplicate_features['k'].unique()

ae_details_to_top_activating = {}


for num_experts in unique_num_experts:
    for k in tqdm(unique_topks):
        filtered_df = duplicate_features[(duplicate_features['num_experts'] == num_experts) & (duplicate_features['k'] == k)]
        feature_index_to_num_dupes = filtered_df.set_index('feature_index')['num_dupes'].to_dict()
        feature_index_tensor = torch.tensor(list(feature_index_to_num_dupes.keys()), device=device)
        feature_index_to_topk_activating = {feature_index: [] for feature_index in feature_index_to_num_dupes.keys()}

        if feature_index_tensor.numel() == 0:
            ae_details_to_top_activating[(num_experts, k)] = {}
            continue

        if num_experts == 1:
            ae = AutoEncoderTopK.from_pretrained(f"../dictionaries/topk/k{k}/ae.pt", k=k, device=device)
        else:
            ae = SwitchAutoEncoder.from_pretrained(f"../dictionaries/fixed-width/{num_experts}_experts/k{k}/ae.pt", k=k, experts=num_experts, device=device)

        key = (num_experts, k)

        for batch_start in range(0, len(data), batch_size):
            batch = data[batch_start:batch_start+batch_size].to(device)
            batch_tokens = tokens[batch_start:batch_start+batch_size].to(device)

            top_acts, top_indices = ae.forward(batch, output_features="all")[2:]
            activations = torch.zeros(batch_tokens.shape[0] * batch_tokens.shape[1], len(ae.decoder), device=device)

            activations.scatter_(1, top_indices, top_acts)
            dupe_feature_activations = activations[..., feature_index_tensor]
            top_activating = torch.topk(dupe_feature_activations, store_topk_activating, dim=0)
            for j, feature_index in enumerate(feature_index_tensor):
                top_activating_indices = top_activating.indices[:, j]
                top_activating_values = top_activating.values[:, j]
                top_activating_contexts = top_activating_indices // ctx_len + batch_start
                top_activating_token_ids = top_activating_indices % ctx_len
                feature_index_to_topk_activating[feature_index.item()].extend(list(zip(top_activating_contexts.cpu().numpy(), top_activating_token_ids.cpu().numpy(), top_activating_values.cpu().numpy())))

        # Get the top k activations for each feature index
        for feature_index, top_activating in feature_index_to_topk_activating.items():
            num_dupes = feature_index_to_num_dupes[feature_index]
            top_activating.sort(key=lambda x: x[2], reverse=True)
            feature_index_to_topk_activating[feature_index] = (top_activating[:store_topk_activating], num_dupes)

        ae_details_to_top_activating[key] = feature_index_to_topk_activating

        torch.save(ae_details_to_top_activating, save_location)

# %%

# Load the top activating for duplicates
ae_details_to_top_activating = torch.load(save_location, map_location=device)

context_limit = 15

# Load GPT2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

for (num_experts, k), feature_index_to_topk_activating in tqdm(ae_details_to_top_activating.items()):
    output_file = f"../data/top_activating_for_dupes_layer8_num-experts{num_experts}_topk{k}.txt"
    
    if num_experts == 1:
        ae = AutoEncoderTopK.from_pretrained(f"../dictionaries/topk/k{k}/ae.pt", k=k, device=device)
    else:
        ae = SwitchAutoEncoder.from_pretrained(f"../dictionaries/fixed-width/{num_experts}_experts/k{k}/ae.pt", k=k, experts=num_experts, device=device)

    decoder_vecs = ae.decoder
    decoder_vecs = decoder_vecs / decoder_vecs.norm(dim=-1, keepdim=True)
    sims = decoder_vecs @ decoder_vecs.T
    threshold = 0.9
    # Zero diagonal of sims
    sims.fill_diagonal_(0)

    encoder_vecs = ae.encoder.weight
    encoder_vecs = encoder_vecs / encoder_vecs.norm(dim=-1, keepdim=True)
    encoder_sims = encoder_vecs @ encoder_vecs.T
    
    with open(output_file, 'w') as f:
        f.write(f"--------------- NUM EXPERTS = {num_experts}, K = {k} ---------------\n\n\n")
        for feature_index, (top_activating, num_dupes) in feature_index_to_topk_activating.items():
            similar = torch.nonzero(sims[feature_index] > threshold).flatten().cpu().numpy()
            nonzero_sims = sims[feature_index][similar].cpu().numpy()
            this_encoder_sims = encoder_sims[feature_index][similar].cpu().numpy()
            f.write(f"feature {feature_index} with {num_dupes} dupes: {similar}, {nonzero_sims}, {this_encoder_sims}\n")
            for context_index, token_index, value in top_activating:
                context_start = max(0, token_index - context_limit)
                context_end = min(token_index, ctx_len)
                token_context = tokens[context_index][context_start:context_end]
                token_context_strs = [tokenizer.decode(token) for token in token_context]
                context = "".join(token_context_strs)
                context = context.replace("\n", " ")
                if context_end + 1 < ctx_len:
                    next_token = tokenizer.decode(tokens[context_index][context_end: context_end + 1][0])
                    next_token = next_token.replace("\n", "<new line>")
                else:
                    next_token = ""
                f.write(f"{value:.4f} {next_token}\n{context}\n")
            f.write("\n")
        f.write("\n")
# %%
