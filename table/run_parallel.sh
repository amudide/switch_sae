#!/bin/bash

# GPT-2 training commands
python3 table/train_switch_table.py --device cuda:0 --layer 2 --lm openai-community/gpt2 --ks 64 --activation_dim 768 --dict_ratio 32 --num_experts 8 --type resid --sae_type switch --steps 20000 &
python3 table/train_switch_table.py --device cuda:1 --layer 2 --lm openai-community/gpt2 --ks 64 --activation_dim 768 --dict_ratio 32 --num_experts 8 --type attn --sae_type switch --steps 20000 &
python3 table/train_switch_table.py --device cuda:2 --layer 2 --lm openai-community/gpt2 --ks 64 --activation_dim 768 --dict_ratio 32 --num_experts 8 --type mlp --sae_type switch --steps 20000 &
python3 table/train_switch_table.py --device cuda:3 --layer 2 --lm openai-community/gpt2 --ks 64 --activation_dim 768 --dict_ratio 32 --num_experts 8 --type resid --sae_type topk --steps 20000 &
python3 table/train_switch_table.py --device cuda:4 --layer 2 --lm openai-community/gpt2 --ks 64 --activation_dim 768 --dict_ratio 32 --num_experts 8 --type attn --sae_type topk --steps 20000 &
python3 table/train_switch_table.py --device cuda:5 --layer 2 --lm openai-community/gpt2 --ks 64 --activation_dim 768 --dict_ratio 32 --num_experts 8 --type mlp --sae_type topk --steps 20000 &
python3 table/train_switch_table.py --device cuda:6 --layer 4 --lm openai-community/gpt2 --ks 64 --activation_dim 768 --dict_ratio 32 --num_experts 8 --type resid --sae_type switch --steps 20000 &
python3 table/train_switch_table.py --device cuda:7 --layer 4 --lm openai-community/gpt2 --ks 64 --activation_dim 768 --dict_ratio 32 --num_experts 8 --type attn --sae_type switch --steps 20000 &

wait

python3 table/train_switch_table.py --device cuda:0 --layer 4 --lm openai-community/gpt2 --ks 64 --activation_dim 768 --dict_ratio 32 --num_experts 8 --type mlp --sae_type switch --steps 20000 &
python3 table/train_switch_table.py --device cuda:1 --layer 4 --lm openai-community/gpt2 --ks 64 --activation_dim 768 --dict_ratio 32 --num_experts 8 --type resid --sae_type topk --steps 20000 &
python3 table/train_switch_table.py --device cuda:2 --layer 4 --lm openai-community/gpt2 --ks 64 --activation_dim 768 --dict_ratio 32 --num_experts 8 --type attn --sae_type topk --steps 20000 &
python3 table/train_switch_table.py --device cuda:3 --layer 4 --lm openai-community/gpt2 --ks 64 --activation_dim 768 --dict_ratio 32 --num_experts 8 --type mlp --sae_type topk --steps 20000 &
python3 table/train_switch_table.py --device cuda:4 --layer 8 --lm openai-community/gpt2 --ks 64 --activation_dim 768 --dict_ratio 32 --num_experts 8 --type resid --sae_type switch --steps 20000 &
python3 table/train_switch_table.py --device cuda:5 --layer 8 --lm openai-community/gpt2 --ks 64 --activation_dim 768 --dict_ratio 32 --num_experts 8 --type attn --sae_type switch --steps 20000 &
python3 table/train_switch_table.py --device cuda:6 --layer 8 --lm openai-community/gpt2 --ks 64 --activation_dim 768 --dict_ratio 32 --num_experts 8 --type mlp --sae_type switch --steps 20000 &
python3 table/train_switch_table.py --device cuda:7 --layer 8 --lm openai-community/gpt2 --ks 64 --activation_dim 768 --dict_ratio 32 --num_experts 8 --type resid --sae_type topk --steps 20000 &

wait

python3 table/train_switch_table.py --device cuda:0 --layer 8 --lm openai-community/gpt2 --ks 64 --activation_dim 768 --dict_ratio 32 --num_experts 8 --type attn --sae_type topk --steps 20000 &
python3 table/train_switch_table.py --device cuda:1 --layer 8 --lm openai-community/gpt2 --ks 64 --activation_dim 768 --dict_ratio 32 --num_experts 8 --type mlp --sae_type topk --steps 20000 &
python3 table/train_switch_table.py --device cuda:2 --layer 10 --lm openai-community/gpt2 --ks 64 --activation_dim 768 --dict_ratio 32 --num_experts 8 --type resid --sae_type switch --steps 20000 &
python3 table/train_switch_table.py --device cuda:3 --layer 10 --lm openai-community/gpt2 --ks 64 --activation_dim 768 --dict_ratio 32 --num_experts 8 --type attn --sae_type switch --steps 20000 &
python3 table/train_switch_table.py --device cuda:4 --layer 10 --lm openai-community/gpt2 --ks 64 --activation_dim 768 --dict_ratio 32 --num_experts 8 --type mlp --sae_type switch --steps 20000 &
python3 table/train_switch_table.py --device cuda:5 --layer 10 --lm openai-community/gpt2 --ks 64 --activation_dim 768 --dict_ratio 32 --num_experts 8 --type resid --sae_type topk --steps 20000 &
python3 table/train_switch_table.py --device cuda:6 --layer 10 --lm openai-community/gpt2 --ks 64 --activation_dim 768 --dict_ratio 32 --num_experts 8 --type attn --sae_type topk --steps 20000 &
python3 table/train_switch_table.py --device cuda:7 --layer 10 --lm openai-community/gpt2 --ks 64 --activation_dim 768 --dict_ratio 32 --num_experts 8 --type mlp --sae_type topk --steps 20000 &

wait

python3 table/train_switch_table.py --device cuda:0 --layer 12 --lm openai-community/gpt2 --ks 64 --activation_dim 768 --dict_ratio 32 --num_experts 8 --type resid --sae_type switch --steps 20000 &
python3 table/train_switch_table.py --device cuda:1 --layer 12 --lm openai-community/gpt2 --ks 64 --activation_dim 768 --dict_ratio 32 --num_experts 8 --type attn --sae_type switch --steps 20000 &
python3 table/train_switch_table.py --device cuda:2 --layer 12 --lm openai-community/gpt2 --ks 64 --activation_dim 768 --dict_ratio 32 --num_experts 8 --type mlp --sae_type switch --steps 20000 &
python3 table/train_switch_table.py --device cuda:3 --layer 12 --lm openai-community/gpt2 --ks 64 --activation_dim 768 --dict_ratio 32 --num_experts 8 --type resid --sae_type topk --steps 20000 &
python3 table/train_switch_table.py --device cuda:4 --layer 12 --lm openai-community/gpt2 --ks 64 --activation_dim 768 --dict_ratio 32 --num_experts 8 --type attn --sae_type topk --steps 20000 &
python3 table/train_switch_table.py --device cuda:5 --layer 12 --lm openai-community/gpt2 --ks 64 --activation_dim 768 --dict_ratio 32 --num_experts 8 --type mlp --sae_type topk --steps 20000 &

wait

# Gemma-2B training commands
python3 table/train_switch_table.py --device cuda:0 --layer 12 --lm google/gemma-2b --ks 64 --activation_dim 2048 --dict_ratio 32 --num_experts 8 --type resid --sae_type switch --steps 2000000 &
python3 table/train_switch_table.py --device cuda:1 --layer 12 --lm google/gemma-2b --ks 64 --activation_dim 2048 --dict_ratio 32 --num_experts 8 --type resid --sae_type topk --steps 2000000 &

wait
