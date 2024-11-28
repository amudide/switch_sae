# %%

import itertools

# Generate all combinations of parameters
layers = [2, 4, 6, 8, 10]
sae_types = ['switch', 'topk']
types = ['resid', 'attn', 'mlp']
devices = [f'cuda:{i}' for i in range(8)]

# First set of commands for GPT-2
gpt2_commands = []
for layer, sae_type, type_ in itertools.product(layers, sae_types, types):
    cmd = f"python3 table/train_switch_table.py --device cuda:{len(gpt2_commands) % 8} --layer {layer} --lm openai-community/gpt2 --ks 64 --activation_dim 768 --dict_ratio 32 --num_experts 8 --type {type_} --sae_type {sae_type} --steps 20000 &"
    gpt2_commands.append(cmd)

# Second set of commands for Gemma-2B
gemma_commands = []
for i, sae_type in enumerate(sae_types):
    cmd = f"python3 table/train_switch_table.py --device cuda:{i} --layer 12 --lm google/gemma-2b --ks 64 --activation_dim 2048 --dict_ratio 32 --num_experts 8 --type resid --sae_type {sae_type} --steps 2000000 &"
    gemma_commands.append(cmd)

# Write commands to a bash script
with open('run_parallel.sh', 'w') as f:
    f.write('#!/bin/bash\n\n')
    
    # Write GPT-2 commands in batches of 8
    f.write('# GPT-2 training commands\n')
    for i, cmd in enumerate(gpt2_commands):
        f.write(f'{cmd}\n')
        if (i + 1) % 8 == 0:
            f.write('\nwait\n\n')  # Wait after every 8 commands
    
    # If there are remaining commands that don't make a full batch of 8
    if len(gpt2_commands) % 8:
        f.write('\nwait\n\n')
    
    # Write Gemma commands
    f.write('# Gemma-2B training commands\n')
    for i, cmd in enumerate(gemma_commands):
        f.write(f'{cmd}\n')
    
    f.write('\nwait\n')  # Wait for all commands to finish

# Make the script executable
import os
os.chmod('run_parallel.sh', 0o755)

print(f"Created run_parallel.sh with {len(gpt2_commands)} GPT-2 commands and {len(gemma_commands)} Gemma commands")

