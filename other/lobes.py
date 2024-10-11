#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
sys.path.append( # the switch_sae directory
    '/om2/user/ericjm/switch_sae'
)
os.environ['HF_HOME'] = '/om/user/ericjm/.cache/huggingface'
import json

import numpy as np
import torch as t
from dictionary_learning.trainers.switch import SwitchAutoEncoder

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# In[3]:


# load up the config from one of the training runs with 8 experts and 16k features
run = "../scaling_laws/attempt0/topk_switch8/dict_class:.SwitchAutoEncoder'>_activation_dim:768_k:32_experts:8_lb_alpha:3.0_heaviside:False_auxk_alpha:0.03125_decay_start:80000_steps:100000_seed:0_device:cuda:0_layer:8_lm_name:openai-communitygpt2_wandb_name:SwitchAutoEncoder_dict_size:16384"
with open(f"{run}/config.json", "r") as f:
    config = json.load(f)


# In[4]:


sae = SwitchAutoEncoder(
    activation_dim=config['trainer']['activation_dim'],
    dict_size=config['trainer']['dict_size'],
    k=config['trainer']['k'],
    experts=config['trainer']['experts'],
    heaviside=config['trainer']['heaviside'],
)

sae.load_state_dict(t.load(f"{run}/ae.pt", map_location=t.device("cpu")))


# In[8]:


sae.decoder.data.shape


# In[9]:


# perform tSNE projection of W_dec features
W_enc = sae.encoder.weight.detach().numpy()
W_dec = sae.decoder.data.detach().numpy()
tsne = TSNE(n_components=2, random_state=0, perplexity=50, learning_rate=100)
W_enc_tsne = tsne.fit_transform(W_enc)
tsne = TSNE(n_components=2, random_state=0, perplexity=50, learning_rate=100)
W_dec_tsne = tsne.fit_transform(W_dec)


# In[10]:


np.save(f"{run}/W_enc_tsne.npy", W_enc_tsne)
np.save(f"{run}/W_dec_tsne.npy", W_dec_tsne)


# In[ ]:


W_enc_tsne = np.load(f"{run}/W_enc_tsne.npy")
W_dec_tsne = np.load(f"{run}/W_dec_tsne.npy")


# In[11]:


f_per_expert = sae.dict_size // sae.experts


# In[12]:


f_per_expert


# In[14]:


colors = ['#FF4136', '#0074D9', '#FFDC00', '#2ECC40', '#F012BE', 
          '#7FDBFF', '#FF851B', '#01FF70', '#B10DC9', '#FF69B4']

cs = []
for i in range(sae.experts):
    cs.extend([colors[i]]*f_per_expert)


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
# plot the tSNE projection
plt.scatter(
    W_enc_tsne[:, 0], W_enc_tsne[:, 1],
    s=1,
    c=cs,
    alpha=0.5
)
plt.axis('off')

plt.subplot(1, 2, 2)
# plot the tSNE projection
plt.scatter(
    W_dec_tsne[:, 0], W_dec_tsne[:, 1],
    s=1,
    c=cs,
    alpha=0.5
)
plt.axis('off')


# # 32k features

# In[2]:


# load up the config from one of the training runs with 8 experts and 16k features
run = "../scaling_laws/attempt0/topk_switch8/dict_class:.SwitchAutoEncoder'>_activation_dim:768_k:32_experts:8_lb_alpha:3.0_heaviside:False_auxk_alpha:0.03125_decay_start:80000_steps:100000_seed:0_device:cuda:0_layer:8_lm_name:openai-communitygpt2_wandb_name:SwitchAutoEncoder_dict_size:32768"
with open(f"{run}/config.json", "r") as f:
    config = json.load(f)


# In[3]:


sae = SwitchAutoEncoder(
    activation_dim=config['trainer']['activation_dim'],
    dict_size=config['trainer']['dict_size'],
    k=config['trainer']['k'],
    experts=config['trainer']['experts'],
    heaviside=config['trainer']['heaviside'],
)

sae.load_state_dict(t.load(f"{run}/ae.pt", map_location=t.device("cpu")))


# In[4]:


# perform tSNE projection of W_dec features
W_enc = sae.encoder.weight.detach().numpy()
W_dec = sae.decoder.data.detach().numpy()
tsne = TSNE(n_components=2, random_state=0, perplexity=50, learning_rate=100)
W_enc_tsne = tsne.fit_transform(W_enc)
tsne = TSNE(n_components=2, random_state=0, perplexity=50, learning_rate=100)
W_dec_tsne = tsne.fit_transform(W_dec)


# In[5]:


# save the tSNE projection
np.save(f"{run}/W_enc_tsne.npy", W_enc_tsne)
np.save(f"{run}/W_dec_tsne.npy", W_dec_tsne)


# In[6]:


f_per_expert = sae.dict_size // sae.experts


# In[7]:


f_per_expert


# In[10]:


colors = ['#FF4136', '#0074D9', '#FFDC00', '#2ECC40', '#F012BE', 
          '#7FDBFF', '#FF851B', '#01FF70', '#B10DC9', '#FF69B4']

cs = []
for i in range(sae.experts):
    cs.extend([colors[i]]*f_per_expert)


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
# plot the tSNE projection
plt.scatter(
    W_enc_tsne[:, 0], W_enc_tsne[:, 1],
    s=1,
    c=cs,
    alpha=0.5
)
plt.title("Encoder features")
plt.axis('off')

plt.subplot(1, 2, 2)
# plot the tSNE projection
plt.scatter(
    W_dec_tsne[:, 0], W_dec_tsne[:, 1],
    s=1,
    c=cs,
    alpha=0.5
)
plt.title("Decoder features")
# add a legend for the experts (colors)
for i in range(sae.experts):
    plt.scatter([], [], color=colors[i], label=f"Expert {i}")
plt.legend(loc='upper right')
plt.axis('off')


# In[38]:


colors = ['#FF4136', '#0074D9', '#FFDC00', '#2ECC40', '#F012BE', 
          '#7FDBFF', '#FF851B', '#01FF70', '#B10DC9', '#FF69B4']

cs = []
for i in range(sae.experts):
    cs.extend([colors[i]]*f_per_expert)

plt.figure(figsize=(6, 6))

# plot the tSNE projection
plt.scatter(
    W_dec_tsne[:, 0], W_dec_tsne[:, 1],
    s=10,
    c=cs,
    alpha=1
)

plt.xlim(-15.275, -15.33)
plt.ylim(-10.345, -10.325)

# plt.axis('off')


# # 65k features

# In[2]:


# load up the config from one of the training runs with 8 experts and 16k features
run = "../scaling_laws/attempt0/topk_switch8/dict_class:.SwitchAutoEncoder'>_activation_dim:768_k:32_experts:8_lb_alpha:3.0_heaviside:False_auxk_alpha:0.03125_decay_start:80000_steps:100000_seed:0_device:cuda:0_layer:8_lm_name:openai-communitygpt2_wandb_name:SwitchAutoEncoder_dict_size:65536"
with open(f"{run}/config.json", "r") as f:
    config = json.load(f)


# In[3]:


sae = SwitchAutoEncoder(
    activation_dim=config['trainer']['activation_dim'],
    dict_size=config['trainer']['dict_size'],
    k=config['trainer']['k'],
    experts=config['trainer']['experts'],
    heaviside=config['trainer']['heaviside'],
)

sae.load_state_dict(t.load(f"{run}/ae.pt", map_location=t.device("cpu")))


# In[4]:


W_enc = sae.encoder.weight.detach().numpy()
W_dec = sae.decoder.data.detach().numpy()


# In[8]:


tsne = TSNE(n_components=2, random_state=0, perplexity=50, learning_rate=100)
W_enc_tsne = tsne.fit_transform(W_enc)


# In[9]:


tsne = TSNE(n_components=2, random_state=0, perplexity=50, learning_rate=100)
W_dec_tsne = tsne.fit_transform(W_dec)


# In[10]:


# save the tSNE projection
np.save(f"{run}/W_enc_tsne.npy", W_enc_tsne)
np.save(f"{run}/W_dec_tsne.npy", W_dec_tsne)


# In[5]:


W_enc_tsne = np.load(f"{run}/W_enc_tsne.npy")
W_dec_tsne = np.load(f"{run}/W_dec_tsne.npy")


# In[6]:


f_per_expert = sae.dict_size // sae.experts


# In[7]:


f_per_expert


# In[10]:


colors = ['#FF4136', '#0074D9', '#FFDC00', '#2ECC40', '#F012BE', 
          '#7FDBFF', '#FF851B', '#01FF70', '#B10DC9', '#FF69B4']

cs = []
for i in range(sae.experts):
    cs.extend([colors[i]]*f_per_expert)


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
# plot the tSNE projection
plt.scatter(
    W_enc_tsne[:, 0], W_enc_tsne[:, 1],
    s=1,
    c=cs,
    alpha=0.5
)
plt.title("Encoder features")
# add a legend for the experts (colors)
for i in range(sae.experts):
    plt.scatter([], [], color=colors[i], label=f"Expert {i}")
plt.legend(loc='lower left', ncol=4, borderaxespad=0., fontsize=8)
plt.axis('off')

plt.subplot(1, 2, 2)
# plot the tSNE projection
plt.scatter(
    W_dec_tsne[:, 0], W_dec_tsne[:, 1],
    s=1,
    c=cs,
    alpha=0.5
)
plt.title("Decoder features")
plt.axis('off')
plt.tight_layout()


# In[34]:


colors = ['#FF4136', '#0074D9', '#FFDC00', '#2ECC40', '#F012BE', 
          '#7FDBFF', '#FF851B', '#01FF70', '#B10DC9', '#FF69B4']

cs = []
for i in range(sae.experts):
    cs.extend([colors[i]]*f_per_expert)


plt.figure(figsize=(5.5, 2.8))

plt.subplot(1, 2, 1)
# plot the tSNE projection
plt.scatter(
    W_enc_tsne[:, 0], W_enc_tsne[:, 1],
    s=0.3,
    c=cs,
    alpha=0.5
)
plt.title("Encoder features", fontsize=9)
# add a legend for the experts (colors)
for i in range(sae.experts):
    plt.scatter([], [], color=colors[i], label=f"Expert {i}")
plt.legend(loc='lower right', ncol=3, borderaxespad=0., fontsize=5)
plt.axis('off')

plt.subplot(1, 2, 2)
# plot the tSNE projection
plt.scatter(
    W_dec_tsne[:, 0], W_dec_tsne[:, 1],
    s=0.3,
    c=cs,
    alpha=0.5
)
plt.title("Decoder features", fontsize=9)
plt.axis('off')
plt.tight_layout()


# In[11]:


colors = ['#FF4136', '#0074D9', '#FFDC00', '#2ECC40', '#F012BE', 
          '#7FDBFF', '#FF851B', '#01FF70', '#B10DC9', '#FF69B4']
cs = []
for i in range(sae.experts):
    cs.extend([colors[i]]*f_per_expert)
cs = np.array(cs)

# break into 20 random splits
n_splits = 20
splits = np.array_split(np.random.permutation(sae.dict_size), n_splits)

fig = plt.figure(figsize=(5.5, 2.75))

# First subplot
ax1 = fig.add_subplot(121)
for split in splits:
    ax1.scatter(
        W_enc_tsne[split, 0], W_enc_tsne[split, 1],
        s=0.3,
        c=cs[split],
        alpha=0.5
    )
ax1.set_title("Encoder features", fontsize=9, pad=-10)
ax1.axis('off')

# Second subplot
ax2 = fig.add_subplot(122)
for split in splits:
    ax2.scatter(
        W_dec_tsne[split, 0], W_dec_tsne[split, 1],
        s=0.3,
        c=cs[split],
        alpha=0.5
    )
ax2.set_title("Decoder features", fontsize=9, pad=-10)
ax2.axis('off')

# Create legend elements
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'Expert {i}',
                              markerfacecolor=colors[i], markersize=5)
                   for i in range(sae.experts)]

# Add the legend to the figure
fig.legend(handles=legend_elements, loc='lower center', ncol=8, fontsize=5, 
           bbox_to_anchor=(0.5, 0.02))

plt.tight_layout(pad=0.1)
# Adjust the subplot positions to make room for the legend
plt.subplots_adjust(bottom=0.08)

# plt.show()
# plt.savefig("tsne-twopanel.png", dpi=300)


# In[12]:


colors = ['#FF4136', '#0074D9', '#FFDC00', '#2ECC40', '#F012BE', 
          '#7FDBFF', '#FF851B', '#01FF70', '#B10DC9', '#FF69B4']
cs = []
for i in range(sae.experts):
    cs.extend([colors[i]]*f_per_expert)
cs = np.array(cs)

# break into 20 random splits
n_splits = 20
splits = np.array_split(np.random.permutation(sae.dict_size), n_splits)

fig = plt.figure(figsize=(5.5, 2))

# First subplot
ax1 = fig.add_subplot(132)
for split in splits:
    ax1.scatter(
        W_enc_tsne[split, 0], W_enc_tsne[split, 1],
        s=0.3,
        c=cs[split],
        alpha=0.5
    )
ax1.set_title("Encoder features", fontsize=8, pad=-10)
ax1.axis('off')

# draw a box around these points
# plt.xlim(-19.4, -18.2)
# plt.ylim(1.8, 3.00)
plt.plot([-19.4, -19.4], [1.8, 3.0], color='black', lw=0.5)
plt.plot([-19.4, -18.2], [1.8, 1.8], color='black', lw=0.5)
plt.plot([-18.2, -18.2], [1.8, 3.0], color='black', lw=0.5)
plt.plot([-19.4, -18.2], [3.0, 3.0], color='black', lw=0.5)


# Second subplot
ax2 = fig.add_subplot(133)
for split in splits:
    ax2.scatter(
        W_dec_tsne[split, 0], W_dec_tsne[split, 1],
        s=0.3,
        c=cs[split],
        alpha=0.5
    )
ax2.set_title("Decoder features", fontsize=8, pad=-10)
ax2.axis('off')

# Create legend elements
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'Expert {i}',
                              markerfacecolor=colors[i], markersize=5)
                   for i in range(sae.experts)]

# Add the legend to the figure
fig.legend(handles=legend_elements, loc='lower center', ncol=8, fontsize=5, 
           bbox_to_anchor=(0.50, 0.02))

plt.tight_layout(pad=0.1)
# Adjust the subplot positions to make room for the legend
plt.subplots_adjust(bottom=0.08)

# plt.show()
plt.savefig("tsne-threepanel.png", dpi=300)


# In[54]:


colors = ['#FF4136', '#0074D9', '#FFDC00', '#2ECC40', '#F012BE', 
          '#7FDBFF', '#FF851B', '#01FF70', '#B10DC9', '#FF69B4']
cs = []
for i in range(sae.experts):
    cs.extend([colors[i]]*f_per_expert)
cs = np.array(cs)

# break into 20 random splits
n_splits = 20
splits = np.array_split(np.random.permutation(sae.dict_size), n_splits)

fig = plt.figure(figsize=(1.5, 1.5))

# First subplot
ax1 = fig.add_subplot(111)
for split in splits:
    ax1.scatter(
        W_enc_tsne[split, 0], W_enc_tsne[split, 1],
        s=8,
        c=cs[split],
        alpha=1
    )
# ax1.set_title("Encoder features", fontsize=9, pad=-10)
# ax1.axis('off')
plt.xticks([])
plt.yticks([])

plt.tight_layout(pad=0.1)
# Adjust the subplot positions to make room for the legend
# plt.subplots_adjust(bottom=0.08)

plt.xlim(-19.5, -18.2)
plt.ylim(1.75, 3.05)

# plt.show()
plt.savefig("tsne-zoomed.png", dpi=200)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[25]:


import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def create_interactive_scatter(W_dec_tsne, cs, sae):
    # Create a mapping of colors to expert numbers
    color_to_expert = {color: f"Expert {i}" for i, color in enumerate(np.unique(cs))}

    # Create the scatter plot
    fig = go.Figure()

    fig.add_trace(go.Scattergl(
        x=W_dec_tsne[:, 0],
        y=W_dec_tsne[:, 1],
        mode='markers',
        marker=dict(
            color=cs,
            size=3,
            opacity=0.5
        ),
        hoverinfo='none'
    ))

    # Update the layout
    fig.update_layout(
        title="Decoder Features",
        width=800,
        height=800,
        xaxis_title="tSNE 1",
        yaxis_title="tSNE 2",
        showlegend=False,
    )

    # Add buttons for resetting axes and toggling spike lines
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(args=[{'xaxis.autorange': True, 'yaxis.autorange': True}],
                         label="Reset Axes",
                         method="relayout"),
                    dict(args=[{"xaxis.showspikes": True, "yaxis.showspikes": True}],
                         label="Show Spike Lines",
                         method="relayout"),
                    dict(args=[{"xaxis.showspikes": False, "yaxis.showspikes": False}],
                         label="Hide Spike Lines",
                         method="relayout")
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.11,
                xanchor="left",
                y=1.1,
                yanchor="top"
            ),
        ]
    )

    # Add annotations for expert colors
    for color, expert in color_to_expert.items():
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            showlegend=True,
            name=expert
        ))

    # Update layout for legend
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=10)
        )
    )

    return fig

# Create and display the plot
fig = create_interactive_scatter(W_enc_tsne, cs, sae)
fig.show()


# In[ ]:





# In[ ]:





# In[21]:


colors = ['#FF4136', '#0074D9', '#FFDC00', '#2ECC40', '#F012BE', 
          '#7FDBFF', '#FF851B', '#01FF70', '#B10DC9', '#FF69B4']

cs = []
for i in range(sae.experts):
    cs.extend([colors[i]]*f_per_expert)

plt.figure(figsize=(6, 6))

# plot the tSNE projection
plt.scatter(
    W_dec_tsne[:, 0], W_dec_tsne[:, 1],
    s=1,
    c=cs,
    alpha=0.5
)
plt.axis('off')


# In[22]:


# run the tsne on the encoder again
tsne = TSNE(n_components=2, random_state=0, perplexity=50, learning_rate=100)
# fit with the encoder weights
tsne.fit(W_enc)
# project the decoder weights with the encoder tSNE
W_dec_tsne_enc_transform = tsne.transform(W_dec)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[16]:





# In[ ]:





# In[17]:


expert_W_decs = [
    sae.encoder.weight[i:i+f_per_expert].detach().cpu().numpy() for i in range(0, sae.dict_size, f_per_expert)
]


# In[22]:


expert_W_decs[0].shape


# In[13]:


sae.encoder.weight


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


parser = argparse.ArgumentParser()
parser.add_argument("--num_experts", type=int, default=8, required=True)
parser.add_argument("--lb_alpha", type=float, default=3.0)
parser.add_argument("--heaviside", type=str2bool, default=False)
args = parser.parse_args()

lm = 'openai-community/gpt2'
activation_dim = 768
layer = 8
hf = 'Skylion007/openwebtext'
steps = 100_000 
n_ctxs = int(1e4)

dict_sizes = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
k = 32

save_dir = f"topk_switch{args.num_experts}"

device = f'cuda:0'
model = LanguageModel(lm, dispatch=True, device_map=device)
submodule = model.transformer.h[layer]
data = hf_dataset_to_generator(hf)
buffer = ActivationBuffer(data, model, submodule, d_submodule=activation_dim, n_ctxs=n_ctxs, device=device)

base_trainer_config = {
    'trainer' : SwitchTrainer,
    'dict_class' : SwitchAutoEncoder,
    'activation_dim' : activation_dim,
    'k': k,
    'experts' : args.num_experts,
    'lb_alpha' : args.lb_alpha,
    'heaviside' : args.heaviside,
    'auxk_alpha' : 1/32,
    'decay_start' : int(steps * 0.8),
    'steps' : steps,
    'seed' : 0,
    'device' : device,
    'layer' : layer,
    'lm_name' : lm,
    'wandb_name' : 'SwitchAutoEncoder'
}

trainer_configs = [(base_trainer_config | {'dict_size': ds}) for ds in dict_sizes]
# {'k': combo[0], 'experts': combo[1], 'heaviside': combo[2], 'lb_alpha': combo[3]}) for combo in itertools.product(args.ks, args.num_experts, args.heavisides, args.lb_alphas)]

wandb.init(entity="ericjmichaud_", project="switch_saes_scaling_laws_attempt_0", config={f"{trainer_config['wandb_name']}-{i}" : trainer_config for i, trainer_config in enumerate(trainer_configs)})
# wandb.init(entity="amudide", project="Switch (LB)", config={f'{trainer_config["wandb_name"]}-{i}' : trainer_config for i, trainer_config in enumerate(trainer_configs)})

trainSAE(buffer, trainer_configs=trainer_configs, save_dir=save_dir, log_steps=1, steps=steps)

print("Training finished. Evaluating SAE...", flush=True)
for i, trainer_config in enumerate(trainer_configs):
    ae = SwitchAutoEncoder.from_pretrained(
        os.path.join(save_dir, f'{cfg_filename(trainer_config)}/ae.pt'),
        k = trainer_config['k'], 
        experts = trainer_config['experts'], 
        heaviside = trainer_config['heaviside'], 
        device=device
    )
    metrics = evaluate(ae, buffer, device=device)
    log = {}
    log.update({f'{trainer_config["wandb_name"]}-{i}/{k}' : v for k, v in metrics.items()})
    wandb.log(log, step=steps+1)
wandb.finish()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




