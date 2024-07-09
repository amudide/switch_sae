<h3 align="center">
  Efficient Dictionary Learning via Switch Sparse Autoencoders (SAEs)
</h3>

## To-Do
* Larger scale runs
  - Check that eval is consistent given a trained SAE (7/7)
  - Check if I can read models from saved state dicts (7/7)
  - Check if scale of MSE is correct (compare with ProLU paper) (7/7)
  - Change ln_1 to blank
  - Geometric median (7/8)
  - Anthropic new loss + unconstrained Adam (7/8)
  - Load balancing (7/9)
  - More sweeps (e, experts, k, lr, dict_ratio, etc.) (7/10)
  - TopK auxiliary loss
  - Experiment with (Heaviside vs. Softmax) & (TopK per expert vs. TopK across chosen experts)
  - Experiment with switch SAE (heaviside vs. weighted)
  - Don't zero out bad experts during evaluation (only during training)
* Interpretability (7/11 - 7/14)
  - Do experts specialize?
* Speed
  - Encoder speed benchmarks
  - Faster activation function
  - PyTorch MoE optimization
  - Kernels
* Questions
  - Should I cache the entire dataset instead of streaming it?
  - Should I change ln_1 to ln_2 (i.e. what does it mean to train a layer 8 residual stream SAE)?
  - Scale of MSE, Delta CE
  - Width of SAEs (32x, 64x, 128x etc.)

## Credits
This repository is adapted from [dictionary_learning](https://github.com/saprmarks/dictionary_learning) by Samuel Marks and Aaron Mueller.
