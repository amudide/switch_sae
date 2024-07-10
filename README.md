<h3 align="center">
  Efficient Dictionary Learning via Switch Sparse Autoencoders (SAEs)
</h3>

## To-Do
* Metrics (7/10 - 7/14)
  - Implement MoE-SAE & Switch SAE (7/9)
  - Load balancing loss (7/10)
  - TopK aux loss, check if its correct and add for MoE (7/10)
  - More sweeps (e, experts, k, lr, load balance coeff, dict_ratio, etc.) (7/10 - 7/14)
    - Experiment with (Heaviside vs. Softmax) & (TopK per expert vs. TopK across chosen experts)
    - Experiment with switch SAE (heaviside vs. weighted)
  - Don't zero out bad experts during evaluation (only during training) [just change the eval code in the train script] [don't need to train new, just use old but eval diff]
* Write blog post & incorporate feedback (7/15 - 7/19)
* Interpretability
  - Do experts specialize?
* Speed
  - Encoder speed benchmarks
  - Faster activation function
  - PyTorch MoE optimization
  - Kernels

## Credits
This repository is adapted from [dictionary_learning](https://github.com/saprmarks/dictionary_learning) by Samuel Marks and Aaron Mueller.
