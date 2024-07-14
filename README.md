<h3 align="center">
  Efficient Dictionary Learning via Switch Sparse Autoencoders (SAEs)
</h3>

## To-Do
* Metrics (7/12 - 7/16)
  - Load balancing loss (7/13)
  - AuxK loss (7/13) [not really important]
  - Think more about softmax issue (7/13)
  - More sweeps (e, experts, k, lr, load balance coeff, dict_ratio, etc.) (7/12 - 7/16)
  - See how much speedup we get for 64x SAEs while retaining ReLU performance
  - Different bias for each expert? Seems bad.
  - Don't zero out bad experts during evaluation (only during training) [just change the eval code in the train script] [don't need to train new, just use old but eval diff]
* Interpretability
  - Do experts specialize?
  - MMCS
  - Spectral Clustering
* Write blog post & incorporate feedback (7/15 - 7/19)
* Speed
  - Speed benchmarks (FLOPs, Clock Time)
  - Fast Switch SAE (Sorting)
  - PyTorch MoE

## Credits
This repository is adapted from [dictionary_learning](https://github.com/saprmarks/dictionary_learning) by Samuel Marks and Aaron Mueller.
