<h3 align="center">
  Efficient Dictionary Learning via Switch Sparse Autoencoders (SAEs)
</h3>

## To-Do
* Metrics (7/12 - 7/16)
  - Load balancing loss (7/12)
  - AuxK loss (7/12) [not really important]
  - Fix softmax issue (-infty) (7/12)
  - Select k//e for each of the e experts, not k total across e experts (same for e = 1) (7/12 - 7/16)
  - More sweeps (e, experts, k, lr, load balance coeff, dict_ratio, etc.) (7/12 - 7/16)
  - See how much speedup we get for 64x SAEs while retaining ReLU performance
  - Different bias for each expert?
  - Don't zero out bad experts during evaluation (only during training) [just change the eval code in the train script] [don't need to train new, just use old but eval diff]
* Write blog post & incorporate feedback (7/15 - 7/19)
* Interpretability
  - Do experts specialize?
  - MMCS
* Speed
  - Speed benchmarks (FLOPs, Clock Time)
  - Fast Switch SAE (Sorting)
  - PyTorch MoE

## Credits
This repository is adapted from [dictionary_learning](https://github.com/saprmarks/dictionary_learning) by Samuel Marks and Aaron Mueller.
