<h3 align="center">
  Efficient Dictionary Learning via Switch Sparse Autoencoders (SAEs)
</h3>

## To-Do
* Training (7/12 - 7/16)
  - More sweeps (experts, k, lb_alpha, flop-matched, etc.) (7/12 - 7/16)
* Post-training (7/17 - 7/19)
  - Don't zero out bad experts during evaluation (only during training) [just change the eval code in the train script] [don't need to train new, just use old but eval diff]
  - Expert distribution (how close to uniform?)
  - Do experts specialize?
  - MMCS
  - Spectral Clustering correspondence?
* Write blog post & incorporate feedback (7/14 - 7/19)

<br>

* True MoE
  - https://github.com/lucidrains/mixture-of-experts/tree/master
  - https://github.com/lucidrains/st-moe-pytorch
* Speed
  - Speed benchmarks (FLOPs, Clock Time)
  - Fast Switch SAE (Sorting)
  - PyTorch MoE
* Appendix
  - See how much speedup we get for 64x, 128x, 256x SAEs while retaining ReLU performance. Scaling laws.


## Credits
This repository is adapted from [dictionary_learning](https://github.com/saprmarks/dictionary_learning) by Samuel Marks and Aaron Mueller.
