<h3 align="center">
  Efficient Dictionary Learning with Switch Sparse Autoencoders (SAEs)
</h3>

## To-Do
* Post-training (7/17 - 7/19)
  - Don't zero out bad experts during evaluation (only during training) [just change the eval code in the train script] [don't need to train new, just use old but eval diff]
  - Expert distribution (how close to uniform?)
  - Do experts specialize?
  - MMCS
  - Spectral Clustering correspondence?
  - Check how similar b_dec and b_router are
* Write blog post (7/16 - 7/20)

<br>

* True MoE
  - https://github.com/lucidrains/mixture-of-experts/tree/master
  - https://github.com/lucidrains/st-moe-pytorch
  - Benchmarks (FLOPs, Clock Time)
  - PyTorch MoE
* Appendix
  - See how much speedup we get for 64x, 128x, 256x SAEs while retaining ReLU performance. Scaling laws.
  - Logit jitter (described in Switch Transformer and ST MoE). Multiply each logit by uniform number in [0.99, 1.01]


## Credits
This repository is adapted from [dictionary_learning](https://github.com/saprmarks/dictionary_learning) by Samuel Marks and Aaron Mueller.
