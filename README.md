# Introduction

Astral is a loss function that is simultaneously error majorant, meaning it can be used for direct error control.

The present repository contains:
1. Detailed description of datasets including instructions on how to download them and the description of equations and upper bound https://github.com/VLSF/astral/blob/main/datasets/datasets_description.md
2. Scripts with PiNN training for different losses and equations https://github.com/VLSF/astral/tree/main/scripts.

All scripts have the same input parameters. A typical way to run them is

```shell
CUDA_VISIBLE_DEVICES=1 python Maxwell_residual.py -path_to_dataset 'Maxwell.npz' -path_to_results 'Maxwell/' -N_features '100' '50' -N_layers '5' '4' '3' -N_drop '10000' '25000' '50000' -learning_rate '1e-4' '5e-3' '1e-3'
```

The code has been tested on [jax](https://github.com/google/jax) 0.4.14, [jaxlib](https://github.com/google/jax) 0.4.14+cuda12.cudnn89, [equinox](https://github.com/patrick-kidger/equinox) 0.10.11, [optax](https://github.com/google-deepmind/optax) 0.1.7. Those are the main dependencies.
