Continual Optimistic INitialization (COIN)
==========================================

This codebase is based on the [Spinning Up repository](https://github.com/openai/spinningup/tree/master). Please follow the installation instructions provided [here](https://spinningup.openai.com/en/latest/user/installation.html).

Running experiments
-------------------

```sh
python -m spinup.run <algo> --env <env_name> --exp_name <log_folder> --epochs <num_epochs> --bonus <b> --bonus_freq <bonus_frequency> --seed <seed>
```

e.g.,

```sh
python -m spinup.run coin --env LunarLander-v2 --exp_name coin_lunarlander_b_0_2_freq_50000 --epochs 60 --bonus 0.2 --bonus_freq 50000 --seed 0
```
