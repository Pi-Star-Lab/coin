Continual Optimistic INitialization (COIN)
==========================================

This codebase is based on the [Spinning Up repository](https://github.com/openai/spinningup/tree/master). Please follow the installation instructions provided [here](https://spinningup.openai.com/en/latest/user/installation.html).

Running experiments
-------------------

```sh
python -m spinup.run <algo> --env <env_name> --exp_name <log_folder> --epochs <num_epochs> --bonus <b> --bonus_freq <bonus_frequency> --seed <seed>
```

e.g.,

For discrete actions
```sh
python -m spinup.run coin_bdual --env LunarLander-v2 --exp_name coin_bdual_lunarlander --epochs 60  --eps_disp 0.05 --eps_b 0.1 --seed 0
```

For continuous control
```
python -m spinup.run coin_td3 --env LunarLanderContinuous-v2 --exp_name coin_td3_lunarlander --epochs 100 --eps_disp 0.05 --eps_b 0.1 --seed 0
```
