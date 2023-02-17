**Status:** Maintenance (expect bug fixes and minor updates)

Welcome to Spinning Up in Deep RL! 
==================================

This is an educational resource produced by OpenAI that makes it easier to learn about deep reinforcement learning (deep RL).

For the unfamiliar: [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) (RL) is a machine learning approach for teaching agents how to solve tasks by trial and error. Deep RL refers to the combination of RL with [deep learning](http://ufldl.stanford.edu/tutorial/).

This module contains a variety of helpful resources, including:

- a short [introduction](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) to RL terminology, kinds of algorithms, and basic theory,
- an [essay](https://spinningup.openai.com/en/latest/spinningup/spinningup.html) about how to grow into an RL research role,
- a [curated list](https://spinningup.openai.com/en/latest/spinningup/keypapers.html) of important papers organized by topic,
- a well-documented [code repo](https://github.com/openai/spinningup) of short, standalone implementations of key algorithms,
- and a few [exercises](https://spinningup.openai.com/en/latest/spinningup/exercises.html) to serve as warm-ups.

Get started at [spinningup.openai.com](https://spinningup.openai.com)!


Citing Spinning Up
------------------

If you reference or use Spinning Up in your research, please cite:

```
@article{SpinningUp2018,
    author = {Achiam, Joshua},
    title = {{Spinning Up in Deep Reinforcement Learning}},
    year = {2018}
}
```


Hyperparameters
---------------

### Bridge-7

```yaml
batch_size=32
update_interval=10
max_ep_len=10
bonus=5.0
bonus_freq=10000
regret_bound=100
log_freq=50
training_starts=10
prior_q_net_path="/home/sheelabhadra/Pi-Star/spinningup/data/dqn_bridge_7/dqn_bridge/pyt_save/model.pt"
```

### LunarLander-v2

```yaml
batch_size=128
update_interval=100
max_ep_len=1000
bonus=1
bonus_freq=10000
regret_bound=100
log_freq=10
training_starts=20
prior_q_net_path="/home/sheelabhadra/Pi-Star/spinningup/data/dqn_ll_run_1/dqn_ll_run_1_s0/pyt_save/model.pt"
```