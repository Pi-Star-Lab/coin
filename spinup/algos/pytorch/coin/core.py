from copy import deepcopy
import numpy as np
import scipy.signal

import torch
import torch.nn as nn


def combined_shape(length, shape=None):
    if shape == None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class MLPQFunction(nn.Module):
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes,
        activation,
    ):
        super().__init__()
        self.q = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def forward(self, obs):
        q = self.q(torch.cat([obs], dim=-1))
        return torch.squeeze(q, -1)


class COINQFunction(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes=(256, 256),
        activation=nn.ReLU,
    ):
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.n

        self.q_coin = MLPQFunction(
            obs_dim,
            act_dim,
            hidden_sizes,
            activation,
        )
        self.q_true = MLPQFunction(
            obs_dim,
            act_dim,
            hidden_sizes,
            activation,
        )

    def act(self, obs):
        """
        Select greedy action.
        """
        with torch.no_grad():
            return self.q_coin(obs).argmax(dim=-1).numpy()
