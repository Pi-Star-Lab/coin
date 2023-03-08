import torch
import torch.nn as nn
import numpy as np

from spinup.utils.policy_utils import cnn, is_image_space, mlp


def combined_shape(length, shape=None):
    if shape == None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


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
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class CNNQFunction(nn.Module):
    def __init__(self, obs_space, act_dim, hidden_sizes, activation):
        super().__init__()
        n_input_channels = obs_space.shape[0]
        self.conv = cnn(n_input_channels)

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.conv(
                torch.as_tensor(obs_space.sample()[None]).float()
            ).shape[1]

        self.linear = MLPQFunction(n_flatten, act_dim, hidden_sizes, activation)

        self.q = nn.Sequential(self.conv, self.linear)

    def forward(self, obs):
        q = self.q(torch.cat([obs], dim=-1))
        return torch.squeeze(q, -1)


class DQNQFunction(nn.Module):
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

        if is_image_space(observation_space):
            # Flatten the image (for minigrid envs)
            self.q = CNNQFunction(
                observation_space, act_dim, hidden_sizes=(512,), activation=activation
            )
        else:
            obs_dim = observation_space.shape[0]
            self.q = MLPQFunction(
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
            return self.q(obs).argmax(dim=-1).numpy()
