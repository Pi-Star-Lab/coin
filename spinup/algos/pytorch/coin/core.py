import warnings

import numpy as np
import torch
import torch.nn as nn
from gym import spaces


def combined_shape(length, shape=None):
    if shape == None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def is_image_space_channels_first(observation_space: spaces.Box) -> bool:
    """
    Check if an image observation space (see ``is_image_space``)
    is channels-first (CxHxW, True) or channels-last (HxWxC, False).
    Use a heuristic that channel dimension is the smallest of the three.
    If second dimension is smallest, raise an exception (no support).
    :param observation_space:
    :return: True if observation space is channels-first image, False if channels-last.
    """
    smallest_dimension = np.argmin(observation_space.shape).item()
    if smallest_dimension == 1:
        warnings.warn(
            "Treating image space as channels-last, while second dimension was smallest of the three."
        )
    return smallest_dimension == 0


def is_image_space(
    observation_space: spaces.Space,
    check_channels: bool = False,
    normalized_image: bool = False,
) -> bool:
    """
    Check if a observation space has the shape, limits and dtype
    of a valid image.
    The check is conservative, so that it returns False if there is a doubt.
    Valid images: RGB, RGBD, GrayScale with values in [0, 255]
    :param observation_space:
    :param check_channels: Whether to do or not the check for the number of channels.
        e.g., with frame-stacking, the observation space may have more channels than expected.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    :return:
    """
    check_dtype = check_bounds = not normalized_image
    if isinstance(observation_space, spaces.Box) and len(observation_space.shape) == 3:
        # Check the type
        if check_dtype and observation_space.dtype != np.uint8:
            return False

        # Check the value range
        incorrect_bounds = np.any(observation_space.low != 0) or np.any(
            observation_space.high != 255
        )
        if check_bounds and incorrect_bounds:
            return False

        # Skip channels check
        if not check_channels:
            return True
        # Check the number of channels
        if is_image_space_channels_first(observation_space):
            n_channels = observation_space.shape[0]
        else:
            n_channels = observation_space.shape[-1]
        # GrayScale, RGB, RGBD
        return n_channels in [1, 3, 4]
    return False


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def cnn(n_input_channels):
    layers = [
        nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0),
        nn.ReLU(),
        nn.Flatten(),
    ]
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


class CNNQFunction(nn.Module):
    def __init__(
        self,
        obs_space,
        act_dim,
        hidden_sizes,
        activation,
    ):
        super().__init__()
        n_input_channels = obs_space.shape[-1]
        self.conv = cnn(n_input_channels)

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.conv(
                torch.as_tensor(obs_space.sample()[None]).float()
            ).shape[1]

        self.linear = mlp([n_flatten] + list(hidden_sizes) + [act_dim], activation)

        self.q = nn.Sequential(self.conv, self.linear)

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
        act_dim = action_space.n

        if is_image_space(observation_space):
            self.q_coin = CNNQFunction(
                observation_space, act_dim, hidden_sizes, activation
            )
        else:
            obs_dim = observation_space.shape[0]
            self.q_coin = MLPQFunction(
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
