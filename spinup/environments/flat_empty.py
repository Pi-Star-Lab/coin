# Author: Sheelabhadra Dey (sheelabhadra@gmail.com)
# Adapted from: https://github.com/mit-acl/gym-minigrid/blob/master/gym_minigrid/envs/fourrooms.py

import numpy as np

from gym import spaces
from gym_minigrid.minigrid import *
from spinup.environments.flat_minigrid import FlatMiniGridEnv


class FlatEmptyEnv(FlatMiniGridEnv):
    """
    Empty grid env, no obstacles, sparse reward with flattened image observations.
    """

    def __init__(
        self,
        size=8,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        super().__init__(
            grid_size=size,
            max_steps=4 * size * size,
            # Set this to True for maximum speed
            see_through_walls=True,
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.grid.set(width - 2, height - 2, Goal())

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"


class FlatEmptyEnv5x5(FlatEmptyEnv):
    def __init__(self):
        super().__init__(size=5)


class FlatEmptyRandomEnv5x5(FlatEmptyEnv):
    def __init__(self):
        super().__init__(size=5, agent_start_pos=None)


class FlatEmptyEnv6x6(FlatEmptyEnv):
    def __init__(self):
        super().__init__(size=6)


class FlatEmptyRandomEnv6x6(FlatEmptyEnv):
    def __init__(self):
        super().__init__(size=6, agent_start_pos=None)


class FlatEmptyEnv16x16(FlatEmptyEnv):
    def __init__(self):
        super().__init__(size=16)
