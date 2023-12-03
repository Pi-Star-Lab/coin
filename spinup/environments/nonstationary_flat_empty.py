# Author: Sheelabhadra Dey (sheelabhadra@gmail.com)
# Adapted from: https://github.com/mit-acl/gym-minigrid/blob/master/gym_minigrid/envs/fourrooms.py

import numpy as np

from gym import spaces
from gym_minigrid.minigrid import *
from spinup.environments.flat_minigrid import FlatMiniGridEnv


class NonStationaryFlatEmptyEnv(FlatMiniGridEnv):
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
        self.episode_num = 0

        super().__init__(
            grid_size=size,
            max_steps=4 * size * size,
            # Set this to True for maximum speed
            see_through_walls=True,
        )

    def reset(self):
        self.episode_num += 1
        # Current position and direction of the agent
        self.agent_pos = None
        self.agent_dir = None

        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        assert self.agent_pos is not None
        assert self.agent_dir is not None

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        # Item picked up, being carried, initially nothing
        self.carrying = None

        # Step count since episode start
        self.step_count = 0

        # Return first observation
        obs = self.gen_obs()
        obs = obs.flatten() / 255.0

        return obs

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        if self.episode_num % 500 == 0 and (self.episode_num // 500) % 2 == 1:
            # Place a goal square in the bottom-right corner
            self.grid.set(width - 2, height - 2, Goal())
        else:
            # Place a goal square in the bottom-left corner
            self.grid.set(width - 2, 1, Goal())

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"


class NonStationaryFlatEmptyEnv6x6(NonStationaryFlatEmptyEnv):
    def __init__(self):
        super().__init__(size=6)


class NonStationaryFlatEmptyEnv16x16(NonStationaryFlatEmptyEnv):
    def __init__(self):
        super().__init__(size=16)
