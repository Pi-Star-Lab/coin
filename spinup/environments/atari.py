"""Implementation of Atari 2600 Preprocessing following the guidelines of Machado et al., 2018."""

"""
Code adapted from: https://github.com/openai/gym/blob/master/gym/wrappers/atari_preprocessing.py
"""

# from collections import deque

# import numpy as np

# import gym
# from gym.spaces import Box

# try:
#     import cv2
# except ImportError:
#     cv2 = None


# class AtariPreprocessing(gym.Wrapper):
#     """Atari 2600 preprocessing wrapper.

#     This class follows the guidelines in Machado et al. (2018),
#     "Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents".

#     Specifically, the following preprocess stages applies to the atari environment:
#     - Noop Reset: Obtains the initial state by taking a random number of no-ops on reset, default max 30 no-ops.
#     - Frame skipping: The number of frames skipped between steps, 4 by default
#     - Max-pooling: Pools over the most recent two observations from the frame skips
#     - Termination signal when a life is lost: When the agent losses a life during the environment, then the environment is terminated.
#         Turned off by default. Not recommended by Machado et al. (2018).
#     - Resize to a square image: Resizes the atari environment original observation shape from 210x180 to 84x84 by default
#     - Grayscale observation: If the observation is colour or greyscale, by default, greyscale.
#     - Scale observation: If to scale the observation between [0, 1) or [0, 255), by default, not scaled.
#     """

#     def __init__(
#         self,
#         env: gym.Env,
#         noop_max: int = 30,
#         frame_skip: int = 4,
#         screen_size: int = 84,
#         terminal_on_life_loss: bool = False,
#         grayscale_obs: bool = True,
#         grayscale_newaxis: bool = True,
#         scale_obs: bool = True,
#     ):
#         """Wrapper for Atari 2600 preprocessing.

#         Args:
#             env (Env): The environment to apply the preprocessing
#             noop_max (int): For No-op reset, the max number no-ops actions are taken at reset, to turn off, set to 0.
#             frame_skip (int): The number of frames between new observation the agents observations effecting the frequency at which the agent experiences the game.
#             screen_size (int): resize Atari frame
#             terminal_on_life_loss (bool): `if True`, then :meth:`step()` returns `terminated=True` whenever a
#                 life is lost.
#             grayscale_obs (bool): if True, then gray scale observation is returned, otherwise, RGB observation
#                 is returned.
#             grayscale_newaxis (bool): `if True and grayscale_obs=True`, then a channel axis is added to
#                 grayscale observations to make them 3-dimensional.
#             scale_obs (bool): if True, then observation normalized in range [0,1) is returned. It also limits memory
#                 optimization benefits of FrameStack Wrapper.

#         Raises:
#             DependencyNotInstalled: opencv-python package not installed
#             ValueError: Disable frame-skipping in the original env
#         """
#         super().__init__(env)
#         if cv2 is None:
#             raise gym.error.DependencyNotInstalled(
#                 "opencv-python package not installed, run `pip install gym[other]` to get dependencies for atari"
#             )
#         assert frame_skip > 0
#         assert screen_size > 0
#         assert noop_max >= 0
#         if frame_skip > 1:
#             if (
#                 "NoFrameskip" not in env.spec.id
#                 and getattr(env.unwrapped, "_frameskip", None) != 1
#             ):
#                 raise ValueError(
#                     "Disable frame-skipping in the original env. Otherwise, more than one "
#                     "frame-skip will happen as through this wrapper"
#                 )
#         self.noop_max = noop_max
#         assert env.unwrapped.get_action_meanings()[0] == "NOOP"

#         self.frame_skip = frame_skip
#         self.screen_size = screen_size
#         self.terminal_on_life_loss = terminal_on_life_loss
#         self.grayscale_obs = grayscale_obs
#         self.grayscale_newaxis = grayscale_newaxis
#         self.scale_obs = scale_obs

#         # buffer of most recent two observations for max pooling
#         assert isinstance(env.observation_space, Box)
#         if grayscale_obs:
#             self.obs_buffer = [
#                 np.empty(env.observation_space.shape[:2], dtype=np.uint8),
#                 np.empty(env.observation_space.shape[:2], dtype=np.uint8),
#             ]
#         else:
#             self.obs_buffer = [
#                 np.empty(env.observation_space.shape, dtype=np.uint8),
#                 np.empty(env.observation_space.shape, dtype=np.uint8),
#             ]

#         self.lives = 0
#         self.game_over = False

#         _low, _high, _obs_dtype = (
#             (0, 255, np.uint8) if not scale_obs else (0, 1, np.float32)
#         )
#         _shape = (1 if grayscale_obs else 3, screen_size, screen_size)
#         if grayscale_obs and not grayscale_newaxis:
#             _shape = _shape[1:]  # Remove channel axis
#         self.observation_space = Box(
#             low=_low, high=_high, shape=_shape, dtype=_obs_dtype
#         )

#     @property
#     def ale(self):
#         """Make ale as a class property to avoid serialization error."""
#         return self.env.unwrapped.ale

#     def step(self, action):
#         """Applies the preprocessing for an :meth:`env.step`."""
#         total_reward, terminated, truncated, info = 0.0, False, False, {}

#         for t in range(self.frame_skip):
#             _, reward, terminated, info = self.env.step(action)
#             total_reward += reward
#             self.game_over = terminated

#             if self.terminal_on_life_loss:
#                 new_lives = self.ale.lives()
#                 terminated = terminated or new_lives < self.lives
#                 self.game_over = terminated
#                 self.lives = new_lives

#             if terminated:
#                 break
#             if t == self.frame_skip - 2:
#                 if self.grayscale_obs:
#                     self.ale.getScreenGrayscale(self.obs_buffer[1])
#                 else:
#                     self.ale.getScreenRGB(self.obs_buffer[1])
#             elif t == self.frame_skip - 1:
#                 if self.grayscale_obs:
#                     self.ale.getScreenGrayscale(self.obs_buffer[0])
#                 else:
#                     self.ale.getScreenRGB(self.obs_buffer[0])
#         return self._get_obs(), total_reward, terminated, info

#     def reset(self, **kwargs):
#         """Resets the environment using preprocessing."""
#         # NoopReset
#         _ = self.env.reset(**kwargs)

#         noops = (
#             self.env.unwrapped.np_random.randint(1, self.noop_max + 1)
#             if self.noop_max > 0
#             else 0
#         )
#         for _ in range(noops):
#             _, _, terminated, _ = self.env.step(0)
#             if terminated:
#                 _ = self.env.reset(**kwargs)

#         self.lives = self.ale.lives()
#         if self.grayscale_obs:
#             self.ale.getScreenGrayscale(self.obs_buffer[0])
#         else:
#             self.ale.getScreenRGB(self.obs_buffer[0])
#         self.obs_buffer[1].fill(0)

#         return self._get_obs()

#     def _get_obs(self):
#         if self.frame_skip > 1:  # more efficient in-place pooling
#             np.maximum(self.obs_buffer[0], self.obs_buffer[1], out=self.obs_buffer[0])
#         assert cv2 is not None
#         obs = cv2.resize(
#             self.obs_buffer[0],
#             (self.screen_size, self.screen_size),
#             interpolation=cv2.INTER_AREA,
#         )

#         if self.scale_obs:
#             obs = np.asarray(obs, dtype=np.float32) / 255.0
#         else:
#             obs = np.asarray(obs, dtype=np.uint8)

#         if self.grayscale_obs and self.grayscale_newaxis:
#             obs = np.expand_dims(obs, axis=0)  # Add a channel axis
#         return obs


import numpy as np
import os

os.environ.setdefault("PATH", "")
from collections import deque
import gym
from gym import spaces
import cv2

cv2.ocl.setUseOpenCL(False)
from .wrappers import TimeLimit


class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info["TimeLimit.truncated"] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        """Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(
                1, self.noop_max + 1
            )  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.

        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(shp[:-1] + (shp[-1] * k,)),
            dtype=env.observation_space.dtype,
        )

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=env.observation_space.shape, dtype=np.float32
        )

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]


def make_atari(env_id, max_episode_steps=None):
    env = gym.make(env_id)
    assert "NoFrameskip" in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def wrap_deepmind(
    env, episode_life=True, clip_rewards=True, frame_stack=False, scale=False
):
    """Configure environment for DeepMind-style Atari."""
    if episode_life:
        env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env
