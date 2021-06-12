from enum import Enum

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import ndim


class Actions(Enum):
    SELL = 0
    BUY = 1
    WATCH = 2


class Positions(Enum):
    SHORT = 0
    LONG = 1
    WATCH = 2

    def opposite(self):
        return Positions.SHORT if self == Positions.LONG else Positions.LONG


class TradingEnv(gym.Env):
    """General trading environment that follows gym interface"""

    metadata = {"render.modes": ["human"]}

    def __init__(self, df, window_size):
        assert df.ndim == 2

        self.seed()
        self.df = df
        self.window_size = window_size
        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1])
        self.fee = 0

        # Define action and observation space
        self.action_space = spaces.Discrete(3)
        # self.position_action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.inf, high=np.inf, shape=self.shape, dtype=np.uint8
        )

        # Define episode
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._done = None
        self._current_tick = None
        self._last_trade_tick = None
        self._position = None
        self._position_history = None
        self._total_reward = None
        self._total_profit = None
        self._first_rendering = None
        self.history = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def step(self, action):
        ...
        return observation, reward, done, info

    def reset(self):
        ...
        return observation  # reward, done, info can't be included

    def render(self, mode="human"):
        ...

    def close(self):
        ...

    def _process_data(self):
        raise NotImplementedError
