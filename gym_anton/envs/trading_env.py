from enum import Enum

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from ..utils import indicator_manager as im 


class Actions(Enum):
    SHORT = 0
    LONG = 1
    WATCH = 2
    EXIT = 3


class TradingEnv(gym.Env):
    """General trading environment that follows gym interface"""

    metadata = {"render.modes": ["human"]}

    def __init__(self, df, window_size):
        assert df.ndim == 2

        self.seed()
        self.df = df
        self.window_size = window_size
        self.frame_bound = (window_size, len(df))
        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1])
        self.fee = 0.0004

        # Define action and observation space
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.inf, high=np.inf, shape=self.shape, dtype=np.uint8
        )

        # Define episode
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._done = None
        self._current_tick = None
        self._last_trade_tick = None
        self._action_history = None
        self._total_reward = None
        self._total_profit = None
        self.history = None



    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def step(self, action):
        ...
        return observation, reward, done, info

    def reset(self):
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._action_history = (self.window_size * [None]) + [self._position]
        self._total_reward = 0.
        self._total_profit = 1.  # unit
        self._first_rendering = True
        self.history = {}
        return self._get_observation()  # reward, done, info can't be included

    def render(self, mode="human"):
        ...

    def close(self):
        ...

    def _process_data(self):
        prices = self.df.loc[:, 'close'].to_numpy()

        # preprocess
        # TODO: neatly
        diff = np.insert(np.diff(prices), 0, 0)
        bollinger = im.bollingerBands(prices)
        rsi = im.rsi(prices)

        signal_features = np.column_stack((prices, diff))
        signal_features = np.column_stack((signal_features, bollinger[0]))
        signal_features = np.column_stack((signal_features, bollinger[1]))
        signal_features = np.column_stack((signal_features, bollinger[2]))
        signal_features = np.column_stack((signal_features, rsi))
        signal_features = np.nan_to_num(signal_features)

        return prices, signal_features
    
    def _get_observation(self):
        return self.signal_features[(self._current_tick-self.window_size):self._current_tick]
