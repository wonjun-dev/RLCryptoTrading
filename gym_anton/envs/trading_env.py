from enum import Enum

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from ..utils import indicator_manager as im
from ..utils import preprocess_manager as pm 


class Actions(Enum):
    SHORT = 0
    LONG = 1
    WATCH = 2
    EXIT = 3

class Position(Enum):
    SHORT = 0
    LONG = 1
    NO_POSITION = 2


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
        self._value = None
        self._start_tick = self.window_size
        self._end_tick = self._start_tick + self.window_size
        self._done = None
        self._current_tick = None
        self._last_trade_tick = None
        self._action = None
        self._action_history = None
        self._position = Position.NO_POSITION
        self._position_history = None
        self._total_reward = None
        self._total_profit = None
        self.history = None



    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def step(self, action):
        self._done = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._done = True
        
        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        # self._update_profit(action)

        if action == 0: # Short
            self._action_short()
        elif action == 1:   # Long
            self._action_long()
        elif action == 2:   # Watch
            self._action_watch()
        elif action == 3:   # Exit
            self._action_exit()

        return observation, reward, done, info

    def reset(self):
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._action_history = (self.window_size * [None]) + [self._action]
        self._total_reward = 0.
        self._total_profit = 1.  # unit
        self._first_rendering = True
        self.history = {}
        return self._get_observation(reset=True)  # reward, done, info can't be included

   
    def _process_data(self):
        open_price = self.df.loc[:, 'open'].to_numpy()
        close_price = self.df.loc[:, 'close'].to_numpy()
        high_price = self.df.loc[:, 'high'].to_numpy()
        low_price = self.df.loc[:, 'low'].to_numpy()
        # diff = np.insert(np.diff(open_price), 0, 0)
        
        # indicator feature
        close_open = pm.normalize_open_price(close_price, open_price)
        high_open = pm.normalize_open_price(high_price, open_price)
        low_open = pm.normalize_open_price(low_price, open_price) 
    
        bu, bm, bl = im.bollingerBands(close_price)
        bu_high = pm.normalize_open_price(bu, high_price)
        bm_open = pm.normalize_open_price(bm, open_price)
        bl_low = pm.normalize_open_price(bl, low_price)


        signal_features = np.column_stack((close_open, high_open))
        signal_features = np.column_stack((signal_features, low_open))
        signal_features = np.column_stack((signal_features, bu_high))
        signal_features = np.column_stack((signal_features, bm_open))
        signal_features = np.column_stack((signal_features, bl_low))
        signal_features = np.nan_to_num(signal_features)

        return open_price, signal_features
    
    def _action_short(self):
        pass

    def _action_long(self):
        pass

    def _action_watch(self):
        pass

    def _action_exit(self):
        pass
    
    def _get_observation(self, reset=False):
        if reset:
            position = self._position
            signal_feature = self.signal_features[(self._current_tick-self.window_size):self._current_tick]
            signal_feature = np.column_stack((signal_feature, position))
        else:
            return self.signal_features[(self._current_tick-self.window_size):self._current_tick]

    def _calculate_reward(self):
        return None

    def _update_profit(self):
        return None
