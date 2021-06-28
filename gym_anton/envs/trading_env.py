from enum import Enum
from mmap import ACCESS_COPY

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
        self._start_tick = self.window_size
        self._end_tick = self._start_tick + self.window_size
        self._done = None
        self._current_tick = None
        self._last_trade_tick = None
        self._action = None
        self._action_history = None
        self._position = None
        self._position_history = None
        self._total_reward = None
        self._total_profit = None
        self._profit_history = None
        self._enter_price = None
        self._enter_price_history = None
        self.history = None



    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def step(self, action):
        self._done = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._done = True

        # change state
        if action == Actions.SHORT.value: # Short
            self._action_short()
        elif action == Actions.LONG.value:   # Long
            self._action_long()
        elif action == Actions.WATCH.value:   # Watch
            self._action_watch()
        elif action == Actions.EXIT.value:   # Exit
            self._action_exit()

        # profit update
        step_profit = self._update_profit()        
        self._total_profit += step_profit
        print(step_profit)

        # calculate reward
        step_reward = self._calculate_reward(action, self._done)
        self._total_reward += step_reward
        print(step_reward)


        
        return observation, reward, done, info

    def reset(self):
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = None
        self._action_history = (self.window_size * [None]) + [self._action]
        self._position = Position.NO_POSITION
        self._position_history = [self._position]
        self._total_reward = 0.
        self._total_profit = 1.  # unit
        self._profit_history = []
        self.history = {}
        return self._get_observation()  # reward, done, info can't be included

   
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
        self._position = Position.SHORT
        self._last_trade_tick = self._current_tick - 1

    def _action_long(self):
        self._position = Position.LONG
        self._last_trade_tick = self._current_tick - 1

    def _action_watch(self):
        pass

    def _action_exit(self):
        self._position = Position.NO_POSITION
        self._last_trade_tick = None

    
    def _get_observation(self):
        signal_feature = self.signal_features[(self._current_tick-self.window_size):self._current_tick]
        position = np.ones(signal_feature.shape[0]) * self._position.value
        signal_feature = np.column_stack((signal_feature, position))
        return signal_feature

    def _calculate_reward(self, action, done):
        step_reward = 0

        if self._position.name == "NO_POSITION":
            return step_reward
        
        if self._position.name == "SHORT":
            if action == Actions.WATCH.value:
                position_reward = self._profit_history[-1] * 100
                tick_reward = (self._profit_history[-1]  - self._profit_history[-2]) * 100

                step_reward = position_reward + tick_reward
                
            elif action == Actions.EXIT.value:
                position_reward = self._profit_history[-1] * 100

                future_price = self.prices[self._current_tick+1]
                current_price = self.prices[self._current_tick]
                price_ratio = future_price/current_price
                exit_reward = 1. if 1 - price_ratio >= 0 else -1.

                step_reward = position_reward + exit_reward

            else:
                step_reward = -1.   # 추가 포지션 패널티
                return step_reward

        if self._position.name == "LONG":
            if action == Actions.WATCH.value:
                position_reward = self._profit_history[-1] * 100
                tick_reward = (self._profit_history[-1]  - self._profit_history[-2]) * 100

                step_reward = position_reward + tick_reward

            elif action == Actions.EXIT.value:
                position_reward = self._profit_history[-1] * 100

                future_price = self.prices[self._current_tick+1]
                current_price = self.prices[self._current_tick]
                price_ratio = future_price/current_price
                exit_reward = 1. if price_ratio - 1 >= 0 else -1.

                step_reward = position_reward + exit_reward
            else:
                step_reward = -1.   # 추가 포지션 패널티
                return step_reward

        if done:
            episode_reward = (self._profit_history[-1] - 1.) * 100
            step_reward += episode_reward
        
        return step_reward

    def _update_profit(self):
        
        if self._position.name == "NO_POSITION":
            profit = None
            self._profit_history.append(profit)
            return 0

        current_price = self.prices[self._current_tick]
        enter_price = self.prices[self._last_trade_tick]
        price_ratio = current_price / enter_price
        if self._position.name == "SHORT":
            profit = 1 - price_ratio
            self._profit_history.append(profit)
            return profit

        if self._position.name == "LONG":
            profit = price_ratio - 1
            self._profit_history.append(profit)
            return profit
