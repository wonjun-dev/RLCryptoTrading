from enum import Enum
from mmap import ACCESS_COPY
from os import close

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
        self.observation_features = None
        self.shape = (window_size, self.signal_features.shape[1])
        self.fee = 0.0004

        # Define action and observation space
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.inf, high=np.inf, shape=self.shape, dtype=np.uint8
        )

        # Define episode
        self._start_tick = self.window_size  # 24
        self._end_tick = self._start_tick + self.window_size  # 48
        self._current_tick = self._start_tick  # 24
        self._last_episode_tick = self._start_tick - 1  # 23
        self._last_trade_tick = None  # current_tick - 1
        self._done = None
        self._invalid_action = False

        self._profit = 1.0
        self._profit_history = [self._profit]
        self._position = Position.NO_POSITION
        self._position_history = [self._position]
        self._action_history = []
        self._reward_history = []
        self._participate = False
        self._timeout = False
        self._liquidate = False

        self._terminate = False

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def step(self, action):
        self._done = False
        self._action_history.append(action)

        # change position
        if action == Actions.SHORT.value:  # Short
            self._action_short()
        elif action == Actions.LONG.value:  # Long
            self._action_long()
        elif action == Actions.WATCH.value:  # Watch
            self._action_watch()
        elif action == Actions.EXIT.value:  # Exit
            self._action_exit()

        # Episode end condition
        # 1. invalid action
        if action != Actions.WATCH.value:
            if action == Actions.EXIT.value:
                self._invalid_action = not self._participate
            else:
                self._invalid_action = self._participate and (
                    action == Actions.SHORT.value or action == Actions.LONG.value
                )

            if self._invalid_action:
                self._done = True
                self._last_episode_tick = self._current_tick
            else:
                if action == Actions.EXIT.value and not self._participate:
                    self._participate = False
                else:
                    self._participate = True

        # 2. timeout
        if self._current_tick + 1 == self._end_tick:
            self._done = True
            self._last_episode_tick = self._current_tick

            if self._participate and action != Actions.EXIT.value:
                self._timeout = True

        # 3. exit action
        if action == Actions.EXIT.value:
            self._done = True
            self._last_episode_tick = self._current_tick

        # 4. liqudating
        if self._profit <= 0.985:
            self._done = True
            self._liquidate = True
            self._last_episode_tick = self._current_tick

        # 5. consume all history
        if self._current_tick >= self.frame_bound[-1]:
            self._done = True
            self._last_episode_tick = self._current_tick
            self._terminate = True

        # profit update
        if not self._done:
            self._update_profit()

        # calculate reward
        # self._print_stats()
        reward = self._calculate_reward(done=self._done)
        self._reward_history.append(reward)

        # get new observation
        self._current_tick += 1
        observation = self._get_observation()

        return observation, reward, self._done, self._terminate

    def reset(self):
        self._start_tick = self._last_episode_tick + 1
        self._end_tick = self._start_tick + self.window_size
        self._current_tick = self._start_tick
        self._last_trade_tick = None
        self._done = None
        self._invalid_action = False

        self._profit = 1.0
        self._profit_history = [self._profit]
        self._position = Position.NO_POSITION
        self._position_history = [self._position]
        self._action_history = []
        self._reward_history = []
        self._participate = False
        self._timeout = False
        self._liquidate = False

        return self._get_observation(reset=True)  # reward, done, info can't be included

    def get_episode_info(self):
        info = {
            "profit_hist": self._profit_history,
            "reward_hist": self._reward_history,
            "action_hist": self._action_history,
            "invalid_action": self._invalid_action,
            "participate": self._participate,
            "liquidate": self._liquidate,
            "timeout": self._timeout,
        }

        return info

    def _process_data(self):
        open_price = self.df.loc[:, "open"].to_numpy()
        close_price = self.df.loc[:, "close"].to_numpy()
        high_price = self.df.loc[:, "high"].to_numpy()
        low_price = self.df.loc[:, "low"].to_numpy()

        # indicator feature
        close_open = pm.normalize_open_price(close_price, open_price)
        high_open = pm.normalize_open_price(high_price, open_price)
        low_open = pm.normalize_open_price(low_price, open_price)

        bu, bm, bl = im.bollingerBands(close_price)
        bu_high = pm.normalize_open_price(bu, high_price)
        bm_open = pm.normalize_open_price(bm, open_price)
        bl_low = pm.normalize_open_price(bl, low_price)

        ma = im.ma(close_price)
        ma_close = pm.normalize_open_price(ma, close_price)

        rsi = im.rsi(close_price)
        adx = im.adx(high_price, low_price, close_price)
        plus_di = im.plus_di(high_price, low_price, close_price)
        minus_di = im.minus_di(high_price, low_price, close_price)

        signal_features = np.column_stack((close_open, high_open))
        signal_features = np.column_stack((signal_features, low_open))
        signal_features = np.column_stack((signal_features, bu_high))
        signal_features = np.column_stack((signal_features, bm_open))
        signal_features = np.column_stack((signal_features, bl_low))
        signal_features = np.column_stack((signal_features, ma_close))
        signal_features = np.column_stack((signal_features, rsi))
        signal_features = np.column_stack((signal_features, adx))
        signal_features = np.column_stack((signal_features, plus_di))
        signal_features = np.column_stack((signal_features, minus_di))

        signal_features = np.nan_to_num(signal_features)

        return close_price, signal_features

    def _action_short(self):
        self._position = Position.SHORT
        self._position_history.append(self._position)
        self._last_trade_tick = self._current_tick - 1

    def _action_long(self):
        self._position = Position.LONG
        self._position_history.append(self._position)
        self._last_trade_tick = self._current_tick - 1

    def _action_watch(self):
        self._position_history.append(self._position)

    def _action_exit(self):
        self._position = Position.NO_POSITION
        self._position_history.append(self._position)
        self._last_trade_tick = None

    def _get_observation(self, reset=False):
        signal_feature = self.signal_features[
            (self._current_tick - self.window_size) : self._current_tick
        ]

        if reset:
            position = np.ones(signal_feature.shape[0]) * self._position_history[-1].value
            profit = np.ones(signal_feature.shape[0]) * self._profit_history[-1]
            timestamp = (
                np.ones(signal_feature.shape[0])
                * (self._current_tick - self._start_tick)
                / self.window_size
            )
            observation = np.column_stack((signal_feature, position))
            observation = np.column_stack((observation, profit))
            observation = np.column_stack((observation, timestamp))
            self.observation_features = observation
        else:
            last_observation = self.observation_features[1:]
            new_observation = np.expand_dims(signal_feature[-1], axis=0)

            position = np.ones(new_observation.shape[0]) * self._position_history[-1].value
            profit = np.ones(new_observation.shape[0]) * self._profit_history[-1]
            timestamp = (
                np.ones(new_observation.shape[0])
                * (self._current_tick - self._start_tick)
                / self.window_size
            )
            new_observation = np.column_stack((new_observation, position))
            new_observation = np.column_stack((new_observation, profit))
            new_observation = np.column_stack((new_observation, timestamp))
            self.observation_features = np.vstack((last_observation, new_observation))
        return self.observation_features

    def _calculate_reward(self, done):

        if self._invalid_action:
            step_reward = -0.2
            return step_reward

        if len(self._profit_history) >= 2 and not done:
            step_reward = (self._profit_history[-1] - self._profit_history[-2]) * 100  # ?????? n%
        else:
            step_reward = 0.0  # ?????????????????? exit

        if done:
            episode_reward = (self._profit_history[-1] - self._profit_history[0]) * 100  # ?????? n%
            return step_reward + episode_reward

        return step_reward

    def _update_profit(self):
        if self._position_history[-1].name == "NO_POSITION":
            self._profit = self._profit_history[-1]
            self._profit_history.append(self._profit)

        else:  # long or short
            current_price = self.prices[self._current_tick]
            enter_price = self.prices[self._last_trade_tick]

            net_worth = (1 - self.fee) * current_price - (1 + self.fee) * enter_price
            net_worth_ratio = (enter_price + net_worth) / enter_price

            if self._position_history[-1].name == "SHORT":
                self._profit = 2 - net_worth_ratio
            if self._position_history[-1].name == "LONG":
                self._profit = net_worth_ratio

            self._profit_history.append(self._profit)

    def _print_stats(self):
        print("Start tick: ", self._current_tick - self.window_size)
        print("End tick: ", self._current_tick - 1)
        print("Action hist: ", self._action_history)
        print("Position hist: ", self._position_history)
        print("Profit hist: ", self._profit_history)
        print("Reward hist: ", self._reward_history)
