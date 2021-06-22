from gym.envs.registration import register, registry
from copy import deepcopy

from . import datasets

register(id='spot-v0', entry_point="gym_anton.envs:TradingEnv", kwargs={})
register(id="future-v0", entry_point="gym_anton.envs:FutureEnv", kwargs={})
