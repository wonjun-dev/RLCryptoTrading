import numpy as np


class StatsManager:
    def __init__(self):
        self.participate_hist = []
        self.inposition_time_hist = []  # 시장 참여한 경우
        self.profit_hist = []  # 시장 참여한 경우
        self.cumulative_reward_hist = []
        self.invalid_action_hist = []
        self.win_hist = []  #  시장 참여한 경우

    def add_stats(self, info):
        pass

    def reset(self):
        self.participate_hist = []
        self.inposition_time_hist = []  # 시장 참여한 경우
        self.profit_hist = []  # 시장 참여한 경우
        self.cumulative_reward_hist = []
        self.invalid_action_hist = []
        self.win_hist = []  #  시장 참여한 경우
