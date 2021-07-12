import numpy as np


class StatsManager:
    def __init__(self):
        self.hist_stats_name = [
            "participate_hist",
            "profit_hist",
            "cumulative_reward_hist",
            "invalid_action_hist",
            "liquidate_hist",
            "timeout_hist",
            "win_hist",
        ]

        self.log_stats_name = [
            "participate_rate",
            "mean_profit",
            "mean_win_profit",
            "mean_loss_profit",
            "mean_cumulative_reward",
            "invalid_action_rate",
            "liquidate_rate",
            "timeout_rate",
            "win_rate",
        ]

        self.hist_stats_dict = {name: [] for name in self.hist_stats_name}
        self.log_stats_dict = {name: None for name in self.log_stats_name}

    def add_stats(self, info):
        self.hist_stats_dict["participate_hist"].append(int(info["participate"]))
        self.hist_stats_dict["profit_hist"].append(info["profit_hist"][-1] - info["profit_hist"][0])
        self.hist_stats_dict["cumulative_reward_hist"].append(sum(info["reward_hist"]))
        self.hist_stats_dict["invalid_action_hist"].append(int(info["invalid_action"]))
        self.hist_stats_dict["liquidate_hist"].append(int(info["liquidate"]))
        self.hist_stats_dict["timeout_hist"].append(int(info["timeout"]))
        self.hist_stats_dict["win_hist"].append(
            1 if info["profit_hist"][-1] - info["profit_hist"][0] > 1 else 0
        )

    def get_log_stats(self, n_epi):
        self.log_stats_dict["participate_rate"] = (
            sum(self.hist_stats_dict["participate_hist"]) / n_epi
        )
        self.log_stats_dict["mean_profit"] = np.mean(self.hist_stats_dict["profit_hist"])

        pass

    def reset(self):
        self.hist_stats_name = {name: [] for name in self.hist_stats_name}

    def _get_participate_only(self, stat):
        return np.multiply(stat, self.hist_stats_name["participate_hist"])
