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
            1 if info["profit_hist"][-1] - info["profit_hist"][0] > 0 else 0
        )

    def get_log_stats(self):
        self.log_stats_dict["participate_rate"] = (
            np.mean(self.hist_stats_dict["participate_hist"]) * 100
        )
        self.log_stats_dict["mean_profit"] = (
            np.mean(self._get_participate_only(self.hist_stats_dict["profit_hist"])) * 100
        )
        self.log_stats_dict["mean_win_profit"] = (
            np.mean(self._get_win_only(self.hist_stats_dict["profit_hist"])) * 100
        )
        self.log_stats_dict["mean_loss_profit"] = (
            np.mean(self._get_lose_only(self.hist_stats_dict["profit_hist"])) * 100
        )
        self.log_stats_dict["mean_cumulative_reward"] = np.mean(
            self.hist_stats_dict["cumulative_reward_hist"]
        )
        self.log_stats_dict["invalid_action_rate"] = (
            np.mean(self.hist_stats_dict["invalid_action_hist"]) * 100
        )
        self.log_stats_dict["liquidate_rate"] = (
            np.mean(self._get_participate_only(self.hist_stats_dict["liquidate_hist"])) * 100
        )
        self.log_stats_dict["timeout_rate"] = np.mean(
            self._get_participate_only(self.hist_stats_dict["timeout_hist"]) * 100
        )
        self.log_stats_dict["win_rate"] = np.mean(
            self._get_participate_only(self.hist_stats_dict["win_hist"]) * 100
        )

    def reset(self):
        self.hist_stats_name = {name: [] for name in self.hist_stats_name}

    def _get_participate_only(self, stat):
        stat = np.array(stat)
        participate = np.array(self.hist_stats_dict["participate_hist"])
        return stat[participate != 0]

    def _get_win_only(self, stat):
        stat = np.multiply(stat, self.hist_stats_dict["win_hist"])
        return self._get_participate_only(stat)

    def _get_lose_only(self, stat):
        stat = np.multiply(
            stat, np.ones_like(self.hist_stats_dict["win_hist"]) - self.hist_stats_dict["win_hist"]
        )
        return self._get_participate_only(stat)
