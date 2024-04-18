from typing import Any

import numpy as np
from backtesting import Strategy
from gymnasium.wrappers.normalize import RunningMeanStd

from strategybuilder.rl.env.long_only_env import LongOnlyStockTradingEnv


def generate_long_only_backtester(obs_columns: list[str], internal_state_obs: list[str], env: Any, agent: Any, max_size: int | None = None):
    class LongOnlyStrategy(Strategy):
        def init(self):
            self.indicators = {}
            # Initialzie indicators
            for col in obs_columns:
                setattr(self, col, self.I(lambda: self.data[col]))
            self.env = env
            # Add internal state obs for the running mean std
            self.obs_rms = {}
            for k in internal_state_obs:
                self.obs_rms[k] = RunningMeanStd(shape=(1,))
            # Add the obs columns to the obs_rms
            self.obs_rms.update({k: RunningMeanStd(shape=(1,)) for k in obs_columns})
            self.agent = agent
            # self.agent = agent
            self.position_pnl_percentage = []
            self.time_since_pos_open = 0
            # self.time_since_pos_open = []
            self.is_position = []
            self.position_entry_time = None
            self.max_len = len(self.data.df)
            self.entry_price = None

        def next(self):
            if len(self.data.index) >= self.max_len - 3:
                return
            if self.position.size != 0:
                self.time_since_pos_open += 1

            mask = LongOnlyStockTradingEnv.mask_actions(self.position.size != 0, self.time_since_pos_open / 100)

            # Commission adjusted pnl
            pnl = 0
            if self.position:
                pnl = self.position.pl_pct * 100
            obs = {
                "position_pnl_percentage": np.array([pnl], dtype=np.float64),
                "is_position": np.array([self.position.size != 0], dtype=np.float64),
            }
            obs.update({k: np.array([getattr(self, k)[-1]], dtype=np.float64) for k in obs_columns})
            for key in obs:
                self.env.obs_rms[key].update(obs[key])

            obs = self.env.normalize_obs(obs)
            action, _ = self.agent.predict(obs, deterministic=True, action_masks=mask)

            if action == 0 and self.position.size == 0:
                if max_size:
                    self.buy(size=max_size)
                else:
                    self.buy()
                self.position_entry_time = self.data.df.index[-1]
                self.time_since_pos_open = 1
                self.entry_price = self.data.Close[-1]
            elif action == 2 and self.position.size != 0:
                self.position.close()
                self.time_since_pos_open = 0
                self.position_entry_time = None
    return LongOnlyStrategy