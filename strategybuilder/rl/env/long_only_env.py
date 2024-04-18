import copy
import random
from typing import Callable

import gymnasium
import numpy as np
import pandas as pd

INTERNAL_STATE_OBS_COLUMNS = ["position_pnl_percentage", "is_position"]

class LongOnlyInternalState:
    """Internal state for the trading environment environment."""

    def __init__(self, cash: float, obs_fields: list[str] | None = None, starting_interval: int = 0,
                 starting_interval_position: int = 0, commission=0):
        self.initial_cash = cash
        self.starting_interval = starting_interval
        self.starting_interval_position = starting_interval_position
        self.current_interval = starting_interval
        self.current_interval_position = starting_interval_position
        self.obs_fields = obs_fields or []
        self.previous_internal_state = None
        self.seen_intervals = {0}
        self.commission = commission

        self.cash = cash

        self.position_pnl = 0
        self.position_pnl_percentage = 0
        self.position_entry_time = None
        self.position_entry_price = None
        self.position_size = 0
        self.position_exit_time = None
        self.position_exit_price = None
        self.cash_before_position = cash
        self.current_time = None

        # Validate obs_fields
        for field in self.obs_fields:
            # Check if field is a property
            if not hasattr(self, field):
                raise ValueError(f"Field {field} is not a valid field in the internal state")

    def was_position_just_closed(self):
        if self.previous_internal_state is None:
            return False
        return self.previous_internal_state.position_size != 0 and self.position_size == 0

    @property
    def is_position(self) -> int:
        return self.position_size != 0

    def get_cash_percentage_difference_after_position_close(self):
        if not self.was_position_just_closed():
            return 0
        return (self.cash - self.cash_before_position) / self.cash_before_position * 100

    def was_position_just_opened(self):
        if self.previous_internal_state is None:
            return False
        return self.previous_internal_state.position_size == 0 and self.position_size != 0

    def holding_position(self):
        if self.previous_internal_state is None:
            return False
        return self.previous_internal_state.position_size != 0 and self.position_size != 0

    def holding_no_position(self):
        if self.previous_internal_state is None:
            return False
        return self.previous_internal_state.position_size == 0 and self.position_size == 0

    @property
    def time_since_pos_open(self):
        if self.current_time is None or self.position_entry_time is None or self.position_size == 0:
            return 0
        return (self.current_time - self.position_entry_time).total_seconds() / 60 / 100

    def reset(self, new_interval: int = 0):
        self.current_interval = new_interval
        self.current_interval_position = self.starting_interval_position
        self.reset_period()

    def reset_period(self):
        self.position_pnl = 0
        self.position_size = 0
        self.cash = self.initial_cash
        self.cash_before_position = self.initial_cash
        self.position_entry_time = None
        self.position_entry_price = None
        self.position_exit_time = None
        self.position_exit_price = None

    def update_pnl(self, current_price: float):
        if self.position_size != 0:
            self.position_pnl = (current_price - self.position_entry_price) * self.position_size
            self.position_pnl_percentage = (current_price - self.position_entry_price) / self.position_entry_price * 100
        else:
            self.position_pnl = 0
            self.position_pnl_percentage = 0


class LongOnlyStockTradingEnv(gymnasium.Env):
    """Environment for a stock trading agent. The agent can buy, sell or hold a stock. The agent can only hold one
    position at a time and the position is long only.

    :param data: pd.DataFrame: Dataframe with the stock data. The dataframe should have a datetime index and columns
    "open", "high", "low", "close", "volume".
    :param obs_columns: list[str]: List of columns to be used as observations for the agent from the data.
    :param initial_cash: float: Initial cash for the agent.
    :param trading_interval: str: Interval for trading. Default is "D" for daily trading. (The data will be split in days)
    :param reward_multiplier: float: Multiplier for the reward. Default is 1.0.
    :param reset_at_random: bool: If True, at the end of the trading interval (e.g. 1 day), the environment will reset
    to a random new day it has not seen before. Default is True.
    :param verbose: bool: If True, the environment will print the cash balance at the end of each trading interval.
    Default is False.
    :param commission: float: Commission for buying or selling a stock. The commission is the percentage of traded value
     e.g. 0.0003 = 0.03%. Default is 0.
    :param custom_reward_function: Callable[[LongOnlyStockTradingEnv], float]: Custom reward function for the agent.
    :param internal_state_obs_fields: list[str]: List of fields from the internal state to be used as observations.
    Available fields are: "position_pnl_percentage", "time_since_pos_open", "is_position".
    Default is ["position_pnl_percentage", "is_position"].
    :param trading_cooldown: int: Cooldown in minutes for closing a position once opened. Default is 2.
    :param close_position_n_min_before_eop: int: Close position n minutes before the end of the period. Default is 1.
    """

    def __init__(self, data: pd.DataFrame, obs_columns: list[str] | None = None, initial_cash: float = 100000,
                 trading_interval="D", reward_multiplier: float = 1.0, reset_at_random=True, verbose=False,
                 commission=0, custom_reward_function: Callable[["LongOnlyStockTradingEnv"], float] | None = None,
                 internal_state_obs_fields: list[str] | None = None, trading_cooldown: int = 2,
                 close_position_n_min_before_eop: int = 5):
        super(LongOnlyStockTradingEnv, self).__init__()
        # Overrides
        self.custom_reward_function = custom_reward_function

        if custom_reward_function is None:
            self.custom_reward_function = self._get_reward

        # External data
        self.data = data
        # Other parameters
        self.obs_columns = obs_columns
        self.initial_cash = initial_cash
        self.reward_multiplier = reward_multiplier
        self.verbose = verbose
        self.commission = commission
        self.trading_cooldown = trading_cooldown
        self.close_position_n_min_before_eop = close_position_n_min_before_eop

        if internal_state_obs_fields is None:
            self.internal_state_obs_fields = INTERNAL_STATE_OBS_COLUMNS
        else:
            self.internal_state_obs_fields = internal_state_obs_fields

        # Reset at random
        self.reset_at_random = reset_at_random

        # Split data into periods
        self.data["period"] = self.data.index.to_period(trading_interval)
        self.intervals = [group for name, group in self.data.groupby("period", as_index=False)]

        # Create the internal state
        self.internal_state = LongOnlyInternalState(cash=initial_cash,
                                                    obs_fields=self.internal_state_obs_fields, commission=commission)

        # Observation space
        self.observation_space = self._create_observation_space()

        # Action space (0: buy, 1: hold, 2: sell)
        self.action_space = gymnasium.spaces.Discrete(3)

    def _create_observation_space(self):
        obs_dict = {}

        for col in self.obs_columns:
            obs_dict[col] = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        for col in self.internal_state.obs_fields:
            obs_dict[col] = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        return gymnasium.spaces.Dict(obs_dict)

    def _get_current_data(self):
        return self.intervals[self.internal_state.current_interval].iloc[self.internal_state.current_interval_position]

    def get_observation(self):
        obs_dict = self._get_current_data()[self.obs_columns].to_dict()

        # Add internal state fields
        for field in self.internal_state.obs_fields:
            obs_dict[field] = getattr(self.internal_state, field)

        return {k: np.array([v], dtype=np.float32) for k, v in obs_dict.items()}

    def _get_time_till_eop(self) -> int | None:
        return len(self.intervals[self.internal_state.current_interval]) - self.internal_state.current_interval_position

    @classmethod
    def get_obs(cls, **kwargs):
        return {k: np.array([v]) for k, v in kwargs.items()}

    @classmethod
    def mask_actions(cls, position, time_since_pos_open=0, trading_cooldown=2, time_till_eop: int | None = None,
                     close_pos_at_eop: int = 1):
        """Only allow buy if no position is open, only allow sell if position is open and time since position open is
        greater than trading_cooldown. Hold is always allowed."""
        # print("position", position, "time_since_pos_open", time_since_pos_open, "trading_cooldown", trading_cooldown, "time_till_eop", time_till_eop, "close_pos_at_eop", close_pos_at_eop)
        if not position:
            # Prevent buying at the end of the period
            if time_till_eop and time_till_eop <= close_pos_at_eop:
                return [False, True, False]
            return [True, True, False]
        else:
            if time_till_eop and time_till_eop <= close_pos_at_eop:
                return [False, False, True]
            if time_since_pos_open < (trading_cooldown + 1) / 100:
                return [False, True, False]

            return [False, True, True]

    def action_masks(self):
        return LongOnlyStockTradingEnv.mask_actions(self.internal_state.position_size != 0,
                                                    self.internal_state.time_since_pos_open,
                                                    self.trading_cooldown,
                                                    self._get_time_till_eop(),
                                                    self.close_position_n_min_before_eop)

    def buy(self):
        if self.internal_state.position_size != 0:
            return

        price = self._get_current_data()["close"]
        # Save cash before position
        self.internal_state.cash_before_position = self.internal_state.cash
        # Buy
        self.internal_state.position_size = (
                (self.internal_state.cash - 2 * self.commission * self.internal_state.cash) // price)
        self.internal_state.position_entry_price = price
        self.internal_state.position_entry_time = self._get_current_data().name
        total_commission = self.internal_state.commission * self.internal_state.position_size * price
        self.internal_state.cash -= self.internal_state.position_size * price + total_commission

    def sell(self):
        if self.internal_state.position_size == 0:
            return

        price = self._get_current_data()["close"]
        # Sell
        total_commission = self.internal_state.commission * self.internal_state.position_size * price
        self.internal_state.cash += self.internal_state.position_size * price - total_commission
        self.internal_state.position_size = 0
        self.internal_state.position_exit_price = price
        self.internal_state.position_exit_time = self._get_current_data().name

    def act(self, action):
        if action == 0:
            self.buy()
        elif action == 2:
            self.sell()
        else:
            pass

    def _move_to_next_step(self):
        done = False

        if self.internal_state.current_interval_position + 1 >= len(
                self.intervals[self.internal_state.current_interval]):
            done = True
            if self.verbose:
                print("End of period cash: ", self.internal_state.cash)
        else:
            self.internal_state.current_interval_position += 1
        if not done:
            # Update internal state
            self.internal_state.update_pnl(self._get_current_data()["close"])
            self.internal_state.current_time = self._get_current_data().name

        return done

    def get_eop_cash_difference_perc(self):
        if self._get_time_till_eop() == self.close_position_n_min_before_eop - 1:
            return (
                        self.internal_state.cash - self.internal_state.initial_cash) / self.internal_state.initial_cash * 100
        return 0

    def _get_reward(self):

        return self.internal_state.get_cash_percentage_difference_after_position_close()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if len(self.internal_state.seen_intervals) == len(self.intervals):
            self.internal_state.seen_intervals = set()
        # Pick an unseen interval
        unseen_intervals = set(range(len(self.intervals))) - self.internal_state.seen_intervals
        if self.reset_at_random:
            # Pick a random unseen interval
            chosen_interval = random.choice(list(unseen_intervals))
        else:
            # Pick the next unseen interval
            chosen_interval = min(unseen_intervals)
        self.internal_state.seen_intervals.add(chosen_interval)
        self.internal_state.reset(new_interval=chosen_interval)
        return self.get_observation(), {}

    def render(self, mode='human'):
        print(f'Balance: {self.internal_state.cash}')

    def get_current_time(self):
        return self._get_current_data().name

    def step(self, action):
        # Save previous internal state
        obs_before_next_step = self.get_observation()
        previous_internal_state = copy.deepcopy(self.internal_state)
        previous_internal_state.previous_internal_state = None
        self.internal_state.previous_internal_state = previous_internal_state

        # Act
        self.act(action)

        reward = self._get_reward() * self.reward_multiplier

        # Move to next step
        done = self._move_to_next_step()

        # Set the previous internal state

        # Observe changes
        obs = self.get_observation()

        # Prepare info
        info = {"action": action, "reward": reward, "done": done, "date": self._get_current_data().name,
                "post_action_obs": obs, "cash": self.internal_state.cash,
                "position_size": self.internal_state.position_size}

        info.update({k: v[0] for (k, v) in obs_before_next_step.items()})
        info.update(self._get_current_data().to_dict())

        return obs, reward, done, False, info



