import pandas as pd
from backtesting import Backtest
from gymnasium import make
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from strategybuilder.backtesting.long_only_backtester import generate_long_only_backtester
from strategybuilder.data.dataframe import DfOp


def multi_day_backtesting(data: pd.DataFrame, initial_equity: int, obs_columns: list[str], env_name: str,
                          agent_path: str, commission: float = 0.0003, verbose: bool = False, max_size: int | None = None):
    days = [group for label, group in data.groupby(data.index.date)]
    all_stats = pd.DataFrame()
    equity = initial_equity

    for day in days:
        mod_data = DfOp(lambda df: df.rename(
            columns={"close": "Close", "volume": "Volume", "open": "Open", "high": "High", "low": "Low"})).transform(
            day)
        env = make(env_name, data=day, obs_columns=obs_columns, reward_multiplier=1,
                   verbose=True, reset_at_random=False)
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        env = VecNormalize(env, clip_obs=100, training=False, norm_reward=False)
        agent = MaskablePPO.load(agent_path, env)
        strategy = generate_long_only_backtester(obs_columns,
                                                 internal_state_obs=["position_pnl_percentage", "is_position"],
                                                 env=env, agent=agent, max_size=max_size)

        bt = Backtest(mod_data, strategy, cash=equity, trade_on_close=True, commission=commission)
        stats = bt.run()
        all_stats = pd.concat([all_stats, pd.DataFrame(stats).T])
        equity = stats["Equity Final [$]"]
        if verbose:
            print(f"Date: {day.index.date[0]}, Equity: {equity}")

    # Add aditional stats
    all_stats["avg_return"] = all_stats["Return [%]"].mean()

    all_stats["avg_winrate"] = all_stats["Win Rate [%]"].mean()
    all_stats["avg_buy_hold_return"] = all_stats["Buy & Hold Return [%]"].mean()
    all_stats.set_index("End", inplace=True)

    # Get all trades
    all_trades = pd.concat(
        [x["_trades"] for _, x in all_stats.sort_index(ascending=False).iterrows() if len(x["_trades"]) > 0])
    all_trades.set_index("ExitTime", inplace=True)
    all_trades.sort_index(ascending=True, inplace=True)

    return all_stats, all_trades
