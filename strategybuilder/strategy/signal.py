from asyncio import Protocol
from scipy.signal import find_peaks

import pandas as pd


class SignalGenerator(Protocol):

    def generate(self, market_data: pd.DataFrame) -> pd.DataFrame:
        ...


class StrategySimulator(Protocol):

    def simulate(self, market_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ...

    def get_pnl_percentage(self) -> float:
        ...

    def get_pnl(self) -> float:
        ...

class InitialBalanceFractionalSharesStrategySimulator:
    def __init__(self, initial_balance: float):
        self.initial_balance = float(initial_balance)
        self.simulation_market_data = None
        self.trading_fee = 0.02

    def get_pnl_percentage(self) -> float:
        """Get the percentage of the profit or loss."""
        return (self.simulation_market_data["PnL"].iloc[-1] / self.initial_balance) * 100

    def get_pnl(self) -> float:
        """Get the profit or loss."""
        return self.simulation_market_data["PnL"].iloc[-1]


    def simulate(self, market_data: pd.DataFrame, sell_eod: bool = True) -> pd.DataFrame:
        """Simulate a strategy based on initial balance."""
        market_data = market_data.copy()
        market_data["Balance"] = self.initial_balance
        market_data["Shares"] = 0.0
        market_data["PnL"] = 0.0
        market_data["PnLPercentage"] = 0.0
        # Set last signal to sell
        last_index = market_data.index[-1]
        if sell_eod:
            market_data.at[last_index, "Signal"] = "Sell"
        last_shares = 0.0
        last_balance = self.initial_balance
        last_pnl = 0.0
        for i, row in market_data.iterrows():
            num_index = int(market_data.index.get_loc(i))
            if num_index == 0:
                continue
            if row["Signal"] == "Buy" and market_data.iloc[num_index - 1]["Balance"] > 0.0:
                market_data.at[i, "Shares"] = (market_data.iloc[num_index - 1]["Balance"] - self.trading_fee) / row["close"]
                # market_data.at[i, "Shares"] = market_data.iloc[num_index - 1]["Balance"] / row["close"]
                last_shares = market_data.at[i, "Shares"]
                market_data.at[i, "Balance"] = 0.0
                last_balance = 0.0
                market_data.at[i, "PnL"] = last_pnl
            elif row["Signal"] == "Sell" and market_data.iloc[num_index - 1]["Shares"] > 0.0:
                market_data.at[i, "Balance"] = (market_data.iloc[num_index - 1]["Shares"] * row["close"] - self.trading_fee)
                market_data.at[i, "Shares"] = 0.0
                last_shares = 0.0
                last_balance = market_data.at[i, "Balance"]
                market_data.at[i, "PnL"] = market_data.at[i, "Balance"] - self.initial_balance
                market_data.at[i, "PnLPercentage"] = (market_data.at[i, "PnL"] / self.initial_balance) * 100
                last_pnl = market_data.at[i, "PnL"]
            else:
                market_data.at[i, "Shares"] = last_shares
                market_data.at[i, "Balance"] = last_balance
                market_data.at[i, "PnL"] = last_pnl
        self.simulation_market_data = market_data
        return market_data


class PeaksAndValleysBCSignalGenerator:
    def __init__(self, distance: int, distance_buy_sell: int = 0, prominance=0, platoe_size=0, source_column: str = "close"):
        self.distance = distance
        self.source_column = source_column
        self.distance_buy_sell = distance_buy_sell
        self.prominence = prominance
        self.platoe_size = platoe_size

    def generate(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy and sell signals based on peaks and valleys."""
        market_data = market_data.copy()
        peaks, _ = find_peaks(market_data[self.source_column], distance=self.distance, prominence=self.prominence, plateau_size=self.platoe_size)
        valleys, _ = find_peaks(-market_data[self.source_column], distance=self.distance, prominence=self.prominence, plateau_size=self.platoe_size)


        peaks_index = market_data.index[peaks]
        valleys_index = market_data.index[valleys]
        market_data["Signal"] = "Hold"
        market_data.loc[peaks_index, "Signal"] = "Sell"
        market_data.loc[valleys_index, "Signal"] = "Buy"

        # Signals should alternate, there should be no consecutive buys or sells even with holds in between
        last_signal = "Hold"
        time_since_last_signal = 0
        for i, row in market_data.iterrows():
            if time_since_last_signal < self.distance_buy_sell:
                time_since_last_signal += 1
                market_data.loc[i, "Signal"] = "Hold"
                continue
            if row["Signal"] == "Hold":
                continue
            if row["Signal"] == "Buy":
                if last_signal == "Buy":
                    market_data.loc[i, "Signal"] = "Hold"
                else:
                    last_signal = "Buy"
                    market_data.loc[i, "Signal"] = "Buy"
                    time_since_last_signal = 0

            if row["Signal"] == "Sell":
                if last_signal == "Sell":
                    market_data.loc[i, "Signal"] = "Hold"
                else:
                    last_signal = "Sell"
                    market_data.loc[i, "Signal"] = "Sell"
                    time_since_last_signal = 0
        return market_data
