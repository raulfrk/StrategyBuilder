from typing import Protocol

import pandas as pd


class DataGetter(Protocol):

    def get_data(self) -> pd.DataFrame: ...


class Indicator(Protocol):
    def apply(self, data: DataGetter) -> pd.DataFrame: ...

    def get_dest_column(self) -> str: ...



class SimpleMovingAverage():
    def __init__(self, window: int, source_column: str, dest_column: str | None = None) -> None:
        self.window = window
        self.source_column = source_column
        self.dest_column = dest_column or f"sma_{self.source_column}_{self.window}"

    def apply(self, data: DataGetter) -> pd.DataFrame:
        dt = data.get_data()
        dest_column = self.dest_column
        dt[dest_column] = dt[self.source_column].rolling(window=self.window).mean()
        return dt


class ExponentialMovingAverage():
    def __init__(self, span: int, source_column: str, dest_column: str | None = None) -> None:
        self.span = span
        self.source_column = source_column
        self.dest_column = dest_column or f"ema_{self.source_column}_{self.span}"

    def apply(self, data: DataGetter) -> pd.DataFrame:
        dt = data.get_data()
        dest_column = self.dest_column
        dt[dest_column] = dt[self.source_column].ewm(span=self.span, adjust=False).mean()
        return dt


class RSIIndicator():
    def __init__(self, window: int, source_column: str, dest_column: str | None = None) -> None:
        self.window = window
        self.source_column = source_column
        self.dest_column = dest_column or f"rsi_{self.source_column}_{self.window}"

    def apply(self, data: DataGetter) -> pd.DataFrame:
        dt = data.get_data()
        dest_column = self.dest_column or self.dest_column
        diff = dt[self.source_column].diff()
        gain = diff.where(diff > 0, 0)
        loss = -diff.where(diff < 0, 0)
        avg_gain = gain.rolling(window=self.window).mean()
        avg_loss = loss.rolling(window=self.window).mean()
        rs = avg_gain / avg_loss
        dt[dest_column] = 100 - (100 / (1 + rs))
        return dt
