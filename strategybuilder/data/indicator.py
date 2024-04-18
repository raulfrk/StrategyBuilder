import pandas as pd
from pandas_ta import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin


class Indicator(TransformerMixin):
    def __init__(self, indicator, col_names: str | list[str] | None = None, **kwargs):
        self.indicator = indicator
        if isinstance(col_names, str) or col_names is None:
            self.col_names = col_names or f"{indicator.__name__}"
        elif isinstance(col_names, list):
            self.col_names = [f"{col_name}" for col_name in col_names]
        self.kwargs = kwargs

    def transform(self, X: DataFrame) -> DataFrame:
        indicator = X.ta.__getattribute__(self.indicator.__name__)(**self.kwargs)
        if isinstance(indicator, pd.Series):
            X[f"{self.col_names}"] = indicator
            return X
        if isinstance(self.col_names, list):
            indicator_cols = self.col_names
        else:
            indicator_cols = indicator.columns
        for i, col in enumerate(indicator_cols):
            X.insert(len(X.columns), col,indicator[indicator.columns[i]])

        return X

class Diff(TransformerMixin):
    def __init__(self, col, new_col: str | None = None, periods=1, pct: bool = False, pct_multiplier: int = 100):
        self.col = col
        self.new_col = new_col or f"{col}_diff{'_pct' if pct else ''}"
        self.pct = pct
        self.pct_multiplier = pct_multiplier
        self.periods = periods

    def transform(self, X: DataFrame) -> DataFrame:
        if self.pct:
            X[self.new_col] = X[self.col].pct_change(self.periods) * self.pct_multiplier
        else:
            X[self.new_col] = X[self.col].diff(self.periods)

        return X

class SelfRatio(TransformerMixin):
    def __init__(self, col, new_col: str | None = None, periods=1):
        self.col = col
        self.new_col = new_col or f"{col}_ratio_{periods}"
        self.periods = periods

    def transform(self, X: DataFrame) -> DataFrame:

        X[self.new_col] = X[self.col] / X[self.col].shift(self.periods)

        return X

class Ratio(TransformerMixin):
    def __init__(self, col1, col2, new_col: str | None = None):
        self.col1 = col1
        self.col2 = col2
        self.new_col = new_col or f"{col1}_ratio_{col2}"

    def transform(self, X: DataFrame) -> DataFrame:
        X[self.new_col] = X[self.col1] / X[self.col2]

        return X

class AvgDiff(TransformerMixin):
    def __init__(self, col, window, new_col: str | None = None, periods=1, pct: bool = False, pct_multiplier: int = 100):
        self.col = col
        self.new_col = new_col or f"{col}_avg_diff{'_pct' if pct else ''}"
        self.pct = pct
        self.pct_multiplier = pct_multiplier
        self.window = window
        self.periods = periods

    def transform(self, X: DataFrame) -> DataFrame:
        if self.pct:
            X[self.new_col] = X[self.col].pct_change(periods=self.periods).rolling(window=self.window).mean() * self.pct_multiplier
        else:
            X[self.new_col] = X[self.col].diff(self.periods).rolling(window=self.window).mean()

        return X