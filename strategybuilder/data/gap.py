from datetime import timedelta, datetime
from typing import Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from strategybuilder.data.utils.gaps import GapFiller


class BarGapExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, min_gap: timedelta = timedelta(minutes=1),
                 within_hours: Optional[tuple[int, int]] = None,
                 within_days: Optional[tuple[datetime, datetime]] = None,
                 on_week_days: Optional[list[int]] = None):
        self.min_gap = min_gap
        self.within_hours = within_hours
        self.within_days = within_days
        self.on_week_days = on_week_days

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        gaps = pd.DataFrame()

        gaps["start"] = X.index[:-1]
        gaps["end"] = X.index[1:]
        gaps["gap"] = gaps["end"] - gaps["start"]
        gaps = gaps[(gaps["gap"] > self.min_gap)]

        if self.within_hours:
            gaps = gaps[(gaps["start"].dt.hour >= self.within_hours[0]) & (gaps["end"].dt.hour < self.within_hours[1]) &
                        (gaps["start"].dt.date == gaps["end"].dt.date)]
        if self.within_days:
            gaps = gaps[(gaps["start"].dt.date >= self.within_days[0]) & (gaps["end"].dt.date < self.within_days[1])]
        if self.on_week_days:
            gaps = gaps[(gaps["start"].dt.weekday.isin(self.on_week_days)) & (gaps["end"].dt.weekday.isin(
                self.on_week_days))]
        return gaps


class BarFillGaps(BaseEstimator, TransformerMixin):
    def __init__(self, gap_filler: GapFiller, resample_rule="min"):
        self.gap_filler = gap_filler
        self.resample_rule = resample_rule

    def transform(self, X: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
        if "gap" in X[0].columns.values:
            gaps, X = X
        else:
            X, gaps = X
        return self.gap_filler.fill(X, gaps, resample_rule=self.resample_rule)
