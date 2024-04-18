from datetime import datetime

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from strategybuilder.data.utils.filters import filter_time


class FilterTime(BaseEstimator, TransformerMixin):
    def __init__(self, within_hours: list[tuple[int, int]] | None = None,
                 within_days: list[tuple[datetime, datetime]] | None = None,
                 exclude_days: list[tuple[datetime, datetime]] | None = None,
                 on_week_days: list[int] | None = None):
        self.within_hours = within_hours
        self.within_days = within_days
        self.exclude_days = exclude_days
        self.on_week_days = on_week_days

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return filter_time(X, self.within_hours, self.within_days, self.exclude_days, self.on_week_days)


class FilterNews(BaseEstimator, TransformerMixin):
    def __init__(self, row_criteria: callable):
        self.row_criteria = row_criteria

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[X.apply(self.row_criteria, axis=1)]


class FilterSentiment(BaseEstimator, TransformerMixin):
    def __init__(self, symbol: list[str] | None = None,
                 sentiment: list[str] | None = None,
                 llm: list[str] | None = None,
                 failed: bool | None = None,
                 sentiment_analysis_process: list[str] | None = None,
                 system_prompt: list[str] | None = None,
                 remove_news_no_sentiment: bool = True, pick_first_on_duplicate: bool = False):
        self.symbol = symbol
        self.sentiment = sentiment
        self.llm = llm
        self.failed = failed
        self.sentiment_analysis_process = sentiment_analysis_process
        self.system_prompt = system_prompt
        self.remove_news_no_sentiment = remove_news_no_sentiment
        self.pick_first_on_duplicate = pick_first_on_duplicate

    def transform(self, data):
        if self.symbol:
            data["sentiments"] = data["sentiments"].apply(lambda x: x[x["symbol"].isin(self.symbol)])

        if self.sentiment:
            data["sentiments"] = data["sentiments"].apply(lambda x: x[x["sentiment"].isin(self.sentiment)])

        if self.llm:
            data["sentiments"] = data["sentiments"].apply(lambda x: x[x["llm"].isin(self.llm)])

        if self.failed is not None:
            data["sentiments"] = data["sentiments"].apply(lambda x: x[x["failed"] == self.failed])

        if self.sentiment_analysis_process:
            data["sentiments"] = data["sentiments"].apply(
                lambda x: x[x["sentiment_analysis_process"].isin(self.sentiment_analysis_process)])
        if self.system_prompt:
            data["sentiments"] = data["sentiments"].apply(lambda x: x[x["system_prompt"].isin(self.system_prompt)])
        if self.remove_news_no_sentiment:
            data = data[data["sentiments"].apply(lambda x: len(x) > 0)]
        if self.pick_first_on_duplicate:
            data["sentiments"] = data["sentiments"].apply(lambda x: x.drop_duplicates(subset=["symbol"], keep="first"))
        return data
