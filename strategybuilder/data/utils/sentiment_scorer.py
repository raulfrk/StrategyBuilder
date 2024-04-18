from typing import Protocol

import pandas as pd


class SentimentScorer(Protocol):

    def score(cls, df: pd.DataFrame) -> pd.DataFrame:
        ...


class SumSentimentScorer:

    @classmethod
    def score(cls, df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby(df.index).sum()


class MostFrequentSentimentScorer:
    @classmethod
    def score(cls, df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby(df.index).agg(lambda x: x.value_counts().idxmax())


class AverageSentimentScorer:
    @classmethod
    def score(cls, df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby(df.index).mean()
