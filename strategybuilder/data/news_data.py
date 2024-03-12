import hashlib
from datetime import datetime, timedelta
from typing import Optional, Callable, Protocol

import pandas as pd
from otpclient.client.enums import ComponentEnum, SentimentAnalysisProcessEnum, LLMProviderEnum, SourceEnum, \
    AssetClassEnum, DatatypeEnum, AccountEnum, TimeFrameEnum
from otpclient.client.user_client import UserClient
from otpclient.proto.news import News

from strategybuilder.data.const import CACHE_PATH
from strategybuilder.data.filters import filter_time


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


class NewsDataWrapper:
    def __init__(self, news_data: pd.DataFrame):
        self.data = news_data

    def get_data(self) -> pd.DataFrame:
        return self.data

    def copy(self) -> "NewsDataWrapper":
        new = NewsDataWrapper(self.data.copy())
        return new

    @classmethod
    async def fetch(cls, symbol: str, analysis_process: SentimentAnalysisProcessEnum, model: str,
                    model_provider: LLMProviderEnum,
                    system_prompt: str, start_time: datetime | timedelta, end_time: datetime | timedelta,
                    load_db_only: bool = False, load_db_only_sentiment: bool = False) -> Optional["NewsDataWrapper"]:
        """Fetch news data."""
        component_to_request = ComponentEnum.DATAPROVIDER if load_db_only else ComponentEnum.SENTIMENT_ANALYZER

        start_t: datetime | None = None
        end_t: datetime | None = None

        if isinstance(start_time, datetime):
            start_t = start_time
        if isinstance(end_time, datetime):
            end_t = end_time
        if isinstance(start_t, timedelta):
            start_t = datetime.now() - start_time
        if isinstance(end_t, timedelta):
            end_t = datetime.now() - end_time

        symbol = symbol.upper()

        if component_to_request == ComponentEnum.DATAPROVIDER:
            client = (await UserClient.new()).dataprovider
            await client.data_get(
                SourceEnum.ALPACA,
                AssetClassEnum.NEWS,
                symbol,
                DatatypeEnum.RAW_TEXT,
                AccountEnum.DEFAULT,
                start_t,
                end_t,
                TimeFrameEnum.ONE_MINUTE,
                no_confirm=True
            )

            return None

        client = (await UserClient.new()).sentimentanalyzer
        if load_db_only_sentiment:
            await client.data_get(
                SourceEnum.ALPACA,
                symbol,
                start_t,
                end_t,
                analysis_process,
                model,
                model_provider,
                system_prompt,
                no_confirm=True
            )
            return None
        data: list[News] = await client.data_get_autoresolve(
            SourceEnum.ALPACA,
            symbol,
            start_t,
            end_t,
            analysis_process,
            model,
            model_provider,
            system_prompt
        )

        news_df: pd.DataFrame = News.list_to_dataframe(data)
        news_df.sort_index(ascending=True, inplace=True)

        return cls(news_df)

    def filter_time(self, within_hours: list[tuple[int, int]] | None = None,
                    within_days: list[tuple[datetime, datetime]] | None = None,
                    exclude_days: list[tuple[datetime, datetime]] | None = None,
                    on_week_days: list[int] | None = None,
                    inplace=False) -> Optional["NewsDataWrapper"]:
        data = self.data.copy()

        data = filter_time(data, within_hours, within_days, exclude_days, on_week_days)

        if inplace:
            self.data = data
            return None

        return NewsDataWrapper(data)

    def filter_news(self, row_criteria: Callable, inplace=False) -> Optional["NewsDataWrapper"]:
        data = self.data.copy()
        data = data[data.apply(row_criteria, axis=1)]
        if inplace:
            self.data = data
            return None
        return NewsDataWrapper(data)

    def filter_sentiment(self, symbol: list[str] | None = None,
                         sentiment: list[str] | None = None,
                         llm: list[str] | None = None,
                         failed: bool | None = None,
                         sentiment_analysis_process: list[str] | None = None,
                         system_prompt: list[str] | None = None,
                         remove_news_no_sentiment: bool = True,
                         inplace=False) -> Optional["NewsDataWrapper"]:
        data = self.data.copy()

        if symbol:
            data["sentiments"] = data["sentiments"].apply(lambda x: x[x["symbol"].isin(symbol)])

        if sentiment:
            data["sentiments"] = data["sentiments"].apply(lambda x: x[x["sentiment"].isin(sentiment)])

        if llm:
            data["sentiments"] = data["sentiments"].apply(lambda x: x[x["llm"].isin(llm)])

        if failed is not None:
            data["sentiments"] = data["sentiments"].apply(lambda x: x[x["failed"] == failed])

        if sentiment_analysis_process:
            data["sentiments"] = data["sentiments"].apply(
                lambda x: x[x["sentiment_analysis_process"].isin(sentiment_analysis_process)])
        if system_prompt:
            data["sentiments"] = data["sentiments"].apply(lambda x: x[x["system_prompt"].isin(system_prompt)])
        if remove_news_no_sentiment:
            data = data[data["sentiments"].apply(lambda x: len(x) > 0)]

        if inplace:
            self.data = data
            return None
        return NewsDataWrapper(data)

    def flatten_sentiment(self, inplace=False) -> Optional["NewsDataWrapper"]:
        data = self.data.copy()

        num_sentiments = [len(x["sentiments"]) for _, x in data.iterrows()]

        if any(x > 1 for x in num_sentiments):
            raise ValueError(
                "Data has multiple sentiments for a single news. Cannot flatten. Considering filtering the data.")

        data["sentiments"] = data["sentiments"].apply(lambda x: x.iloc[0]["sentiment"])
        # Fill empty strings with NaN
        data["sentiments"] = data["sentiments"].replace("", float("nan"))
        if inplace:
            self.data = data
            return None
        return NewsDataWrapper(data)

    def get_sentiment_score(self, timeframe: str, scoring_strategy: SentimentScorer):
        data = self.data.copy()
        # Round index to timeframe
        data.index = data.index.round(timeframe)

        # Replace "positive" with 1, "negative" with -1, "neutral" with 0
        data["sentiment_score"] = data["sentiments"].map({"positive": 1, "negative": -1, "neutral": 0}).infer_objects(
            copy=False)

        # Create new dataframe with same index and sentiment
        df = pd.DataFrame(data["sentiment_score"], index=data.index)
        # Rename index to timeframe
        df.index.name = "timestamp"

        return scoring_strategy.score(df)
