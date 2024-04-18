import asyncio
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from io import StringIO

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from otpclient.client.dataprovider_client import DataproviderClient
from otpclient.client.datastorage_client import DatastorageClient
from otpclient.client.enums import AssetClassEnum, ComponentEnum, SourceEnum, DatatypeEnum, AccountEnum, TimeFrameEnum, \
    SentimentAnalysisProcessEnum, LLMProviderEnum
from otpclient.client.user_client import UserClient
from otpclient.proto.bar import Bar
from otpclient.proto.news import News
from strategybuilder.data.const import CACHE_PATH


def runner(func, queue, insubprocess=False):
    loop = asyncio.get_event_loop()

    result = loop.run_until_complete(func())

    queue.put(result)


class BarDataFetch(BaseEstimator, TransformerMixin):
    def __init__(self, asset_class: AssetClassEnum, symbol: str,
                 start_time: datetime | timedelta | None = None, end_time: datetime | timedelta | None = None,
                 cache: bool = True, load_db_only=False):
        self.asset_class = asset_class
        self.symbol = symbol
        self.start_time = start_time
        self.end_time = end_time
        self.cache = cache
        self.load_db_only = load_db_only

    def fit(self, X, y=None):
        return self

    async def fetch(self) -> pd.DataFrame:

        component_to_request = ComponentEnum.DATAPROVIDER if self.load_db_only else ComponentEnum.DATASTORAGE

        start_t: datetime | None = None
        end_t: datetime | None = None

        if isinstance(self.start_time, datetime):
            start_t = self.start_time
        if isinstance(self.end_time, datetime):
            end_t = self.end_time
        if isinstance(start_t, timedelta):
            start_t = datetime.now() - self.start_time
        if isinstance(end_t, timedelta):
            end_t = datetime.now() - self.end_time

        symbol = self.symbol.upper()
        symbol_name_file = symbol.replace("/", "_")

        cache_file_name = f"{self.asset_class.value}_{symbol_name_file}_{int(start_t.timestamp())}_{int(end_t.timestamp())}_bar.feather"
        if self.cache:
            if (CACHE_PATH / cache_file_name).exists():
                return pd.read_feather(CACHE_PATH / cache_file_name)

        client: DatastorageClient | DataproviderClient = ((await UserClient.new()).
                                                          __getattribute__(component_to_request.value))
        print(f"Fetching data for {symbol} from {start_t} to {end_t}")
        data: list[Bar] = await client.data_get_autoresolve(
            SourceEnum.ALPACA,
            self.asset_class,
            symbol,
            DatatypeEnum.BAR,
            AccountEnum.DEFAULT,
            start_t,
            end_t,
            TimeFrameEnum.ONE_MINUTE,
        )

        bar_df: pd.DataFrame = Bar.list_to_dataframe(data)
        bar_df.sort_index(ascending=True, inplace=True)
        if self.cache:
            bar_df.to_feather(CACHE_PATH / cache_file_name)
        return bar_df

    def transform(self, X):
        create_subprocess = asyncio.get_event_loop().is_running()
        # print(f"Subprocess PID: {subprocess_pid}, Main PID: {main_pid}")
        queue = multiprocessing.Queue()
        if not create_subprocess:
            runner(self.fetch, queue)
            data = queue.get()
            return data
        process = multiprocessing.Process(target=runner, args=(self.fetch, queue,))
        process.start()
        data = queue.get()
        process.join()
        return data


class NewsDataFetch(BaseEstimator, TransformerMixin):
    def __init__(self, symbol: str, analysis_process: SentimentAnalysisProcessEnum, model: str,
                 model_provider: LLMProviderEnum,
                 system_prompt: str, start_time: datetime | timedelta, end_time: datetime | timedelta,
                 load_db_only: bool = False, load_db_only_sentiment: bool = False):
        self.symbol = symbol
        self.analysis_process = analysis_process
        self.model = model
        self.model_provider = model_provider
        self.system_prompt = system_prompt
        self.start_time = start_time
        self.end_time = end_time
        self.load_db_only = load_db_only
        self.load_db_only_sentiment = load_db_only_sentiment

    async def fetch(self):
        """Fetch news data."""
        component_to_request = ComponentEnum.DATAPROVIDER if self.load_db_only else ComponentEnum.SENTIMENT_ANALYZER

        start_t: datetime | None = None
        end_t: datetime | None = None

        if isinstance(self.start_time, datetime):
            start_t = self.start_time
        if isinstance(self.end_time, datetime):
            end_t = self.end_time
        if isinstance(start_t, timedelta):
            start_t = datetime.now() - self.start_time
        if isinstance(end_t, timedelta):
            end_t = datetime.now() - self.end_time

        symbol = self.symbol.upper()

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
        if self.load_db_only_sentiment:
            await client.data_get(
                SourceEnum.ALPACA,
                symbol,
                start_t,
                end_t,
                self.analysis_process,
                self.model,
                self.model_provider,
                self.system_prompt,
                no_confirm=True
            )
            return None
        data: list[News] = await client.data_get_autoresolve(
            SourceEnum.ALPACA,
            symbol,
            start_t,
            end_t,
            self.analysis_process,
            self.model,
            self.model_provider,
            self.system_prompt
        )

        news_df: pd.DataFrame = News.list_to_dataframe(data)
        news_df["created_at"] = news_df.index
        news_df.sort_index(ascending=True, inplace=True)
        sentiments = [x.to_json(orient="records") for x in news_df["sentiments"]]
        symbols = [list(x) for x in news_df["symbols"].to_list()]
        news_df.drop(columns=["sentiments", "symbols"], inplace=True)
        return (news_df.to_json(orient="records"), sentiments, symbols)

    def reunite_sentiment(self, data, sentiments, symbols):
        data = pd.read_json(StringIO(data))

        data["sentiments"] = [pd.read_json(StringIO(x), orient="records") for x in sentiments]
        data["symbols"] = symbols
        data["sentiments"] = data["sentiments"].apply(lambda x: x.assign(news=""))

        return data

    def transform(self, X):
        create_subprocess = asyncio.get_event_loop().is_running()

        queue = multiprocessing.Queue()
        if not create_subprocess:
            runner(self.fetch, queue)
            data = queue.get()
            return self.reunite_sentiment(data[0], data[1], data[2])
        process = multiprocessing.Process(target=runner, args=(self.fetch, queue,))
        process.start()
        data = queue.get()
        process.join()
        return self.reunite_sentiment(data[0], data[1], data[2])


