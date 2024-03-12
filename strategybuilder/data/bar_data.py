import copy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Any

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from otpclient.client.user_client import UserClient
from otpclient.client.datastorage_client import DatastorageClient
from otpclient.client.dataprovider_client import DataproviderClient

from otpclient.client.enums import *
from otpclient.proto.bar import Bar

from strategybuilder.data.const import CACHE_PATH
from strategybuilder.data.filters import filter_time
from strategybuilder.data.indicators import Indicator
from strategybuilder.data.utils.gaps import GapFiller, GapWrapper
from strategybuilder.data.utils.scaling import Scaler
from strategybuilder.strategy.signal import SignalGenerator, StrategySimulator


class BarDataGapWrapper:

    def __init__(self, gap_data: pd.DataFrame) -> None:
        self.data = gap_data

    def get_max_gap(self) -> "BarDataGapWrapper":
        return BarDataGapWrapper(self.data[self.data["gap"].max() == self.data["gap"]])

    def get_gap_start_end(self) -> pd.DataFrame:
        return self.data[["start", "end"]]


class BarDataWrapper:
    def __init__(self, bar_data: pd.DataFrame, scaler: dict[str, Scaler] | None = None) -> None:
        self.data: pd.DataFrame = bar_data
        # self.scaler: Scaler | None = scaler
        self.scaler: dict[str, Scaler] = scaler
        self.single_scaler: Scaler | None = None
        self.indicators: list[Indicator] = []
        self.simulator: StrategySimulator | None = None

    def get_data(self) -> pd.DataFrame:
        return self.data

    def copy(self) -> "BarDataWrapper":
        new = BarDataWrapper(self.data.copy(), self.scaler)
        new.indicators = self.indicators
        return new

    def add_indicator(self, indicator: Indicator, inplace=False) -> Optional["BarDataWrapper"]:
        if inplace:
            self.indicators.append(indicator)
            self.data = indicator.apply(self)
            return None
        new = self.copy()
        new.add_indicator(indicator, inplace=True)
        return new

    def remove_indicator(self, indicator: Indicator, inplace=False) -> Optional["BarDataWrapper"]:
        col: str = ""
        i_to_remove: Indicator | None = None
        for i in self.indicators:
            if i.get_dest_column() == indicator.get_dest_column():
                col = i.get_dest_column()
                i_to_remove = i
        if inplace:
            self.indicators.remove(i_to_remove)
            self.data.drop(columns=[col], inplace=True)
            return None

        new = self.copy()
        new.remove_indicator(indicator, inplace=True)
        return new

    @classmethod
    async def fetch(cls, asset_class: AssetClassEnum, symbol: str,
                    start_time: datetime | timedelta | None = None, end_time: datetime | timedelta | None = None,
                    cache: bool = True, load_db_only=False) -> Optional["BarDataWrapper"]:

        component_to_request = ComponentEnum.DATAPROVIDER if load_db_only else ComponentEnum.DATASTORAGE

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
        symbol_name_file = symbol.replace("/", "_")

        cache_file_name = f"{asset_class.value}_{symbol_name_file}_{int(start_t.timestamp())}_{int(end_t.timestamp())}_bar.feather"
        if cache:
            if (CACHE_PATH / cache_file_name).exists():
                return BarDataWrapper(pd.read_feather(CACHE_PATH / cache_file_name))

        client: DatastorageClient | DataproviderClient = ((await UserClient.new()).
                                                          __getattribute__(component_to_request.value))

        data: list[Bar] = await client.data_get_autoresolve(
            SourceEnum.ALPACA,
            asset_class,
            symbol,
            DatatypeEnum.BAR,
            AccountEnum.DEFAULT,
            start_t,
            end_t,
            TimeFrameEnum.ONE_MINUTE,
        )

        bar_df: pd.DataFrame = Bar.list_to_dataframe(data)
        bar_df.sort_index(ascending=True, inplace=True)
        if cache:
            bar_df.to_feather(CACHE_PATH / cache_file_name)
        return BarDataWrapper(bar_df)

    def get_temporal_gaps(self,
                          min_gap: timedelta = timedelta(minutes=1),
                          within_hours: Optional[tuple[int, int]] = None,
                          within_days: Optional[tuple[datetime, datetime]] = None,
                          on_week_days: Optional[list[int]] = None

                          ) -> BarDataGapWrapper:

        gaps = pd.DataFrame()
        gaps["start"] = self.data.index[:-1]
        gaps["end"] = self.data.index[1:]
        gaps["gap"] = gaps["end"] - gaps["start"]
        gaps = gaps[(gaps["gap"] > min_gap)]

        if within_hours:
            gaps = gaps[(gaps["start"].dt.hour >= within_hours[0]) & (gaps["end"].dt.hour < within_hours[1]) &
                        (gaps["start"].dt.date == gaps["end"].dt.date)]
        if within_days:
            gaps = gaps[(gaps["start"].dt.date >= within_days[0]) & (gaps["end"].dt.date < within_days[1])]
        if on_week_days:
            gaps = gaps[(gaps["start"].dt.weekday.isin(on_week_days)) & (gaps["end"].dt.weekday.isin(on_week_days))]

        return BarDataGapWrapper(gaps)

    def filter_time(self, within_hours: list[tuple[int, int]] | None = None,
                    within_days: list[tuple[datetime, datetime]] | None = None,
                    exclude_days: list[tuple[datetime, datetime]] | None = None,
                    on_week_days: list[int] | None = None,
                    inplace=False) -> Optional["BarDataWrapper"]:
        data = self.data.copy()

        data = filter_time(data, within_hours, within_days, exclude_days, on_week_days)

        if inplace:
            self.data = data
            return None

        return BarDataWrapper(data)

    def fill_time_gaps(self, gaps: GapWrapper, gap_filler: GapFiller, resample_rule="min",
                       in_place=False) -> Optional["BarDataWrapper"]:
        data = gap_filler.fill(self.data, gaps, resample_rule=resample_rule)

        if in_place:
            self.data = data
            return None
        return BarDataWrapper(data)

    def resample(self, time_frame: str, inplace=False) -> Optional["BarDataWrapper"]:
        # Calculate the volume * typical price for each minute
        data = self.data.copy()
        data["typical_price"] = (data["high"] + data["low"] + data["close"]) / 3
        data["typical_volume"] = data["typical_price"] * data["volume"]
        agg = {col: "" for col in data.columns}
        agg.update({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "trade_count": "sum",
            'typical_volume': 'sum',
        })

        # Get columns that have objects
        for k, v in agg.items():
            if v == "":
                agg[k] = "first"

        data = data.resample(time_frame).agg(agg)
        data['vwap'] = data['typical_volume'] / data['volume']

        data.drop(columns=["typical_price", "typical_volume"], inplace=True)
        data.dropna(inplace=True)
        if inplace:
            self.data = data
            return None
        return BarDataWrapper(data)

    def resample_many(self, time_frames: list[str]) -> dict[str, "BarDataWrapper"]:
        return {time_frame: self.resample(time_frame, inplace=False) for time_frame in time_frames}

    def scale_one_scaler(self, scaler: Scaler, inplace=False) -> Optional["BarDataWrapper"]:
        data = self.data.copy()
        non_object_cols = data.select_dtypes(exclude=["object"]).columns
        data[non_object_cols] = scaler.fit_transform(data[non_object_cols])
        if inplace:
            self.data = data
            self.single_scaler = scaler
            return None
        new = BarDataWrapper(data)
        new.single_scaler = scaler
        return new

    def reverse_scale_one_scaler(self, inplace=False) -> Optional["BarDataWrapper"]:
        data = self.data.copy()
        if not self.single_scaler:
            raise ValueError("No scaler for this dataset!")
        non_object_cols = data.select_dtypes(exclude=["object"]).columns
        data[non_object_cols] = self.single_scaler.inverse_transform(data[non_object_cols])
        if inplace:
            self.data = data
            self.single_scaler = None
            return None
        return BarDataWrapper(data)

    def scale(self, scaler: Scaler, inplace=False) -> Optional["BarDataWrapper"]:
        data = self.data.copy()
        scalers: dict[str, Scaler] = {}
        # Get non object cols
        non_object_cols = data.select_dtypes(exclude=["object"]).columns
        for col in non_object_cols:
            s = copy.deepcopy(scaler)
            data[col] = s.fit_transform(data[col].values.reshape(-1, 1))
            scalers[col] = s
        if inplace:
            self.data = data
            self.scaler = scalers
            return None
        return BarDataWrapper(data, scaler=scalers)

    def reverse_scale(self, inplace=False) -> Optional["BarDataWrapper"]:
        data = self.data.copy()
        if not self.scaler:
            raise ValueError("No scaler for this dataset!")
            # Get non object cols
        non_object_cols = data.select_dtypes(exclude=["object"]).columns
        for col, s in self.scaler.items():
            data[col] = s.inverse_transform(pd.DataFrame(data[col]))
        if inplace:
            self.data = data
            self.scaler = None
            return None
        return BarDataWrapper(data)

    def get_degree_of_change(self, timeframes: list[str], use_high_low=False) -> pd.DataFrame:
        out = []
        cols = ["open", "close"]
        if use_high_low:
            cols = ["low", "high"]

        for timeframe in timeframes:
            odict = {}
            data = self.resample(timeframe, inplace=False).data
            odict["timeframe"] = timeframe
            odict["min"] = (data[cols[0]] - data[cols[1]]).abs().min()
            odict["max"] = (data[cols[0]] - data[cols[1]]).abs().max()
            odict["mean"] = (data[cols[0]] - data[cols[1]]).abs().mean()
            odict["mean_pec"] = (
                    (((data[cols[0]] - data[cols[1]]) / ((data[cols[1]] + data[cols[0]]) / 2))).abs() * 100).mean()
            odict["std"] = (data[cols[0]] - data[cols[1]]).abs().std()
            odict["median"] = (data[cols[0]] - data[cols[1]]).abs().median()
            out.append(odict)
        return pd.DataFrame(out).set_index("timeframe")

    def add_score(self, dest_column: str, score_column: str, score_df: pd.DataFrame, empty_fill: int = 0,
                  inplace=False) -> Optional["BarDataWrapper"]:
        data = self.data.copy()
        data = data.join(score_df[score_column], how="left", on="timestamp")
        data.rename(columns={score_column: dest_column}, inplace=True)
        data[dest_column] = data[dest_column].fillna(empty_fill)
        if inplace:
            self.data = data
            return None
        return BarDataWrapper(data)

    def add_signal(self, generator: SignalGenerator, inplace=False) -> Optional["BarDataWrapper"]:
        data = self.data.copy()
        data = generator.generate(data)
        if inplace:
            self.data = data
            return None
        return BarDataWrapper(data)

    def add_strategy_simulator(self, simulator: StrategySimulator, inplace=False) -> Optional["BarDataWrapper"]:
        data = self.data.copy()
        data = simulator.simulate(data)
        if inplace:
            self.data = data
            self.simulator = simulator
            return None
        new = BarDataWrapper(data)
        new.simulator = simulator
        return new
