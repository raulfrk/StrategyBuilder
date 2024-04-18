from datetime import datetime

import pandas_ta as ta
from sklearn.pipeline import Pipeline

from otpclient.client.enums import AssetClassEnum
from strategybuilder.data.dataframe import Passthrough, DfOp, AddColDiff, CombineFirst
from strategybuilder.data.fetcher import BarDataFetch
from strategybuilder.data.filter import FilterTime
from strategybuilder.data.gap import BarGapExtractor, BarFillGaps
from strategybuilder.data.indicator import Diff, Ratio, Indicator
from strategybuilder.data.parallel import Union
from strategybuilder.data.sentiment import ApplySentimentScore, AddSentimentDecay
from strategybuilder.data.utils.gaps import LinearGapFiller

ASSET_CLASS = AssetClassEnum.STOCK

CANDLE_DATA_OBSERVABLE_COLUMNS = ["open_1_min_diff", "close_1_min_diff", "high_1_min_diff", "low_1_min_diff",
                                  "close_open_ratio", "high_low_ratio", "close_vwap_diff", "9_min_ema_close_diff",
                                  "9_min_rsi", "bolinger_diff", "bolinger_close_upper_diff",
                                  "bolinger_close_lower_diff", "psar_diff_close", "reversal", "21_min_ema_close_diff",
                                  "50_min_ema_close_diff"]

CANDLE_DATA_OBSERVABLE_WITH_SENTIMENT_COLUMNS = CANDLE_DATA_OBSERVABLE_COLUMNS + ["sentiment_score"]

def generate_candle_data_pipeline(
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        with_sentiment: bool = False,
        with_sentiment_decay: bool = True,
        sentiment_decay: float = 0.60,
        sentiment_decay_cutoff_timeframe: str = "1d",
        news_sentiment_pipeline: Pipeline = None,
        asset_class: AssetClassEnum = ASSET_CLASS,
):
    """
    Generate a pipeline that fetches candle data with default indicators.

    :param symbol: The symbol of the stock to fetch candles for.
    :param start_time: The start time of the candles.
    :param end_time: The end time of the candles.
    :param with_sentiment: Whether to include sentiment in the pipeline.
    :param with_sentiment_decay: Whether to include sentiment decay in the pipeline.
    :param sentiment_decay: The decay to use for sentiment.
    :param sentiment_decay_cutoff_timeframe: The timeframe to use for sentiment decay (e.g. "1d" will cut the decay off
    at eod).
    :param news_sentiment_pipeline: The news sentiment pipeline to use for sentiment.
    (Required if with_sentiment is True)
    :param asset_class: The asset class to use for the pipeline.
    :return: The pipeline.
    """
    if with_sentiment and news_sentiment_pipeline is None:
        raise ValueError("News sentiment sub-pipeline is required when sentiment is enabled.")

    sentiment_sub_ops = [
        ("sentiment_branching", Union([
            ("data_passthrough", Passthrough()),
            ("get_sentiment_score", news_sentiment_pipeline)
        ]
        )),
        ("apply_sentiment_score", ApplySentimentScore()),

    ]
    if with_sentiment_decay:
        sentiment_sub_ops.append(("add_sentiment_decay", AddSentimentDecay(decay=sentiment_decay,
                                                                           timeframe=sentiment_decay_cutoff_timeframe)))

    if not with_sentiment:
        sentiment_sub_ops = [("data_passthrough", Passthrough())]

    bar_data_pipeline = Pipeline(
        [
            ("data_fetch", BarDataFetch(
                asset_class, symbol, start_time=start_time, end_time=end_time
            )),
            ("filter_trading_hours", FilterTime(on_week_days=[x for x in range(6)], within_hours=[(15, 21)])),
            ("split_gaps", Union(
                [
                    ("data_passthrough", Passthrough()),
                    ("gap_getter", BarGapExtractor(within_hours=(15, 21)))

                ]
            )),
            ("gap_filler", BarFillGaps(LinearGapFiller())),
            ("sort_index", DfOp(lambda x: x.sort_index())),

            # Sentiment
            *sentiment_sub_ops,

            ("open_prev_diff", Diff("open", "open_1_min_diff", pct=True)),  # INDICATOR
            ("close_prev_diff", Diff("close", "close_1_min_diff", pct=True)),  # INDICATOR
            ("high_prev_diff", Diff("high", "high_1_min_diff", pct=True)),  # INDICATOR
            ("low_prev_diff", Diff("low", "low_1_min_diff", pct=True)),  # INDICATOR
            ("close_open_ratio", Ratio("close", "open", new_col="close_open_ratio")),  # INDICATOR
            ("high_low_ratio", Ratio("close", "open", new_col="high_low_ratio")),  # INDICATOR
            ("close_vwap_diff", AddColDiff("close", "vwap", new_col="close_vwap_diff")),  # INDICATOR
            ("9_min_ema", Indicator(ta.ema, col_names="9_min_ema", length=10)),
            ("21_min_ema", Indicator(ta.ema, col_names="21_min_ema", length=21)),
            ("50_min_ema", Indicator(ta.ema, col_names="50_min_ema", length=50)),

            ("9_min_ema_close_diff", AddColDiff("close", "9_min_ema", new_col="9_min_ema_close_diff")),  # INDICATOR
            ("21_min_ema_close_diff", AddColDiff("close", "21_min_ema", new_col="21_min_ema_close_diff")),  # INDICATOR
            ("50_min_ema_close_diff", AddColDiff("close", "50_min_ema", new_col="50_min_ema_close_diff")),  # INDICATOR

            ("9_min_rsi", Indicator(ta.rsi, length=9, col_names="9_min_rsi")),
            ("9_min_rsi_scaled", DfOp(lambda x: x.assign(**{"9_min_rsi": 2 * ((x["9_min_rsi"] - 0) / 100) - 1}))),
            # INDICATOR

            ("bolinger_bands",
             Indicator(ta.bbands, col_names=["lower", "mid", "upper", "bandwidth", "perc"], length=20, std=2)),
            ("bolinger_diff", AddColDiff("upper", "lower", new_col="bolinger_diff")),  # INDICATOR
            ("bolinger_upper_close_diff", AddColDiff("close", "upper", new_col="bolinger_close_upper_diff")),
            # INDICATOR
            ("bolinger_lower_close_diff", AddColDiff("close", "lower", new_col="bolinger_close_lower_diff")),
            # INDICATOR

            ("psar", Indicator(ta.psar, col_names=["short", "long", "af", "reversal"], af0=0.01, af=0.02, max_af=0.05)),
            ("psar_long_short_combined", CombineFirst("long", "short", new_col="psar_long_short_combined")),
            ("psar_diff_close", AddColDiff("close", "psar_long_short_combined", new_col="psar_diff_close")),
            # INDICATOR
            ("drop_psar_original_cols", DfOp(lambda x: x.drop(columns=["short", "long"]))),

            ("drop_na", DfOp(lambda x: x.dropna()))
        ]

    )

    return bar_data_pipeline
