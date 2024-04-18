import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class BarResampler(BaseEstimator, TransformerMixin):
    def __init__(self, time_frame: str):
        self.time_frame = time_frame

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        data = X.copy()
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

        data = data.resample(self.time_frame).agg(agg)
        data['vwap'] = data['typical_volume'] / data['volume']

        data.drop(columns=["typical_price", "typical_volume"], inplace=True)
        data.dropna(inplace=True)
        return data
