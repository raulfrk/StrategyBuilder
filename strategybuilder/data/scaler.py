import copy

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class Scale(BaseEstimator, TransformerMixin):
    def __init__(self, scaler):
        self.scaler = scaler

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        data = X
        # Get non object cols
        non_object_cols = data.select_dtypes(exclude=["object"]).columns
        for col in non_object_cols:
            s = copy.deepcopy(self.scaler)
            data[col] = s.fit_transform(data[col].values.reshape(-1, 1))
        return data
