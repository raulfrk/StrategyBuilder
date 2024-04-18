import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DfOp(BaseEstimator, TransformerMixin):
    def __init__(self, operation):
        self.operation = operation

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.operation(X)

class AddColDiff(BaseEstimator, TransformerMixin):
    def __init__(self, col1, col2, new_col: str | None = None):
        self.col1 = col1
        self.col2 = col2
        self.new_col = new_col or f"{col1}_{col2}_diff"

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X[self.new_col] = X[self.col1] - X[self.col2]
        return X

class CombineFirst(BaseEstimator, TransformerMixin):
    def __init__(self, col1, col2, new_col: str | None = None):
        self.col1 = col1
        self.col2 = col2
        self.new_col = new_col or f"{col1}_{col2}_combined"

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Get first column index
        first_col_idx = X.columns.get_loc(self.col2)
        X.insert(first_col_idx, self.new_col, X[self.col1].combine_first(X[self.col2]))
        return X

class Passthrough(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X
