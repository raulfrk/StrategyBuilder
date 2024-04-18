import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from strategybuilder.data.utils.sentiment_scorer import SentimentScorer


class FlattenSentiment(BaseEstimator, TransformerMixin):
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        num_sentiments = [len(x["sentiments"]) for _, x in data.iterrows()]

        if any(x > 1 for x in num_sentiments):
            raise ValueError(
                "Data has multiple sentiments for a single news. Cannot flatten. Considering filtering the data.")

        data["sentiments"] = data["sentiments"].apply(lambda x: x.iloc[0]["sentiment"])
        # Fill empty strings with NaN
        data["sentiments"] = data["sentiments"].replace("", float("nan"))
        return data


class ExtractSentimentScore(BaseEstimator, TransformerMixin):
    def __init__(self, timeframe: str, scoring_strategy: SentimentScorer):
        self.timeframe = timeframe
        self.scoring_strategy = scoring_strategy

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        # Round index to timeframe
        data.index = data.index.round(self.timeframe)

        # Replace "positive" with 1, "negative" with -1, "neutral" with 0
        data["sentiment_score"] = data["sentiments"].map({"positive": 1, "negative": -1, "neutral": 0}).infer_objects(
            copy=False)

        # Create new dataframe with same index and sentiment
        df = pd.DataFrame(data["sentiment_score"], index=data.index)
        # Rename index to timeframe
        df.index.name = "timestamp"

        return self.scoring_strategy.score(df)


class ApplySentimentScore(BaseEstimator, TransformerMixin):
    def __init__(self, score_column: str = "sentiment_score", dest_column: str = "sentiment_score", empty_fill=0):
        self.dest_column = dest_column
        self.score_column = score_column
        self.empty_fill = empty_fill

    def transform(self, ops: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:

        if self.score_column in ops[0].columns.values:
            score_df, data = ops
        else:
            data, score_df = ops

        data = data.join(score_df[self.score_column], how="left", on="timestamp")
        data.rename(columns={self.score_column: self.dest_column}, inplace=True)
        data[self.dest_column] = data[self.dest_column].fillna(self.empty_fill)
        return data

class AddSentimentDecay(BaseEstimator, TransformerMixin):
    def __init__(self, decay: float, timeframe: str, score_column: str = "sentiment_score", dest_column: str = "sentiment_score"):
        self.decay = decay
        self.dest_column = dest_column
        self.score_column = score_column
        self.timeframe = timeframe

    def decay_values(self, col: pd.Series):

        decay_in_progress = False
        current_value = 0
        is_negative = False
        for i in range(0, len(col)):
            if col.iloc[i] != 0:
                if not decay_in_progress:
                    is_negative = col.iloc[i] < 0
                    decay_in_progress = True
                    current_value = col.iloc[i]
                    continue
                else:
                    current_value = col.iloc[i] + current_value

            if decay_in_progress and (current_value < 0.01 and not is_negative) or (current_value > -0.01 and is_negative):
                decay_in_progress = False
            if decay_in_progress:
                current_value *= self.decay
                col.iloc[i] = current_value
        return col


    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        # When encoutering a non zero sentiment score, as it moves further away from the current time, the sentiment decays (i.e. becomes closer to 0)
        # Split the data in timeframe
        data["period"] = data.index.round(self.timeframe)
        # Group by period
        grouped = data.groupby("period")
        # For each group, apply decay
        data[self.dest_column] = grouped[self.score_column].transform(self.decay_values)
        return data