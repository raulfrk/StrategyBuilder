from asyncio import Protocol

import pandas as pd


class GapFiller(Protocol):
    def fill(self, data: pd.DataFrame, gaps: pd.DataFrame, resample_rule="min", **kwargs) -> pd.DataFrame: ...


class LinearGapFiller:

    @classmethod
    def fill(cls, data: pd.DataFrame, gaps: pd.DataFrame, resample_rule="min", **kwargs) -> pd.DataFrame:
        out_data = data.copy()

        # Resample and interpolate non-object columns
        resampled = data.resample(resample_rule).asfreq()
        non_object_cols = resampled.select_dtypes(exclude=["object"]).columns
        resampled[non_object_cols] = resampled[non_object_cols].interpolate(method="linear", axis=0)

        # Forward fill object columns
        object_cols = resampled.select_dtypes(include=["object"]).columns
        resampled[object_cols] = resampled[object_cols].ffill()

        gap_start_times = gaps["start"]

        gap_end_times = gaps["end"]
        actual_gap_start_times = [resampled.index.get_loc(x) for x in
                                  resampled.index[resampled.index.isin(gap_start_times)]]
        actual_gap_end_times = [resampled.index.get_loc(x) for x in
                                resampled.index[resampled.index.isin(gap_end_times)]]

        # Collect slices for gaps excluding the end times
        gap_slices = [resampled[start:end] for start, end in zip(actual_gap_start_times, actual_gap_end_times)]

        # Concatenate collected slices
        out_data = pd.concat([out_data] + gap_slices, axis=0)

        # Sort the final DataFrame
        out_data.sort_index(inplace=True)
        out_data = out_data[~out_data.index.duplicated(keep='first')]
        return out_data
