from datetime import datetime

import pandas as pd


def filter_time(data: pd.DataFrame, within_hours: list[tuple[int, int]] | None = None,
                within_days: list[tuple[datetime, datetime]] | None = None,
                exclude_days: list[tuple[datetime, datetime]] | None = None,
                on_week_days: list[int] | None = None) -> pd.DataFrame:
    """
    Given a dataframe, filter the data based on the time of the day, the days of the week, and the days of the month.
    """
    if within_days:
        flt = None
        for days in within_days:
            if flt is None:
                flt = ((data.index.date >= days[0].date()) & (data.index.date <= days[1].date()))
            flt = flt | ((data.index.date >= days[0].date()) & (data.index.date <= days[1].date()))
        data = data[flt]
    if exclude_days:
        flt = None
        for days in exclude_days:
            if flt is None:
                flt = ((data.index.date < days[0].date()) | (data.index.date > days[1].date()))
            flt = flt | ((data.index.date < days[0].date()) | (data.index.date > days[1].date()))
        data = data[flt]
    if within_hours:
        flt = None
        for hours in within_hours:
            if flt is None:
                flt = ((data.index.hour >= hours[0]) & (data.index.hour < hours[1]))
            flt = flt | ((data.index.hour >= hours[0]) & (data.index.hour < hours[1]))
        data = data[flt]

    if on_week_days:
        flt = None
        for day in on_week_days:
            if flt is None:
                flt = (data.index.weekday == day)
            flt = flt | (data.index.weekday == day)
        data = data[flt]

    return data
