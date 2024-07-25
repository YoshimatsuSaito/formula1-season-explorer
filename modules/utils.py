from datetime import datetime
from pathlib import Path

import pandas as pd
import pytz
import yaml

japan_timezone = pytz.timezone("Asia/Tokyo")


def load_config(path: str | Path) -> dict:
    """Load config"""
    with open(path) as f:
        return yaml.safe_load(f)


def get_latest_grandprix(df_calendar: pd.DataFrame) -> pd.Series:
    """Get latest future grandprix"""
    current_date = datetime.now(japan_timezone).date()
    df_calendar["date_combined"] = pd.to_datetime(
        df_calendar.apply(
            lambda row: f"{row['season']}-{row['month']:02d}-{row['date']:02d}", axis=1
        )
    )

    df_future = df_calendar[
        df_calendar["date_combined"] >= pd.to_datetime(current_date)
    ]
    if not df_future.empty:
        ser_next_day = df_future.iloc[0]
    else:
        ser_next_day = df_calendar.iloc[-1]

    return ser_next_day
