from dataclasses import dataclass

import pandas as pd

from .load_csv_data import load_csv_data


@dataclass
class ModelInputData:
    """Data for model input"""

    df: pd.DataFrame
    list_col_X: list[str] | None = None
    col_y: str | None = None

    def __post_init__(self):
        if self.list_col_X is None:
            self.list_col_X = [
                "recent_position",
                "season_position",
                "season_q_relative_performance",
                "prev_position",
            ]
        if self.col_y is None:
            self.col_y = "position"



def make_datamart(bucket_name: str) -> pd.DataFrame:
    """Concatenate dfs for feature engineering"""

    df_race_result = load_csv_data(bucket_name=bucket_name, key="data/race_result.csv")
    df_qualify = load_csv_data(bucket_name=bucket_name, key="data/qualify.csv")

    # merge dfs into one
    df = pd.merge(
        df_race_result.loc[
            :, 
            [
                "season",
                "round",
                "grandprix",
                "driver", 
                "position", 
            ]
        ], 
        df_qualify.loc[
            :,
            [
                "season",
                "round",
                "grandprix",
                "driver",
                "q1_sec",
                "q2_sec",
                "q3_sec",
            ]
        ], 
        on=[
            "season", 
            "round",
            "grandprix",
            "driver"
        ], 
        how="left", 
        suffixes=["_race", "_qualify"]
    )
    return df


def add_future_race_row(df_datamart: pd.DataFrame, df_calendar: pd.DataFrame) -> pd.DataFrame:
    """Add row of future races for prediction"""
    # Consider latest season only
    latest_season = df_datamart["season"].max()
    # Extract all drivers who drove in this season
    list_latest_driver = df_datamart.loc[df_datamart["season"]==latest_season, "driver"].unique().tolist()
    # Extract latest past round
    latest_round = df_datamart.loc[df_datamart["season"]==latest_season, "round"].max()

    # Add rows of future races
    df_future = df_calendar.loc[df_calendar["round"]>latest_round]
    if not df_future.empty:
        for _, row in df_future.iterrows():
            # Create all drivers' row
            df_tmp = pd.DataFrame({"driver": list_latest_driver})
            df_tmp["season"] = latest_season
            df_tmp["round"] = row["round"]
            df_tmp["grandprix"] = row["grandprix"]
            # Concatenate
            df_datamart = pd.concat([df_datamart, df_tmp])
    return df_datamart


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features to datamart for model training and inference"""

    # Make relative performance about qualifying
    df["q1_sec_fastest"] = df.groupby(["season", "round"])["q1_sec"].transform("min")
    df["q2_sec_fastest"] = df.groupby(["season", "round"])["q2_sec"].transform("min")
    df["q3_sec_fastest"] = df.groupby(["season", "round"])["q3_sec"].transform("min")
    df["q1_sec_relative_performance"] = (df["q1_sec"] - df["q1_sec_fastest"]) / df["q1_sec_fastest"]
    df["q2_sec_relative_performance"] = (df["q2_sec"] - df["q2_sec_fastest"]) / df["q2_sec_fastest"]
    df["q3_sec_relative_performance"] = (df["q3_sec"] - df["q3_sec_fastest"]) / df["q3_sec_fastest"]
    df["q_relative_performance"] = (
        df[
            [
                "q1_sec_relative_performance",
                "q2_sec_relative_performance",
                "q3_sec_relative_performance",
            ]
        ].mean(axis=1, skipna=True)
    ) * 100

    # Make prev position data
    df.sort_values(by=["driver", "grandprix", "season"], inplace=True)
    df["prev_position"] = df.groupby(["driver", "grandprix"])["position"].shift(1)

    # Make rolling data
    df.sort_values(by=["driver", "season", "round"], inplace=True)
    df["recent_position"] = df.groupby(["driver", "season"])["position"].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )
    df["season_position"] = df.groupby(["driver", "season"])["position"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
    df["season_q_relative_performance"] = df.groupby(["driver", "season"])[
        "q_relative_performance"
    ].transform(lambda x: x.shift(1).expanding(min_periods=1).mean())

    return df
