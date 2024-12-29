from dataclasses import dataclass

import pandas as pd

from .load_csv_data import load_csv_data

PREV_NUM = 5
PREV_NUM_SEASON = 3


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
            ],
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
            ],
        ],
        on=["season", "round", "grandprix", "driver"],
        how="left",
        suffixes=["_race", "_qualify"],
    )
    return df


def add_future_race_row(
    df_datamart: pd.DataFrame, df_calendar: pd.DataFrame
) -> pd.DataFrame:
    """Add row of future races for prediction"""
    # Consider latest season only
    latest_season = df_datamart["season"].max()
    # Extract all drivers who drove in this season
    list_latest_driver = (
        df_datamart.loc[df_datamart["season"] == latest_season, "driver"]
        .unique()
        .tolist()
    )
    # Extract latest past round
    latest_round = df_datamart.loc[
        df_datamart["season"] == latest_season, "round"
    ].max()

    # Add rows of future races
    df_future = df_calendar.loc[df_calendar["round"] > latest_round]
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
    df["q1_sec_relative_performance"] = (
        (df["q1_sec"] - df["q1_sec_fastest"]) / df["q1_sec_fastest"]
    ) * 100
    df["q2_sec_relative_performance"] = (
        (df["q2_sec"] - df["q2_sec_fastest"]) / df["q2_sec_fastest"]
    ) * 100
    df["q3_sec_relative_performance"] = (
        (df["q3_sec"] - df["q3_sec_fastest"]) / df["q3_sec_fastest"]
    ) * 100

    # Make previous position data
    for prev_num in range(1, PREV_NUM + 1):
        df[f"position_prev{prev_num}"] = df.groupby(["season", "driver"])[
            "position"
        ].shift(prev_num)

    # Make previous qualifying data
    for q in range(1, 4):
        for prev_num in range(1, PREV_NUM + 1):
            df[f"q{q}_performance_prev{prev_num}"] = df.groupby(["season", "driver"])[
                f"q{q}_sec_relative_performance"
            ].shift(prev_num)

    # Make previous season position data
    df.sort_values(by=["driver", "grandprix", "season"], inplace=True)
    for prev_num in range(1, PREV_NUM_SEASON + 1):
        df[f"position_prev_season{prev_num}"] = df.groupby(["driver", "grandprix"])[
            "position"
        ].shift(prev_num)

    return df
