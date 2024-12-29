import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from modules.inmemory_db import InmemoryDB


@st.cache_resource(ttl=60 * 10)
def create_winner_prediction_plot(df_winner_prediction: pd.DataFrame) -> None:
    """Create winner prediction figure
    NOTE: This is adhoc function for design"""
    df = df_winner_prediction.loc[df_winner_prediction["y_pred_winner"] > 0.5]
    df.sort_values(by="y_pred_winner", ascending=False, inplace=True)

    df["driver"] = df["driver"].apply(
        lambda x: x.split("_")[1].upper() if len(x.split("_")) > 1 else x.upper()
    )

    for _, row in df.iterrows():
        st.metric(
            label=f"{round(row['y_pred_winner'] * 100, 2)} % to win",
            value=row["driver"],
        )


@st.cache_resource(ttl=60 * 10)
def create_position_prediction_plot(
    df_position_prediction: pd.DataFrame, _db: InmemoryDB
) -> None:
    """Create position prediction figure
    NOTE: This is adhoc function for design"""
    grandprix = df_position_prediction["grandprix"].unique()[0]
    current_season = df_position_prediction["season"].max()
    query = f"""
        SELECT
            driver,
            position AS position_prev
        FROM
            race_result
        WHERE
            season = {current_season - 1}
            AND grandprix = '{grandprix}'
    """
    df_prev_race = _db.execute_query(query)

    df = df_position_prediction.copy()
    df = pd.merge(df, df_prev_race, on=["driver"], how="left")
    df["diff_from_prev_season"] = df["position_prev"] - df["y_pred_position"]

    df.sort_values(by="y_pred_position", ascending=True, inplace=True)

    df["driver"] = df["driver"].apply(
        lambda x: x.split("_")[1].upper() if len(x.split("_")) > 1 else x.upper()
    )

    for pos, df_grouped in df.groupby("y_pred_position"):
        dict_suffixes = {1: "st", 2: "nd", 3: "rd"}
        suffix = dict_suffixes.get(pos, "th")
        for _, row in df_grouped.iterrows():
            st.metric(
                label=f"{pos} {suffix}",
                value=row["driver"],
                delta=f'{row["diff_from_prev_season"]} from last year',
            )
