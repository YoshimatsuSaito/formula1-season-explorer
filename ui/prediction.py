import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from modules.inmemory_db import InmemoryDB


@st.cache_resource(ttl=60 * 10)
def create_winner_prediction_plot(df_winner_prediction: pd.DataFrame) -> Figure:
    """Create winner prediction figure
    NOTE: This is adhoc function for design"""
    df = df_winner_prediction.loc[df_winner_prediction["y_pred_winner"] > 0.5]
    df.sort_values(by="y_pred_winner", ascending=False, inplace=True)

    df.rename(
        columns={"driver": "Driver", "y_pred_winner": "Winning Probability"},
        inplace=True,
    )
    df["Driver"] = df["Driver"].apply(
        lambda x: x.split("_")[1].upper() if len(x.split("_")) > 1 else x.upper()
    )
    fig, ax = plt.subplots()
    sns.barplot(
        y="Driver",
        x="Winning Probability",
        data=df,
        orient="h",
        ax=ax,
        color="skyblue",
    )
    ax.set_ylabel("")
    ax.set_xlim(0.5, 1)
    return fig


@st.cache_resource(ttl=60 * 10)
def create_position_prediction_plot(
    df_position_prediction: pd.DataFrame, _db: InmemoryDB
) -> None:
    """Create position prediction figure
    NOTE: This is adhoc function for design"""
    current_round_num = df_position_prediction["round"].max()
    current_season = df_position_prediction["season"].max()
    query = f"""
        SELECT
            driver,
            position AS position_prev
        FROM
            race_result
        WHERE
            season = {current_season}
            AND round = {current_round_num - 1}
    """
    df_prev_race = _db.execute_query(query)

    df = df_position_prediction.copy()
    df = pd.merge(df, df_prev_race, on=["driver"], how="left")
    df.sort_values(by="y_pred_position", ascending=True, inplace=True)
    df["diff_from_prev_round"] = df["position_prev"] - df["y_pred_position"]

    df.rename(
        columns={"driver": "Driver", "y_pred_position": "Predicted Position"},
        inplace=True,
    )
    df["Driver"] = df["Driver"].apply(
        lambda x: x.split("_")[1].upper() if len(x.split("_")) > 1 else x.upper()
    )

    for pos, df_grouped in df.groupby("Predicted Position"):
        dict_suffixes = {1: "st", 2: "nd", 3: "rd"}
        suffix = dict_suffixes.get(pos, "th")
        for _, row in df_grouped.iterrows():
            st.metric(
                label=f"{pos} {suffix}",
                value=row["Driver"],
                delta=f'{row["diff_from_prev_round"]} from prev round',
            )
