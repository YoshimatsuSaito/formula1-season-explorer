import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from modules.inmemory_db import InmemoryDB


@st.cache_data(ttl=60 * 10)
def create_race_top3_table(_db: InmemoryDB, grandprix: str) -> pd.DataFrame:
    """Create Qualify Top3 drivers table"""
    query = f"""
        SELECT
            driver,
            position,
            season,
        FROM
            race_result
        WHERE
            grandprix = '{grandprix}'
            AND
            position IN (1, 2, 3)
    """
    df = _db.execute_query(query)
    df = df.pivot_table(
        index="season", columns="position", values="driver", aggfunc="first"
    )
    df.index.name = "Year"
    df.sort_index(ascending=False, inplace=True)
    return df





@st.cache_resource(ttl=60 * 10)
def create_completion_ratio_plot(
    _db: InmemoryDB, grandprix: str, ser_grandprix_this_season: pd.Series
) -> Figure:
    """Create Finish Ratio time"""

    def completion_ratio(group) -> float:
        """Finish Ratio for each group"""
        total_count = len(group)
        none_count = group.isnull().sum()
        return (total_count - none_count) / total_count

    # Detect Finish Ratio by position columns (None means DNF etc)
    query = """
        SELECT
            season,
            grandprix,
            position
        FROM
            race_result
    """
    df = _db.execute_query(query)
    df = df.loc[df["grandprix"].isin(ser_grandprix_this_season)]
    df = pd.DataFrame(
        df.groupby(["grandprix", "season"])["position"].apply(completion_ratio)
    ).reset_index()
    df = pd.DataFrame(df.groupby("grandprix")["position"].mean()).reset_index()
    df.sort_values(by="position", ascending=False, inplace=True)
    colors = ["red" if gp == grandprix else "skyblue" for gp in df["grandprix"]]
    fig, ax = plt.subplots()
    sns.barplot(
        y="grandprix",
        x="position",
        data=df,
        orient="h",
        ax=ax,
        palette=colors,
    )
    ax.set_xlabel("Finish Ratio")
    ax.set_ylabel("Grand Prix")

    plt.tight_layout()

    return fig


@st.cache_resource(ttl=60 * 10)
def create_race_time_plot(_db: InmemoryDB, grandprix: str) -> Figure:
    """Create race time plot"""
    query = f"""
    SELECT season, time_sec 
    FROM race_result 
    WHERE grandprix = '{grandprix}' 
    AND position = 1
    """
    df = _db.execute_query(query)
    df["time_minutes"] = df["time_sec"] / 60

    fig, ax = plt.subplots()
    sns.lineplot(
        data=df,
        x="season",
        y="time_minutes",
        marker="o",
        color="skyblue",
        ax=ax,
        linestyle="-",
    )
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Minutes", fontsize=14)
    ax.set_xticks(df["season"])
    ax.set_xticklabels(df["season"], rotation=45)

    # Annotate each point with the time value
    for i, row in df.iterrows():
        ax.annotate(
            f"{row['time_minutes']:.2f}",
            (row["season"], row["time_minutes"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    plt.tight_layout()

    return fig

