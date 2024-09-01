import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from modules.inmemory_db import InmemoryDB


@st.cache_resource(ttl=60 * 10)
def create_fastest_lap_plot(_db: InmemoryDB, grandprix: str) -> Figure:
    """Create Fastest Lap Trajectory"""
    query = f"""
    SELECT season, time_sec 
    FROM fastest_lap 
    WHERE grandprix = '{grandprix}' 
    AND position = 1
    """
    df = _db.execute_query(query)

    fig, ax = plt.subplots()
    sns.lineplot(
        data=df,
        x="season",
        y="time_sec",
        marker="o",
        color="skyblue",
        ax=ax,
        linestyle="-",
    )
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Seconds", fontsize=14)
    ax.set_xticks(df["season"])
    ax.set_xticklabels(df["season"], rotation=45)

    # Annotate each point with the time value
    for i, row in df.iterrows():
        ax.annotate(
            f"{row['time_sec']:.2f}",
            (row["season"], row["time_sec"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    plt.tight_layout()

    return fig


@st.cache_resource(ttl=60 * 10)
def create_fastest_lap_timing_plot(_db: InmemoryDB, grandprix: str) -> Figure:
    """Create Fastest Lap Timing Plot"""
    query = f"""
        SELECT 
            lap
        FROM 
            fastest_lap 
        WHERE 
            grandprix = '{grandprix}' 
            AND
            lap >= 10
    """
    df = _db.execute_query(query)

    fig, ax = plt.subplots()
    sns.histplot(data=df, x="lap", color="skyblue", ax=ax, bins=40)
    ax.set_xlabel("Lap Number", fontsize=14)
    ax.set_ylabel("Number of Fastest Laps Recorded", fontsize=14)

    plt.tight_layout()

    return fig


@st.cache_resource(ttl=60 * 10)
def create_diff_fastest_lap_and_pole_time_plot(
    _db: InmemoryDB, grandprix: str, ser_grandprix_this_season: pd.Series
) -> Figure:
    """Create Difference Between Fastest Lap and Pole Time"""

    query = """
        SELECT
            q.season,
            q.grandprix,
            q.q3_sec AS q3_fastest,
            f.time_sec AS race_fastest,
            ((f.time_sec - q.q3_sec) / q.q3_sec) * 100 AS time_diff_percent
        FROM
            qualifying q
        JOIN
            fastest_lap f 
        ON 
            q.season = f.season 
            AND 
            q.grandprix = f.grandprix
        WHERE
            q.position = 1
            AND 
            f.position = 1
    """
    df = _db.execute_query(query)
    df = df.loc[df["grandprix"].isin(ser_grandprix_this_season)]
    df = pd.DataFrame(
        df.groupby("grandprix")["time_diff_percent"].median()
    ).reset_index()
    df.sort_values(by="time_diff_percent", ascending=False, inplace=True)
    colors = ["red" if gp == grandprix else "skyblue" for gp in df["grandprix"]]
    fig, ax = plt.subplots()
    sns.barplot(
        y="grandprix",
        x="time_diff_percent",
        data=df,
        orient="h",
        ax=ax,
        palette=colors,
    )
    ax.set_xlabel("Time Difference (%): Fastest Lap / Pole Position Time", fontsize=14)
    ax.set_ylabel("Grand Prix", fontsize=14)

    plt.tight_layout()

    return fig
