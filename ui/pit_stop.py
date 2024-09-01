import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from modules.inmemory_db import InmemoryDB


@st.cache_resource(ttl=60 * 10)
def create_pit_stop_count_plot(_db: InmemoryDB, grandprix: str) -> Figure:
    """Create Pit Stop Count Proportion Plot"""
    query = f"""
        SELECT
            driver,
            season,
            COUNT(*) AS pit_stop_count
        FROM
            pit_stop
        WHERE
            grandprix = '{grandprix}'
            AND
            season >= 2011
        GROUP BY
            driver, season
    """
    df = _db.execute_query(query)

    df_pit_stop_percentage = (
        df.groupby(["season", "pit_stop_count"])
        .size()
        .reset_index(name="count")
        .pivot(index="season", columns="pit_stop_count", values="count")
        .fillna(0)
    )
    df_pit_stop_percentage = (
        df_pit_stop_percentage.div(df_pit_stop_percentage.sum(axis=1), axis=0) * 100
    )

    fig, ax = plt.subplots()
    bottom = pd.Series(
        [0] * len(df_pit_stop_percentage), index=df_pit_stop_percentage.index
    )
    palette = sns.color_palette("coolwarm", len(df_pit_stop_percentage.columns))
    for i, column in enumerate(df_pit_stop_percentage.columns):
        sns.barplot(
            x=df_pit_stop_percentage.index,
            y=df_pit_stop_percentage[column],
            color=palette[i],
            label=f"{column} stop",
            bottom=bottom,
            ax=ax,
        )
        bottom += df_pit_stop_percentage[column]
    ax.set_ylabel("Pit Stop Strategy Distribution (%)", fontsize=14)
    ax.set_xlabel("Year", fontsize=14)

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], title="", loc="upper left")
    plt.tight_layout()

    return fig


@st.cache_resource(ttl=60 * 10)
def create_first_pit_stop_timing_plot(_db: InmemoryDB, grandprix: str) -> Figure:
    """Create First Pit Stop Timing Plot"""
    query = f"""
        SELECT
            driver,
            season,
            lap
        FROM
            pit_stop
        WHERE
            grandprix = '{grandprix}'
            AND
            season >= 2011
    """
    df = _db.execute_query(query)

    fig, ax = plt.subplots()
    sns.histplot(
        df.drop_duplicates(subset=["driver", "season"], keep="first")["lap"],
        bins=20,
        color="skyblue",
        ax=ax,
    )
    ax.set_xlabel("Lap Number", fontsize=14)
    ax.set_ylabel("Number of First Pit Stops", fontsize=14)
    plt.tight_layout()

    return fig
