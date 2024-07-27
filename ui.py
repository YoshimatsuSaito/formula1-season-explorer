import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from modules.inmemory_db import InmemoryDB
from modules.utils import get_latest_grandprix


def get_round_grandprix_from_sidebar(df_calendar: pd.DataFrame) -> tuple[int, str]:
    """Identify the target round to show"""
    ser_default = get_latest_grandprix(df_calendar=df_calendar)
    dict_options = {
        round_num: f"Rd. {round_num}: {grandprix}"
        for round_num, grandprix in zip(
            df_calendar["round"].tolist(), df_calendar["grandprix"].tolist()
        )
    }
    round_to_show = st.sidebar.selectbox(
        "Choose Grand Prix",
        options=list(dict_options.keys()),
        format_func=lambda x: dict_options[x],
        index=int(ser_default["round"]) - 1,
    )
    grandprix_to_show = df_calendar.loc[
        df_calendar["round"] == round_to_show, "grandprix"
    ].iloc[0]
    return round_to_show, grandprix_to_show


def plot_calendar(df_calendar: pd.DataFrame) -> None:
    """Create grandprix race date calendar"""
    with st.expander("Calendar"):
        for _, row in df_calendar.iterrows():
            str_expr = (
                f"Rd. {row['round']}: {row['grandprix']} ({row['month']}/{row['date']})"
            )
            st.caption(str_expr)


@st.cache_resource(ttl=60 * 10)
def plot_circuit(array_geo: np.array) -> Figure:
    longitude = array_geo[:, 0]
    latitude = array_geo[:, 1]
    fig, ax = plt.subplots()
    ax.plot(longitude, latitude, linestyle="-", color="r")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    return fig


@st.cache_resource(ttl=60 * 10)
def create_winner_prediction_plot(df_winner_prediction_result: pd.DataFrame) -> Figure:
    """Create winner prediction figure
    NOTE: This is adhoc function for design"""
    df_winner_prediction_result.rename(
        columns={"driver": "Driver", "y_pred": "Winning Probability"}, inplace=True
    )
    df_winner_prediction_result["Driver"] = df_winner_prediction_result["Driver"].apply(
        lambda x: x.split("_")[1].upper() if len(x.split("_")) > 1 else x.upper()
    )
    fig, ax = plt.subplots()
    sns.barplot(
        y="Driver",
        x="Winning Probability",
        data=df_winner_prediction_result,
        orient="h",
        ax=ax,
        color="skyblue",
    )
    ax.set_ylabel("")
    ax.set_xlim(0, 1)
    return fig


@st.cache_resource(ttl=60 * 10)
def create_pole_position_time_plot(_db: InmemoryDB, grandprix: str) -> Figure:
    """Create pole position time"""
    query = f"""
    SELECT season, q3_sec 
    FROM qualifying 
    WHERE grandprix = '{grandprix}' 
    AND position = 1
    """
    df = _db.execute_query(query)

    fig, ax = plt.subplots()
    sns.lineplot(data=df, x="season", y="q3_sec", marker="o", color="skyblue", ax=ax, linestyle="-")
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("sec", fontsize=14)
    ax.set_xticks(df["season"])
    ax.set_xticklabels(df["season"], rotation=45)

    # Annotate each point with the time value
    for i, row in df.iterrows():
        ax.annotate(
            f"{row['q3_sec']:.2f}",
            (row["season"], row["q3_sec"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    plt.tight_layout()

    return fig


@st.cache_resource(ttl=60 * 10)
def create_q1_threshold(_db: InmemoryDB, grandprix: str) -> Figure:
    """Create Q1 -> Q2 thredhold"""
    query = f"""
    SELECT season, q1_sec 
    FROM qualifying 
    WHERE grandprix = '{grandprix}' 
    AND position = 15
    """
    df = _db.execute_query(query)

    fig, ax = plt.subplots()
    sns.lineplot(data=df, x="season", y="q1_sec", marker="o", color="skyblue", ax=ax, linestyle="-")
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("sec", fontsize=14)
    ax.set_xticks(df["season"])
    ax.set_xticklabels(df["season"], rotation=45)

    # Annotate each point with the time value
    for i, row in df.iterrows():
        ax.annotate(
            f"{row['q1_sec']:.2f}",
            (row["season"], row["q1_sec"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    plt.tight_layout()

    return fig



@st.cache_resource(ttl=60 * 10)
def create_q2_threshold(_db: InmemoryDB, grandprix: str) -> Figure:
    """Create Q2 -> Q3 threshold"""
    query = f"""
    SELECT season, q2_sec 
    FROM qualifying 
    WHERE grandprix = '{grandprix}' 
    AND position = 10
    """
    df = _db.execute_query(query)

    fig, ax = plt.subplots()
    sns.lineplot(data=df, x="season", y="q2_sec", marker="o", color="skyblue", ax=ax, linestyle="-")
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("sec", fontsize=14)
    ax.set_xticks(df["season"])
    ax.set_xticklabels(df["season"], rotation=45)

    # Annotate each point with the time value
    for i, row in df.iterrows():
        ax.annotate(
            f"{row['q2_sec']:.2f}",
            (row["season"], row["q2_sec"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    plt.tight_layout()

    return fig



@st.cache_resource(ttl=60 * 10)
def create_qualify_diff_1st_2nd(_db: InmemoryDB, grandprix: str) -> Figure:
    """Create time difference between pole sitter and 2nd"""
    query = f"""
        SELECT
            q_1st.season,
            q_1st.q3_sec AS q3_sec_1st,
            q_2nd.q3_sec AS q3_sec_2nd,
            q_2nd.q3_sec - q_1st.q3_sec AS diff_1st_2nd
        FROM
            qualifying q_1st
        JOIN
            qualifying q_2nd ON q_1st.season = q_2nd.season
        WHERE
            q_1st.grandprix = '{grandprix}'
            AND q_2nd.grandprix = '{grandprix}'
            AND q_1st.position = 1
            AND q_2nd.position = 2
    """
    df = _db.execute_query(query)

    fig, ax = plt.subplots()
    sns.lineplot(data=df, x="season", y="diff_1st_2nd", color="skyblue", ax=ax, marker="o")
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("sec", fontsize=14)
    ax.set_xticks(df["season"])
    ax.set_xticklabels(df["season"], rotation=45)

    # Annotate each point with the time value
    for i, row in df.iterrows():
        ax.annotate(
            f"{row['diff_1st_2nd']:.2f}",
            (row["season"], row["diff_1st_2nd"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
    plt.tight_layout()

    return fig


@st.cache_resource(ttl=60 * 10)
def create_pit_stop_count_plot(_db: InmemoryDB, grandprix: str) -> Figure:
    """Create pit stop count proportion"""
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
            season >= 2022
        GROUP BY
            driver, season
    """
    df = _db.execute_query(query)

    # Distribution of pit stop count proportion
    df_pit_stop_count_distribution = (
        df["pit_stop_count"].value_counts(normalize=True).reset_index()
    )
    df_pit_stop_count_distribution.columns = ["stop_count", "proportion"]

    fig, ax = plt.subplots()
    sns.barplot(
        x="stop_count",
        y="proportion",
        data=df_pit_stop_count_distribution,
        color="skyblue",
        ax=ax,
    )
    ax.set_ylim([0, 1])
    ax.set_xlabel("Pit Stop Count")
    ax.set_ylabel("Proportion")
    ax.set_title("2022 season ~")
    plt.tight_layout()

    return fig


@st.cache_resource(ttl=60 * 10)
def create_first_pit_stop_timing_plot(_db: InmemoryDB, grandprix: str) -> Figure:
    """Create first pit stop timing"""
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
            season >= 2022
    """
    df = _db.execute_query(query)

    fig, ax = plt.subplots()
    sns.histplot(
        df.drop_duplicates(subset=["driver", "season"], keep="first")["lap"],
        bins=20,
        color="skyblue",
        ax=ax,
    )
    ax.set_xlabel("Number of laps")
    ax.set_ylabel("Frequency")
    ax.set_title("2022 season ~")
    plt.tight_layout()

    return fig


@st.cache_resource(ttl=60 * 10)
def create_probability_from_each_grid_plots(
    _db: InmemoryDB, grandprix: str
) -> tuple[Figure, Figure, Figure]:
    """Create probability from each grid"""
    query = f"""
        SELECT
            grid.season,
            grid.driver,
            grid.position AS grid_position,
            race_result.position
        FROM
            grid
        JOIN
            race_result
        ON
            grid.season = race_result.season
            AND grid.driver = race_result.driver
        WHERE
            grid.grandprix = '{grandprix}'
            AND race_result.grandprix = '{grandprix}'
        ORDER BY
            grid.season,
            grid.position;
    """
    df = _db.execute_query(query)

    # Winning probability
    df_win_counts_from_each_grid = (
        df[df["position"] == 1].groupby("grid_position").size()
    )
    df_total_counts_of_each_grid = df.groupby("grid_position").size()
    df_win_rates_from_each_grid = (
        df_win_counts_from_each_grid / df_total_counts_of_each_grid
    ).reset_index()
    df_win_rates_from_each_grid.columns = ["grid_position", "win_rate"]

    fig1, ax1 = plt.subplots()
    sns.barplot(
        x="grid_position",
        y="win_rate",
        data=df_win_rates_from_each_grid,
        color="skyblue",
        ax=ax1,
    )
    ax1.set_ylim([0, 1])
    ax1.set_xlabel("Grid position")
    ax1.set_ylabel("Proportion")
    ax1.set_title("Winning rate")
    plt.tight_layout()

    # Podium probability
    df_podium_counts_from_each_grid = (
        df[df["position"] <= 3].groupby("grid_position").size()
    )
    df_podium_rates_from_each_grid = (
        df_podium_counts_from_each_grid / df_total_counts_of_each_grid
    ).reset_index()
    df_podium_rates_from_each_grid.columns = ["grid_position", "podium_rate"]

    fig2, ax2 = plt.subplots()
    sns.barplot(
        x="grid_position",
        y="podium_rate",
        data=df_podium_rates_from_each_grid,
        color="skyblue",
        ax=ax2,
    )
    ax2.set_ylim([0, 1])
    ax2.set_xlabel("Grid position")
    ax2.set_ylabel("Proportion")
    ax2.set_title("Podium rate")
    plt.tight_layout()

    # Point probability
    df_point_counts_from_each_grid = (
        df[df["position"] <= 10].groupby("grid_position").size()
    )
    df_point_rates_from_each_grid = (
        df_point_counts_from_each_grid / df_total_counts_of_each_grid
    ).reset_index()
    df_point_rates_from_each_grid.columns = ["grid_position", "point_rate"]

    fig3, ax3 = plt.subplots()
    sns.barplot(
        x="grid_position",
        y="point_rate",
        data=df_point_rates_from_each_grid,
        color="skyblue",
        ax=ax3,
    )
    ax3.set_ylim([0, 1])
    ax3.set_xlabel("Grid position")
    ax3.set_ylabel("Proportion")
    ax3.set_title("Point positions rate")
    plt.tight_layout()

    return fig1, fig2, fig3


def show_user_search_result(
    db: InmemoryDB, list_result_type: list[str], grandprix: str
) -> None:
    """Show past result which user want to see"""

    list_result_type = [x if x != "qualify" else f"{x}ing" for x in list_result_type]

    with st.expander("Search past results"):
        # Users choose season
        df_season = db.execute_query(
            "SELECT DISTINCT season FROM race_result ORDER BY season"
        )
        season = st.selectbox(
            label="Season",
            options=df_season["season"],
        )
        # Show the result
        dict_df = dict()
        for result_type in list_result_type:
            df = db.execute_query(
                f"SELECT * FROM {result_type} WHERE season = {season} AND grandprix = '{grandprix}'"
            )
            if "position" in df.columns:
                df.sort_values(by="position", inplace=True)
            dict_df[result_type] = df

        for result_type in list_result_type:
            st.markdown(f"##### {result_type}")

            st.dataframe(dict_df[result_type], hide_index=True)
