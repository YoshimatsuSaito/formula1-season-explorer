import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt

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
    """Plot grandprix race date calendar"""
    with st.expander("Calendar"):
        for _, row in df_calendar.iterrows():
            str_expr = (
                f"Rd. {row['round']}: {row['grandprix']} ({row['month']}/{row['date']})"
            )
            st.caption(str_expr)


def plot_circuit(array_geo: np.array) -> None:
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
    with st.expander("Circuit layout"):
        st.pyplot(fig)


def plot_winner_prediction(df_winner_prediction_result: pd.DataFrame) -> None:
    """Plot winner prediction figure
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
    st.pyplot(fig)


def plot_pole_position_time(db: InmemoryDB, grandprix: str) -> None:
    """Plot pole position time"""
    query = f"""
    SELECT season, q3_sec 
    FROM qualifying 
    WHERE grandprix = '{grandprix}' 
    AND position = 1
    """
    df = db.execute_query(query)

    plt.figure()
    ax = sns.lineplot(data=df, x="season", y="q3_sec", marker="o", color="skyblue")
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

    st.pyplot(plt)


def plot_pit_stop_count(db: InmemoryDB, grandprix: str) -> None:
    """Plot pit stop count proportion"""
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
    df = db.execute_query(query)

    # Distribution of pit stop count proportion
    df_pit_stop_count_distribution = (
        df["pit_stop_count"].value_counts(normalize=True).reset_index()
    )
    df_pit_stop_count_distribution.columns = ["stop_count", "proportion"]

    plt.figure()
    sns.barplot(
        x="stop_count",
        y="proportion",
        data=df_pit_stop_count_distribution,
        color="skyblue",
    )
    plt.ylim([0, 1])
    plt.xlabel("Pit Stop Count")
    plt.ylabel("Proportion")
    plt.title("2022 season ~")
    plt.tight_layout()

    st.pyplot(plt)


def plot_first_pit_stop_timing(db: InmemoryDB, grandprix: str) -> None:
    """Plot first pit stop timing"""
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
    df = db.execute_query(query)

    plt.figure()
    sns.histplot(
        df.drop_duplicates(subset=["driver", "season"], keep="first")["lap"],
        bins=20,
        color="skyblue",
    )
    plt.xlabel("Number of laps")
    plt.ylabel("Frequency")
    plt.title("2022 season ~")
    plt.tight_layout()

    st.pyplot(plt)


def plot_probability_from_each_grid(db: InmemoryDB, grandprix: str) -> None:
    """Plot probability from each grid"""
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
    df = db.execute_query(query)

    # Winning probability
    df_win_counts_from_each_grid = (
        df[df["position"] == 1].groupby("grid_position").size()
    )
    df_total_counts_of_each_grid = df.groupby("grid_position").size()
    df_win_rates_from_each_grid = (
        df_win_counts_from_each_grid / df_total_counts_of_each_grid
    ).reset_index()
    df_win_rates_from_each_grid.columns = ["grid_position", "win_rate"]

    plt.figure()
    sns.barplot(
        x="grid_position",
        y="win_rate",
        data=df_win_rates_from_each_grid,
        color="skyblue",
    )
    plt.ylim([0, 1])
    plt.xlabel("Grid position")
    plt.ylabel("Proportion")
    plt.title("Winning rate")
    plt.tight_layout()

    st.pyplot(plt)

    # Podium probability
    df_podium_counts_from_each_grid = (
        df[df["position"] <= 3].groupby("grid_position").size()
    )
    df_podium_rates_from_each_grid = (
        df_podium_counts_from_each_grid / df_total_counts_of_each_grid
    ).reset_index()
    df_podium_rates_from_each_grid.columns = ["grid_position", "podium_rate"]

    plt.figure()
    sns.barplot(
        x="grid_position",
        y="podium_rate",
        data=df_podium_rates_from_each_grid,
        color="skyblue",
    )
    plt.ylim([0, 1])
    plt.xlabel("Grid position")
    plt.ylabel("Proportion")
    plt.title("Podium rate")
    plt.tight_layout()

    st.pyplot(plt)

    # Point probability
    df_point_counts_from_each_grid = (
        df[df["position"] <= 10].groupby("grid_position").size()
    )
    df_point_rates_from_each_grid = (
        df_point_counts_from_each_grid / df_total_counts_of_each_grid
    ).reset_index()
    df_point_rates_from_each_grid.columns = ["grid_position", "point_rate"]

    plt.figure()
    sns.barplot(
        x="grid_position",
        y="point_rate",
        data=df_point_rates_from_each_grid,
        color="skyblue",
    )
    plt.ylim([0, 1])
    plt.xlabel("Grid position")
    plt.ylabel("Proportion")
    plt.title("Point positions rate")
    plt.tight_layout()

    st.pyplot(plt)


def show_user_search_result(db: InmemoryDB, list_result_type: list[str]) -> None:
    """Show past result which user want to see"""

    with st.expander("Search past results"):
        # Users choose result type
        result_type = st.selectbox(label="Result type", options=list_result_type)
        df_result_type_from_query = db.execute_query(
            f"SELECT DISTINCT season, round, grandprix FROM {result_type}"
        )
        # Users choose season
        season = st.selectbox(
            label="Season",
            options=df_result_type_from_query["season"].unique().tolist(),
        )
        df_season_from_query = db.execute_query(
            f"SELECT DISTINCT round, grandprix FROM {result_type} WHERE season = {season}"
        )
        df_season_from_query.sort_values(by=["round"], ascending=True, inplace=True)
        # Users choose round (grandprix)
        round_options = df_season_from_query.apply(
            lambda row: f"Round. {row['round']}: {row['grandprix']}", axis=1
        )
        selected_round = st.selectbox(
            "Select a Grand Prix", round_options, index=len(round_options) - 1
        )
        # Show the grandprix result
        selected_rows = df_season_from_query.loc[round_options == selected_round]
        round_num = selected_rows["round"].iloc[0]
        df_target = db.execute_query(
            f"SELECT * FROM {result_type} WHERE season = {season} AND round = {round_num}"
        )
        st.dataframe(df_target, hide_index=True)
