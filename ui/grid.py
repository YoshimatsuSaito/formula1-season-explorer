import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from modules.inmemory_db import InmemoryDB


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
    df_win_rates_from_each_grid.columns = ["grid_position", "win_percentage"]
    df_win_rates_from_each_grid["win_percentage"] = (
        df_win_rates_from_each_grid["win_percentage"] * 100
    )

    fig1, ax1 = plt.subplots()
    sns.barplot(
        x="grid_position",
        y="win_percentage",
        data=df_win_rates_from_each_grid,
        color="skyblue",
        ax=ax1,
    )
    ax1.set_ylim([0, 100])
    ax1.set_xlabel("Starting Grid Position")
    ax1.set_ylabel("Winning Percentage (%)")
    ax1.set_title("Winning Probability by Starting Grid")
    plt.tight_layout()

    # Podium probability
    df_podium_counts_from_each_grid = (
        df[df["position"] <= 3].groupby("grid_position").size()
    )
    df_podium_rates_from_each_grid = (
        df_podium_counts_from_each_grid / df_total_counts_of_each_grid
    ).reset_index()
    df_podium_rates_from_each_grid.columns = ["grid_position", "podium_percentage"]
    df_podium_rates_from_each_grid["podium_percentage"] = (
        df_podium_rates_from_each_grid["podium_percentage"] * 100
    )

    fig2, ax2 = plt.subplots()
    sns.barplot(
        x="grid_position",
        y="podium_percentage",
        data=df_podium_rates_from_each_grid,
        color="skyblue",
        ax=ax2,
    )
    ax2.set_ylim([0, 100])
    ax2.set_xlabel("Starting Grid position")
    ax2.set_ylabel("Podium Finish Percentage (%)")
    ax2.set_title("Podium Probability by Starting Grid")
    plt.tight_layout()

    # Point probability
    df_point_counts_from_each_grid = (
        df[df["position"] <= 10].groupby("grid_position").size()
    )
    df_point_rates_from_each_grid = (
        df_point_counts_from_each_grid / df_total_counts_of_each_grid
    ).reset_index()
    df_point_rates_from_each_grid.columns = ["grid_position", "point_percentage"]
    df_point_rates_from_each_grid["point_percentage"] = (
        df_point_rates_from_each_grid["point_percentage"] * 100
    )

    fig3, ax3 = plt.subplots()
    sns.barplot(
        x="grid_position",
        y="point_percentage",
        data=df_point_rates_from_each_grid,
        color="skyblue",
        ax=ax3,
    )
    ax3.set_ylim([0, 100])
    ax3.set_xlabel("Starting Grid position")
    ax3.set_ylabel("Points Finish Percentage (%)")
    ax3.set_title("Points Finish Percentage (%)")
    plt.tight_layout()

    return fig1, fig2, fig3
