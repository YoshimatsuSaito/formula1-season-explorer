import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from modules.inmemory_db import InmemoryDB


@st.cache_resource(ttl=60 * 10)
def create_driver_standings_plot(
    _db: InmemoryDB,
    season: int,
    df_calendar: pd.DataFrame,
    num_for_projection: int,
) -> Figure:
    """Create drivers point plot"""

    query = f"""
        SELECT 
            r.driver,
            r.driver_abbreviation,
            r.grandprix,
            r.round,
            COALESCE(r.points, 0) + COALESCE(s.points, 0) AS points
        FROM 
            race_result AS r
        LEFT JOIN
            sprint AS s
        ON
            r.driver = s.driver
            AND
            r.grandprix = s.grandprix
            AND
            s.season = {season}
        WHERE 
            r.season = {season}
        ORDER BY
            r.round
    """

    df = _db.execute_query(query)

    # Create simple future projection
    recent_rounds = df["round"].unique()[-num_for_projection:]
    if num_for_projection > 0:
        df_recent_result = df[df["round"].isin(recent_rounds)]
        df_recent_avg_point = (
            df_recent_result.groupby(["driver", "driver_abbreviation"])["points"]
            .mean()
            .reset_index()
        )
        df_future_race = df_calendar.loc[
            ~df_calendar["round"].isin(df["round"]), ["round", "grandprix"]
        ]
        df_future_projection = pd.merge(
            df_future_race, df_recent_avg_point, how="cross"
        )
        df = pd.concat([df, df_future_projection])

    df["total_point_at_each_round"] = df.groupby(["driver"])["points"].cumsum()
    df["total_point_at_each_round"] = df["total_point_at_each_round"].round(0)
    df["total_point_at_each_round"] = df["total_point_at_each_round"].astype(int)

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 5 * 2))
    df_driver_rank_sort = df.loc[
        df["round"] == df["round"].max(), ["driver", "total_point_at_each_round"]
    ]
    df_driver_rank_sort.sort_values(
        by="total_point_at_each_round", inplace=True, ascending=False
    )

    df_driver_top10 = df_driver_rank_sort.iloc[:10]
    df_driver_under10 = df_driver_rank_sort.iloc[10:]

    for driver in df_driver_top10["driver"].unique():
        df_driver = df.loc[df["driver"] == driver]
        df_driver_past = df_driver.loc[df_driver["round"] <= recent_rounds[-1]]
        df_driver_projection = df_driver.loc[df_driver["round"] > recent_rounds[-1]]
        sns.lineplot(
            data=df_driver_past,
            x="round",
            y="total_point_at_each_round",
            marker="o",
            ax=ax[0],
            linestyle="-",
            color="gray",
        )
        if not df_driver_projection.empty:
            sns.lineplot(
                data=df_driver_projection,
                x="round",
                y="total_point_at_each_round",
                marker="o",
                ax=ax[0],
                linestyle="-",
                color="red",
            )

        ax[0].annotate(
            f"{df_driver['driver_abbreviation'].iloc[-1]} ({df_driver['total_point_at_each_round'].max()})",
            (
                df_driver["round"].iloc[-1],
                df_driver["total_point_at_each_round"].iloc[-1],
            ),
            textcoords="offset points",
            xytext=(15, 0),
            ha="center",
        )
    ax[0].set_xticks(df_calendar["round"])
    ax[0].set_ylabel("Points")
    ax[0].set_xlabel("Round")
    ax[0].set_xlim(0, df_calendar["round"].max() + 1)
    ax[0].set_title("Top10 drivers")

    for driver in df_driver_under10["driver"].unique():
        df_driver = df.loc[df["driver"] == driver]
        df_driver_past = df_driver.loc[df_driver["round"] <= recent_rounds[-1]]
        df_driver_projection = df_driver.loc[df_driver["round"] > recent_rounds[-1]]
        sns.lineplot(
            data=df_driver_past,
            x="round",
            y="total_point_at_each_round",
            marker="o",
            ax=ax[1],
            linestyle="-",
            color="gray",
        )
        if not df_driver_projection.empty:
            sns.lineplot(
                data=df_driver_projection,
                x="round",
                y="total_point_at_each_round",
                marker="o",
                ax=ax[1],
                linestyle="-",
                color="red",
            )

        ax[1].annotate(
            f"{df_driver['driver_abbreviation'].iloc[-1]} ({df_driver['total_point_at_each_round'].max():.1f})",
            (
                df_driver["round"].iloc[-1],
                df_driver["total_point_at_each_round"].iloc[-1],
            ),
            textcoords="offset points",
            xytext=(20, 0),
            ha="center",
        )
    ax[1].set_xticks(df_calendar["round"])
    ax[1].set_ylabel("Points")
    ax[1].set_xlabel("Round")
    ax[1].set_xlim(0, df_calendar["round"].max() + 1)
    ax[1].set_title("Remaining drivers")

    plt.tight_layout()
    return fig
