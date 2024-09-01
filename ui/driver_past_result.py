import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from modules.inmemory_db import InmemoryDB

_DRIVER_NUM_CONSTANT = 20


@st.cache_resource(ttl=60 * 10)
def create_driver_past_qualify_result_plot(
    _db: InmemoryDB, grandprix: str, season: int, driver: str
) -> Figure:
    """Create past qualify result plot with teammate result"""

    def create_query(driver: str) -> str:
        """Return query for each driver"""
        return f"""
            SELECT 
                q1.season, 
                q1.driver, 
                q1.position,
                q1.constructor,
                q1.grandprix,
                q2.driver AS teammate,
                q2.position AS position_teammate
            FROM 
                qualifying AS q1
            JOIN
                qualifying AS q2
            ON
                q1.season = q2.season
                AND
                q1.grandprix = q2.grandprix
                AND
                q1.constructor = q2.constructor
            WHERE 
                q1.grandprix = '{grandprix}'
                AND
                q1.driver = '{driver}'
                AND
                q1.driver != q2.driver
        """

    # Get drivers of the season
    df_driver_points = _db.execute_query(
        f"""
            SELECT
                r.driver,
                SUM(COALESCE(r.points, 0)) + SUM(COALESCE(s.points, 0)) as total_points
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
            GROUP BY
                r.driver
            ORDER BY
                total_points DESC
        """
    )

    fig, ax = plt.subplots()

    query = create_query(driver=driver)
    df = _db.execute_query(query)

    # get winning rate
    df["position"].fillna(_DRIVER_NUM_CONSTANT, inplace=True)
    df["position_teammate"].fillna(_DRIVER_NUM_CONSTANT, inplace=True)
    df_tmp = df.loc[df["position"] != df["position_teammate"]].copy()
    winning_percentage = (
        (df_tmp["position"] < df_tmp["position_teammate"]).sum() / len(df_tmp)
    ) * 100

    df_driver = df.loc[:, ["season", "driver", "position", "constructor", "grandprix"]]
    df_driver["driver"] = "Self"
    df_teammate = df.loc[
        :, ["season", "teammate", "position_teammate", "constructor", "grandprix"]
    ]
    df_teammate.rename(
        columns={"teammate": "driver", "position_teammate": "position"},
        inplace=True,
    )
    df_teammate["driver"] = "Teammate"
    df_concat = pd.concat([df_driver, df_teammate]).reset_index()

    sns.lineplot(
        data=df_concat,
        x="season",
        y="position",
        hue="driver",
        marker="o",
        palette={"Self": "skyblue", "Teammate": "gray"},
        ax=ax,
        linestyle="-",
    )
    ax.set_xlabel("")
    ax.set_ylim(0, _DRIVER_NUM_CONSTANT)
    ax.set_ylabel("Qualifying Position", fontsize=14)
    ax.set_xticks(df["season"])
    ax.set_xticklabels(df["season"], rotation=45)
    ax.set_yticks(range(0, len(df_driver_points) + 1))
    ax.set_yticklabels(
        [str(x) if x % 2 != 0 else "" for x in range(0, len(df_driver_points) + 1)]
    )
    ax.set_title(f"{driver} ({winning_percentage:.2f}% win)")
    ax.legend(title="")
    for p in [3, 10]:
        ax.axhline(y=p, color="gray", alpha=0.5, ls="--")

    plt.tight_layout()
    return fig


@st.cache_resource(ttl=60 * 10)
def create_driver_past_race_result_plot(
    _db: InmemoryDB, grandprix: str, season: int, driver: str
) -> Figure:
    """Create past race result plot with teammate result"""

    def create_query(driver: str) -> str:
        """Return query for each driver"""
        return f"""
            SELECT 
                r1.season, 
                r1.driver, 
                r1.position,
                r1.constructor,
                r1.grandprix,
                r2.driver AS teammate,
                r2.position AS position_teammate
            FROM 
                race_result AS r1
            JOIN
                race_result AS r2
            ON
                r1.season = r2.season
                AND
                r1.grandprix = r2.grandprix
                AND
                r1.constructor = r2.constructor
            WHERE 
                r1.grandprix = '{grandprix}'
                AND
                r1.driver = '{driver}'
                AND
                r1.driver != r2.driver
        """

    fig, ax = plt.subplots()
    query = create_query(driver=driver)
    df = _db.execute_query(query)

    # get winning rate
    df["position"].fillna(_DRIVER_NUM_CONSTANT, inplace=True)
    df["position_teammate"].fillna(_DRIVER_NUM_CONSTANT, inplace=True)
    df_tmp = df.loc[df["position"] != df["position_teammate"]].copy()
    winning_percentage = (
        (df_tmp["position"] < df_tmp["position_teammate"]).sum() / len(df_tmp)
    ) * 100

    df_driver = df.loc[:, ["season", "driver", "position", "constructor", "grandprix"]]
    df_driver["driver"] = "Self"
    df_teammate = df.loc[
        :, ["season", "teammate", "position_teammate", "constructor", "grandprix"]
    ]
    df_teammate.rename(
        columns={"teammate": "driver", "position_teammate": "position"},
        inplace=True,
    )
    df_teammate["driver"] = "Teammate"
    df_concat = pd.concat([df_driver, df_teammate]).reset_index()

    sns.lineplot(
        data=df_concat,
        x="season",
        y="position",
        hue="driver",
        marker="o",
        palette={"Self": "skyblue", "Teammate": "gray"},
        ax=ax,
        linestyle="-",
    )
    ax.set_xlabel("")
    ax.set_ylim(0, _DRIVER_NUM_CONSTANT)
    ax.set_ylabel("Finish Position", fontsize=14)
    ax.set_xticks(df["season"])
    ax.set_xticklabels(df["season"], rotation=45)
    ax.set_yticks(range(0, _DRIVER_NUM_CONSTANT + 1))
    ax.set_yticklabels(
        [str(x) if x % 2 != 0 else "" for x in range(0, _DRIVER_NUM_CONSTANT + 1)]
    )
    ax.set_title(f"{driver} ({winning_percentage:.2f}% win)")
    ax.legend(title="")
    for p in [3, 10]:
        ax.axhline(y=p, color="gray", alpha=0.5, ls="--")

    plt.tight_layout()
    return fig


@st.cache_resource(ttl=60 * 10)
def create_drivers_point_plot(
    _db: InmemoryDB, season: int, driver_target: str
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
    df["total_point_at_each_round"] = df.groupby(["driver"])["points"].cumsum()

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
        if driver == driver_target:
            color = "red"
        else:
            color = "gray"
        df_driver = df.loc[df["driver"] == driver]
        sns.lineplot(
            data=df_driver,
            x="round",
            y="total_point_at_each_round",
            marker="o",
            ax=ax[0],
            linestyle="-",
            color=color,
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
    ax[0].set_xticks(df["round"])
    ax[0].set_ylabel("Points")
    ax[0].set_xlabel("Round")
    ax[0].set_title("Top10 drivers")

    for driver in df_driver_under10["driver"].unique():
        if driver == driver_target:
            color = "red"
        else:
            color = "gray"
        df_driver = df.loc[df["driver"] == driver]
        sns.lineplot(
            data=df_driver,
            x="round",
            y="total_point_at_each_round",
            marker="o",
            ax=ax[1],
            linestyle="-",
            color=color,
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
    ax[1].set_xticks(df["round"])
    ax[1].set_ylabel("Points")
    ax[1].set_xlabel("Round")
    ax[1].set_title("Remaining drivers")

    plt.tight_layout()
    return fig
