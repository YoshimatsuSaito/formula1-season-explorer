import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from modules.inmemory_db import InmemoryDB


@st.cache_resource(ttl=60 * 10)
def create_practice_time_plot(
    _db: InmemoryDB, grandprix: str, practice_num: int
) -> Figure:
    """Create Practice Time Plot"""

    query = f"""
    SELECT
        season, 
        time_sec,
        1 AS practice
    FROM 
        practice{practice_num}
    WHERE 
        grandprix = '{grandprix}' 
    AND 
        practice{practice_num}.position = 1    
    """
    df = _db.execute_query(query)

    fig, ax = plt.subplots()
    sns.lineplot(
        data=df,
        x="season",
        y="time_sec",
        marker="o",
        ax=ax,
        linestyle="-",
        color="skyblue",
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
def create_practice_race_relation_plot(
    _db: InmemoryDB, grandprix: str
) -> tuple[Figure, Figure, Figure]:
    """Create relation between practice position and race result"""

    query = f"""
        SELECT
            practice1.season,
            practice1.driver,
            practice1.position AS practice_position,
            race_result.position AS race_position,
            1 AS practice
        FROM
            practice1
        JOIN
            race_result
        ON
            practice1.season = race_result.season
            AND practice1.driver = race_result.driver
        WHERE
            practice1.grandprix = '{grandprix}'
            AND race_result.grandprix = '{grandprix}'

        UNION ALL

        SELECT
            practice2.season,
            practice2.driver,
            practice2.position AS practice_position,
            race_result.position AS race_position,
            2 AS practice
        FROM
            practice2
        JOIN
            race_result
        ON
            practice2.season = race_result.season
            AND practice2.driver = race_result.driver
        WHERE
            practice2.grandprix = '{grandprix}'
            AND race_result.grandprix = '{grandprix}'

        UNION ALL

        SELECT
            practice3.season,
            practice3.driver,
            practice3.position AS practice_position,
            race_result.position AS race_position,
            3 AS practice
        FROM
            practice3
        JOIN
            race_result
        ON
            practice3.season = race_result.season
            AND practice3.driver = race_result.driver
        WHERE
            practice3.grandprix = '{grandprix}'
            AND race_result.grandprix = '{grandprix}'

        ORDER BY
            season,
            practice,
            practice_position;
    """

    df = _db.execute_query(query)

    fig, ax = plt.subplots()
    palette = sns.color_palette("cool", 3)
    for practice_num in range(1, 4):
        sns.lineplot(
            data=df.loc[df["practice"] == practice_num],
            x="practice_position",
            y="race_position",
            ax=ax,
            color=palette[practice_num - 1],
            estimator="median",
            errorbar=("ci", 50),
            label=f"Practice {practice_num}",
        )
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(0.5, 20.5)
    ax.set_yticks(range(1, 21))
    ax.set_xticks(range(1, 21))
    ax.set_xlabel("Practice Position", fontsize=14)
    ax.set_ylabel("Race Result", fontsize=14)
    ax.invert_xaxis()
    ax.invert_yaxis()

    handles, labels = ax.get_legend_handles_labels()
    labels.append("Line: median, Fill: CI (50%)")
    handles.append(plt.Line2D([0], [0], color="gray", linestyle="-", linewidth=2))

    ax.legend(handles, labels, loc="lower right", frameon=True)

    plt.tight_layout()

    return fig


@st.cache_resource(ttl=60 * 10)
def create_practice_qualify_relation_plot(
    _db: InmemoryDB, grandprix: str
) -> tuple[Figure, Figure, Figure]:
    """Create relation between practice position and qualify result"""

    query = f"""
        SELECT
            practice1.season,
            practice1.driver,
            practice1.position AS practice_position,
            qualifying.position AS qualifying_position,
            1 AS practice
        FROM
            practice1
        JOIN
            qualifying
        ON
            practice1.season = qualifying.season
            AND practice1.driver = qualifying.driver
        WHERE
            practice1.grandprix = '{grandprix}'
            AND qualifying.grandprix = '{grandprix}'

        UNION ALL

        SELECT
            practice2.season,
            practice2.driver,
            practice2.position AS practice_position,
            qualifying.position AS qualifying_position,
            2 AS practice
        FROM
            practice2
        JOIN
            qualifying
        ON
            practice2.season = qualifying.season
            AND practice2.driver = qualifying.driver
        WHERE
            practice2.grandprix = '{grandprix}'
            AND qualifying.grandprix = '{grandprix}'

        UNION ALL

        SELECT
            practice3.season,
            practice3.driver,
            practice3.position AS practice_position,
            qualifying.position AS qualifying_position,
            3 AS practice
        FROM
            practice3
        JOIN
            qualifying
        ON
            practice3.season = qualifying.season
            AND practice3.driver = qualifying.driver
        WHERE
            practice3.grandprix = '{grandprix}'
            AND qualifying.grandprix = '{grandprix}'

        ORDER BY
            season,
            practice,
            practice_position;
    """

    df = _db.execute_query(query)

    fig, ax = plt.subplots()
    palette = sns.color_palette("cool", 3)
    for practice_num in range(1, 4):
        sns.lineplot(
            data=df.loc[df["practice"] == practice_num],
            x="practice_position",
            y="qualifying_position",
            ax=ax,
            color=palette[practice_num - 1],
            estimator="median",
            errorbar=("ci", 50),
            label=f"Practice {practice_num}",
        )
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(0.5, 20.5)
    ax.set_yticks(range(1, 21))
    ax.set_xticks(range(1, 21))
    ax.set_xlabel("Practice Position", fontsize=14)
    ax.set_ylabel("Qualifying Position", fontsize=14)
    ax.invert_xaxis()
    ax.invert_yaxis()

    handles, labels = ax.get_legend_handles_labels()
    labels.append("Line: median, Fill: CI (50%)")
    handles.append(plt.Line2D([0], [0], color="gray", linestyle="-", linewidth=2))

    ax.legend(handles, labels, loc="lower right", frameon=True)

    plt.tight_layout()

    return fig


@st.cache_resource(ttl=60 * 10)
def create_practice_race_correlation_comparison_plot(
    _db: InmemoryDB, grandprix: str, ser_grandprix_this_season: pd.Series
) -> Figure:
    """Create Correlation comparison between grandprixs"""

    query = """
        SELECT
            practice1.season,
            practice1.driver,
            practice1.position AS practice_position,
            race_result.position AS race_position,
            practice1.grandprix,
            1 AS practice
        FROM
            practice1
        JOIN
            race_result
        ON
            practice1.season = race_result.season
            AND practice1.driver = race_result.driver

        UNION ALL

        SELECT
            practice2.season,
            practice2.driver,
            practice2.position AS practice_position,
            race_result.position AS race_position,
            practice2.grandprix,
            2 AS practice
        FROM
            practice2
        JOIN
            race_result
        ON
            practice2.season = race_result.season
            AND practice2.driver = race_result.driver

        UNION ALL

        SELECT
            practice3.season,
            practice3.driver,
            practice3.position AS practice_position,
            race_result.position AS race_position,
            practice3.grandprix,
            3 AS practice
        FROM
            practice3
        JOIN
            race_result
        ON
            practice3.season = race_result.season
            AND practice3.driver = race_result.driver

        ORDER BY
            season,
            grandprix,
            practice,
            practice_position;
    """

    df = _db.execute_query(query)
    df = df.loc[df["grandprix"].isin(ser_grandprix_this_season)]

    df = pd.DataFrame(
        df.groupby("grandprix")[["race_position", "practice_position"]].apply(
            lambda x: x["race_position"].corr(x["practice_position"], method="spearman")
        ),
    ).reset_index()
    df.columns = ["grandprix", "correlation"]

    df.sort_values(by="correlation", ascending=False, inplace=True)
    colors = ["red" if gp == grandprix else "skyblue" for gp in df["grandprix"]]
    fig, ax = plt.subplots()
    sns.barplot(
        y="grandprix",
        x="correlation",
        data=df,
        orient="h",
        ax=ax,
        palette=colors,
    )
    ax.set_xlabel("Spearman Correlation between Practice and Race Result")
    ax.set_ylabel("Grand Prix")

    plt.tight_layout()

    return fig
