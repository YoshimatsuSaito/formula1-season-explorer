import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from modules.inmemory_db import InmemoryDB


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
            data=df.loc[df["practice"]==practice_num], 
            x="practice_position", 
            y="race_position", 
            ax=ax, 
            color=palette[practice_num-1],
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
    handles.append(plt.Line2D([0], [0], color='gray', linestyle='-', linewidth=2))

    ax.legend(handles, labels, loc='lower right', frameon=True)

    plt.tight_layout()

    return fig
