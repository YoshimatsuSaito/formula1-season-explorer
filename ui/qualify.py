import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from scipy.optimize import curve_fit

from modules.inmemory_db import InmemoryDB


@st.cache_data(ttl=60 * 10)
def create_qualify_top3_table(_db: InmemoryDB, grandprix: str) -> pd.DataFrame:
    """Create Qualify Top3 drivers table"""
    query = f"""
        SELECT
            driver,
            position,
            season,
        FROM
            qualifying
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
def create_pole_position_time_plot(_db: InmemoryDB, grandprix: str) -> Figure:
    """Create Pole Position Time Plot"""
    query = f"""
    SELECT season, q3_sec 
    FROM qualifying 
    WHERE grandprix = '{grandprix}' 
    AND position = 1
    """
    df = _db.execute_query(query)

    fig, ax = plt.subplots()
    sns.lineplot(
        data=df,
        x="season",
        y="q3_sec",
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
            f"{row['q3_sec']:.2f}",
            (row["season"], row["q3_sec"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    plt.tight_layout()

    return fig


@st.cache_resource(ttl=60 * 10)
def create_qualify_marginal_gain_plot(
    _db: InmemoryDB,
    grandprix: str,
    q_session: str,
    reference_position: int,
    lower_position: int,
) -> Figure:
    """Create Qualifying Marginal Gain Plot"""
    query = f"""
        SELECT 
            season, {q_session}_sec
        FROM 
            qualifying
        WHERE 
            grandprix = '{grandprix}'
        ORDER BY
            season ASC,
            {q_session}_sec ASC
    """
    df = _db.execute_query(query)

    def calculate_differences(group):
        base_time = group[group["position"] == -1 * reference_position][
            f"{q_session}_sec"
        ].values[0]
        group["gap_from_reference_position"] = base_time - group[f"{q_session}_sec"]

        return group

    df["position"] = -1 * df.groupby("season")[f"{q_session}_sec"].rank(method="first")
    df = (
        df.groupby("season")
        .apply(calculate_differences, include_groups=True)
        .reset_index(drop=True)
    )
    df.dropna(inplace=True)
    df = df.loc[
        (df["gap_from_reference_position"].between(-2, 2))
        & (df["position"].between(-1 * lower_position, -1))
    ]

    x = df["gap_from_reference_position"].values
    y = df["position"].values

    def shifted_sigmoid(x, x0, k):
        L = lower_position - 1
        C = -1 * lower_position
        return L / (1 + np.exp(-k * (x - x0))) + C

    initial_params = [0, 1]
    popt, pcov = curve_fit(shifted_sigmoid, x, y, p0=initial_params)

    x_line = np.linspace(min(x), max(x), 300)
    y_line = shifted_sigmoid(x_line, *popt)

    def derivative_sigmoid(x, x0, k, L=lower_position - 1):
        return (L * k * np.exp(-k * (x - x0))) / ((1 + np.exp(-k * (x - x0))) ** 2)

    # 0.1秒単位とする
    slope_at_x0 = derivative_sigmoid(0, *popt) / 10

    fig, ax = plt.subplots()

    for i in range(0, -1 * (lower_position + 1), -1):
        ax.axhline(y=i, alpha=0.3, color="gray", ls="--", lw=1)
    ax.axhline(
        y=-1 * reference_position,
        alpha=0.8,
        color="lightgreen",
        ls="-",
        label="Qualifying Cut-off Line",
    )
    for i in np.arange(-2, 2.1, 0.1):
        ax.axvline(x=i, alpha=0.3, color="gray", ls="--", lw=1)

    sns.lineplot(
        x=x_line,
        y=y_line,
        color="red",
        ax=ax,
        linestyle="-",
        label=f"Fitted Curve ({df['season'].min()} - {df['season'].max()})",
    )

    for idx, (season, group) in enumerate(df.groupby("season")):
        if season == df["season"].max():
            label = f"Most Recent Session ({season})"
            color = "skyblue"
            alpha = 1
        else:
            label = None
            color = "gray"
            alpha = 0.2
        sns.lineplot(
            x="gap_from_reference_position",
            y="position",
            data=group,
            ax=ax,
            linestyle="-",
            marker="o",
            color=color,
            alpha=alpha,
            label=label,
        )

    ax.set_ylabel("Position", fontsize=14)
    ax.set_xlabel(
        f"Time Difference (seconds) from Cut-off ({reference_position}th Position)",
        fontsize=14,
    )
    ax.set_xticks(np.arange(-2, 2.1, 0.1))
    ax.set_xticklabels(
        [
            (
                str(-1 * round(x, 1))
                if round(x, 1) in [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
                else ""
            )
            for x in np.arange(-2, 2.1, 0.1)
        ]
    )
    ax.set_yticks(range(0, -1 * (lower_position + 2), -1))
    ax.set_yticklabels([""] + list(range(1, lower_position + 1)) + [""])
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1 * (lower_position + 1), 0)

    ax.set_title(
        f"Impact of 0.1 Second Gain/Loss Around Cut-off ({reference_position}th Position): Change of {slope_at_x0:.2f} Positions"
    )
    ax.legend(loc="upper left")

    plt.tight_layout()

    return fig


@st.cache_resource(ttl=60 * 10)
def create_q3_marginal_gain_plot(
    _db: InmemoryDB,
    grandprix: str,
    q_session: str = "q3",
    reference_position: int = 1,
    lower_position: int = 10,
) -> Figure:
    """Create Q3 Marginal Gain Plot"""
    query = f"""
        SELECT 
            season, {q_session}_sec
        FROM 
            qualifying
        WHERE 
            grandprix = '{grandprix}'
        ORDER BY
            season ASC,
            {q_session}_sec ASC
    """
    df = _db.execute_query(query)

    def calculate_differences(group):
        base_time = group[group["position"] == -1 * reference_position][
            f"{q_session}_sec"
        ].values[0]
        group["gap_from_reference_position"] = base_time - group[f"{q_session}_sec"]

        return group

    df["position"] = -1 * df.groupby("season")[f"{q_session}_sec"].rank(method="first")
    df = (
        df.groupby("season")
        .apply(calculate_differences, include_groups=True)
        .reset_index(drop=True)
    )
    df.dropna(inplace=True)
    df = df.loc[
        (df["gap_from_reference_position"].between(-2, 2))
        & (df["position"].between(-1 * lower_position, -1))
    ]

    fig, ax = plt.subplots()

    for i in range(0, -1 * (lower_position + 1), -1):
        ax.axhline(y=i, alpha=0.3, color="gray", ls="--", lw=1)
    for i in np.arange(-2, 2.1, 0.1):
        ax.axvline(x=i, alpha=0.3, color="gray", ls="--", lw=1)

    for idx, (season, group) in enumerate(df.groupby("season")):
        if season == df["season"].max():
            label = f"Most Recent Session ({season})"
            color = "skyblue"
            alpha = 1
        else:
            label = None
            color = "gray"
            alpha = 0.2
        sns.lineplot(
            x="gap_from_reference_position",
            y="position",
            data=group,
            ax=ax,
            linestyle="-",
            marker="o",
            color=color,
            alpha=alpha,
            label=label,
        )

    ax.set_ylabel("Position", fontsize=14)
    ax.set_xlabel("Time Difference (seconds) from Pole Position", fontsize=14)
    ax.set_xticks(np.arange(-2, 0.1, 0.1))
    ax.set_xticklabels(
        [
            str(-1 * round(x, 1)) if round(x, 1) in [-2, -1.5, -1, -0.5, 0] else ""
            for x in np.arange(-2, 0.1, 0.1)
        ]
    )
    ax.set_yticks(range(0, -1 * (lower_position + 2), -1))
    ax.set_yticklabels([""] + list(range(1, lower_position + 1)) + [""])
    ax.set_xlim(-2, 0.1)
    ax.set_ylim(-1 * (lower_position + 1), 0)

    ax.set_title("")
    ax.legend(loc="upper left")

    plt.tight_layout()

    return fig


@st.cache_resource(ttl=60 * 10)
def create_qualify_diff_1st_2nd_plot(_db: InmemoryDB, grandprix: str) -> Figure:
    """Create Time Difference Between Pole Sitter and 2nd Place"""
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
    sns.lineplot(
        data=df, x="season", y="diff_1st_2nd", color="skyblue", ax=ax, marker="o"
    )
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Time Difference (Seconds)", fontsize=14)
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
