import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from scipy.optimize import curve_fit

from modules.inmemory_db import InmemoryDB
from modules.utils import get_latest_grandprix

_DRIVER_NUM_CONSTANT = 20


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


@st.cache_data(ttl=60 * 10)
def create_race_top3_table(_db: InmemoryDB, grandprix: str) -> pd.DataFrame:
    """Create Qualify Top3 drivers table"""
    query = f"""
        SELECT
            driver,
            position,
            season,
        FROM
            race_result
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

    slope = (
        (df["position"] - (-1 * reference_position))
        .div(df["gap_from_reference_position"])
        .mean()
    )
    intercept = -1 * reference_position - slope * 0

    x_line = np.linspace(
        min(df["gap_from_reference_position"]),
        max(df["gap_from_reference_position"]),
        300,
    )
    y_line = slope * x_line + intercept

    fig, ax = plt.subplots()

    for i in range(0, -1 * (lower_position + 1), -1):
        ax.axhline(y=i, alpha=0.3, color="gray", ls="--", lw=1)
    for i in np.arange(-2, 2.1, 0.1):
        ax.axvline(x=i, alpha=0.3, color="gray", ls="--", lw=1)

    sns.lineplot(
        x=x_line,
        y=y_line,
        color="red",
        ax=ax,
        linestyle="-",
        label=f"Fitted Line ({df['season'].min()} - {df['season'].max()})",
    )

    for idx, (season, group) in enumerate(df.groupby("season")):
        if season == df["season"].max():
            label = (f"Most Recent Session ({season})",)
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

    ax.set_title(
        f"Impact of 0.1 Second Gain/Loss Around Pole Position: Change of {(slope/10):.2f} Positions"
    )
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
    ax.set_ylabel("Lap Time (Seconds)", fontsize=14)
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
    ax.set_xlabel("Lap", fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)

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
    ax.set_ylabel("Percentage", fontsize=14)
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
    ax.set_ylabel("Frequency", fontsize=14)
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


@st.cache_resource(ttl=60 * 10)
def create_completion_ratio_plot(
    _db: InmemoryDB, grandprix: str, ser_grandprix_this_season: pd.Series
) -> Figure:
    """Create completion ratio time"""

    def completion_ratio(group) -> float:
        """Completion ratio for each group"""
        total_count = len(group)
        none_count = group.isnull().sum()
        return (total_count - none_count) / total_count

    # Detect completion ratio by position columns (None means DNF etc)
    query = """
        SELECT
            season,
            grandprix,
            position
        FROM
            race_result
    """
    df = _db.execute_query(query)
    df = df.loc[df["grandprix"].isin(ser_grandprix_this_season)]
    df = pd.DataFrame(
        df.groupby(["grandprix", "season"])["position"].apply(completion_ratio)
    ).reset_index()
    df = pd.DataFrame(df.groupby("grandprix")["position"].mean()).reset_index()
    df.sort_values(by="position", ascending=False, inplace=True)
    colors = ["red" if gp == grandprix else "skyblue" for gp in df["grandprix"]]
    fig, ax = plt.subplots()
    sns.barplot(
        y="grandprix",
        x="position",
        data=df,
        orient="h",
        ax=ax,
        palette=colors,
    )
    ax.set_xlabel("Completion ratio")
    ax.set_ylabel("Grandprix")

    plt.tight_layout()

    return fig


@st.cache_resource(ttl=60 * 10)
def create_race_time_plot(_db: InmemoryDB, grandprix: str) -> Figure:
    """Create race time plot"""
    query = f"""
    SELECT season, time_sec 
    FROM race_result 
    WHERE grandprix = '{grandprix}' 
    AND position = 1
    """
    df = _db.execute_query(query)
    df["time_minutes"] = df["time_sec"] / 60

    fig, ax = plt.subplots()
    sns.lineplot(
        data=df,
        x="season",
        y="time_minutes",
        marker="o",
        color="skyblue",
        ax=ax,
        linestyle="-",
    )
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("minutes", fontsize=14)
    ax.set_xticks(df["season"])
    ax.set_xticklabels(df["season"], rotation=45)

    # Annotate each point with the time value
    for i, row in df.iterrows():
        ax.annotate(
            f"{row['time_minutes']:.2f}",
            (row["season"], row["time_minutes"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

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
    df_teammate["driver"] = "teammate"
    df_concat = pd.concat([df_driver, df_teammate]).reset_index()

    sns.lineplot(
        data=df_concat,
        x="season",
        y="position",
        hue="driver",
        marker="o",
        palette={"Self": "skyblue", "teammate": "gray"},
        ax=ax,
        linestyle="-",
    )
    ax.set_xlabel("")
    ax.set_ylim(0, _DRIVER_NUM_CONSTANT)
    ax.set_ylabel("position", fontsize=14)
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
                driver,
                SUM(points) as total_points
            FROM
                race_result
            WHERE
                season = {season}
            GROUP BY
                driver
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
    df_teammate["driver"] = "teammate"
    df_concat = pd.concat([df_driver, df_teammate]).reset_index()

    sns.lineplot(
        data=df_concat,
        x="season",
        y="position",
        hue="driver",
        marker="o",
        palette={"Self": "skyblue", "teammate": "gray"},
        ax=ax,
        linestyle="-",
    )
    ax.set_xlabel("")
    ax.set_ylim(0, _DRIVER_NUM_CONSTANT)
    ax.set_ylabel("position", fontsize=14)
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
def create_drivers_point_plot(
    _db: InmemoryDB, season: int, driver_target: str
) -> Figure:
    """Create drivers point plot"""

    query = f"""
        SELECT 
            driver,
            driver_abbreviation,
            grandprix,
            round,
            points
        FROM 
            race_result
        WHERE 
            season = {season}
        ORDER BY
            round
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


def show_user_search_result(
    db: InmemoryDB, list_result_type: list[str], grandprix: str
) -> None:
    """Show past result which user want to see"""

    list_result_type = [x if x != "qualify" else f"{x}ing" for x in list_result_type]

    # Users choose season
    df_season = db.execute_query(
        "SELECT DISTINCT season FROM race_result ORDER BY season"
    )
    season = st.selectbox(
        label="Season", options=df_season["season"], index=len(df_season) - 2
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
