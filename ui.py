import pandas as pd
import pydeck as pdk
import pytz
import seaborn as sns
import streamlit as st
from matplotlib import dates as mdates
from matplotlib import pyplot as plt

from modules.ergast_api import RoundData


def plot_schedule(round_data: RoundData) -> None:
    """Plot schedule figure
    NOTE: This is adhoc function for design"""
    df = pd.DataFrame(
        [
            round_data.fp1_time,
            round_data.fp2_time,
            round_data.fp3_time,
            round_data.qualifying_time,
            round_data.sprint_time,
            round_data.race_time,
        ],
        columns=["Time"],
        index=["FP1", "FP2", "FP3", "Qualifying", "Sprint", "Race"],
    )
    df.dropna(axis=0, how="any", inplace=True)
    df.sort_values(by="Time", inplace=True, ascending=False)

    fig, ax = plt.subplots()
    ax.xaxis_date(tz=pytz.timezone("Asia/Tokyo"))
    colors = plt.cm.Pastel1.colors

    min_date = df["Time"].dt.date.min()
    max_date = df["Time"].dt.date.max()
    date_range = pd.date_range(min_date, max_date, freq="D")

    for i, date in enumerate(date_range):
        start_time = pd.Timestamp(date).tz_localize(pytz.timezone("Asia/Tokyo"))
        end_time = start_time + pd.Timedelta(days=1)
        ax.fill_betweenx(
            y=[-1, len(df)],
            x1=start_time,
            x2=end_time,
            color=colors[i % len(colors)],
            alpha=0.3,
        )
        ax.annotate(
            f"{date:%m/%d %a}",
            xy=(start_time, len(df)),
            xytext=(10, -10),
            textcoords="offset points",
            ha="left",
            va="top",
        )

    for i, (_, row) in enumerate(df.iterrows()):
        ax.scatter(row["Time"], i, color="red", s=20)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.index)
    ax.set_ylim(-0.5, len(df))

    ax.set_xticks(df["Time"])
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%H:%M", tz=pytz.timezone("Asia/Tokyo"))
    )
    plt.xticks(rotation=45)
    plt.tight_layout()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect(0.2)
    st.pyplot(fig)


def show_actual_results(round_data: RoundData) -> None:
    """Show actual results"""
    df = pd.merge(
        round_data.df_race_result,
        round_data.df_qualifying_results,
        on="driver",
        how="left",
        suffixes=("_race", "_qualifying"),
    )
    df.sort_values(by="position_race", ascending=True, inplace=True)
    df["driver"] = df["driver"].apply(
        lambda x: x.split("_")[1].upper() if len(x.split("_")) > 1 else x.upper()
    )
    df.rename(
        columns={
            "driver": "Driver",
            "position_race": "Pos.",
            "position_qualifying": "Grid",
            "count": "Pitstops",
            "time": "Finish Time",
            "fastest_lap": "Fastest",
            "q1": "Q1",
            "q2": "Q2",
            "q3": "Q3",
        },
        inplace=True,
    )
    df["Finish Time"] = df["Finish Time"].apply(
        lambda x: f"{x.hour * 60 + x.minute}min{x.second}sec" if pd.notnull(x) else x
    )
    for col in ["Fastest", "Q1", "Q2", "Q3"]:
        df[col] = df[col].apply(
            lambda x: (
                f"{x.minute*60 + x.second + x.microsecond/1e6:.3f}"
                if pd.notnull(x)
                else x
            )
        )
    st.dataframe(df, hide_index=True)


def plot_rank_prediction(df: pd.DataFrame) -> None:
    """Plot rank prediction figure
    NOTE: This is adhoc function for design"""
    df["driver"] = df["driver"].apply(
        lambda x: x.split("_")[1].upper() if len(x.split("_")) > 1 else x.upper()
    )
    grouped = df.groupby("y_pred")["driver"].apply(lambda x: ", ".join(x))

    fig, ax = plt.subplots()
    positions = range(len(grouped), 0, -1)
    for pos, (rank, drivers) in zip(positions, grouped.items()):
        ax.text(
            x=0,
            y=pos,
            s=f"P{rank}: {drivers}",
            verticalalignment="center",
            horizontalalignment="left",
        )
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(grouped) + 1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    st.pyplot(fig)


def plot_winner_prediction(df: pd.DataFrame) -> None:
    """Plot winner prediction figure
    NOTE: This is adhoc function for design"""
    df.rename(
        columns={"driver": "Driver", "y_pred": "Winning Probability"}, inplace=True
    )
    df["Driver"] = df["Driver"].apply(
        lambda x: x.split("_")[1].upper() if len(x.split("_")) > 1 else x.upper()
    )
    fig, ax = plt.subplots()
    sns.barplot(
        y="Driver",
        x="Winning Probability",
        data=df,
        orient="h",
        ax=ax,
        color="skyblue",
    )
    ax.set_ylabel("")
    ax.set_xlim(0, 1)
    st.pyplot(fig)


def plot_circuit(df: pd.DataFrame) -> None:
    """Plot circuit layout figure
    NOTE: This is adhoc function for design"""
    cols = st.columns([1, 3, 1])

    df["start_lon"] = df["lon"].shift(1)
    df["start_lat"] = df["lat"].shift(1)
    df = df.dropna()

    layer = pdk.Layer(
        "LineLayer",
        df,
        get_source_position="[start_lon, start_lat]",
        get_target_position="[lon, lat]",
        get_width=5,
        get_color=[255, 0, 0, 160],
        pickable=False,
        auto_highlight=False,
    )
    view_state = pdk.ViewState(
        longitude=df["lon"].mean(),
        latitude=df["lat"].mean(),
        zoom=14,
        pitch=0,
        bearing=0,
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/light-v9",
        tooltip=False,
        views=[pdk.View(type="MapView", controller=False)],
    )

    cols[1].pydeck_chart(deck)
