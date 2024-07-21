import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from modules.utils import get_latest_grandprix
from modules.inmemory_db import InmemoryDB


def get_round_grandprix_from_sidebar(df_calendar: pd.DataFrame) -> tuple[int, str]:
    """ Identify the target round to show """
    ser_default = get_latest_grandprix(df_calendar=df_calendar)
    dict_options = {
        round_num: f"Rd. {round_num}: {grandprix}"
        for round_num, grandprix in zip(df_calendar["round"].tolist(), df_calendar["grandprix"].tolist())
    }
    round_to_show = st.sidebar.selectbox(
        "Choose Grand Prix",
        options=list(dict_options.keys()),
        format_func=lambda x: dict_options[x],
        index=int(ser_default["round"]) - 1,
    )
    grandprix_to_show = df_calendar.loc[df_calendar["round"]==round_to_show, "grandprix"].iloc[0]
    return round_to_show, grandprix_to_show


def plot_calendar(df_calendar: pd.DataFrame) -> None:
    """Plot grandprix race date calendar"""
    with st.expander("Calendar"):
        for _, row in df_calendar.iterrows():
            str_expr = f"Rd. {row['round']}: {row['grandprix']} ({row['month']}/{row['date']})"
            st.caption(str_expr)


def plot_circuit(array_geo: np.array) -> None:
    longitude = array_geo[:, 0]
    latitude = array_geo[:, 1]
    fig, ax = plt.subplots()
    ax.plot(longitude, latitude, linestyle='-', color='r')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
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
    
    plt.figure(figsize=(12, 8))
    ax = sns.lineplot(data=df, x='season', y='q3_sec', marker='o')
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('sec', fontsize=14)
    ax.set_xticks(df['season'])
    ax.set_xticklabels(df['season'], rotation=45)
    
    # Annotate each point with the time value
    for i, row in df.iterrows():
        ax.annotate(f"{row['q3_sec']:.2f}", (row['season'], row['q3_sec']), textcoords="offset points", xytext=(0,10), ha='center')

    plt.tight_layout()
    
    st.pyplot(plt)
