import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

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
    round_to_show = st.sidebar.radio(
        label="Choose Grand Prix",
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
