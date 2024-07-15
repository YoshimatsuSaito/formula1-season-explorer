import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt


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
