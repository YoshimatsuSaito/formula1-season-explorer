import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from modules.model import Classifier
from modules.preprocess import ModelInputData, make_datamart, add_features
from modules.utils import load_config, get_latest_grandprix
from modules.load_csv_data import load_csv_data
from ui import plot_winner_prediction

load_dotenv()

BUCKET_NAME = os.environ.get("BUCKET_NAME")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")

if BUCKET_NAME is None or AWS_ACCESS_KEY_ID is None or AWS_SECRET_ACCESS_KEY is None:
    BUCKET_NAME = st.secrets["BUCKET_NAME"]
    AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
    AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]

DICT_CONFIG = load_config("./config/config.yml")
SEASON = DICT_CONFIG["season"]

@st.cache_data(ttl=60 * 60 * 24 * 7, show_spinner=True)
def cached_calendar() -> pd.DataFrame:
    df = load_csv_data(bucket_name=BUCKET_NAME, key=f"calendar/{SEASON}.csv")
    return df

@st.cache_data(ttl=60 * 60, show_spinner=True)
def cached_make_model_input_data() -> pd.DataFrame:
    df = make_datamart(bucket_name=BUCKET_NAME)
    df = add_features(df)
    model_input_data = ModelInputData(df=df)
    return model_input_data


@st.cache_resource(ttl=60 * 60 * 24, show_spinner=True)
def cached_classifier() -> Classifier:
    classifier = Classifier(
        bucket_name=BUCKET_NAME,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        s3_model_key=DICT_CONFIG["s3_model_key"],
    )
    classifier.load_model()
    return classifier


@st.cache_data(ttl=60 * 60, show_spinner=True)
def cached_winner_prediction_result(
    model_input_data: ModelInputData,
    _classifier: Classifier,
    season: int,
    round_to_show: int,
) -> pd.DataFrame:

    df = model_input_data.df.copy()
    df = df.loc[(df["season"] == season) & (df["round"] == round_to_show)]
    X = df[model_input_data.list_col_X]
    y_pred = _classifier.predict(X)
    df["y_pred"] = y_pred
    df.sort_values(by="y_pred", ascending=False, inplace=True)

    return df


# Identify the target round to show
df_calendar = cached_calendar()
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

# Show the data
st.header(f"{SEASON} Round {round_to_show}: {grandprix_to_show}")

# Show race prediction
model_input_data = cached_make_model_input_data()
classifier = cached_classifier()

st.subheader("Winner Prediction")
df_winner_prediction_result = cached_winner_prediction_result(
    model_input_data, classifier, SEASON, round_to_show
)
plot_winner_prediction(df_winner_prediction_result)
