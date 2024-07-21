import os

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from modules.model import Classifier
from modules.preprocess import ModelInputData, make_datamart, add_features, add_future_race_row
from modules.utils import load_config
from modules.load_csv_data import load_csv_data
from ui import plot_winner_prediction, plot_calendar, get_round_grandprix_from_sidebar, plot_circuit, plot_pole_position_time
from modules.geo import load_geo_data
from modules.inmemory_db import InmemoryDB

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
def _cached_calendar() -> pd.DataFrame:
    df = load_csv_data(bucket_name=BUCKET_NAME, key=f"calendar/{SEASON}.csv")
    return df


@st.cache_data(ttl=60 * 60 * 24 * 7, show_spinner=True)
def _cached_geodata(grandprix: str) -> np.array:
    geo_json = load_geo_data(bucket_name=BUCKET_NAME, geo_file_name=DICT_CONFIG["gp_circuits"][grandprix], s3_geo_data_key=DICT_CONFIG["s3_geo_data_key"])
    return np.array(geo_json["features"][0]["geometry"]["coordinates"])

@st.cache_data(ttl=60 * 60, show_spinner=True)
def _cached_make_model_input_data(df_calendar: pd.DataFrame) -> pd.DataFrame:
    # Make datamart from past result
    df = make_datamart(bucket_name=BUCKET_NAME)
    # Add future race rows to datamart
    df = add_future_race_row(df_datamart=df, df_calendar=df_calendar)
    # Make features to predict
    df = add_features(df)
    model_input_data = ModelInputData(df=df)
    return model_input_data


@st.cache_resource(ttl=60 * 60 * 24, show_spinner=True)
def _cached_classifier() -> Classifier:
    classifier = Classifier(
        bucket_name=BUCKET_NAME,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        s3_model_key=DICT_CONFIG["s3_model_key"],
    )
    classifier.load_model()
    return classifier


@st.cache_data(ttl=60 * 60, show_spinner=True)
def _cached_winner_prediction_result(
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


@st.cache_resource(ttl=60 * 60, show_spinner=True)
def _cached_inmemory_db() -> InmemoryDB:
    db = InmemoryDB(bucket_name=BUCKET_NAME)
    db.create_inmemory_db(dict_csv_key=DICT_CONFIG["s3_grandprix_result_data_key"])
    return db


# Create sidebar and get round and grandprix
df_calendar = _cached_calendar()
round_to_show, grandprix_to_show = get_round_grandprix_from_sidebar(df_calendar=df_calendar)

# Show page title and circuit layout
st.header(f"{SEASON} Round {round_to_show}: {grandprix_to_show}")
array_geo = _cached_geodata(grandprix=grandprix_to_show)
plot_circuit(array_geo=array_geo)

# Show calendar
plot_calendar(df_calendar=df_calendar)

# Show past result
st.subheader("Past result")
db = _cached_inmemory_db()

# Pole time
st.markdown("#### Pole position time")
plot_pole_position_time(db=db, grandprix=grandprix_to_show)


# Show race prediction
st.subheader("Winner Prediction")
model_input_data = _cached_make_model_input_data(df_calendar=df_calendar)
classifier = _cached_classifier()
df_winner_prediction_result = _cached_winner_prediction_result(
    model_input_data, classifier, SEASON, round_to_show
)
plot_winner_prediction(df_winner_prediction_result=df_winner_prediction_result)


if __name__ == "__main__":
    df_calendar = load_csv_data(bucket_name=BUCKET_NAME, key=f"calendar/{SEASON}.csv")
    # Make datamart from past result
    df = make_datamart(bucket_name=BUCKET_NAME)
    # Add future race rows to datamart
    df = add_future_race_row(df_datamart=df, df_calendar=df_calendar)
    # Make features to predict
    df = add_features(df)