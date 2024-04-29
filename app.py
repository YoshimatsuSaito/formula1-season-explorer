import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from modules.ergast_api import RoundData, retrieve_data
from modules.geo import load_geo_data
from modules.model.current_data_retriever import load_current_data
from modules.model.model import Classifier, Ranker
from modules.model.preprocess import ModelInputData, make_model_input_data
from modules.utils import (get_target_previous_season_round, get_target_round,
                           get_target_season, load_config, retrieve_basic_info)
from ui import (plot_circuit, plot_rank_prediction, plot_schedule,
                plot_winner_prediction, show_actual_results)

load_dotenv()

BUCKET_NAME = os.environ.get("BUCKET_NAME")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")

if BUCKET_NAME is None or AWS_ACCESS_KEY_ID is None or AWS_SECRET_ACCESS_KEY is None:
    BUCKET_NAME = st.secrets["BUCKET_NAME"]
    AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
    AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]

DICT_CONFIG = load_config("./config/config.yml")


# Make cached function
@st.cache_data(ttl=60 * 60 * 24, show_spinner=True)
def cached_get_target_previous_season_round(
    current_season: int, current_round_num: int
) -> tuple[int, int] | None:
    return get_target_previous_season_round(current_season, current_round_num)


@st.cache_data(ttl=60 * 60 * 24, show_spinner=True)
def cached_get_target_round(season: int) -> int:
    return get_target_round(season)


@st.cache_data(ttl=60 * 60 * 24, show_spinner=True)
def cached_get_target_season() -> int:
    return get_target_season()


@st.cache_data(ttl=60 * 60 * 24, show_spinner=True)
def cached_retrieve_basic_info(season: int) -> dict:
    return retrieve_basic_info(season)


@st.cache_data(ttl=60 * 60, show_spinner=True)
def cached_retrieve_data(season: int, round_num: int) -> RoundData:
    return retrieve_data(season, round_num)


@st.cache_data(ttl=60 * 60 * 24, show_spinner=True)
def cached_load_geo_data(geo_file_name: str) -> dict:
    return load_geo_data(
        geo_file_name=geo_file_name,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        bucket_name=BUCKET_NAME,
        s3_geo_data_key=DICT_CONFIG["s3_geo_data_key"],
    )


@st.cache_data(ttl=60 * 60, show_spinner=True)
def cached_make_model_input_data(season: int) -> pd.DataFrame:
    df = load_current_data(
        season,
        bucket_name=BUCKET_NAME,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        s3_current_data_key=DICT_CONFIG["s3_current_data_key"],
    )

    model_input_data = make_model_input_data(df)
    return model_input_data


@st.cache_resource(ttl=60 * 60 * 24, show_spinner=True)
def cached_ranker() -> Ranker:
    ranker = Ranker(
        bucket_name=BUCKET_NAME,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        s3_model_key=DICT_CONFIG["s3_model_key"],
    )
    ranker.load_model()
    return ranker


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
    df = df.loc[(df["year"] == season) & (df["round"] == round_to_show)]
    X = df[model_input_data.list_col_X]
    y_pred = _classifier.predict(X)
    df["y_pred"] = y_pred
    df.sort_values(by="y_pred", ascending=False, inplace=True)

    return df


@st.cache_data(ttl=60 * 60, show_spinner=True)
def cached_rank_prediction_result(
    model_input_data: ModelInputData,
    _ranker: Ranker,
    season: int,
    round_to_show: int,
) -> pd.DataFrame:

    df = model_input_data.df.copy()
    df = df.loc[(df["year"] == season) & (df["round"] == round_to_show)]
    X = df[model_input_data.list_col_X]
    y_pred = _ranker.predict(X)
    df["y_pred"] = y_pred
    df.sort_values(by="y_pred", ascending=True, inplace=True)

    return df


# Identify the target round to show
SEASON = cached_get_target_season()
round_to_show = cached_get_target_round(SEASON)
dict_basic_info = cached_retrieve_basic_info(SEASON)
dict_options = {
    round_num: f"Rd. {round_num}: {gp_name}"
    for round_num, gp_name in zip(
        dict_basic_info["round_num"], dict_basic_info["gp_name"]
    )
}
default_idx = round_to_show - 1
round_to_show = st.sidebar.selectbox(
    "Choose Grand Prix",
    options=list(dict_options.keys()),
    format_func=lambda x: dict_options[x],
    index=default_idx,
)
previous_season, previous_round = cached_get_target_previous_season_round(
    SEASON, round_to_show
)

# Get the data of the target round
round_data = cached_retrieve_data(SEASON, round_to_show)
if previous_round is not None:
    previous_data = cached_retrieve_data(previous_season, previous_round)
else:
    previous_data = None

# Show the data
st.header(f"{SEASON} Round {round_data.round_num}: {round_data.gp_name}")

# Show the schedule
st.subheader("Schedule")
plot_schedule(round_data)


# Show the circuits layout
st.subheader("Circuit Layout")
geojson_data = cached_load_geo_data(
    geo_file_name=DICT_CONFIG["gp_circuits"][round_data.gp_name]
)
chart_data = pd.DataFrame(
    geojson_data["features"][0]["geometry"]["coordinates"], columns=["lon", "lat"]
)
plot_circuit(chart_data)


# Show the results of previous hosting
st.subheader(f"Results from last hosting: {previous_season}")
if previous_data is not None:
    show_actual_results(previous_data)
else:
    st.warning("There is not available data")

# Show race prediction
model_input_data = cached_make_model_input_data(SEASON)
ranker = cached_ranker()
classifier = cached_classifier()

st.subheader("Winner Prediction")
df_winner_prediction_result = cached_winner_prediction_result(
    model_input_data, classifier, SEASON, round_to_show
)
plot_winner_prediction(df_winner_prediction_result)

st.subheader("Rank Prediction")
df_rank_prediction_result = cached_rank_prediction_result(
    model_input_data, ranker, SEASON, round_to_show
)
plot_rank_prediction(df_rank_prediction_result)
