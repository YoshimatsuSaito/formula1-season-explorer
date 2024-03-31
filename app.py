import os

import pandas as pd
import pydeck as pdk
import streamlit as st
from dotenv import load_dotenv

from modules.ergast_api import RoundData, retrieve_data
from modules.geo import load_geo_data
from modules.model.current_data_retriever import load_current_data
from modules.model.model import Classifier, Ranker
from modules.model.preprocess import ModelInputData, make_model_input_data
from modules.utils import (get_target_previous_season_round, get_target_round,
                           get_target_season, load_config, retrieve_basic_info)

load_dotenv()

BUCKET_NAME = os.environ.get("BUCKET_NAME")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
DICT_CONFIG = load_config("./config/config.yml")


# Make cached function
@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def cached_get_target_previous_season_round(
    current_season: int, current_round_num: int
) -> tuple[int, int] | None:
    return get_target_previous_season_round(current_season, current_round_num)


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def cached_get_target_round(season: int) -> int:
    return get_target_round(season)


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def cached_get_target_season() -> int:
    return get_target_season()


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def cached_retrieve_basic_info(season: int) -> dict:
    return retrieve_basic_info(season)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def cached_retrieve_data(season: int, round_num: int) -> RoundData:
    return retrieve_data(season, round_num)


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def cached_load_geo_data(geo_file_name: str) -> dict:
    return load_geo_data(
        geo_file_name=geo_file_name,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        bucket_name=BUCKET_NAME,
        s3_geo_data_key=DICT_CONFIG["s3_geo_data_key"],
    )


@st.cache_data(ttl=60 * 60, show_spinner=False)
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


@st.cache_resource(ttl=60 * 60 * 24, show_spinner=False)
def cached_ranker() -> Ranker:
    ranker = Ranker(
        bucket_name=BUCKET_NAME,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        s3_model_key=DICT_CONFIG["s3_model_key"],
    )
    ranker.load_model()
    return ranker


@st.cache_resource(ttl=60 * 60 * 24, show_spinner=False)
def cached_classifier() -> Classifier:
    classifier = Classifier(
        bucket_name=BUCKET_NAME,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        s3_model_key=DICT_CONFIG["s3_model_key"],
    )
    classifier.load_model()
    return classifier


@st.cache_data(ttl=60 * 60, show_spinner=False)
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


@st.cache_data(ttl=60 * 60, show_spinner=False)
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
st.write(
    f"FP1: {round_data.fp1_time.strftime('%m/%d %H:%M') if round_data.fp1_time else 'TBA/Unknown'}"
)
st.write(
    f"FP2: {round_data.fp2_time.strftime('%m/%d %H:%M') if round_data.fp2_time else 'TBA/Unknown'}"
)
st.write(
    f"FP3: {round_data.fp3_time.strftime('%m/%d %H:%M') if round_data.fp3_time else 'TBA/Unknown'}"
)
st.write(
    f"Qualifying: {round_data.qualifying_time.strftime('%m/%d %H:%M') if round_data.qualifying_time else 'TBA/Unknown'}"
)
st.write(
    f"Sprint: {round_data.sprint_time.strftime('%m/%d %H:%M') if round_data.sprint_time else 'TBA/Unknown'}"
)
st.write(
    f"Race: {round_data.race_time.strftime('%m/%d %H:%M') if round_data.race_time else 'TBA/Unknown'}"
)


# Show the circuits layout
st.subheader("Circuit Layout")
geojson_data = cached_load_geo_data(
    geo_file_name=DICT_CONFIG["gp_circuits"][round_data.gp_name]
)
chart_data = pd.DataFrame(
    geojson_data["features"][0]["geometry"]["coordinates"], columns=["lon", "lat"]
)
st.map(chart_data, size=10)


# Show the results of previous hosting
st.subheader(f"Results from last hosting: {previous_season}")
if previous_data is not None:
    st.dataframe(previous_data.df_qualifying_results)
    st.dataframe(previous_data.df_race_result)
else:
    st.warning("There is not available data")

# Show race prediction
st.subheader("AI Prediction")
model_input_data = cached_make_model_input_data(SEASON)
ranker = cached_ranker()
classifier = cached_classifier()
df_winner_prediction_result = cached_winner_prediction_result(
    model_input_data, classifier, SEASON, round_to_show
)
st.dataframe(df_winner_prediction_result)

df_rank_prediction_result = cached_rank_prediction_result(
    model_input_data, ranker, SEASON, round_to_show
)
st.dataframe(df_rank_prediction_result)
