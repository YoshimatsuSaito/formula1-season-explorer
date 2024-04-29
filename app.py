import os
import pytz

import pandas as pd
import pydeck as pdk
import seaborn as sns
import streamlit as st
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from matplotlib import dates as mdates

from modules.ergast_api import RoundData, retrieve_data
from modules.geo import load_geo_data
from modules.model.current_data_retriever import load_current_data
from modules.model.model import Classifier, Ranker
from modules.model.preprocess import ModelInputData, make_model_input_data
from modules.utils import (
    get_target_previous_season_round,
    get_target_round,
    get_target_season,
    load_config,
    retrieve_basic_info,
)

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


def plot_schedule(roud_data: RoundData) -> None:
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
    ax.xaxis_date(tz=pytz.timezone('Asia/Tokyo'))
    colors = plt.cm.Pastel1.colors
    
    min_date = df['Time'].dt.date.min()
    max_date = df['Time'].dt.date.max()
    date_range = pd.date_range(min_date, max_date, freq='D')

    for i, date in enumerate(date_range):
        start_time = pd.Timestamp(date).tz_localize(pytz.timezone('Asia/Tokyo'))
        end_time = start_time + pd.Timedelta(days=1)
        ax.fill_betweenx(y=[-1, len(df)], x1=start_time, x2=end_time, color = colors[i % len(colors)] , alpha=0.3)
        ax.annotate(f'{date:%m/%d %a}', xy=(start_time, len(df)), xytext=(10, -10),
                    textcoords='offset points', ha='left', va='top')

    for i, (_, row) in enumerate(df.iterrows()):
        ax.scatter(row['Time'], i, color="red", s=20)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.index)
    ax.set_ylim(-0.5, len(df))

    ax.set_xticks(df['Time'])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=pytz.timezone('Asia/Tokyo')))
    plt.xticks(rotation=45)
    plt.tight_layout()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect(0.2)
    st.pyplot(fig)


def plot_rank_prediction(df: pd.DataFrame) -> None:
    """Plot rank prediction figure
    NOTE: This is adhoc function for design"""
    df["driver"] = df["driver"].apply(
        lambda x: x.split("_")[1].upper() if len(x.split("_")) > 1 else x.upper()
    )
    grouped = df.groupby('y_pred')['driver'].apply(lambda x: ', '.join(x))

    fig, ax = plt.subplots()
    positions = range(len(grouped), 0, -1)
    for pos, (rank, drivers) in zip(positions, grouped.items()):
        ax.text(x=0, y=pos, s=f"P{rank}: {drivers}", verticalalignment='center', horizontalalignment="left")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(grouped)+1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    st.pyplot(fig)


def plot_winner_prediction(df: pd.DataFrame) -> None:
    """Plot winner prediction figure
    NOTE: This is adhoc function for design"""
    df_winner_prediction_result.rename(
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

lon = [139.759455, 139.744728, 139.730001, 139.715274]
lat = [35.682839, 35.663452, 35.644065, 35.624678]

# データフレームを作成します。
df = pd.DataFrame({"lon": lon, "lat": lat})

# 各行が線の始点と終点を表すようにデータフレームを変換します。
df["start_lon"] = df["lon"].shift(1)
df["start_lat"] = df["lat"].shift(1)
df = df.dropna()

# LineLayerを作成します。
line_layer = pdk.Layer(
    "LineLayer",
    df,
    get_source_position="[start_lon, start_lat]",
    get_target_position="[lon, lat]",
    get_width=5,
    get_color=[255, 0, 0],
    pickable=True,
    auto_highlight=True,
)

# デッキを作成し、レイヤーを追加します。
deck = pdk.Deck(
    layers=[line_layer],
    initial_view_state=pdk.ViewState(
        latitude=35.682839, longitude=139.759455, zoom=11, pitch=50
    ),
)
st.pydeck_chart(deck)


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

st.dataframe(df_winner_prediction_result)
