import os

import pandas as pd
import pydeck as pdk
import streamlit as st
from dotenv import load_dotenv

from modules.ergast_api import retrieve_data
from modules.geo import load_geo_data
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
DICT_CONFIG = load_config("./config/config.yml")


# Identify the target round to show
SEASON = get_target_season()
round_to_show = get_target_round(SEASON)
dict_basic_info = retrieve_basic_info(SEASON)
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
previous_season, previous_round = get_target_previous_season_round(
    SEASON, round_to_show
)

# Get the data of the target round
round_data = retrieve_data(SEASON, round_to_show)
if previous_round is not None:
    previous_data = retrieve_data(previous_season, previous_round)
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
geojson_data = load_geo_data(
    geo_file_name=DICT_CONFIG["gp_circuits"][round_data.gp_name],
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    bucket_name=BUCKET_NAME,
    s3_geo_data_key=DICT_CONFIG["s3_geo_data_key"],
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
