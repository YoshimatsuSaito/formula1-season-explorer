import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from modules.inmemory_db import InmemoryDB
from modules.load_csv_data import load_csv_data
from modules.model import Classifier
from modules.preprocess import (ModelInputData, add_features,
                                add_future_race_row, make_datamart)
from modules.utils import load_config
from ui import (create_completion_ratio_plot,
                create_diff_fastest_lap_and_pole_time_plot,
                create_driver_past_qualify_result_plot,
                create_driver_past_race_result_plot, create_drivers_point_plot,
                create_fastest_lap_plot, create_fastest_lap_timing_plot,
                create_first_pit_stop_timing_plot, create_pit_stop_count_plot,
                create_pole_position_time_plot,
                create_probability_from_each_grid_plots,
                create_q1_threshold_plot, create_q2_threshold_plot,
                create_qualify_diff_1st_2nd_plot, create_race_time_plot,
                create_winner_prediction_plot,
                get_round_grandprix_from_sidebar,
                show_user_search_result)

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
round_to_show, grandprix_to_show = get_round_grandprix_from_sidebar(
    df_calendar=df_calendar
)

# Setup duckdb
db = _cached_inmemory_db()

# Get drivers list in this season
df_driver = db.execute_query(
    f"""
        SELECT
            driver,
            SUM(points) as total_points
        FROM
            race_result
        WHERE
            season = {SEASON}
        GROUP BY
            driver
        ORDER BY
            total_points DESC
    """
)

# Show page title and circuit layout
st.header(f"{SEASON} Round {round_to_show}: {grandprix_to_show}")

page = st.radio(
    "",
    ["Grandprix", "Drivers", "Winner prediction", "Custom search", "Others"],
    captions=[
        "Previous stats about this GP",
        "Previous stats of drivers for this GP",
        "Made by AI",
        "Search your own interest",
        ""
    ],
    index=0,
)

if page == "Grandprix":
    # Show stats
    st.subheader(f"Stats of the GrandPrix")

    genre = st.selectbox("Select data type", ["Qualify", "Grid", "Race", "Pit stop", "Fastest lap"], index=0)

    if genre == "Qualify":
        ## Pole position time
        st.markdown("#### Pole position time")
        fig_pole = create_pole_position_time_plot(_db=db, grandprix=grandprix_to_show)
        st.pyplot(fig_pole)

        ## Q1 -> Q2 threshold
        st.markdown("#### Q1 -> Q2 threshold")
        fig_q1_thr = create_q1_threshold_plot(_db=db, grandprix=grandprix_to_show)
        st.pyplot(fig_q1_thr)

        ## Q2 -> Q3 threshold
        st.markdown("#### Q2 -> Q3 threshold")
        fig_q2_thr = create_q2_threshold_plot(_db=db, grandprix=grandprix_to_show)
        st.pyplot(fig_q2_thr)

        ## Q3 sec Difference between 1st and 2nd
        st.markdown("#### Difference between pole sitter and 2nd")
        fig_q_diff = create_qualify_diff_1st_2nd_plot(_db=db, grandprix=grandprix_to_show)
        st.pyplot(fig_q_diff)


    if genre == "Grid":
        ## Winning probability from each grid
        st.markdown("#### Result Probabilities from each grid")
        fig_win_prob, fig_pod_prob, fig_point_prob = (
            create_probability_from_each_grid_plots(_db=db, grandprix=grandprix_to_show)
        )
        st.pyplot(fig_win_prob)
        st.pyplot(fig_pod_prob)
        st.pyplot(fig_point_prob)


    if genre == "Race":
        ### Completion ratio
        st.markdown("#### Completion ratio")
        fig_completion_ratio = create_completion_ratio_plot(
            _db=db,
            grandprix=grandprix_to_show,
            ser_grandprix_this_season=df_calendar["grandprix"],
        )
        st.pyplot(fig_completion_ratio)

        ### Race time
        st.markdown("#### Race time")
        fig_race_time = create_race_time_plot(_db=db, grandprix=grandprix_to_show)
        st.pyplot(fig_race_time)

    if genre == "Pit stop":
        ## Pit stop count proportion
        st.markdown("#### Pit stop count proportion")
        fig_pit_count = create_pit_stop_count_plot(_db=db, grandprix=grandprix_to_show)
        st.pyplot(fig_pit_count)

        ## First pit stop timing
        st.markdown("#### First pit stop timing")
        fig_first_pit = create_first_pit_stop_timing_plot(
            _db=db, grandprix=grandprix_to_show
        )
        st.pyplot(fig_first_pit)

    if genre == "Fastest lap":
        ## Fastest lap
        st.markdown("#### Fastest lap")
        fig_fastest_lap = create_fastest_lap_plot(_db=db, grandprix=grandprix_to_show)
        st.pyplot(fig_fastest_lap)

        ## Fastest lap timing
        st.markdown("#### Fastest lap timing (after 10 laps)")
        fig_fastest_lap_timing = create_fastest_lap_timing_plot(
            _db=db, grandprix=grandprix_to_show
        )
        st.pyplot(fig_fastest_lap_timing)

        ## Fastest lap / Pole position
        st.markdown("#### Fastest lap / Pole position")
        fig_diff_fastest_pole = create_diff_fastest_lap_and_pole_time_plot(
            _db=db,
            grandprix=grandprix_to_show,
            ser_grandprix_this_season=df_calendar["grandprix"],
        )
        st.pyplot(fig_diff_fastest_pole)

if page == "Drivers":
    st.subheader(f"Stats of drivers")
    driver = st.selectbox("Select a driver", df_driver["driver"].tolist(), index=0)

    st.markdown("#### Past qualify result")
    fig_past_qualify_performance = create_driver_past_qualify_result_plot(
        _db=db, grandprix=grandprix_to_show, season=SEASON, driver=driver
    )
    st.pyplot(fig_past_qualify_performance)

    ## Past performance at the grandprix
    st.markdown("#### Past race result")
    fig_past_race_performance = create_driver_past_race_result_plot(
        _db=db, grandprix=grandprix_to_show, season=SEASON, driver=driver
    )
    st.pyplot(fig_past_race_performance)

    st.markdown("#### Point standings")
    fig_point_standings = create_drivers_point_plot(_db=db, season=SEASON, driver_target=driver)
    st.pyplot(fig_point_standings)

if page == "Winner prediction":
    st.markdown("### Winner prediction")    
    # Show race prediction
    model_input_data = _cached_make_model_input_data(df_calendar=df_calendar)
    classifier = _cached_classifier()
    df_winner_prediction_result = _cached_winner_prediction_result(
        model_input_data, classifier, SEASON, round_to_show
    )
    fig_pred = create_winner_prediction_plot(
        df_winner_prediction_result=df_winner_prediction_result
    )
    st.pyplot(fig_pred)

if page == "Custom search":
    st.markdown("### Search stats")
    ## Users search
    show_user_search_result(
        db=db,
        list_result_type=list(DICT_CONFIG["s3_grandprix_result_data_key"].keys()),
        grandprix=grandprix_to_show,
    )


if page == "Others":
    st.markdown("### Others")
    # Show calendar
    st.link_button("Season Full Schedule", "https://www.formula1.com/en/racing/2024")


if __name__ == "__main__":
    df_calendar = load_csv_data(bucket_name=BUCKET_NAME, key=f"calendar/{SEASON}.csv")
    # Make datamart from past result
    df = make_datamart(bucket_name=BUCKET_NAME)
    # Add future race rows to datamart
    df = add_future_race_row(df_datamart=df, df_calendar=df_calendar)
    # Make features to predict
    df = add_features(df)
