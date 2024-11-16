import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from modules.inmemory_db import InmemoryDB
from modules.load_csv_data import load_csv_data
from modules.model import Classifier
from modules.preprocess import (
    ModelInputData,
    add_features,
    add_future_race_row,
    make_datamart,
)
from modules.utils import load_config
from ui.custom_search import show_user_search_result
from ui.driver_past_result import (
    create_driver_past_qualify_result_plot,
    create_driver_past_race_result_plot,
    create_drivers_point_plot,
)
from ui.fastest_lap import (
    create_diff_fastest_lap_and_pole_time_plot,
    create_fastest_lap_plot,
    create_fastest_lap_timing_plot,
)
from ui.grid import create_probability_from_each_grid_plots
from ui.pit_stop import create_first_pit_stop_timing_plot, create_pit_stop_count_plot
from ui.practice import (
    create_practice_race_relation_plot,
    create_practice_race_correlation_comparison_plot,
)
from ui.qualify import (
    create_pole_position_time_plot,
    create_q3_marginal_gain_plot,
    create_qualify_diff_1st_2nd_plot,
    create_qualify_marginal_gain_plot,
    create_qualify_top3_table,
)
from ui.race import (
    create_completion_ratio_plot,
    create_race_time_plot,
    create_race_top3_table,
)
from ui.sidebar import get_round_grandprix_from_sidebar
from ui.standings import create_driver_standings_plot
from ui.winner_prediction import create_winner_prediction_plot

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


@st.cache_data(ttl=60 * 60, show_spinner=True)
def _cached_df_driver(_db: InmemoryDB) -> pd.DataFrame:
    df = _db.execute_query(
        f"""
            SELECT
                r.driver_abbreviation,
                r.driver,
                SUM(COALESCE(r.points, 0)) + SUM(COALESCE(s.points, 0)) as total_points
            FROM
                race_result AS r
            LEFT JOIN
                sprint AS s
            ON
                r.driver = s.driver
                AND
                r.grandprix = s.grandprix
                AND
                s.season = {SEASON}
            WHERE
                r.season = {SEASON}
            GROUP BY
                r.driver, r.driver_abbreviation
            ORDER BY
                total_points DESC
        """
    )
    return df


# Create sidebar and get round and grandprix
df_calendar = _cached_calendar()
round_to_show, grandprix_to_show = get_round_grandprix_from_sidebar(
    df_calendar=df_calendar
)

# Setup duckdb
db = _cached_inmemory_db()

# Get driver list ordered by total points
df_driver = _cached_df_driver(_db=db)

# Show page title and circuit layout
st.header(f"{SEASON} Round {round_to_show}: {grandprix_to_show}")
st.markdown(
    f"This page provides various information about the {grandprix_to_show} Grand Prix, as well as current standings and statistics for the {SEASON} season."
)

page = st.radio(
    "Select the type of information to display",
    [
        "Grand Prix",
        "Drivers",
        "Prediction",
        "Standings",
        "Custom Search",
        "Miscellaneous",
    ],
    index=0,
    horizontal=True,
)

st.markdown("---")

if page == "Grand Prix":
    # Show stats
    st.subheader("Grand Prix Statistics")

    genre = st.radio(
        "Select Data Type",
        ["Practice", "Qualifying", "Grid", "Race", "Pit Stops", "Fastest Laps"],
        index=0,
        horizontal=True,
    )

    if genre == "Practice":
        ## Relation between practice and race
        st.markdown("#### Relation between Practice and Race Result")
        fig_practice = create_practice_race_relation_plot(
            _db=db, grandprix=grandprix_to_show
        )
        st.pyplot(fig_practice)

        ## Comparison across Grandprix
        st.markdown("#### Comparison of Correlation Coefficients Between Grand Prix")
        fig_correlation_comparison = create_practice_race_correlation_comparison_plot(
            _db=db,
            grandprix=grandprix_to_show,
            ser_grandprix_this_season=df_calendar["grandprix"],
        )
        st.pyplot(fig_correlation_comparison)

    if genre == "Qualifying":
        ## Pole Position Time
        st.markdown("#### Pole Position Time")
        fig_pole = create_pole_position_time_plot(_db=db, grandprix=grandprix_to_show)
        st.pyplot(fig_pole)

        ## Q1
        st.markdown("#### Q1: Impact of Lap Time Changes on Position")
        fig_q1_thr = create_qualify_marginal_gain_plot(
            _db=db,
            grandprix=grandprix_to_show,
            q_session="q1",
            reference_position=15,
            lower_position=20,
        )
        st.pyplot(fig_q1_thr)

        ## Q2
        st.markdown("#### Q2: Impact of Lap Time Changes on Position")
        fig_q2_thr = create_qualify_marginal_gain_plot(
            _db=db,
            grandprix=grandprix_to_show,
            q_session="q2",
            reference_position=10,
            lower_position=15,
        )
        st.pyplot(fig_q2_thr)

        ## Q3
        st.markdown("#### Q3: Impact of Lap Time Changes on Position")
        fig_q3_thr = create_q3_marginal_gain_plot(
            _db=db,
            grandprix=grandprix_to_show,
            q_session="q3",
            reference_position=1,
            lower_position=10,
        )
        st.pyplot(fig_q3_thr)

        ## Q3 sec Difference between 1st and 2nd
        st.markdown("#### Difference Between Pole Sitter and 2nd Place")
        fig_q_diff = create_qualify_diff_1st_2nd_plot(
            _db=db, grandprix=grandprix_to_show
        )
        st.pyplot(fig_q_diff)

        ## Past Top 3 Qualifiers
        st.markdown("#### Past Top 3 Drivers")
        df = create_qualify_top3_table(_db=db, grandprix=grandprix_to_show)
        st.dataframe(df)

    if genre == "Grid":
        ## Winning Probability from Each Grid
        st.markdown("#### Result Probabilities by Starting Grid")
        fig_win_prob, fig_pod_prob, fig_point_prob = (
            create_probability_from_each_grid_plots(_db=db, grandprix=grandprix_to_show)
        )
        st.pyplot(fig_win_prob)
        st.pyplot(fig_pod_prob)
        st.pyplot(fig_point_prob)

    if genre == "Race":
        ### Finish Ratio
        st.markdown("#### Finish Ratio")
        fig_completion_ratio = create_completion_ratio_plot(
            _db=db,
            grandprix=grandprix_to_show,
            ser_grandprix_this_season=df_calendar["grandprix"],
        )
        st.pyplot(fig_completion_ratio)

        ### Race Duration
        st.markdown("#### Race Duration")
        fig_race_time = create_race_time_plot(_db=db, grandprix=grandprix_to_show)
        st.pyplot(fig_race_time)

        ## Past Top 3 Finishers
        st.markdown("#### Past Top 3 Drivers")
        df = create_race_top3_table(_db=db, grandprix=grandprix_to_show)
        st.dataframe(df)

    if genre == "Pit Stops":
        ## Pit Stop Frequency
        st.markdown("#### Pit Stop Frequency")
        fig_pit_count = create_pit_stop_count_plot(_db=db, grandprix=grandprix_to_show)
        st.pyplot(fig_pit_count)

        ## Timing of First Pit Stop
        st.markdown("#### Timing of First Pit Stop")
        fig_first_pit = create_first_pit_stop_timing_plot(
            _db=db, grandprix=grandprix_to_show
        )
        st.pyplot(fig_first_pit)

    if genre == "Fastest Laps":
        ## Fastest Lap
        st.markdown("#### Fastest Lap")
        fig_fastest_lap = create_fastest_lap_plot(_db=db, grandprix=grandprix_to_show)
        st.pyplot(fig_fastest_lap)

        ## Fastest Lap Timing
        st.markdown("#### Fastest Lap Timing (Post 10 Laps)")
        fig_fastest_lap_timing = create_fastest_lap_timing_plot(
            _db=db, grandprix=grandprix_to_show
        )
        st.pyplot(fig_fastest_lap_timing)

        ## Fastest Lap / Pole Position
        st.markdown("#### Fastest Lap to Pole Position Ratio")
        fig_diff_fastest_pole = create_diff_fastest_lap_and_pole_time_plot(
            _db=db,
            grandprix=grandprix_to_show,
            ser_grandprix_this_season=df_calendar["grandprix"],
        )
        st.pyplot(fig_diff_fastest_pole)

if page == "Drivers":
    st.subheader("Driver Statistics")
    driver_abbreviation = st.radio(
        "Select a Driver",
        df_driver["driver_abbreviation"].tolist(),
        index=0,
        horizontal=True,
    )
    driver = df_driver.loc[
        df_driver["driver_abbreviation"] == driver_abbreviation, "driver"
    ].iloc[0]

    st.markdown("#### Past Qualifying Results")
    fig_past_qualify_performance = create_driver_past_qualify_result_plot(
        _db=db, grandprix=grandprix_to_show, season=SEASON, driver=driver
    )
    st.pyplot(fig_past_qualify_performance)

    ## Past performance at the Grand Prix
    st.markdown("#### Past Race Results")
    fig_past_race_performance = create_driver_past_race_result_plot(
        _db=db, grandprix=grandprix_to_show, season=SEASON, driver=driver
    )
    st.pyplot(fig_past_race_performance)

    st.markdown("#### Points Standings")
    fig_point_standings = create_drivers_point_plot(
        _db=db, season=SEASON, driver_target=driver
    )
    st.pyplot(fig_point_standings)

if page == "Prediction":
    st.markdown("### Race Winner Prediction")
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

if page == "Standings":
    st.markdown(f"### Points Standings for the {SEASON} Season")

    projection_num = st.radio(
        "Number of Races for Season End Projection",
        options=[0, 1, 2, 3, 4, 5],
        index=0,
        horizontal=True,
    )

    fig_standings = create_driver_standings_plot(
        _db=db,
        season=SEASON,
        df_calendar=df_calendar,
        num_for_projection=projection_num,
    )
    st.pyplot(fig_standings)

if page == "Custom Search":
    st.markdown("### Custom Statistics Search")
    ## Users' search
    show_user_search_result(
        db=db,
        list_result_type=list(DICT_CONFIG["s3_grandprix_result_data_key"].keys()),
        grandprix=grandprix_to_show,
    )

if page == "Miscellaneous":
    st.markdown("### Miscellaneous")
    # Show calendar
    st.link_button("Season Full Schedule", "https://www.formula1.com/en/racing/2024")
