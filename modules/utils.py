from datetime import datetime
from functools import cache

import pytz
from ergast_py import Ergast

japan_timezone = pytz.timezone("Asia/Tokyo")


def cvt_datetime_to_sec(datetime_value) -> float:
    """Convert datetime to sec"""
    return (
        datetime_value.hour * 3600
        + datetime_value.minute * 60
        + datetime_value.second
        + datetime_value.microsecond / 1000000
    )


@cache
def retrieve_basic_info(season: int) -> dict[str, list[datetime | int | str]]:
    """Retrieve basic information of the season"""
    res = Ergast().season(season).get_races()
    dict_res = {
        "race_time": [x.date.astimezone(japan_timezone) for x in res],
        "round_num": [x.round_no for x in res],
        "gp_name": [x.race_name for x in res],
    }
    return dict_res


def get_target_season() -> int:
    """Get the target season and round"""
    # Candidate years
    # Possible patterns
    # 1. The current year is the target year(yyyy-01-01 to final race)
    # 2. The next year is the target year(final race to yyyy-12-31)
    year_candidate_1 = datetime.now().year
    year_candidate_2 = year_candidate_1 + 1

    # Retrieve basic information of the candidate years
    dict_res_1 = retrieve_basic_info(year_candidate_1)
    dict_res_2 = retrieve_basic_info(year_candidate_2)

    # Identify the target season
    today = datetime.now(japan_timezone)
    for race_time_1 in dict_res_1["race_time"]:
        if race_time_1 is None:
            continue
        elif race_time_1 > today:
            return year_candidate_1
    for race_time_2 in dict_res_2["race_time"]:
        if race_time_2 is None:
            continue
        elif race_time_2 > today:
            return year_candidate_2


def get_target_round(season: int) -> int:
    """Get the target round"""
    dict_res = retrieve_basic_info(season)
    today = datetime.now(japan_timezone)

    # If today is after final race, return the final round number
    if today > max(dict_res["race_time"]):
        return max(dict_res["round_num"])
    latest_gp_time = min(x for x in dict_res["race_time"] if x >= today)
    latest_gp_idx = dict_res["race_time"].index(latest_gp_time)
    return dict_res["round_num"][latest_gp_idx]


def get_target_previous_season_round(
    current_season: int, current_round_num: int
) -> tuple[int, int] | None:
    """Get the round number of the previous year"""
    dict_res_current_year = retrieve_basic_info(current_season)

    # Identify grand prix name of the target round
    gp_round_idx_current_year = dict_res_current_year["round_num"].index(current_round_num)
    gp_name = dict_res_current_year["gp_name"][gp_round_idx_current_year]

    # Check last 10 years
    for y in range(1, 21):
        season_to_check = current_season - y
        dict_res_previous_year = retrieve_basic_info(season_to_check)
        try:
            gp_name_idx_previous_year = dict_res_previous_year["gp_name"].index(gp_name)
            return season_to_check, dict_res_previous_year["round_num"][gp_name_idx_previous_year]
        except:
            continue
    return None
