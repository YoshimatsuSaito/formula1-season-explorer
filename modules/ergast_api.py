import os
import sys
from dataclasses import dataclass
from datetime import datetime
from functools import cache, cached_property
from typing import Any

import pandas as pd
from ergast_py import Ergast

current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir, "../"))

from modules.utils import japan_timezone


@dataclass
class RoundData:
    """Round data retrieved from ergast api"""

    basic_info: Any | None
    race_results: list | None
    qualifying_results: list | None
    sprint_results: list | None
    pit_stops: list | None

    @property
    def circuit_name(self) -> str:
        """Get circuit name of the round"""
        return self.basic_info.circuit.circuit_name

    @property
    def race_time(self, timezone=japan_timezone) -> datetime | None:
        """Get race time of the round"""
        return (
            self.basic_info.date.astimezone(timezone)
            if self.basic_info.date is not None
            else None
        )

    @property
    def fp1_time(self, timezone=japan_timezone) -> datetime | None:
        """Get fp1 time of the round"""
        return (
            self.basic_info.first_practice.astimezone(timezone)
            if self.basic_info.first_practice is not None
            else None
        )

    @property
    def fp2_time(self, timezone=japan_timezone) -> datetime | None:
        """Get fp2 time of the round"""
        return (
            self.basic_info.second_practice.astimezone(timezone)
            if self.basic_info.second_practice is not None
            else None
        )

    @property
    def fp3_time(self, timezone=japan_timezone) -> datetime | None:
        """Get fp3 time of the round"""
        return (
            self.basic_info.third_practice.astimezone(timezone)
            if self.basic_info.third_practice is not None
            else None
        )

    @property
    def sprint_time(self, timezone=japan_timezone) -> datetime | None:
        """Get sprint time of the round"""
        return (
            self.basic_info.sprint.astimezone(timezone)
            if self.basic_info.sprint is not None
            else None
        )

    @property
    def qualifying_time(self, timezone=japan_timezone) -> datetime | None:
        """Get qualifying time of the round"""
        return (
            self.basic_info.qualifying.astimezone(timezone)
            if self.basic_info.qualifying is not None
            else None
        )

    @property
    def round_num(self) -> int:
        """Get round number"""
        return self.basic_info.round_no

    @property
    def gp_name(self) -> str:
        """Get grand prix name"""
        return self.basic_info.race_name

    @property
    def wiki_url(self) -> str:
        """Get wiki url of the round"""
        return self.basic_info.url

    @property
    def latitude(self) -> float:
        """Get latitude of the round"""
        return self.basic_info.circuit.location.latitude

    @property
    def longitude(self) -> float:
        """Get longitude of the round"""
        return self.basic_info.circuit.location.longitude

    @cached_property
    def df_race_result(self) -> pd.DataFrame | None:
        """Return dataframe of race results"""
        if self.race_results is None:
            return None

        positions = []
        drivers = []
        times = []
        fastest_laps = []
        for race_result in self.race_results:
            positions.append(race_result.position)
            drivers.append(race_result.driver.driver_id)
            times.append(race_result.time.millis)
            fastest_laps.append(race_result.fastest_lap.time)
        df_race_result = pd.DataFrame(
            {
                "position": positions,
                "driver": drivers,
                "time": times,
                "fastest_lap": fastest_laps,
            }
        )

        # add pit stop info
        if self.pit_stops is None:
            df_race_result["pit_stop"] = None
            return df_race_result
        ser_pit_stop = pd.DataFrame(self.pit_stops)["driver_id"].value_counts()
        df_race_result = pd.merge(
            df_race_result, ser_pit_stop, left_on="driver", right_index=True, how="left"
        )
        return df_race_result

    @cached_property
    def df_qualifying_results(self) -> pd.DataFrame | None:
        """Return dataframe of qualifying results"""
        if self.qualifying_results is None:
            return None
        positions = []
        drivers = []
        q1s = []
        q2s = []
        q3s = []
        for qualifying_result in self.qualifying_results:
            positions.append(qualifying_result.position)
            drivers.append(qualifying_result.driver.driver_id)
            q1s.append(qualifying_result.qual_1)
            q2s.append(qualifying_result.qual_2)
            q3s.append(qualifying_result.qual_3)
        return pd.DataFrame(
            {
                "position": positions,
                "driver": drivers,
                "q1": q1s,
                "q2": q2s,
                "q3": q3s,
            }
        )


@cache
def retrieve_data(season: int, round_num: int) -> RoundData:
    """Retrieve info from api
    NOTE: This is the only access point to the api
    """
    basic_info = Ergast().season(season).round(round_num).get_race()

    try:
        race_results = Ergast().season(season).round(round_num).get_result().results
    except:
        race_results = None

    try:
        qualifying_results = (
            Ergast().season(season).round(round_num).get_qualifying().qualifying_results
        )
    except:
        qualifying_results = None

    try:
        sprint_results = (
            Ergast().season(season).round(round_num).get_sprint().sprint_results
        )
    except:
        sprint_results = None

    try:
        pit_stops = Ergast().season(season).round(round_num).get_pit_stop().pit_stops
    except:
        pit_stops = None

    return RoundData(
        basic_info=basic_info,
        race_results=race_results,
        qualifying_results=qualifying_results,
        sprint_results=sprint_results,
        pit_stops=pit_stops,
    )
