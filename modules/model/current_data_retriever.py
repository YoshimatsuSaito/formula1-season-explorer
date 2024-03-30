import datetime
from io import BytesIO, StringIO

import boto3
import pandas as pd
from ergast_py import Ergast


def _cvt_datetime_to_sec(x: datetime.datetime | None) -> float | None:
    """Convert datetime to sec"""
    if x is None:
        return None
    return x.hour * 3600 + x.minute * 60 + x.second + x.microsecond * 1e-6


def _retrieve_race_data(season: int) -> pd.DataFrame:
    """Get result of race of target season"""
    # dict to store data
    dict_res_race = {
        "position": [],
        "year": [],
        "round": [],
        "grandprix": [],
        "driver": [],
    }

    # Loop for season
    list_race = Ergast().season(season).get_races()
    list_driver = [x.driver_id for x in Ergast().season(season).get_drivers()]

    # Loop for race
    for race in list_race:

        grandprix = race.race_name
        n_round = race.round_no
        print(f"Collect data of round {n_round}: {grandprix} of {season}")

        # If the race is future, store dummy data in dict
        if race.date > datetime.datetime.now(datetime.timezone.utc):
            dict_res_race["year"] += [season] * len(list_driver)
            dict_res_race["grandprix"] += [grandprix] * len(list_driver)
            dict_res_race["round"] += [n_round] * len(list_driver)
            dict_res_race["driver"] += list_driver
            dict_res_race["position"] += [None] * len(list_driver)

        else:
            list_result_of_driver = (
                Ergast().season(season).round(n_round).get_result().results
            )

            # Loop for race result for each driver
            for result_of_driver in list_result_of_driver:
                dict_res_race["year"].append(season)
                dict_res_race["grandprix"].append(grandprix)
                dict_res_race["round"].append(n_round)
                dict_res_race["driver"].append(result_of_driver.driver.driver_id)
                dict_res_race["position"].append(result_of_driver.position)

    df = pd.DataFrame(dict_res_race)
    df = df.sort_values(by=["year", "round", "driver"]).reset_index(drop=True)
    return df


def _retrieve_qualifying_data(season: int) -> pd.DataFrame:
    """Get result of quelifying of target season"""
    dict_res_quali = {
        "year": [],
        "round": [],
        "grandprix": [],
        "driver": [],
        "q1": [],
        "q2": [],
        "q3": [],
    }

    # Loop for season
    list_race = Ergast().season(season).get_races()
    list_driver = [x.driver_id for x in Ergast().season(season).get_drivers()]

    # Loop for race
    for race in list_race:

        grandprix = race.race_name
        n_round = race.round_no
        print(f"Collect data of round {n_round}: {grandprix} of {season}")

        # If the race is future, store dummy data in dict
        if race.date > datetime.datetime.now(datetime.timezone.utc):

            dict_res_quali["year"] += [season] * len(list_driver)
            dict_res_quali["grandprix"] += [grandprix] * len(list_driver)
            dict_res_quali["round"] += [n_round] * len(list_driver)
            dict_res_quali["driver"] += list_driver
            dict_res_quali["q1"] += [None] * len(list_driver)
            dict_res_quali["q2"] += [None] * len(list_driver)
            dict_res_quali["q3"] += [None] * len(list_driver)

        else:
            list_result_of_driver = (
                Ergast()
                .season(season)
                .round(n_round)
                .get_qualifying()
                .qualifying_results
            )

            # Loop for quali result for each driver
            for result_of_driver in list_result_of_driver:
                dict_res_quali["year"].append(season)
                dict_res_quali["grandprix"].append(grandprix)
                dict_res_quali["round"].append(n_round)
                dict_res_quali["driver"].append(result_of_driver.driver.driver_id)
                dict_res_quali["q1"].append(
                    _cvt_datetime_to_sec(result_of_driver.qual_1)
                )
                dict_res_quali["q2"].append(
                    _cvt_datetime_to_sec(result_of_driver.qual_2)
                )
                dict_res_quali["q3"].append(
                    _cvt_datetime_to_sec(result_of_driver.qual_3)
                )

    df = pd.DataFrame(dict_res_quali)
    df = df.sort_values(by=["year", "round", "driver"]).reset_index(drop=True)
    return df


def retrieve_data(season: int) -> pd.DataFrame:
    """Get result of race and qualifying of target season
    NOTE: This is used in AWS lambda function
    """
    try:
        df_race = _retrieve_race_data(season)
        df_qualifying = _retrieve_qualifying_data(season)
    except:
        return pd.DataFrame()

    df = pd.merge(
        df_race, df_qualifying, on=["year", "round", "grandprix", "driver"], how="left"
    )
    return df


def save_data(
    df: pd.DataFrame,
    season: int,
    s3_current_data_key: str,
    bucket_name: str,
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
) -> None:
    """Save data to S3
    NOTE: This is used in AWS lambda function
    """

    if aws_access_key_id is None or aws_secret_access_key is None:
        s3 = boto3.client("s3")
    else:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(
        Bucket=bucket_name,
        Key=f"{s3_current_data_key}/{season}.csv",
        Body=csv_buffer.getvalue(),
    )


def load_current_data(
    season: int,
    bucket_name: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    s3_current_data_key: str,
) -> pd.DataFrame:
    """Load current data from S3 which is stored by AWS Lambda
    NOTE: This is intended to be used in local environment
    """

    s3 = boto3.resource(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    # Model needs data of current season and previous season
    list_df = []
    for s in [season, season - 1]:
        content_object = s3.Object(
            bucket_name=bucket_name,
            key=f"{s3_current_data_key}/{s}.csv",
        )
        csv_content = content_object.get()["Body"].read().decode("utf-8")
        list_df.append(pd.read_csv(StringIO(csv_content)))
    df = pd.concat(list_df, ignore_index=True, axis=0)

    return df
