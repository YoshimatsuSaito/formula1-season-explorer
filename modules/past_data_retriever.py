import os
import boto3
import pandas as pd
import sys
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import pandas as pd

current_dir = Path(__file__).resolve().parent
sys.path.append(str(Path(current_dir).parent))

from modules.utils import load_config

AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
DICT_CONFIG = load_config(Path(current_dir.parent) / "config/config.yml")


@dataclass
class Formula1Data:
    """Data class to store Formula 1 data from 1950 to 2022"""

    circuits: pd.DataFrame | None = None
    constructor_results: pd.DataFrame | None = None
    constructor_standings: pd.DataFrame | None = None
    constructors: pd.DataFrame | None = None
    driver_standings: pd.DataFrame | None = None
    drivers: pd.DataFrame | None = None
    lap_times: pd.DataFrame | None = None
    pit_stops: pd.DataFrame | None = None
    qualifying: pd.DataFrame | None = None
    races: pd.DataFrame | None = None
    results: pd.DataFrame | None = None
    seasons: pd.DataFrame | None = None
    sprint_results: pd.DataFrame | None = None
    status: pd.DataFrame | None = None


def load_csv_data(data_name: str) -> pd.DataFrame:
    """Load CSV data from S3 bucket with the given prefix"""

    s3 = boto3.resource(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )

    content_object = s3.Object(
        bucket_name=BUCKET_NAME,
        key=f"{DICT_CONFIG['s3_past_data_key']}/{data_name}.csv",
    )
    csv_content = content_object.get()["Body"].read().decode("utf-8")
    df = pd.read_csv(StringIO(csv_content))

    return df
