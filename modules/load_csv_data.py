from io import StringIO

import boto3
import pandas as pd

dict_dtype_columns = {
    "position": "Int64",
    "car_number": str,
    "driver": str,
    "driver_avvreviation": str,
    "constructor": str,
    "q1": str,
    "q2": str,
    "q3": str,
    "q1_sec": float,
    "q2_sec": float,
    "q3_sec": float,
    "season": int,
    "round": int,
    "grandprix": str,
    "laps": "Int64",
    "time": str,
    "points": float,
    "time_sec": float,
    "lap": "Int64",
    "avg_speed": float,
    "stops": "Int64",
    "month": int,
    "date": int,
}


def load_csv_data(
    bucket_name: str,
    key: str,
) -> pd.DataFrame:
    """Load CSV data from S3 bucket with the given prefix"""

    s3 = boto3.resource("s3")
    content_object = s3.Object(
        bucket_name=bucket_name,
        key=key,
    )
    csv_content = content_object.get()["Body"].read().decode("utf-8")
    df = pd.read_csv(StringIO(csv_content), dtype=dict_dtype_columns)

    return df
