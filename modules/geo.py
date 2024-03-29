import json
from functools import cache
from typing import Any

import boto3


@cache
def load_geo_data(
    geo_file_name: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    bucket_name: str,
    s3_geo_data_key: str,
) -> dict[str, Any]:
    """Load geo data of the grand prix circuit"""

    s3 = boto3.resource(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    content_object = s3.Object(
        bucket_name=bucket_name,
        key=f"{s3_geo_data_key}/{geo_file_name}",
    )
    file_content = content_object.get()["Body"].read().decode("utf-8")
    geojson_data = json.loads(file_content)

    return geojson_data
