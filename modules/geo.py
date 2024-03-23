import json
import os
import sys
from functools import cache
from pathlib import Path
from typing import Any

import boto3

current_dir = Path(__file__).resolve().parent
sys.path.append(str(Path(current_dir).parent))

from modules.utils import load_config

BUCKET_NAME = os.environ.get("BUCKET_NAME")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")


@cache
def load_geo_data(gp_name: str) -> dict[str, Any]:
    """Load geo data of the grand prix circuit"""

    s3 = boto3.resource(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
    dict_config = load_config(Path(current_dir.parent) / "config/config.yml")

    content_object = s3.Object(
        bucket_name=BUCKET_NAME, key=f"{dict_config['s3_key']}/{dict_config['gp_circuits'][gp_name]}"
    )
    file_content = content_object.get()["Body"].read().decode("utf-8")
    geojson_data = json.loads(file_content)

    return geojson_data
