import os
from datetime import datetime

from current_data_retriever import retrieve_data, save_data

BUCKET_NAME = os.environ.get("BUCKET_NAME")
S3_CURRENT_DATA_KEY = os.environ.get("S3_CURRENT_DATA_KEY")
SEASON = datetime.now().year


def lambda_handler(event, context):

    df = retrieve_data(SEASON)
    save_data(
        df=df,
        season=SEASON,
        s3_current_data_key=S3_CURRENT_DATA_KEY,
        bucket_name=BUCKET_NAME,
    )

    return {"statusCode": 200, "body": "Success"}
