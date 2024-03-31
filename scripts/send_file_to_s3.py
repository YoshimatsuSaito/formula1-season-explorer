import argparse
import os

import boto3
from dotenv import load_dotenv

load_dotenv()


def upload_file_to_s3(local_file_path: str, object_name: str) -> None:
    """Upload file to S3"""
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    s3_client.upload_file(local_file_path, os.getenv("BUCKET_NAME"), object_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("local_file_path")
    parser.add_argument("object_name")
    args = parser.parse_args()
    upload_file_to_s3(args.local_file_path, args.object_name)
