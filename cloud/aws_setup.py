import boto3
import os
import sys
sys.path.append(os.path.dirname(__file__))
from aws_config import BUCKET_NAME, REGION, S3_PATHS

def create_bucket():
    s3 = boto3.client("s3", region_name=REGION)
    try:
        s3.create_bucket(Bucket=BUCKET_NAME)
        print(f"Bucket created: {BUCKET_NAME}")
    except Exception as e:
        print(f"Bucket note: {e}")

def upload_all():
    s3 = boto3.client("s3", region_name=REGION)
    local_files = {
        S3_PATHS["raw"]:         "scripts/dataclean/orders_cleaned.csv",
        S3_PATHS["predictions"]: "scripts/dataclean/sales_predictions_final.csv",
        S3_PATHS["segments"]:    "scripts/dataclean/customer_segments.csv",
        S3_PATHS["tableau"]:     "scripts/dataclean/tableau_ready.csv",
    }
    for s3_key, local_path in local_files.items():
        print(f"Checking: {local_path}")
        if os.path.exists(local_path):
            print(f"Uploading: {local_path}") 
            s3.upload_file(local_path, BUCKET_NAME, s3_key)
            print(f"Uploaded: {local_path} → s3://{BUCKET_NAME}/{s3_key}")
        else:
            print(f"Not found: {local_path}")

if __name__ == "__main__":
    create_bucket()
    upload_all()
    print("\nAll files uploaded to S3")