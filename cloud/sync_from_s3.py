import boto3
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from aws_config import BUCKET_NAME, REGION, S3_PATHS

def sync_from_s3():
    s3 = boto3.client("s3", region_name=REGION)
    downloads = {
        S3_PATHS["tableau"]: "scripts/dataclean/tableau_ready.csv",
    }
    for s3_key, local_path in downloads.items():
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(BUCKET_NAME, s3_key, local_path)
        print(f"Downloaded: s3://{BUCKET_NAME}/{s3_key} → {local_path}")
    print("\nSync complete.")

if __name__ == "__main__":
    sync_from_s3()