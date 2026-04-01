import boto3
import pandas as pd
from io import StringIO
import os
import sys
sys.path.append(os.path.dirname(__file__))
from aws_config import BUCKET_NAME, REGION, S3_PATHS

def load_from_s3(s3_key):
    s3 = boto3.client("s3", region_name=REGION)
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=s3_key)
    df = pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))
    print(f"Loaded from S3: {s3_key} — shape: {df.shape}")
    return df

def save_to_s3(df, s3_key):
    s3 = boto3.client("s3", region_name=REGION)
    buf = StringIO()
    df.to_csv(buf, index=False)
    s3.put_object(Bucket=BUCKET_NAME, Key=s3_key, Body=buf.getvalue())
    print(f"Saved to S3: s3://{BUCKET_NAME}/{s3_key}")

if __name__ == "__main__":
    df = load_from_s3(S3_PATHS["raw"])
    print(df.head())