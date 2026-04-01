BUCKET_NAME = "sales-cloud-analytics-anshika"
REGION = "us-east-1"

S3_PATHS = {
    "raw":         "data/raw/orders_cleaned.csv",
    "predictions": "data/processed/sales_predictions_final.csv",
    "segments":    "data/processed/customer_segments.csv",
    "tableau":     "data/processed/tableau_ready.csv",
}