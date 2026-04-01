#!/bin/bash
echo "=== Sales Analytics Pipeline Starting ==="
cd /home/ec2-user/sales-cloud-analytics

echo "Step 1: Feature engineering"
python3 scripts/feature_engineering.py

echo "Step 2: Model training"
python3 scripts/model_training.py

echo "Step 3: Model evaluation"
python3 scripts/model_evaluation.py

echo "Step 4: Tableau export"
python3 scripts/tableau_export.py

echo "Step 5: Upload results to S3"
python3 cloud/aws_setup.py

echo "=== Pipeline Complete ==="