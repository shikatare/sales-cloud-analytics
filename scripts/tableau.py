import pandas as pd
import os

DATA_DIR = "scripts/dataclean"

orders   = pd.read_csv(f"{DATA_DIR}/orders_cleaned.csv")
preds    = pd.read_csv(f"{DATA_DIR}/sales_predictions_final.csv")
segments = pd.read_csv(f"{DATA_DIR}/customer_segments.csv")

print("ORDER COLUMNS:   ", list(orders.columns))
print("PRED COLUMNS:    ", list(preds.columns))
print("SEGMENT COLUMNS: ", list(segments.columns))
print(f"\nOrders rows: {len(orders)}")
print(f"Predictions rows: {len(preds)}")
print(f"Segments rows: {len(segments)}")

# find the predicted sales column name
pred_col = [c for c in preds.columns if "predict" in c.lower()]
if not pred_col:
    pred_col = [c for c in preds.columns if "sales" in c.lower()]
pred_col = pred_col[0] if pred_col else preds.columns[-1]
print(f"\nUsing prediction column: {pred_col}")

# find actual sales column in predictions file
actual_col = [c for c in preds.columns if "actual" in c.lower()]
if actual_col:
    actual_col = actual_col[0]
    print(f"Found actual column: {actual_col}")

# check if predictions file has Order ID to merge on
order_id_col = [c for c in preds.columns if "order" in c.lower() and "id" in c.lower()]

if order_id_col:
    # merge on Order ID
    print(f"Merging on: {order_id_col[0]}")
    preds_slim = preds[[order_id_col[0], pred_col]].rename(
        columns={pred_col: "Predicted_Sales"}
    )
    orders = orders.merge(preds_slim, on=order_id_col[0], how="left")

else:
    # predictions are test-set only — add as NaN for train rows, value for test rows
    # use the row_index column if it exists
    row_col = [c for c in preds.columns if "index" in c.lower() or "row" in c.lower()]
    
    if row_col:
        print(f"Merging on row index column: {row_col[0]}")
        preds_slim = preds[[row_col[0], pred_col]].rename(
            columns={row_col[0]: "_row_idx", pred_col: "Predicted_Sales"}
        )
        orders["_row_idx"] = orders.index
        orders = orders.merge(preds_slim, on="_row_idx", how="left")
        orders.drop(columns=["_row_idx"], inplace=True)
    else:
        # last resort — just mark test rows by position (last 20%)
        print("No join key found — mapping predictions to last 20% of orders")
        orders["Predicted_Sales"] = None
        test_start = len(orders) - len(preds)
        orders.loc[test_start:, "Predicted_Sales"] = preds[pred_col].values

# merge segments on Customer ID
if "Customer ID" in segments.columns and "Customer ID" in orders.columns:
    cluster_cols = [c for c in segments.columns if "cluster" in c.lower()]
    cluster_col = cluster_cols[0] if cluster_cols else "Cluster"
    orders = orders.merge(
        segments[["Customer ID", cluster_col]].rename(columns={cluster_col: "Cluster"}),
        on="Customer ID",
        how="left"
    )
    print(f"Segments merged. Unique clusters: {orders['Cluster'].nunique()}")

# save
out_path = f"{DATA_DIR}/tableau_ready.csv"
orders.to_csv(out_path, index=False)
print(f"\nDone! Saved to {out_path}")
print(f"Final shape: {orders.shape}")
print(f"Final columns: {list(orders.columns)}")
print(f"Predicted_Sales non-null: {orders['Predicted_Sales'].notna().sum()}")