import numpy as np
import pandas as pd
from sklearn.preprocessing   import LabelEncoder
from sklearn.model_selection import train_test_split

TARGET_RAW = "Sales"
TARGET_LOG = "log_Sales"
SEED       = 42


def run(input_path: str):

    print("=" * 55)
    print("  STEP 1 — FEATURE ENGINEERING")
    print("=" * 55)

    # ── Load ─────────────────────────────────
    df = pd.read_csv(input_path)
    print(f"\n  Loaded  :  {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"  Sales skew (raw) :  {df[TARGET_RAW].skew():.2f}  ← needs fixing")

    # ── Fix 11 missing Postal Codes ──────────
    df["Postal Code"] = df["Postal Code"].fillna(df["Postal Code"].median())

    # ── Outlier cap  (IQR × 3) ───────────────
    q1, q3 = df[TARGET_RAW].quantile(0.25), df[TARGET_RAW].quantile(0.75)
    cap_hi  = q3 + 3.0 * (q3 - q1)
    n_capped = (df[TARGET_RAW] > cap_hi).sum()
    df[TARGET_RAW] = df[TARGET_RAW].clip(upper=cap_hi)
    print(f"\n  Outlier cap  :  upper = ${cap_hi:,.2f}  ({n_capped} rows capped)")

    # ── Log-transform target ─────────────────
    df[TARGET_LOG] = np.log1p(df[TARGET_RAW])
    print(f"  Skew after log   :  {df[TARGET_LOG].skew():.3f}  ← much better")

    # ── Date features ────────────────────────
    df["Order Date"] = pd.to_datetime(df["Order Date"],
                                      infer_datetime_format=True, errors="coerce")
    df["Ship Date"]  = pd.to_datetime(df["Ship Date"],
                                      infer_datetime_format=True, errors="coerce")

    df["Order_Year"]       = df["Order Date"].dt.year
    df["Order_Month"]      = df["Order Date"].dt.month
    df["Order_Quarter"]    = df["Order Date"].dt.quarter
    df["Order_DayOfWeek"]  = df["Order Date"].dt.dayofweek
    df["Shipping_Days"]    = (df["Ship Date"] - df["Order Date"]).dt.days.clip(lower=0)
    df["Is_HolidaySeason"] = df["Order_Month"].isin([11, 12]).astype(int)
    df["Is_BackToSchool"]  = df["Order_Month"].isin([8, 9]).astype(int)
    df["Is_Q1_Slow"]       = df["Order_Month"].isin([1, 2]).astype(int)

    # ── Label encode low-cardinality cols ────
    le = LabelEncoder()
    for col in ["Category", "Sub-Category", "Segment",
                "Region", "Ship Mode", "State"]:
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))

    # ── Frequency encode high-cardinality cols
    for col in ["City", "Product Name", "Customer Name"]:
        freq = df[col].value_counts(normalize=True)
        df[col + "_freq"] = df[col].map(freq)

    # ── Interaction features ─────────────────
    df["Region_x_Category"] = df["Region_enc"]    * df["Category_enc"]
    df["Segment_x_SubCat"]  = df["Segment_enc"]   * df["Sub-Category_enc"]
    df["Month_x_Category"]  = df["Order_Month"]   * df["Category_enc"]
    df["Quarter_x_Segment"] = df["Order_Quarter"] * df["Segment_enc"]

    # ── Drop raw / ID columns ────────────────
    drop_cols = [
        "Row ID", "Order ID", "Customer ID", "Product ID",
        "Order Date", "Ship Date",
        "Customer Name", "Product Name", "City",   # replaced by _freq
        "Country",                                  # only 1 value: United States
        "Category", "Sub-Category", "Segment",      # replaced by _enc
        "Region", "Ship Mode", "State",             # replaced by _enc
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    feature_cols = [c for c in df.columns if c not in [TARGET_RAW, TARGET_LOG]]
    # fix: fill any NaNs produced by frequency encoding or date parsing
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
    print(f"\n  Features ({len(feature_cols)}) :  {feature_cols}")

    # ── Train / test split ───────────────────
    X = df[feature_cols].values
    y = df[TARGET_LOG].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    y_test_dollars = np.expm1(y_test)

    print(f"\n  Train : {X_train.shape[0]:,}  |  Test : {X_test.shape[0]:,}")
    print("\n  Feature engineering complete ✓\n")

    return X_train, X_test, y_train, y_test, y_test_dollars, feature_cols


if __name__ == "__main__":
    # standalone test
    import os
    BASE = os.path.dirname(os.path.abspath(__file__))
    run(os.path.join(BASE, "dataclean", "orders_cleaned.csv"))