# ================================
# Feature Engineering Script (SAFE)
# ================================

import pandas as pd

# Load dataset
df = pd.read_csv("scripts/dataclean/orders_cleaned.csv")

print("Original Shape:", df.shape)
print("Columns:", df.columns)

# -------------------------------
# 1. Convert Dates
# -------------------------------

if "Order Date" in df.columns:
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")

if "Ship Date" in df.columns:
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")

# -------------------------------
# 2. Create Features
# -------------------------------

# Shipping Days
if "Order Date" in df.columns and "Ship Date" in df.columns:
    df["Shipping_Days"] = (df["Ship Date"] - df["Order Date"]).dt.days

# Time Features
if "Order Date" in df.columns:
    df["Order_Month"] = df["Order Date"].dt.month
    df["Order_Year"] = df["Order Date"].dt.year

# Interaction Feature (SAFE)
if "Quantity" in df.columns:
    df["Sales_per_Quantity"] = df["Sales"] / (df["Quantity"] + 1)

# -------------------------------
# 3. Clean Data
# -------------------------------

df = df.drop(columns=["Order Date", "Ship Date"], errors="ignore")

# Remove outliers
df = df[df["Sales"] < df["Sales"].quantile(0.99)]

# Drop missing values
df = df.dropna()

print("After Feature Engineering:", df.shape)

# -------------------------------
# 4. Save Dataset
# -------------------------------

df.to_csv("scripts/dataclean/orders_engineered.csv", index=False)

print("Feature engineered dataset saved successfully!")