# ==========================================
# Customer Segmentation Pipeline
# ==========================================

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ==========================================
# 1. Load Dataset
# ==========================================

df = pd.read_csv("scripts/dataclean/orders_cleaned.csv")

print("Initial Shape:", df.shape)

# ==========================================
# 2. Preprocessing
# ==========================================

# Convert dates
df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")

# Remove missing
df = df.dropna()

# ==========================================
# 3. Feature Engineering (RFM-lite)
# ==========================================

# Total sales per customer
sales = df.groupby("Customer ID")["Sales"].sum().reset_index()

# Frequency (number of orders)
freq = df.groupby("Customer ID").size().reset_index(name="Frequency")

# Recency
last_order = df.groupby("Customer ID")["Order Date"].max().reset_index()
last_date = df["Order Date"].max()

last_order["Recency"] = (last_date - last_order["Order Date"]).dt.days
last_order = last_order.drop(columns=["Order Date"])

# Merge all features
customer_df = sales.merge(freq, on="Customer ID")
customer_df = customer_df.merge(last_order, on="Customer ID")

print("Customer-level data:", customer_df.shape)

# ==========================================
# 4. Scaling
# ==========================================

features = ["Sales", "Frequency", "Recency"]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_df[features])

# ==========================================
# 5. K-Means Clustering
# ==========================================

kmeans = KMeans(n_clusters=4, random_state=42)
customer_df["Cluster"] = kmeans.fit_predict(scaled_data)

# ==========================================
# 6. Save Output
# ==========================================

customer_df.to_csv("scripts/dataclean/customer_segments.csv", index=False)

print("Segmentation complete. File saved.")