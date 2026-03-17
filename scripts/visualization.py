import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("scripts/dataclean/customer_segments.csv")

# ---------------------------
# 1. Cluster distribution
# ---------------------------
plt.figure()
df["Cluster"].value_counts().plot(kind="bar")
plt.title("Customers per Cluster")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.show()

# ---------------------------
# 2. Sales per cluster
# ---------------------------
plt.figure()
df.groupby("Cluster")["Sales"].mean().plot(kind="bar")
plt.title("Average Sales per Cluster")
plt.xlabel("Cluster")
plt.ylabel("Sales")
plt.show()

# ---------------------------
# 3. Frequency per cluster
# ---------------------------
plt.figure()
df.groupby("Cluster")["Frequency"].mean().plot(kind="bar")
plt.title("Average Frequency per Cluster")
plt.xlabel("Cluster")
plt.ylabel("Frequency")
plt.show()

# ---------------------------
# 4. Recency per cluster
# ---------------------------
plt.figure()
df.groupby("Cluster")["Recency"].mean().plot(kind="bar")
plt.title("Average Recency per Cluster")
plt.xlabel("Cluster")
plt.ylabel("Days")
plt.show()