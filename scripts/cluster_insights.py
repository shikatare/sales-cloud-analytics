import pandas as pd

df = pd.read_csv("scripts/dataclean/customer_segments.csv")

# Only numeric aggregation
summary = df.groupby("Cluster")[["Sales", "Frequency", "Recency"]].mean()

# ---------------------------
# Custom labeling (DATA-DRIVEN)
# ---------------------------
def label_cluster(row):
    if row["Sales"] > 9000 and row["Frequency"] > 20:
        return "Gold Customers (High Value)"
    
    elif row["Sales"] > 3000 and row["Frequency"] > 15:
        return "Silver Customers (Loyal)"
    
    elif row["Recency"] > 400:
        return "Lost Customers"
    
    else:
        return "Occasional Customers"

summary["Segment"] = summary.apply(label_cluster, axis=1)

# ---------------------------
# Add Business Insight Column
# ---------------------------
def business_action(segment):
    if "Gold" in segment:
        return "Retain with premium offers and loyalty rewards"
    
    elif "Silver" in segment:
        return "Upsell and convert to high-value customers"
    
    elif "Lost" in segment:
        return "Run re-engagement campaigns"
    
    else:
        return "Target with promotions to increase frequency"

summary["Business_Action"] = summary["Segment"].apply(business_action)

print("\n===== FINAL INSIGHTS =====")
print(summary)

# Save
summary.to_csv("scripts/dataclean/cluster_summary_final.csv")