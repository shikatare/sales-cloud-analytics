import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("scripts/dataclean/customer_segments.csv")

features = ["Sales", "Frequency", "Recency"]

scaler = StandardScaler()
scaled = scaler.fit_transform(df[features])

wcss = []

for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled)
    wcss.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, 10), wcss)
plt.title("Elbow Method")
plt.xlabel("Clusters")
plt.ylabel("WCSS")
plt.show()