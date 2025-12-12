import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

file_path = r"G:\task02_customer_clustering\Mall_Customers.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"CSV file not found at: {file_path}")

df = pd.read_csv(file_path)
print("Dataset Loaded Successfully!")
print(df.head())

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method to Determine Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)

print("Clustering Completed!")
print(df.head())

plt.figure(figsize=(8, 6))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=df['Cluster'], cmap='viridis', s=80, alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=300, c='red', marker='X', label='Centroids')
plt.title('Customer Segments (K-Means Clustering)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.show()

output_file = r"G:\task02_customer_clustering\clustered_customers.csv"
df.to_csv(output_file, index=False)
print("clustered_customers.csv saved successfully!")



