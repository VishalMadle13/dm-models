from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Assuming you have a dataset X
# X should be a 2D array where each row represents a data point, and each column represents a feature

# Choose the number of clusters (K)
k = 3

# Create a KMeans instance
kmeans = KMeans(n_clusters=k, random_state=42)

# Fit the model to your data
kmeans.fit(X)

# Get cluster assignments and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Visualize the clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200)
plt.title(f'K-Means Clustering with {k} clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
