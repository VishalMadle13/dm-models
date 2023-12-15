from sklearn_extra.cluster import KMedoids
import numpy as np
import matplotlib.pyplot as plt

# Assuming you have a dataset X
# X should be a 2D array where each row represents a data point, and each column represents a feature

# Choose the number of clusters (K)
k = 3

# Create a KMedoids instance
kmedoids = KMedoids(n_clusters=k, random_state=42)

# Fit the model to your data
kmedoids.fit(X)

# Get cluster assignments and medoids
labels = kmedoids.labels_
medoids = kmedoids.medoid_indices_

# Visualize the clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.5)
plt.scatter(X[medoids, 0], X[medoids, 1], c='red', marker='X', s=200)
plt.title(f'K-Medoids Clustering with {k} clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
