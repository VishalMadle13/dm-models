from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

# Assuming you have a dataset X
# X should be a 2D array where each row represents a data point, and each column represents a feature

# Create a DBSCAN instance
dbscan = DBSCAN(eps=0.5, min_samples=5)

# Fit the model to your data
dbscan.fit(X)

# Get cluster assignments and labels (-1 for noise)
labels = dbscan.labels_

# Visualize the clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.5)
plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
