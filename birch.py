from sklearn.cluster import Birch
import numpy as np
import matplotlib.pyplot as plt

# Assuming you have a dataset X
# X should be a 2D array where each row represents a data point, and each column represents a feature

# Create a BIRCH instance
birch = Birch(threshold=0.5, n_clusters=3)

# Fit the model to your data
birch.fit(X)

# Get cluster assignments
labels = birch.labels_

# Visualize the clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.5)
plt.title('BIRCH Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
