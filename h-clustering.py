import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets._samples_generator import make_blobs
from scipy.cluster.hierarchy import linkage, dendrogram, inconsistent
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

plt.rcParams["figure.figsize"] = (12, 8)
sns.set_theme()

X, _ = make_blobs(n_samples=2000, centers=7, cluster_std=0.5)

plt.scatter(X[:, 0], X[:, 1], s=50)
plt.show()

links = linkage(X, method="ward")

pd.DataFrame(links)

dendrogram(links)

hc = AgglomerativeClustering(n_clusters=7)
Y = hc.fit_predict(X)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap="viridis", edgecolors="black")
plt.show()

# shiluette validation
silhouette = silhouette_score(X, Y)
print("Silhouette score:", silhouette)

# iteration on methods
methods = ["single", "complete", "average", "weighted", "centroid", "median"]

for method in methods:
    links = linkage(X, method=method)

    hc = AgglomerativeClustering(n_clusters=3)
    Y = hc.fit_predict(X)

    silhouette = silhouette_score(X, Y)
    print(f"Method: {method} / Score: {silhouette}")
