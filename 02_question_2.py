import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

df = pd.read_csv("iris_clustering_dataset.csv")
X = df.drop(columns=["species", "target"])
Xs = StandardScaler().fit_transform(X)
print("First five:\n",df.head()); print("\nLast five:\n",df.tail())
