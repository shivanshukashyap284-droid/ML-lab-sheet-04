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
a=KMeans(n_clusters=3,random_state=42,n_init=10).fit_predict(Xs); b=AgglomerativeClustering(n_clusters=3,linkage="ward").fit_predict(Xs); print("K-Means:",silhouette_score(Xs,a)); print("Hierarchical:",silhouette_score(Xs,b)); print("Higher score = better separation.")
