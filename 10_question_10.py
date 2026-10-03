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
vals=[]
for k in range(1,9): vals.append(KMeans(n_clusters=k,random_state=42,n_init=10).fit(Xs).inertia_)
print([round(v,3) for v in vals])
plt.plot(range(1,9),vals,marker="o"); plt.xlabel("K"); plt.ylabel("Inertia"); plt.title("Elbow Method"); plt.show()
