import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA

# ----------------------------------------------------------
# Paths & basic config
# ----------------------------------------------------------
model_name = "umberto"
input_csv = f"df_fse_hdbscan_grid_2.csv"
output_csv = f"df_fse_hdb_kmeans_agg.csv"

df = pd.read_csv(input_csv)

# Identify embedding columns (UmBERTo vectors)
embedding_cols = [c for c in df.columns if c.startswith("U") and c[1:2].isdigit()]
print(f"Found {len(embedding_cols)} embedding dimensions.")

# Dimensionality reduction
X = df[embedding_cols].values
pca = PCA(n_components=50, random_state=42)
X_red = pca.fit_transform(X)
print("Dimensionality reduction completed:", X_red.shape)

# Identify HDBSCAN cluster columns
hdb_cols = [c for c in df.columns if c.startswith("hdb_")]

# Count effective clusters (ignoring noise label -1)
def n_clusters(labels):
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    return len(uniq[uniq != -1])

hdb_cluster_counts = [n_clusters(df[c]) for c in hdb_cols]

k_min = int(np.min(hdb_cluster_counts))
k_med = int(np.median(hdb_cluster_counts))
k_max = int(np.max(hdb_cluster_counts))

print("HDBSCAN cluster counts:", hdb_cluster_counts)
print("k_min =", k_min, "k_med =", k_med, "k_max =", k_max)

baseline_cols = []

for k in [k_min, k_med, k_max]:

    # ---------------------------
    # K-means
    # ---------------------------
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    col_km = f"kmeans_k{k}"
    df[col_km] = km.fit_predict(X_red)
    baseline_cols.append(col_km)

    # ---------------------------
    # Agglomerative clustering (Ward)
    # ---------------------------
    agg = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward",
        metric="euclidean"
    )
    col_agg = f"agg_ward_k{k}"
    df[col_agg] = agg.fit_predict(X_red)
    baseline_cols.append(col_agg)

print("Added baseline cluster columns:", baseline_cols)

df.to_csv(output_csv, index=False)
print("Saved with k-means + hierarchical columns to:", output_csv)
