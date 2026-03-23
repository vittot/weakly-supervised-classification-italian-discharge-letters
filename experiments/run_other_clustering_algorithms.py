import argparse
import os
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA

from utils.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run K-means and Agglomerative baselines using k derived from HDBSCAN outputs."
    )

    parser.add_argument(
        "--input_csv",
        type=str,
        default="df_fse_hdbscan_grid_2.csv",
        help="Input CSV containing embeddings and HDBSCAN cluster columns."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="df_fse_hdb_kmeans_agg.csv",
        help="Output CSV with added baseline clustering columns."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed."
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=50,
        help="Number of PCA components."
    )
    parser.add_argument(
        "--hdb_prefix",
        type=str,
        default="hdb_",
        help="Prefix identifying HDBSCAN cluster columns."
    )
    parser.add_argument(
        "--k_values",
        type=str,
        nargs="+",
        default=["min", "med", "max"],
        help="Which k values to use among: min med max or explicit integers, e.g. --k_values 10 20 30"
    )
    parser.add_argument(
        "--overwrite_existing",
        action="store_true",
        help="Recompute baseline columns even if they already exist."
    )

    return parser.parse_args()


def n_clusters(labels):
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    return len(uniq[uniq != -1])


def resolve_k_values(hdb_cluster_counts, requested_k_values):
    k_min = int(np.min(hdb_cluster_counts))
    k_med = int(np.median(hdb_cluster_counts))
    k_max = int(np.max(hdb_cluster_counts))

    lookup = {
        "min": k_min,
        "med": k_med,
        "median": k_med,
        "max": k_max,
    }

    resolved = []
    for item in requested_k_values:
        item_lower = str(item).lower()
        if item_lower in lookup:
            resolved.append(lookup[item_lower])
        else:
            resolved.append(int(item))

    # remove duplicates while preserving order
    seen = set()
    out = []
    for k in resolved:
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out, k_min, k_med, k_max


def main():
    args = parse_args()
    set_seed(args.seed)

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Input file not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)

    # Identify embedding columns
    embedding_cols = [c for c in df.columns if c.startswith("U") and c[1:].isdigit()]
    if len(embedding_cols) == 0:
        raise ValueError("No embedding columns found. Expected columns like U0, U1, ...")

    print(f"Found {len(embedding_cols)} embedding dimensions.")

    # PCA
    X = df[embedding_cols].values
    pca = PCA(n_components=args.n_components)
    X_red = pca.fit_transform(X)
    print("Dimensionality reduction completed:", X_red.shape)

    # HDBSCAN columns
    hdb_cols = [c for c in df.columns if c.startswith(args.hdb_prefix)]
    if len(hdb_cols) == 0:
        raise ValueError(f"No HDBSCAN columns found with prefix '{args.hdb_prefix}'.")

    hdb_cluster_counts = [n_clusters(df[c]) for c in hdb_cols]
    print("HDBSCAN cluster counts:", dict(zip(hdb_cols, hdb_cluster_counts)))

    k_values, k_min, k_med, k_max = resolve_k_values(hdb_cluster_counts, args.k_values)
    print("k_min =", k_min, "k_med =", k_med, "k_max =", k_max)
    print("Using k values:", k_values)

    baseline_cols = []

    for k in k_values:
        # ---------------------------
        # K-means
        # ---------------------------
        col_km = f"kmeans_k{k}"
        if col_km in df.columns and not args.overwrite_existing:
            print(f"Skipping {col_km}: already present.")
        else:
            km = KMeans(n_clusters=k, random_state=args.seed, n_init="auto")
            df[col_km] = km.fit_predict(X_red)
            baseline_cols.append(col_km)

        # ---------------------------
        # Agglomerative clustering (Ward)
        # ---------------------------
        col_agg = f"agg_ward_k{k}"
        if col_agg in df.columns and not args.overwrite_existing:
            print(f"Skipping {col_agg}: already present.")
        else:
            agg = AgglomerativeClustering(
                n_clusters=k,
                linkage="ward",
                metric="euclidean"
            )
            df[col_agg] = agg.fit_predict(X_red)
            baseline_cols.append(col_agg)

    print("Added baseline cluster columns:", baseline_cols)

    df.to_csv(args.output_csv, index=False)
    print("Saved with k-means + hierarchical columns to:", args.output_csv)


if __name__ == "__main__":
    main()