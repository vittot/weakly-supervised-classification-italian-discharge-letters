import sys
import argparse
import pandas as pd
from pipeline.diagnosis_clustering import load_umberto, add_bert_embeddings, cluster_hdbscan
from sklearn.decomposition import PCA
import os
from tqdm import tqdm
from utils.utils import set_seed


# -------------------
# Argument parser
# -------------------
def parse_args():
    parser = argparse.ArgumentParser(description="HDBSCAN clustering grid search")

    # --- core ---
    parser.add_argument("model_name", type=str,
                        help="HuggingFace model name for embeddings")

    parser.add_argument("--input_csv", type=str, default="df_fse_embeddings.csv")
    parser.add_argument("--output_csv", type=str, default="df_fse_hdbscan_grid.csv")
    parser.add_argument("--text_col", type=str, default="testo_clean")

    # --- clustering grid ---
    parser.add_argument("--min_cluster_sizes", type=int, nargs="+", default=[5, 10, 20],
                        help="List of min_cluster_size values")
    parser.add_argument("--min_samples", type=int, nargs="+", default=[1, 5, 10],
                        help="List of min_samples values")

    parser.add_argument("--cluster_selection_epsilon", type=float, default=0.0)
    parser.add_argument("--min_cluster_size_end", type=int, default=5)

    # --- PCA ---
    parser.add_argument("--n_components", type=int, default=100)

    # --- misc ---
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--force_recompute_embeddings", action="store_true",
                        help="Recompute embeddings even if present")

    return parser.parse_args()


# -------------------
# Main
# -------------------
def main():
    args = parse_args()

    set_seed(args.seed)

    # -------------------
    # Load data
    # -------------------
    if os.path.exists(args.output_csv):
        print(f"Loading existing file with clusters: {args.output_csv}")
        df = pd.read_csv(args.output_csv)
    else:
        print(f"Loading original file: {args.input_csv}")
        df = pd.read_csv(args.input_csv)

    # -------------------
    # Embeddings
    # -------------------
    embedding_cols = [c for c in df.columns if c.startswith("U") and c[1:].isdigit()]

    if len(embedding_cols) == 0 or args.force_recompute_embeddings:
        print("Computing BERT embeddings...")
        model, tokenizer = load_umberto(args.model_name)
        df = add_bert_embeddings(model, tokenizer, df, args.text_col)
        embedding_cols = [c for c in df.columns if c.startswith("U") and c[1:].isdigit()]
    else:
        print(f"Found {len(embedding_cols)} embedding columns, reusing them.")

    # -------------------
    # PCA
    # -------------------
    print("Running dimensionality reduction...")
    X = df[embedding_cols]
    pca = PCA(n_components=args.n_components)
    X_red = pca.fit_transform(X)
    X_red = pd.DataFrame(X_red, index=df.index)

    print("Starting HDBSCAN grid search...")

    # -------------------
    # Grid search
    # -------------------
    for mcs in tqdm(args.min_cluster_sizes, desc="min_cluster_size"):
        for ms in tqdm(args.min_samples, desc="min_samples", leave=False):

            cluster_col = f"hdb_mcs{mcs}_ms{ms}"
            summary_col = f"summary_{cluster_col}"

            if cluster_col in df.columns and summary_col in df.columns:
                print(f"Skipping {cluster_col}: already computed.")
                continue

            print(f"\n=== Running HDBSCAN mcs={mcs}, ms={ms} ===")

            df = cluster_hdbscan(
                X_red=X_red,
                df=df,
                text_col=args.text_col,
                min_cluster_size_start=mcs,
                min_cluster_size_end=args.min_cluster_size_end,
                cluster_col=cluster_col,
                min_samples=ms,
                cluster_selection_epsilon=args.cluster_selection_epsilon,
            )

            df.to_csv(args.output_csv, index=False)
            print(f"Saved results to {args.output_csv}")

    print("\nFinished all HDBSCAN grid runs!")


if __name__ == "__main__":
    main()