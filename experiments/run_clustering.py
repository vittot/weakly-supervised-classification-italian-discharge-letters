import sys
import pandas as pd
from pipeline.diagnosis_clustering import load_umberto, add_bert_embeddings, cluster_hdbscan
from sklearn.decomposition import PCA
import os
from tqdm import tqdm

# -------------------
# Config
# -------------------
input_csv = "df_fse_embeddings.csv"
weak_label_col = "our_bronchio"
gold_label_col = "bronchiolite"
text_col = "testo_clean"
output_csv = f"df_fse_hdbscan_grid.csv"

# grid of hyperparameters
min_cluster_sizes = [5, 10, 20]
min_samples_list = [1, 5, 10]
cluster_selection_epsilon = 0.0

# model name from command line
model_name = sys.argv[1]


# -------------------
# Load data
# -------------------
if os.path.exists(output_csv):
    print(f"Loading existing file with clusters: {output_csv}")
    df = pd.read_csv(output_csv)
else:
    print(f"Loading original file: {input_csv}")
    df = pd.read_csv(input_csv)

# -------------------
# Embeddings
# -------------------
embedding_cols = [c for c in df.columns if c.startswith("U") and c[1:2].isdigit()]

if len(embedding_cols) == 0:
    print("No embedding columns found, computing BERT embeddings...")
    model, tokenizer = load_umberto(model_name)
    df = add_bert_embeddings(model, tokenizer, df, text_col)
    # update embedding_cols after adding embeddings
    embedding_cols = [c for c in df.columns if c.startswith("U")]
else:
    print(f"Found {len(embedding_cols)} embedding columns, reusing them.")

# -------------------
# Dimensionality reduction
# -------------------
print("Running Dimensionality Reduction...")
X = df[embedding_cols]
pca = PCA(n_components=50)
X_red = pca.fit_transform(X)
# keep as DataFrame to preserve index alignment
X_red = pd.DataFrame(X_red, index=df.index)

print("Starting HDBSCAN grid search...")

# -------------------
# Grid search over (min_cluster_size, min_samples)
# -------------------
for mcs in tqdm(min_cluster_sizes):
    for ms in tqdm(min_samples_list):
        cluster_col = f"hdb_mcs{mcs}_ms{ms}"

        # Skip if we already computed this configuration
        if cluster_col in df.columns:
            print(f"Skipping {cluster_col}: already present in df.")
            continue

        print(f"\n=== Running HDBSCAN with min_cluster_size={mcs}, min_samples={ms} ===")
        msc_end = 5
        df = cluster_hdbscan(
            X_red,
            df,
            text_col,
            mcs,                  
            msc_end,                  
            cluster_col,          
            ms,                  
            cluster_selection_epsilon
        )

        df.to_csv(output_csv, index=False)
        print(f"Saved df with column '{cluster_col}' to {output_csv}")

print("\nFinished all HDBSCAN grid runs!")
