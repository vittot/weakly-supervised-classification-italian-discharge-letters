import os
import sys
import pandas as pd

from classification import run_experiment_all_weak_sources_100, setup_dataset

df = pd.read_csv('df_fse_hdb_kmeans_agg.csv')

weak_label_col = 'our_bronchio'
gold_label_col = 'bronchiolite'
text_col = 'testo_clean'
model = sys.argv[1]



# all HDBSCAN + k-means + Ward clusterings as weak labels
weak_label_cols = [
    c for c in df.columns
    if c.startswith("hdb_") or c.startswith("kmeans_") or c.startswith("agg_ward_")
]

print(f"Using {len(weak_label_cols)} weak label columns for classification.")
print(weak_label_cols)

base_experiment_name = "weaklabels_allclusters"
all_results = run_experiment_all_weak_sources_100(
    df=df,
    model=model,
    text_col=text_col,
    weak_label_cols=weak_label_cols,
    gold_label_col=gold_label_col,
    base_experiment_name=base_experiment_name,
    n_reps=1
)

print('Finished!')