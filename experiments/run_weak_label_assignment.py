import argparse
import os
import pandas as pd

from utils.utils import set_seed

from pipeline.weak_labels_selection import (
    get_disease_definitions,
    build_cluster_table,
    apply_weak_labels,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply weak-label assignment from clustering results."
    )

    parser.add_argument(
        "--input_csv",
        type=str,
        default="df_fse_hdbscan_grid.csv",
        help="CSV containing clustering results."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="df_fse_weak_labels.csv",
        help="Output CSV with document-level weak labels added."
    )
    parser.add_argument(
        "--output_clusters_csv",
        type=str,
        default="selected_clusters.csv",
        help="Output CSV with cluster-level keyword summaries and selected clusters."
    )
    parser.add_argument(
        "--cluster_col",
        type=str,
        required=True,
        help="Cluster assignment column to use, e.g. hdb_mcs10_ms5"
    )
    parser.add_argument(
        "--summary_col",
        type=str,
        default=None,
        help="Cluster summary column. Default: summary_<cluster_col>"
    )
    parser.add_argument(
        "--disease",
        type=str,
        default="bronchiolitis",
        help="Disease name used to load keyword definitions."
    )
    parser.add_argument(
        "--output_label_col",
        type=str,
        default="our_bronchio",
        help="Name of the new weak-label column."
    )
    parser.add_argument(
        "--drop_unlabeled",
        action="store_true",
        help="Drop rows with missing weak label after assignment."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed."
    )

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Input file not found: {args.input_csv}")

    summary_col = args.summary_col if args.summary_col is not None else f"summary_{args.cluster_col}"

    df = pd.read_csv(args.input_csv)

    if args.cluster_col not in df.columns:
        raise KeyError(f"Cluster column '{args.cluster_col}' not found in input file.")
    if summary_col not in df.columns:
        raise KeyError(f"Summary column '{summary_col}' not found in input file.")

    disease_definitions = get_disease_definitions(args.disease)

    cluster_table = build_cluster_table(
        df=df,
        cluster_col=args.cluster_col,
        summary_col=summary_col,
        disease_definitions=disease_definitions,
    )

    df_labeled = apply_weak_labels(
        df=df,
        cluster_table=cluster_table,
        cluster_col=args.cluster_col,
        output_label_col=args.output_label_col,
    )

    if args.drop_unlabeled:
        before = len(df_labeled)
        df_labeled = df_labeled[df_labeled[args.output_label_col].notna()].copy()
        after = len(df_labeled)
        print(f"Dropped {before - after} unlabeled rows.")

    cluster_table_to_save = cluster_table.copy()
    if "parsed_keywords" in cluster_table_to_save.columns:
        cluster_table_to_save["parsed_keywords"] = cluster_table_to_save["parsed_keywords"].apply(
            lambda x: ";".join(x) if isinstance(x, list) else x
        )

    df_labeled.to_csv(args.output_csv, index=False)
    cluster_table_to_save.to_csv(args.output_clusters_csv, index=False)

    n_total = len(df_labeled)
    n_nonmissing = df_labeled[args.output_label_col].notna().sum()
    n_pos = (df_labeled[args.output_label_col] == 1).sum()
    n_neg = (df_labeled[args.output_label_col] == 0).sum()

    print(f"Saved labeled dataframe to: {args.output_csv}")
    print(f"Saved cluster-level table to: {args.output_clusters_csv}")
    print()
    print(f"Cluster column used: {args.cluster_col}")
    print(f"Summary column used: {summary_col}")
    print(f"Weak label column created: {args.output_label_col}")
    print(f"Total rows: {n_total}")
    print(f"Rows with weak label: {n_nonmissing}")
    print(f"Weak positives: {n_pos}")
    print(f"Weak negatives: {n_neg}")


if __name__ == "__main__":
    main()