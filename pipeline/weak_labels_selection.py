import argparse
import json
import os
import re
from typing import Dict, List, Set, Any

import numpy as np
import pandas as pd

from utils.utils import set_seed
set_seed(1234)


# ---------------------------------------------------------------------
# Disease definitions
# ---------------------------------------------------------------------

def get_disease_definitions(disease: str) -> List[Dict[str, List[str]]]:
    """
    Return disease definitions as a list of rules.
    Each rule has:
      - positive: keywords that must all be present
      - negative: keywords that must all be absent
    Matching is substring-based after normalization.

    You can extend this dictionary for other diseases.
    """
    disease = disease.lower().strip()

    definitions = {
        "bronchiolitis": [
            {
                "positive": ["bronchiolite"],
                "negative": [],
            },
            {
                "positive": ["broncospasmo", "febbre"],
                "negative": [],
            },
        ],
        "bronchitis": [
            {
                "positive": ["bronchite"],
                "negative": [],
            },
            {
                "positive": ["infezione", "vie", "respiratorie", "inferiori"],
                "negative": ["superiori"],
            }
        ],
    }

    if disease not in definitions:
        raise ValueError(
            f"Unknown disease '{disease}'. "
            f"Available diseases: {list(definitions.keys())}"
        )

    return definitions[disease]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def normalize_text(s: Any) -> str:
    """Lowercase + strip + collapse spaces."""
    if pd.isna(s):
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def parse_summary_keywords(summary_value: Any) -> List[str]:
    """
    Parse cluster summary into a list of keywords.

    Expected formats:
    - 'bronchiolite;broncospasmo;febbre'
    - ['bronchiolite', 'broncospasmo']
    - 'bronchiolite, broncospasmo'
    """
    if pd.isna(summary_value):
        return []

    if isinstance(summary_value, list):
        return [normalize_text(x) for x in summary_value if normalize_text(x)]

    s = normalize_text(summary_value)

    if not s:
        return []

    # try JSON-like list first
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [normalize_text(x) for x in parsed if normalize_text(x)]
        except Exception:
            pass

    # fallback: split on ; or ,
    parts = re.split(r"[;,]", s)
    return [normalize_text(x) for x in parts if normalize_text(x)]


def keyword_rule_matches(
    cluster_keywords: List[str],
    positive_keywords: List[str],
    negative_keywords: List[str],
) -> bool:
    """
    Substring-based matching, consistent with your current code logic:
    - every positive keyword must appear in at least one cluster keyword
    - no negative keyword may appear in any cluster keyword
    """
    positives_ok = all(
        any(pk in kw for kw in cluster_keywords)
        for pk in positive_keywords
    )

    negatives_ok = not any(
        nk in kw
        for nk in negative_keywords
        for kw in cluster_keywords
    )

    return positives_ok and negatives_ok


def cluster_is_positive(
    cluster_keywords: List[str],
    disease_definitions: List[Dict[str, List[str]]],
) -> bool:
    """Check if cluster satisfies at least one disease definition."""
    for rule in disease_definitions:
        if keyword_rule_matches(
            cluster_keywords=cluster_keywords,
            positive_keywords=[normalize_text(x) for x in rule["positive"]],
            negative_keywords=[normalize_text(x) for x in rule["negative"]],
        ):
            return True
    return False


# ---------------------------------------------------------------------
# Main labeling logic
# ---------------------------------------------------------------------

def build_cluster_table(
    df: pd.DataFrame,
    cluster_col: str,
    summary_col: str,
    disease_definitions: List[Dict[str, List[str]]],
) -> pd.DataFrame:
    """
    Build one row per cluster with:
    - cluster id
    - parsed keywords
    - positive/negative label according to rules
    """
    if cluster_col not in df.columns:
        raise KeyError(f"Column '{cluster_col}' not found in input dataframe.")
    if summary_col not in df.columns:
        raise KeyError(f"Column '{summary_col}' not found in input dataframe.")

    cluster_df = (
        df[[cluster_col, summary_col]]
        .drop_duplicates(subset=[cluster_col])
        .copy()
        .reset_index(drop=True)
    )

    cluster_df["parsed_keywords"] = cluster_df[summary_col].apply(parse_summary_keywords)
    cluster_df["is_selected_cluster"] = cluster_df["parsed_keywords"].apply(
        lambda kws: cluster_is_positive(kws, disease_definitions)
        if len(kws) > 0 else False
    )

    return cluster_df


def apply_weak_labels(
    df: pd.DataFrame,
    cluster_table: pd.DataFrame,
    cluster_col: str,
    output_label_col: str = "weak_label",
) -> pd.DataFrame:
    """
    Apply weak labels to documents:
    - 1 if cluster selected
    - 0 if clustered and not selected
    - NaN if cluster == -1 or missing
    """
    cluster_to_label = dict(
        zip(cluster_table[cluster_col], cluster_table["is_selected_cluster"])
    )

    out = df.copy()

    def map_doc_label(cluster_id):
        if pd.isna(cluster_id):
            return np.nan
        try:
            # HDBSCAN noise convention
            if int(cluster_id) == -1:
                return np.nan
        except Exception:
            pass

        if cluster_id not in cluster_to_label:
            return np.nan

        return int(bool(cluster_to_label[cluster_id]))

    out[output_label_col] = out[cluster_col].apply(map_doc_label)
    return out


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--output_clusters_csv", type=str, required=True)

    parser.add_argument("--cluster_col", type=str, required=True)
    parser.add_argument("--summary_col", type=str, required=True)

    parser.add_argument("--disease", type=str, default="bronchiolitis")
    parser.add_argument("--output_label_col", type=str, default="weak_label")

    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Input file not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)

    disease_definitions = get_disease_definitions(args.disease)

    cluster_table = build_cluster_table(
        df=df,
        cluster_col=args.cluster_col,
        summary_col=args.summary_col,
        disease_definitions=disease_definitions,
    )

    df_labeled = apply_weak_labels(
        df=df,
        cluster_table=cluster_table,
        cluster_col=args.cluster_col,
        output_label_col=args.output_label_col,
    )

    # nicer export of keyword lists
    cluster_table_to_save = cluster_table.copy()
    cluster_table_to_save["parsed_keywords"] = cluster_table_to_save["parsed_keywords"].apply(
        lambda x: ";".join(x)
    )

    df_labeled.to_csv(args.output_csv, index=False)
    cluster_table_to_save.to_csv(args.output_clusters_csv, index=False)

    n_total = len(df_labeled)
    n_clustered = df_labeled[args.output_label_col].notna().sum()
    n_pos = (df_labeled[args.output_label_col] == 1).sum()
    n_neg = (df_labeled[args.output_label_col] == 0).sum()
    n_selected_clusters = cluster_table["is_selected_cluster"].sum()

    print(f"Saved labeled documents to: {args.output_csv}")
    print(f"Saved cluster table to: {args.output_clusters_csv}")
    print()
    print(f"Total documents: {n_total}")
    print(f"Clustered documents with weak label: {n_clustered}")
    print(f"Weak positives: {n_pos}")
    print(f"Weak negatives: {n_neg}")
    print(f"Selected positive clusters: {n_selected_clusters}")


if __name__ == "__main__":
    main()