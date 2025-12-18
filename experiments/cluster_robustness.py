import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# ----------------------------------------------------------
# Load dataframe with all clustering columns
# ----------------------------------------------------------
input = sys.argv[1] if len(sys.argv)>1 else "df_fse_hdbscan_grid.csv"
out = sys.argv[2] if len(sys.argv)>2 else ""

print(input)
print(out)

if not (os.path.exists(f"cluster_robustness_ARI_{out}.xlsx") and os.path.exists(f"cluster_robustness_NMI_{out}.xlsx")):

    df = pd.read_csv(input)

    hdb_cols = [c for c in df.columns if c.startswith("hdb_")]
    baseline_cols = [c for c in df.columns if c.startswith("kmeans_k") or c.startswith("agg_ward_k")]

    cluster_cols = hdb_cols + baseline_cols
    print(f"Found {len(cluster_cols)} cluster columns.")

    # ----------------------------------------------------------
    # Compute ARI and NMI matrices
    # ----------------------------------------------------------
    ari_matrix = pd.DataFrame(index=cluster_cols, columns=cluster_cols, dtype=float)
    nmi_matrix = pd.DataFrame(index=cluster_cols, columns=cluster_cols, dtype=float)

    for c1 in cluster_cols:
        for c2 in cluster_cols:
            labels1 = df[c1].fillna(-2)   # safety if NaNs appear
            labels2 = df[c2].fillna(-2)
            ari_matrix.loc[c1, c2] = adjusted_rand_score(labels1, labels2)
            nmi_matrix.loc[c1, c2] = normalized_mutual_info_score(labels1, labels2)

    # Save
    ari_matrix.to_excel(f"cluster_robustness_ARI_{out}.xlsx")
    nmi_matrix.to_excel(f"cluster_robustness_NMI_{out}.xlsx")

    print("ARI/NMI matrices saved.")

else:
    print("Loading existing ARI/NMI matrices.")
    ari_matrix = pd.read_excel(f"cluster_robustness_ARI_{out}.xlsx", index_col=0)
    nmi_matrix = pd.read_excel(f"cluster_robustness_NMI_{out}.xlsx", index_col=0)

# ----------------------------------------------------------
# Compute mean ARI / NMI excluding diagonal
# ----------------------------------------------------------
def matrix_mean_offdiag(M):
    arr = M.values
    n = arr.shape[0]
    offdiag = arr[~np.eye(n, dtype=bool)]
    return offdiag.mean()

mean_ari = matrix_mean_offdiag(ari_matrix)
mean_nmi = matrix_mean_offdiag(nmi_matrix)

print("\n===== ROBUSTNESS SUMMARY =====")
print(f"Mean ARI (off-diagonal): {mean_ari:.4f}")
print(f"Mean NMI (off-diagonal): {mean_nmi:.4f}")

ari_clean = ari_matrix.copy()
# ari_clean.index = ari_clean.index.str.replace("^hdb_", "", regex=True)
# ari_clean.columns = ari_clean.columns.str.replace("^hdb_", "", regex=True)
nmi_clean = nmi_matrix.copy()
# nmi_clean.index = nmi_clean.index.str.replace("^hdb_", "", regex=True)
# nmi_clean.columns = nmi_clean.columns.str.replace("^hdb_", "", regex=True)

# -------------------------------
# Figure with 2 stacked axes
# -------------------------------
fig, (ax1, ax2) = plt.subplots(
    nrows=2, ncols=1,
    figsize=(12, 16),
    constrained_layout=True   # let matplotlib manage spacing
)

# -----------------------------------------
# Panel A - ARI heatmap (no colorbar here)
# -----------------------------------------
hm1 = sns.heatmap(
    ari_clean.astype(float),
    cmap="viridis",
    vmin=0, vmax=1,
    square=True,
    annot=True,
    annot_kws={"fontsize": 9},
    cbar=False,
    ax=ax1
)
ax1.set_title("Clustering comparison — ARI", fontsize=16)
# Rotate x ticks a little to limit overlap
plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

# Panel letter A
ax1.text(
    -0.20, 1.02, "A",
    transform=ax1.transAxes,
    fontsize=20,
    fontweight="bold",
    va="bottom",
    ha="center"
)
# Add extra padding between x labels and the bottom of ax1
ax1.tick_params(axis='x', pad=10)

# -----------------------------------------
# Panel B - NMI heatmap (no colorbar here)
# -----------------------------------------
hm2 = sns.heatmap(
    nmi_clean.astype(float),
    cmap="viridis",
    vmin=0, vmax=1,
    square=True,
    annot=True,
    annot_kws={"fontsize": 9},
    cbar=False,
    ax=ax2
)
ax2.set_title("Clustering comparison — NMI", fontsize=16)
plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")

# Panel letter B
ax2.text(
    -0.20, 1.02, "B",
    transform=ax2.transAxes,
    fontsize=20,
    fontweight="bold",
    va="bottom",
    ha="center"
)

# Add padding between x ticks and the colorbar
ax2.tick_params(axis='x', pad=10)

# -----------------------------------------
# Create a narrower colorbar axis
# -----------------------------------------
# Get position of bottom heatmap
pos = ax2.get_position()

# Define a colorbar axis that is narrower:
cbar_width = (pos.x1 - pos.x0) * 0.8     
cbar_x = pos.x0 + (pos.x1 - pos.x0 - cbar_width) / 2  # center it
cbar_y = pos.y0 - 0.05                   # move below heatmap
cbar_height = 0.02

cax = fig.add_axes([cbar_x, cbar_y, cbar_width, cbar_height])

norm = plt.Normalize(vmin=0, vmax=1)
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])

fig.colorbar(sm, cax=cax, orientation="horizontal", label="")

fig.savefig(f"cluster_robustness_ARI_NMI_{out}.png", dpi=300, bbox_inches="tight")