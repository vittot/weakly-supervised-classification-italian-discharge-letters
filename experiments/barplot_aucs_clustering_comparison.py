import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ----------------------------------------------------------
# Load results
# ----------------------------------------------------------
if not os.path.exists("df_summary_aucs_allsettings.xlsx"):

    results_path = "results_weaklabels_allclusters_allweak.pkl"  # adjust name
    with open(results_path, "rb") as f:
        all_results = pickle.load(f)

    # ----------------------------------------------------------
    # Extract AUCs into a long DataFrame
    # ----------------------------------------------------------
    METRIC_KEY = "aucs_gold"

    records = []

    for weak_col, reps_dict in all_results.items():
        for rep, cv_results in reps_dict.items():
                    auc_list = cv_results[METRIC_KEY]  # list of 10 AUCs
                    for i, auc_value in enumerate(auc_list):
                        records.append({
                            "setting": weak_col,
                            "rep": rep,
                            "fold": i,
                            "auc": auc_value
                        })

    df_auc = pd.DataFrame(records)
    print(df_auc.head())
    print(df_auc.shape)

    # ----------------------------------------------------------
    # Summary stats per clustering setting
    # ----------------------------------------------------------
    summary = (
        df_auc
        .groupby("setting")["auc"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    summary["se"] = summary["std"] / np.sqrt(summary["count"])
    z = 1.96  # normal approx
    summary["ci_halfwidth"] = z * summary["se"]
    summary["ci_low"] = summary["mean"] - summary["ci_halfwidth"]
    summary["ci_high"] = summary["mean"] + summary["ci_halfwidth"]

    #summary = summary.sort_values("mean", ascending=False)
    print(summary)
    summary.to_excel("df_summary_aucs_allsettings.xlsx", index=False)
else:
    summary = pd.read_excel("df_summary_aucs_allsettings.xlsx")
    summary["se"] = summary["std"] / np.sqrt(summary["count"])
    z = 1.96  # normal approx
    summary["ci_halfwidth"] = z * summary["se"]
    summary["ci_low"] = summary["mean"] - summary["ci_halfwidth"]
    summary["ci_high"] = summary["mean"] + summary["ci_halfwidth"]

    #summary = summary.sort_values("mean", ascending=False)
    print(summary)
    summary.to_excel("df_summary_aucs_allsettings.xlsx", index=False)


# ----------------------------------------------------------
# Barplot with 95% CI error bars
# ----------------------------------------------------------
summary_sorted = summary.sort_values("mean", ascending=True)

plt.figure(figsize=(10, 8))

y_pos = np.arange(len(summary_sorted))
means = summary_sorted["mean"].values
xerr = summary_sorted["ci_halfwidth"].values

plt.barh(
    y_pos,
    means,
    xerr=xerr,
    capsize=5,
    color="steelblue"
)

plt.yticks(y_pos, summary_sorted["setting"])
plt.xlabel("AUC ± 95% CI over 10-fold CV")
plt.xlim(0.0, 1.0)

plt.title("AUC comparing weak labels from different clusterings")

plt.tight_layout()
plt.savefig("auc_horizontal_barplot_all_weakcols.png", dpi=300)
plt.show()
