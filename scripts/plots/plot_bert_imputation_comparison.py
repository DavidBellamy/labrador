import os.path as op
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

results_path = "data/results/"
random_seed = 3141592
time_string = time.strftime("%Y%m%d-%H%M%S")

# Config
left_plot_title = "BERT 194M"
right_plot_title = "BERT 68.5M"
left_datafile = "intrinsic_imputation_BERT194M_20230522-105746.csv"
left_ablation_datafile = "intrinsic_imputation_BERT194M_20230522-110330_ablation.csv"
right_datafile = "intrinsic_imputation_discrete_transformer_20231022-222541.csv"
right_ablation_datafile = (
    "intrinsic_imputation_discrete_transformer_20231022-222725_ablation.csv"
)
outfile_name = f"bert_intrinsic_imputations_comparison_{time_string}.png"
left_data_path = op.join(results_path, left_datafile)
left_ablated_data_path = op.join(results_path, left_ablation_datafile)
right_data_path = op.join(results_path, right_datafile)
right_ablated_data_path = op.join(results_path, right_ablation_datafile)
plt.rcParams.update({"font.size": 18})

left_df = pd.read_csv(left_data_path)
left_df_ablated = pd.read_csv(left_ablated_data_path)
right_df = pd.read_csv(right_data_path)
right_df_ablated = pd.read_csv(right_ablated_data_path)

# Drop rows that had NaN ytrue (no value to impute)
# Note: BERT's softmax was weighted to compute ypred, and the logit for the NaN token was ignored.
# so there is no point in assessing the distribution of ypred's for NaN ytrue.
left_df.dropna(inplace=True)
left_df_ablated.dropna(inplace=True)
right_df.dropna(inplace=True)
right_df_ablated.dropna(inplace=True)

# Calculate the global Pearson correlation for all imputations
pearson_correlation_left = np.round(np.corrcoef(left_df.ypred, left_df.ytrue)[0][1], 3)
pearson_correlation_ablated_left = np.round(
    np.corrcoef(left_df_ablated.ypred, left_df_ablated.ytrue)[0][1], 3
)
pearson_correlation_right = np.round(
    np.corrcoef(right_df.ypred, right_df.ytrue)[0][1], 3
)
pearson_correlation_ablated_right = np.round(
    np.corrcoef(right_df_ablated.ypred, right_df_ablated.ytrue)[0][1], 3
)

left_concatenated = pd.concat(
    [
        left_df.sample(frac=0.01, random_state=random_seed).assign(
            Model=f"Pre-trained $R^2={pearson_correlation_left}$"
        ),
        left_df_ablated.sample(frac=0.01, random_state=random_seed).assign(
            Model=f"Ablated $R^2={pearson_correlation_ablated_left}$"
        ),
    ]
)

right_concatenated = pd.concat(
    [
        right_df.sample(frac=0.01, random_state=random_seed).assign(
            Model=f"Pre-trained $R^2={pearson_correlation_right}$"
        ),
        right_df_ablated.sample(frac=0.01, random_state=random_seed).assign(
            Model=f"Ablated $R^2={pearson_correlation_ablated_right}$"
        ),
    ]
)

fig = plt.figure(figsize=(16, 16))
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.subplots_adjust(hspace=0.3)
subfigs = fig.subfigures(3, 1, height_ratios=[3, 2, 2])
subfigs[0].suptitle("A", x=0.05, y=0.9, fontsize=36)

axes = subfigs[0].subplots(1, 2)
sns.scatterplot(
    data=left_concatenated, x="ypred", y="ytrue", alpha=0.2, hue="Model", ax=axes[0]
)
axes[0].title.set_text(left_plot_title)
axes[0].set_xlabel("$\hat{y}$")
axes[0].set_ylabel("$y$")
axes[0].plot([0, 1], [0, 1], alpha=0.75, color="r")
axes[0].legend(title=None, framealpha=0.5, borderpad=0.2, borderaxespad=0.25)

sns.scatterplot(
    data=right_concatenated, x="ypred", y="ytrue", alpha=0.2, hue="Model", ax=axes[1]
)
axes[1].title.set_text(right_plot_title)
axes[1].set_xlabel("$\hat{y}$")
axes[1].set_ylabel("$y$")
axes[1].plot([0, 1], [0, 1], alpha=0.75, color="r")
axes[1].legend(title=None, framealpha=0.5, borderpad=0.2, borderaxespad=0.25)

# Calculate the Pearson correlation for each itemid and examine the best and worst itemid's (with >= 1500 samples)
right_itemid_corrs = right_df.groupby("itemid")[["ypred", "ytrue"]].apply(
    lambda x: np.corrcoef(x.ypred, x.ytrue)[0][1]
)
right_itemid_corrs = pd.DataFrame(right_itemid_corrs).reset_index()
right_itemid_corrs.rename(columns={0: "correlation"}, inplace=True)
codebook = pd.read_csv("data/processed/labcode_codebook_labrador.csv")
right_itemid_corrs = right_itemid_corrs.merge(codebook, how="left")
counts = (
    right_df["itemid"]
    .value_counts()
    .reset_index()
    .rename(columns={"index": "itemid", "itemid": "n_samples"})
)
right_itemid_corrs = right_itemid_corrs.merge(counts, how="inner")
right_itemid_corrs = right_itemid_corrs[
    [
        "itemid",
        "correlation",
        "n_samples",
        "frequency_rank",
        "label",
        "fluid",
        "category",
        "loinc_code",
    ]
]

left_itemid_corrs = left_df.groupby("itemid")[["ypred", "ytrue"]].apply(
    lambda x: np.corrcoef(x.ypred, x.ytrue)[0][1]
)
left_itemid_corrs = pd.DataFrame(left_itemid_corrs).reset_index()
left_itemid_corrs.rename(columns={0: "correlation"}, inplace=True)
codebook = pd.read_csv("data/processed/labcode_codebook_labrador.csv")
left_itemid_corrs = left_itemid_corrs.merge(codebook, how="left")
counts = (
    left_df["itemid"]
    .value_counts()
    .reset_index()
    .rename(columns={"index": "itemid", "itemid": "n_samples"})
)
left_itemid_corrs = left_itemid_corrs.merge(counts, how="inner")
left_itemid_corrs = left_itemid_corrs[
    [
        "itemid",
        "correlation",
        "n_samples",
        "frequency_rank",
        "label",
        "fluid",
        "category",
        "loinc_code",
    ]
]

# Make scatter plots for the best and worst 4 itemid's by Pearson correlation
right_itemid_corrs[right_itemid_corrs.n_samples >= 1500].sort_values("correlation")
left_itemid_corrs[left_itemid_corrs.n_samples >= 1500].sort_values("correlation")

right_top4_itemids = [50810, 51613, 51256, 51222]
right_bottom4_itemids = [50993, 50960, 51491, 51010]
left_top4_itemids = right_top4_itemids
left_bottom4_itemids = right_bottom4_itemids

best_itemids_fig = subfigs[1].subfigures(1, 2)
worst_itemids_fig = subfigs[2].subfigures(1, 2)

best_itemids_fig[0].suptitle("B", x=0.1, y=1.1, fontsize=36)
worst_itemids_fig[0].suptitle("C", x=0.1, y=1.1, fontsize=36)

left_best_itemids_axes = best_itemids_fig[0].subplots(2, 2)
left_worst_itemids_axes = worst_itemids_fig[0].subplots(2, 2)
right_best_itemids_axes = best_itemids_fig[1].subplots(2, 2)
right_worst_itemids_axes = worst_itemids_fig[1].subplots(2, 2)

for subfigure, itemid_list, df in zip(
    [
        left_best_itemids_axes,
        left_worst_itemids_axes,
        right_best_itemids_axes,
        right_worst_itemids_axes,
    ],
    [
        left_top4_itemids,
        left_bottom4_itemids,
        right_top4_itemids,
        right_bottom4_itemids,
    ],
    [left_df, left_df, right_df, right_df],
):
    for ax, itemid in zip(subfigure.flatten(), itemid_list):
        sns.scatterplot(
            data=df[df.itemid == itemid], x="ypred", y="ytrue", alpha=1, ax=ax
        )
        ax.plot([0, 1], [0, 1], alpha=0.75, color="r")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_title(f"{codebook[codebook.itemid == itemid]['label'].item()}")

plt.savefig(op.join("figures/", outfile_name), dpi=300, bbox_inches="tight")

plt.show()
