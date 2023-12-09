import os.path as op
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

results_path = "data/results/intrinsic_imputation"
random_seed = 3141592
time_string = time.strftime("%Y%m%d-%H%M%S")

# Config
labrador_plot_title = "Labrador"
bert_plot_title = "BERT"
labrador_datafile = "intrinsic_imputation_continuous_transformer_20230216-095644.csv"
labrador_ablation_datafile = (
    "intrinsic_imputation_continuous_transformer_20230216-100547_ablation.csv"
)
bert_datafile = "intrinsic_imputation_BERT194M_20230522-105746.csv"
bert_ablation_datafile = "intrinsic_imputation_BERT194M_20230522-110330_ablation.csv"
outfile_names = [
    f"intrinsic_imputations_{time_string}.eps",
    f"intrinsic_imputations_{time_string}.png",
]
labrador_data_path = op.join(results_path, labrador_datafile)
labrador_ablated_data_path = op.join(results_path, labrador_ablation_datafile)
bert_data_path = op.join(results_path, bert_datafile)
bert_ablated_data_path = op.join(results_path, bert_ablation_datafile)
plt.rcParams.update({"font.size": 18})

labrador_df = pd.read_csv(labrador_data_path)
labrador_df_ablated = pd.read_csv(labrador_ablated_data_path)
bert_df = pd.read_csv(bert_data_path)
bert_df_ablated = pd.read_csv(bert_ablated_data_path)

# Drop rows that had NaN ytrue (no value to impute)
# Note: BERT's softmax was weighted to compute ypred, and the logit for the NaN token was ignored.
# so there is no point in assessing the distribution of ypred's for NaN ytrue.
labrador_df.dropna(inplace=True)
labrador_df_ablated.dropna(inplace=True)
bert_df.dropna(inplace=True)
bert_df_ablated.dropna(inplace=True)

# Calculate the global Pearson correlation for all imputations
pearson_correlation_labrador = np.round(
    np.corrcoef(labrador_df.ypred, labrador_df.ytrue)[0][1], 3
)
pearson_correlation_ablated_labrador = np.round(
    np.corrcoef(labrador_df_ablated.ypred, labrador_df_ablated.ytrue)[0][1], 3
)
pearson_correlation_bert = np.round(np.corrcoef(bert_df.ypred, bert_df.ytrue)[0][1], 3)
pearson_correlation_ablated_bert = np.round(
    np.corrcoef(bert_df_ablated.ypred, bert_df_ablated.ytrue)[0][1], 3
)

labrador_concatenated = pd.concat(
    [
        labrador_df.sample(frac=0.01, random_state=random_seed).assign(
            Model=f"Pre-trained $R^2={pearson_correlation_labrador}$"
        ),
        labrador_df_ablated.sample(frac=0.01, random_state=random_seed).assign(
            Model=f"Ablated $R^2={pearson_correlation_ablated_labrador}$"
        ),
    ]
)

bert_df_concatenated = pd.concat(
    [
        bert_df.sample(frac=0.01, random_state=random_seed).assign(
            Model=f"Pre-trained $R^2={pearson_correlation_bert}$"
        ),
        bert_df_ablated.sample(frac=0.01, random_state=random_seed).assign(
            Model=f"Ablated $R^2={pearson_correlation_ablated_bert}$"
        ),
    ]
)

fig = plt.figure(figsize=(16, 16))
plt.subplots_adjust(hspace=0.3)
subfigs = fig.subfigures(3, 1, height_ratios=[3, 2, 2])
subfigs[0].suptitle("A", x=0.05, y=0.9, fontsize=36)

axes = subfigs[0].subplots(1, 2)
sns.scatterplot(
    data=labrador_concatenated,
    x="ypred",
    y="ytrue",
    alpha=0.2,
    hue="Model",
    ax=axes[0],
)
axes[0].title.set_text(labrador_plot_title)
axes[0].set_xlabel("$\hat{y}$")
axes[0].set_ylabel("$y$")
axes[0].plot([0, 1], [0, 1], alpha=0.75, color="r")
axes[0].legend(title=None, framealpha=0.5, borderpad=0.2, borderaxespad=0.25)

sns.scatterplot(
    data=bert_df_concatenated, x="ypred", y="ytrue", alpha=0.2, hue="Model", ax=axes[1]
)
axes[1].title.set_text(bert_plot_title)
axes[1].set_xlabel("$\hat{y}$")
axes[1].set_ylabel("$y$")
axes[1].plot([0, 1], [0, 1], alpha=0.75, color="r")
axes[1].legend(title=None, framealpha=0.5, borderpad=0.2, borderaxespad=0.25)

# The best and worst 4 itemid's by Pearson correlation (with >= 1500 samples; so the plot is not too sparse)
labrador_top4_itemids = [51279, 51248, 51250, 51222]
labrador_bottom4_itemids = [51255, 50853, 51143, 51010]
bert_top4_itemids = [
    50810,
    51613,
    51256,
    51222,
]  # Note: I chose 51222 hemoglobin over 50811 hemoglobin so that it would align with Labrador
bert_bottom4_itemids = [50993, 50960, 51491, 51010]

codebook = pd.read_csv("data/processed/labcode_codebook_labrador.csv")
best_itemids_fig = subfigs[1].subfigures(1, 2)
worst_itemids_fig = subfigs[2].subfigures(1, 2)

best_itemids_fig[0].suptitle("B", x=0.1, y=1, fontsize=36)
worst_itemids_fig[0].suptitle("C", x=0.1, y=1, fontsize=36)

labrador_best_itemids_axes = best_itemids_fig[0].subplots(2, 2)
labrador_worst_itemids_axes = worst_itemids_fig[0].subplots(2, 2)
bert_best_itemids_axes = best_itemids_fig[1].subplots(2, 2)
bert_worst_itemids_axes = worst_itemids_fig[1].subplots(2, 2)

for subfigure, itemid_list, df in zip(
    [
        labrador_best_itemids_axes,
        labrador_worst_itemids_axes,
        bert_best_itemids_axes,
        bert_worst_itemids_axes,
    ],
    [
        labrador_top4_itemids,
        labrador_bottom4_itemids,
        bert_top4_itemids,
        bert_bottom4_itemids,
    ],
    [labrador_df, labrador_df, bert_df, bert_df],
):
    for ax, itemid in zip(subfigure.flatten(), itemid_list):
        sns.scatterplot(
            data=df[df.itemid == itemid], x="ypred", y="ytrue", alpha=1, ax=ax
        )
        ax.plot([0, 1], [0, 1], alpha=0.75, color="r")
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_title(f"{codebook[codebook.itemid == itemid]['label'].item()}")

for outfile in outfile_names:
    plt.savefig(op.join("figures/", outfile), dpi=300, bbox_inches="tight")

plt.show()
