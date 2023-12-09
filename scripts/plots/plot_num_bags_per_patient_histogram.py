import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

results_dir = "data/results/"

train_df = pd.read_csv(os.path.join(results_dir, "train_num_bags_per_patient.csv"))
val_df = pd.read_csv(os.path.join(results_dir, "val_num_bags_per_patient.csv"))
test_df = pd.read_csv(os.path.join(results_dir, "test_num_bags_per_patient.csv"))

fig, ax = plt.subplots(figsize=(12, 8))
sns.set(font_scale=1.5)
sns.histplot(data=train_df, x="num_bags", label="train", alpha=0.5, binwidth=1, ax=ax)
sns.histplot(data=val_df, x="num_bags", label="val", alpha=0.5, binwidth=1, ax=ax)
sns.histplot(data=test_df, x="num_bags", label="test", alpha=0.5, binwidth=1, ax=ax)
ax.set_xlim(0, 100)
ax.set_xlabel("Number of bags per patient", fontsize=18)
ax.set_ylabel("Count", fontsize=18)
ax.tick_params(axis="both", which="major", labelsize=16)
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[::2], labels[::2])
plt.savefig(os.path.join("figures/", "num_bags_per_patient_histogram.png"), dpi=300)
