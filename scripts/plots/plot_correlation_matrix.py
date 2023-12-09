import os.path as op
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

results_dir = "data/results"
processed_data_dir = "data/processed/"
outfile_path = "figures/"

codebook = pd.read_csv(op.join(processed_data_dir, "labcode_codebook_panelized.csv"))

renaming = {
    "Calculated Bicarbonate, Whole Blood": "Bicarb",
    "Alanine Aminotransferase (ALT)": "ALT",
    "Asparate Aminotransferase (AST)": "AST",
    "Alkaline Phosphatase": "ALP",
    "Gamma Glutamyltransferase": "GGT",
    "Bilirubin, Total": "Bilirubin",
    "White Blood Cells": "WBC",
    "Red Blood Cells": "RBC",
    "Platelet Count": "Platelets",
}

codebook["label"] = codebook["label"].replace(renaming)

df = pd.read_csv(op.join(results_dir, "correlation_matrix.csv"), index_col=0)
df = df.merge(codebook[["itemid", "label"]], on="itemid").drop(columns=["itemid"])

# Set `label` column as index and rename columns with labels
df.set_index("label", inplace=True)
df.rename(
    columns={df.columns[i]: df.index[i] for i in range(len(df.columns))}, inplace=True
)

# Drop columns and rows with all np.nan
df.dropna(axis=0, how="all", inplace=True)
df.dropna(axis=1, how="all", inplace=True)

# Create plot
fig, ax = plt.subplots(figsize=(18, 12))
sns.set(font_scale=2)
sns.heatmap(df, annot=False, vmin=-1, vmax=1, center=0, cmap="seismic", ax=ax)
ax.hlines([0, 11, 14, 20, 30, 40, 46, 50, 53, 60], *ax.get_xlim(), colors="black")
ax.vlines([0, 11, 14, 20, 30, 40, 46, 50, 53, 60], *ax.get_ylim(), colors="black")
ax.set(xticklabels=[])
ax.set(yticklabels=[])
ax.tick_params(left=False, bottom=False)
ax.set_ylabel(None)
ax.text(-0.2, 5.5, "BMP", horizontalalignment="right")
ax.text(-0.2, 13, "Bilirubin", horizontalalignment="right")
ax.text(-0.2, 17.5, "Blood Gases", horizontalalignment="right")
ax.text(-0.2, 25.5, "CBC", horizontalalignment="right")
ax.text(-0.2, 35.5, "CBC Diff", horizontalalignment="right")
ax.text(-0.2, 43.5, "Lipids", horizontalalignment="right")
ax.text(-0.2, 48.5, "Liver Enzymes", horizontalalignment="right")
ax.text(-0.2, 52, "Tox. (Blood)", horizontalalignment="right")
ax.text(-0.2, 57, "Urinalysis", horizontalalignment="right")

plt.savefig(op.join(outfile_path, f"lab_correlation_matrix.eps"), dpi=300)
plt.savefig(op.join(outfile_path, f"lab_correlation_matrix.png"), dpi=300)
plt.close()
