import os
import os.path as op
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from lab_transformers.utils import json_lines_loader

data_path = "data/processed/labrador_train_patients.jsonl"
codebook_path = "data/processed/labcode_codebook_panelized.csv"
outfile_path = "data/results/"

# Load codebook with lab test panel information
codebook = pd.read_csv(codebook_path)

# Load training bags of labs
train_data = json_lines_loader(data_path)

tokens = codebook[~pd.isna(codebook.mimic_panel)].frequency_rank.to_numpy()

df_rows = []
for patient in tqdm(train_data, desc="Computing correlations"):
    for token_bag, lab_value_bag in zip(patient["code_bags"], patient["value_bags"]):
        if len(set(token_bag).intersection(tokens)) < 2:
            continue

        row = dict()
        for token in tokens:
            if token in token_bag:
                row = row | {
                    str(token): lab_value_bag[np.where(token_bag == token)[0][0]]
                }

        # replace <NULL> strings with np.nan
        row = {k: row[k] if row[k] != "<NULL>" else np.nan for k in row.keys()}
        df_rows.append(row)

df = pd.DataFrame(df_rows)

# Rename columns with MIMIC itemid's
df.columns = df.columns.map(
    lambda x: codebook[codebook.frequency_rank == int(x)].itemid.item()
)

# Sort columns in df by which mimic_panel they have in codebook
df = df[
    codebook[~pd.isna(codebook.mimic_panel)]
    .sort_values(by="mimic_panel")
    .itemid.to_numpy()
]

print("Compute pairwise Pearson correlations\n")
correlation_matrix = df.corr()
print("Success\n")

# Create outfile directory if it doesn't exist
if not op.exists(outfile_path):
    os.makedirs(outfile_path)

correlation_matrix.to_csv(
    op.join(outfile_path, "correlation_matrix.csv"), index=True, index_label="itemid"
)
