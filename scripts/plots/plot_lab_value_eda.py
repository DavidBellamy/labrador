import os.path as op
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from lab_transformers.utils import json_lines_loader

data_dir = "data/"
processed_data_dir = "data/processed"

print("Loading train bags...")
train_bags = json_lines_loader(op.join(processed_data_dir, "labrador_train_bags.jsonl"))

print("Loading validation bags...")
val_bags = json_lines_loader(
    op.join(processed_data_dir, "labrador_validation_bags.jsonl")
)

print("Loading test bags...")
test_bags = json_lines_loader(op.join(processed_data_dir, "labrador_test_bags.jsonl"))

# Convert JSON lines with bags of labs to dataframes (much easier for summarizing/plotting)
print("Converting train bags to df...")
train_df = (
    pd.DataFrame.from_dict(train_bags)
    .explode(["code_bags", "value_bags", "hadm_id", "charttime"])
    .explode(["code_bags", "value_bags"])
)

print("Converting validation bags to df...")
val_df = (
    pd.DataFrame.from_dict(val_bags)
    .explode(["code_bags", "value_bags", "hadm_id", "charttime"])
    .explode(["code_bags", "value_bags"])
)

print("Converting test bags to df...")
test_df = (
    pd.DataFrame.from_dict(test_bags)
    .explode(["code_bags", "value_bags", "hadm_id", "charttime"])
    .explode(["code_bags", "value_bags"])
)

# Add a column to each df indicating which split the data comes from
train_df["split"] = "train"
val_df["split"] = "validation"
test_df["split"] = "test"

# Concatenate the three splits by row
print("Concatenating splits...")
df = pd.concat([train_df, val_df, test_df])

# Rename the columns
df.rename(
    columns={"code_bags": "frequency_rank", "value_bags": "lab_value"}, inplace=True
)

# Replace '<NULL>' lab_value with NA
print("Replacing <NULL> strings...")
df.replace("<NULL>", np.nan, inplace=True)

# Load the raw lab data
labevents = pd.read_csv(
    op.join(data_dir, "raw", "labevents.csv"),
    dtype={
        "labevent_id": int,
        "subject_id": int,
        "hadm_id": "Int64",
        # Pandas nullable Int type
        "specimen_id": int,
        "itemid": int,
        "charttime": "string",
        "storetime": "string",
        "value": object,
        "valuenum": float,
        "valueuom": "string",
        "ref_range_lower": float,
        "ref_range_upper": float,
        "flag": "string",
        "priority": "string",
        "comments": "string",
    },
)

# Load in lab codebook and summary statistics from train split
lab_codebook = pd.read_csv(op.join(processed_data_dir, "labcode_codebook_labrador.csv"))
train_summary_stats = pd.read_csv(
    op.join(processed_data_dir, "train_summary_stats.csv")
)

# Make kernel density estimate plots for every lab code
codes = df.frequency_rank.unique()

for lab_code in tqdm(codes, desc="Creating density plots for each lab code..."):
    values = df[df.frequency_rank == lab_code].lab_value.dropna()

    # Skip lab values that have no numeric entries
    if len(values) == 0:
        continue

    # Make density plot
    values.plot.kde()

    # Create plot title
    itemid = lab_codebook[lab_codebook.frequency_rank == lab_code].itemid.values[0]
    label = lab_codebook[lab_codebook.frequency_rank == lab_code].label.values[0]
    fluid = lab_codebook[lab_codebook.frequency_rank == lab_code].fluid.values[0]
    category = lab_codebook[lab_codebook.frequency_rank == lab_code].category.values[0]
    loinc = lab_codebook[lab_codebook.frequency_rank == lab_code].loinc_code.values[0]
    mean = round(
        train_summary_stats[train_summary_stats.frequency_rank == lab_code][
            "mean"
        ].values[0],
        2,
    )
    std = round(
        train_summary_stats[train_summary_stats.frequency_rank == lab_code][
            "std"
        ].values[0],
        2,
    )

    title = f"{itemid} ({lab_code}): {label}, {fluid}, {category}, {loinc} \n mean: {mean}, st dev: {std}"

    plt.title(title)
    plt.xlabel("Standardized units")
    plt.savefig(op.join(data_dir, "lab_value_figures", f"{lab_code}_density.png"))
    plt.clf()

    # Make scatter plot
    plt.scatter(range(len(values)), values, alpha=0.1, s=4)
    plt.title(title)
    plt.ylabel("Standardized units")
    plt.xlabel("Arbitrary index")
    plt.savefig(op.join(data_dir, "lab_value_figures", f"{lab_code}_scatter.png"))
    plt.clf()
