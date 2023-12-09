import os.path as op
import sys

import pandas as pd

from lab_transformers.utils import json_lines_loader

processed_data_dir = "data/processed/"
outfile_dir = "data/results/"

print("Loading train bags...")
train_bags = json_lines_loader(op.join(processed_data_dir, "labrador_train_bags.jsonl"))

print("Loading validation bags...")
val_bags = json_lines_loader(
    op.join(processed_data_dir, "labrador_validation_bags.jsonl")
)

print("Loading test bags...")
test_bags = json_lines_loader(op.join(processed_data_dir, "labrador_test_bags.jsonl"))

train_hist_data = dict()
for p in train_bags:
    train_hist_data[p["subject_id"]] = len(p["code_bags"])

val_hist_data = dict()
for p in val_bags:
    val_hist_data[p["subject_id"]] = len(p["code_bags"])

test_hist_data = dict()
for p in test_bags:
    test_hist_data[p["subject_id"]] = len(p["code_bags"])

train_df = (
    pd.DataFrame.from_dict(train_hist_data, orient="index")
    .reset_index()
    .rename(columns={"index": "subject_id", 0: "num_bags"})
)
val_df = (
    pd.DataFrame.from_dict(val_hist_data, orient="index")
    .reset_index()
    .rename(columns={"index": "subject_id", 0: "num_bags"})
)
test_df = (
    pd.DataFrame.from_dict(test_hist_data, orient="index")
    .reset_index()
    .rename(columns={"index": "subject_id", 0: "num_bags"})
)

# Save the dataframes
train_df.to_csv(op.join(outfile_dir, "train_num_bags_per_patient.csv"), index=False)
val_df.to_csv(op.join(outfile_dir, "val_num_bags_per_patient.csv"), index=False)
test_df.to_csv(op.join(outfile_dir, "test_num_bags_per_patient.csv"), index=False)
