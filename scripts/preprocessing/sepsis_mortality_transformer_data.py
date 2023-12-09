import os.path as op
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from lab_transformers.data.tokenize_tabular_data import (
    make_bert_inputs,
    make_labrador_inputs,
    map_lab_values_to_eCDF_values,
)
from lab_transformers.utils import json_lines_loader

processed_data_dir = sys.argv[1]
evaluation_data_dir = sys.argv[2]
ecdf_path = sys.argv[3]
labrador_codebook_path = sys.argv[4]
bert_codebook_path = sys.argv[5]
labrador_null_token = int(sys.argv[6])

labrador_codebook = pd.read_csv(labrador_codebook_path)
bert_codebook = pd.read_csv(bert_codebook_path)

# Keep the top 23 most frequent lab tests (during the first 24 hours) and impute their missingness
# Note: less frequent lab tests have too many missing values to be used by tabular models
itemids_to_keep = [
    50912,
    51006,
    50902,
    51221,
    50971,
    50882,
    50868,
    51265,
    51301,
    51222,
    51248,
    51249,
    51250,
    51279,
    51277,
    50983,
    50931,
    50960,
    50893,
    50970,
    51237,
    51274,
    51275,
]

df = json_lines_loader(op.join(processed_data_dir, "sepsis.jsonl"))

mortality_df = {}
LOS_df = {}
# for each line, build a row of the eventual tabular dataset
for i, stay in tqdm(enumerate(df), desc="Creating tabular datasets"):
    mortality_row = {}
    LOS_row = {}
    mortality_row["mortality_indicator"] = stay["mortality_indicator"]
    LOS_row["length_of_stay"] = stay["length_of_stay"]

    for itemid in itemids_to_keep:
        # get index(es)
        itemid_ixs = np.argwhere(np.array(stay["itemids"]) == itemid).squeeze()

        if itemid_ixs.size == 0:
            mortality_row[itemid] = np.nan
            LOS_row[itemid] = np.nan
            continue

        if itemid_ixs.size > 1:
            # if the lab test occurred multiple times in the first 24-hours, take the latest result
            chart_times = np.take(stay["charttime"], itemid_ixs)
            ix = itemid_ixs[np.argmax(chart_times)]
        else:
            ix = itemid_ixs

        mortality_row[itemid] = stay["lab_values"][ix]
        LOS_row[itemid] = stay["lab_values"][ix]

    mortality_df[i] = mortality_row
    LOS_df[i] = LOS_row

# Build dataframe from dictionary as rows
mortality_df = pd.DataFrame.from_dict(mortality_df, orient="index")
LOS_df = pd.DataFrame.from_dict(LOS_df, orient="index")

# Convert column names to strings
mortality_df.columns = mortality_df.columns.astype(str)
LOS_df.columns = LOS_df.columns.astype(str)

# Convert lab values to their eCDF values
ecdfs = np.load(ecdf_path)
mortality_df = map_lab_values_to_eCDF_values(mortality_df, ecdfs)
LOS_df = map_lab_values_to_eCDF_values(LOS_df, ecdfs)

# Convert data to Labrador inputs
labrador_mortality_inputs, labels, _ = make_labrador_inputs(
    mortality_df,
    "mortality_indicator",
    labrador_codebook,
    labrador_null_token,
)
labrador_LOS_inputs, _, _ = make_labrador_inputs(
    LOS_df,
    "length_of_stay",
    labrador_codebook,
    labrador_null_token,
)

# Convert data to BERT inputs
bert_mortality_inputs, _, _ = make_bert_inputs(
    mortality_df, "mortality_indicator", bert_codebook
)
bert_LOS_inputs, _, _ = make_bert_inputs(LOS_df, "length_of_stay", bert_codebook)

# Save transformer_inputs and labels to numpy zip archive
labrador_mortality_inputs["label"] = labels
labrador_LOS_inputs["label"] = labels
bert_mortality_inputs["label"] = labels
bert_LOS_inputs["label"] = labels

np.savez(
    op.join(evaluation_data_dir, "sepsis_mortality_labrador"),
    **labrador_mortality_inputs
)
np.savez(op.join(evaluation_data_dir, "sepsis_LOS_labrador"), **labrador_LOS_inputs)
np.savez(op.join(evaluation_data_dir, "sepsis_mortality_bert"), **bert_mortality_inputs)
np.savez(op.join(evaluation_data_dir, "sepsis_LOS_bert"), **bert_LOS_inputs)
