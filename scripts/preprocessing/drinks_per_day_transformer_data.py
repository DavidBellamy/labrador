import os.path as op

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer

from lab_transformers.data.tokenize_tabular_data import (
    make_bert_inputs,
    make_labrador_inputs,
    map_lab_values_to_eCDF_values,
)

config = {
    "dataset_dir": "data/raw/",
    "dataset_name": "drinks_per_day.csv",
    "outfile_root": "drinks_per_day",
    "outfile_dir": "data/evaluations/",
    "label_column": "drinks",
    "ecdf_path": "data/processed/mimic4_ecdfs.npz",
    "labrador_codebook_path": "data/processed/labcode_codebook_labrador.csv",
    "bert_codebook_path": "data/processed/labcode_codebook_bert.csv",
    "task_type": "regression",
    "labrador_null_token": 531,
}

dataset_path = op.join(config["dataset_dir"], config["dataset_name"])
df = pd.read_csv(dataset_path)
label_col = config["label_column"]
feature_cols = list(df.columns)
feature_cols.remove(label_col)
ecdfs = np.load(config["ecdf_path"]) if config.get("ecdf_path") is not None else None
labrador_codebook = pd.read_csv(config["labrador_codebook_path"])
bert_codebook = pd.read_csv(config["bert_codebook_path"])

print("\n Converting lab values to their eCDF values \n", flush=True)
df = map_lab_values_to_eCDF_values(df, ecdfs)

print("\n Converting data to Labrador inputs \n", flush=True)
labrador_inputs, labels, non_mimic_feature_values = make_labrador_inputs(
    df,
    label_col,
    labrador_codebook,
    config["labrador_null_token"],
)

print("\n Converting data to BERT inputs \n", flush=True)
bert_inputs, _, _ = make_bert_inputs(df, label_col, bert_codebook)

label_encoder = LabelBinarizer()
labels = label_encoder.fit_transform(labels)

# Save transformer_inputs, labels and non_mimic_features to numpy zip archive
labrador_inputs["label"] = labels
bert_inputs["label"] = labels

if non_mimic_feature_values is not None:
    labrador_inputs["non_mimic_features"] = non_mimic_feature_values.to_numpy()
    bert_inputs["non_mimic_features"] = non_mimic_feature_values.to_numpy()

np.savez(
    op.join(config["outfile_dir"], config["outfile_root"] + "_labrador_inputs"),
    **labrador_inputs
)
np.savez(
    op.join(config["outfile_dir"], config["outfile_root"] + "bert_inputs"),
    **bert_inputs
)
