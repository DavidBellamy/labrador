import os.path as op
import sys

import numpy as np
import pandas as pd
from sklearn import preprocessing

from lab_transformers.data.tokenize_tabular_data import (
    make_bert_inputs,
    make_labrador_inputs,
    map_lab_values_to_eCDF_values,
)

raw_data_dir = sys.argv[1]
evaluation_data_dir = sys.argv[2]
ecdf_path = sys.argv[3]
label_column = str(sys.argv[4])
labrador_codebook_path = sys.argv[5]
bert_codebook_path = sys.argv[6]
labrador_null_token = int(sys.argv[7])

# The non-MIMIC features in this dataset
# are: Sex, Age, Suspect, PCR, KAL, NAT
df = pd.read_csv(op.join(raw_data_dir, "covid_diagnosis.csv"))
labrador_codebook = pd.read_csv(labrador_codebook_path)
bert_codebook = pd.read_csv(bert_codebook_path)
ecdfs = np.load(ecdf_path)

# Drop columns with too much missingness or that are not useful
df.drop(columns=["CK", "UREA"], inplace=True)

# Drop patients with >25% missing features
df.dropna(thresh=25, inplace=True)
df.reset_index(inplace=True, drop=True)

# Rename known lab tests with MIMIC-IV itemid's
df = df.rename(
    columns={
        "CA": "50893",
        "CREA": "50912",
        "ALP": "50863",
        "GGT": "50927",
        "GLU": "50931",
        "AST": "50878",
        "ALT": "50861",
        "LDH": "50954",
        "WBC": "51301",
        "RBC": "51279",
        "HGB": "51222",
        "HCT": "51221",
        "MCV": "51250",
        "MCH": "51248",
        "MCHC": "51249",
        "PLT1": "51265",
        "NE": "51256",
        "LY": "51244",
        "MO": "51254",
        "EO": "51200",
        "BA": "51146",
        "NET": "52075",
        "LYT": "51133",
        "MOT": "52074",
        "EOT": "52073",
        "BAT": "52069",
    }
)

# Multiply `CA` values by 4 to convert from mmol/L to mg/dL
df["50893"] = df["50893"] * 4

# Encode categorical feature variables
categorical_variables = ["Suspect", "Sex"]
enc = preprocessing.OrdinalEncoder()
df[categorical_variables] = enc.fit_transform(df[categorical_variables])

print("\n Converting lab values to their eCDF values \n", flush=True)
df_ecdf = map_lab_values_to_eCDF_values(df, ecdfs)

print("\n Converting data to Labrador inputs \n", flush=True)
labrador_inputs, labels, non_mimic_feature_values = make_labrador_inputs(
    df_ecdf,
    label_column,
    labrador_codebook,
    labrador_null_token,
)

print("\n Converting data to BERT inputs \n", flush=True)
bert_inputs, _, _ = make_bert_inputs(df_ecdf, label_column, bert_codebook)

# Save transformer_inputs, labels and non_mimic_features to numpy zip archive
labrador_inputs["label"] = labels
bert_inputs["label"] = labels

if non_mimic_feature_values is not None:
    continuous_variables = [
        col
        for col in non_mimic_feature_values.columns
        if col not in categorical_variables
    ]
    labrador_inputs["non_mimic_features_discrete"] = non_mimic_feature_values[
        categorical_variables
    ].to_numpy()
    labrador_inputs["non_mimic_features_continuous"] = non_mimic_feature_values[
        continuous_variables
    ].to_numpy()
    bert_inputs["non_mimic_features_discrete"] = non_mimic_feature_values[
        categorical_variables
    ].to_numpy()
    bert_inputs["non_mimic_features_continuous"] = non_mimic_feature_values[
        continuous_variables
    ].to_numpy()

np.savez(op.join(evaluation_data_dir, "covid_diagnosis_labrador"), **labrador_inputs)
np.savez(op.join(evaluation_data_dir, "covid_diagnosis_bert"), **bert_inputs)
