import sys

import numpy as np
import pandas as pd
from sklearn import preprocessing

from lab_transformers.data.tokenize_tabular_data import (
    make_bert_inputs,
    make_labrador_inputs,
    map_lab_values_to_eCDF_values,
)

evaluation_data_dir = sys.argv[1]
ecdf_path = sys.argv[2]
label_column = str(sys.argv[3])
labrador_codebook_path = sys.argv[4]
bert_codebook_path = sys.argv[5]
labrador_null_token = int(sys.argv[6])

df = pd.read_csv("data/raw/cancer_diagnosis.csv")
labrador_codebook = pd.read_csv(labrador_codebook_path)
bert_codebook = pd.read_csv(bert_codebook_path)
ecdfs = np.load(ecdf_path)

# Drop columns with too much missingness or that are not useful
df.drop(
    columns=[
        "Patient Number",
        "Urine epitheilum (UL)",
        "A/G Ratio",
        "Urine Ketone",
        "Urine Glucose",
        "Strip WBC",
    ],
    inplace=True,
)

# Rename known lab tests with MIMIC-IV itemid's (and correct typos from the raw dataset)
df = df.rename(
    columns={
        "Albumin": "50862",
        "Alk": "50863",
        "ALT (GPT)": "50861",
        "AST (GOT)": "50878",
        "BUN": "51006",
        "Calcium": "50893",
        "Chloride": "50902",
        "Creatinine": "50912",
        "Direct Bilirubin": "50883",
        "Glucose AC": "50931",
        "pH": "51491",
        "Potassium": "50971",
        "Sodium": "50983",
        "Specific Gravity": "51498",
        "Total Bilirubin": "50885",
        "Total Cholesterol": "50907",
        "Total Protein": "50976",
        "Triglyceride": "51000",
        "Urine epithelium count": "51476",
        "Uric acid": "51007",
        "Hyper1en1ion": "Hypertension",
        "Diabe1es": "Diabetes",
        "Bee1leNu1": "BetelNut",
        "FamilyHis1ory": "FamilyHistory",
    }
)

# Encode categorical feature variables
categorical_variables = [
    "FamilyHistory",
    "BetelNut",
    "Drinking",
    "Smoking",
    "Diabetes",
    "Hypertension",
    "gender",
    "Nitrite",
    "Urine Bilirubin",
]
enc = preprocessing.OrdinalEncoder(encoded_missing_value=-1)
df[categorical_variables] = enc.fit_transform(df[categorical_variables])

# Integer-Encode the label
le = preprocessing.LabelEncoder()
le.fit(df[label_column])
df[label_column] = le.transform(df[label_column])

# In the non-mimic-lab columns, impute the remaining miss values with the column mean
non_mimic_lab_columns = [col for col in df.columns if not col.isdigit()]
df[non_mimic_lab_columns] = df[non_mimic_lab_columns].fillna(
    df[non_mimic_lab_columns].mean()
)

print("\n Converting lab values to their eCDF values \n", flush=True)
df = map_lab_values_to_eCDF_values(df, ecdfs)

print("\n Converting data to Labrador inputs \n", flush=True)
(
    labrador_inputs,
    labels,
    non_mimic_feature_values,
) = make_labrador_inputs(df, label_column, labrador_codebook, labrador_null_token)

print("\n Converting data to BERT inputs \n", flush=True)
bert_inputs, _, _ = make_bert_inputs(df, label_column, bert_codebook)

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

np.savez("data/evaluations/cancer_diagnosis_labrador_data", **labrador_inputs)
np.savez("data/evaluations/cancer_diagnosis_bert_data", **bert_inputs)
