import os.path as op
import sys

import pandas as pd
from sklearn import preprocessing

raw_data_dir = sys.argv[1]
evaluation_data_dir = sys.argv[2]

# The non-MIMIC features in this dataset
# are: Sex, Age, Suspect, PCR, KAL, NAT
df = pd.read_csv(op.join(raw_data_dir, "covid_diagnosis.csv"))

# Drop columns with too much missingness
df.drop(columns=["CK", "UREA"], inplace=True)

# Drop patients with >25% missing features
df.dropna(thresh=25, inplace=True)

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
categorical_variables = ["Suspect"]
enc = preprocessing.OrdinalEncoder()
df[categorical_variables] = enc.fit_transform(df[categorical_variables])

# Save the cleaned dataset
df.to_csv(
    op.join(evaluation_data_dir, "covid_diagnosis_baselines_data.csv"), index=False
)
