import json
import os.path as op
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from lab_transformers.utils import NpEncoder

# Arguments
raw_data_directory = sys.argv[1]
processed_data_directory = sys.argv[2]

# Configuration
outfile_path = op.join(processed_data_directory, "sepsis.jsonl")
admissions_path = op.join(raw_data_directory, "admissions.csv")
sepsis_cohort_path = op.join(raw_data_directory, "mimic4_sepsis_cohort.csv")
labevents_path = op.join(raw_data_directory, "labevents.csv")

# Get outcome variables (mortality & LOS) for each septic patient
admission = pd.read_csv(admissions_path)
sepsis_cohort = pd.read_csv(sepsis_cohort_path)

# Shift the ICU admittime's for linking to sepsis cohort
admission["admittime"] = pd.to_datetime(admission["admittime"])
admission["dischtime"] = pd.to_datetime(admission["dischtime"])
admission["admittime_shifted"] = admission["admittime"] - pd.Timedelta(hours=4)

df = admission.merge(sepsis_cohort, how="left", on=["subject_id"]).query(
    "sofa_time.between(`admittime_shifted`, `dischtime`)"
)

# Create binary indicator for mortality
df["mortality_indicator"] = 0
df.loc[
    ~df.deathtime.isna() | (df.discharge_location == "DIED"), "mortality_indicator"
] = 1

# Create length of stay variable (in days): difference between dischtime and admittime
df["length_of_stay"] = (df["dischtime"] - df["admittime"]).dt.total_seconds() / 86_400

# Create survival time variable (i.e. LOS for those who died)
df["survival_time"] = np.nan
df.loc[df.mortality_indicator == 1, "survival_time"] = df.loc[
    df.mortality_indicator == 1, "length_of_stay"
]

# Get lab features for each septic patient (from first 24hr of their ICU stay, since suspected of infection)
print("\n Loading MIMIC-IV labevents.csv \n", flush=True)
labevents = pd.read_csv(
    labevents_path,
    dtype={
        "labevent_id": int,
        "subject_id": int,
        "hadm_id": "Int64",  # Pandas nullable Int type
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

labevents["charttime"] = pd.to_datetime(labevents["charttime"])

print("\n Linking sepsis cohort to lab tests \n", flush=True)
df["24h_end"] = df["admittime"] + pd.Timedelta(hours=24)
labs = labevents.merge(df, how="left", on=["subject_id"]).query(
    "charttime.between(`admittime`, `24h_end`)"
)

# Subset to just the columns that are needed
labs = labs[
    [
        "subject_id",
        "stay_id",
        "admittime",
        "dischtime",
        "itemid",
        "valuenum",
        "charttime",
        "mortality_indicator",
        "length_of_stay",
        "sofa_score",
    ]
]

# Generate JSON lines
first_line = True
mode = "w"

# Make an index out of subject_id for faster subsetting of the df
stay_ids = labs["stay_id"].unique()
labs.set_index("stay_id", inplace=True)

for stay_id in tqdm(stay_ids, desc=f"Writing JSON lines..."):
    temp = labs.loc[labs.index == stay_id]

    # Filter out patients that only have a single lab (no bag to learn context from)
    if len(temp) < 10:
        continue  # skip this patient

    # Create individual patient JSON line
    patient_jsonl = {
        "subject_id": int(temp.subject_id.values[0]),
        "stay_id": int(stay_id),
        "admittime": np.datetime_as_string(temp.admittime.values[0], unit="m"),
        "dischtime": np.datetime_as_string(temp.dischtime.values[0], unit="m"),
        "itemids": temp.itemid.values.tolist(),
        "lab_values": temp.valuenum.values.tolist(),
        "charttime": np.datetime_as_string(temp.charttime, unit="m").tolist(),
        "mortality_indicator": int(temp.mortality_indicator.values[0]),
        "length_of_stay": round(temp.length_of_stay.values[0], 3),
        "sofa_score": temp.sofa_score.values[0],
    }

    # Write it to file
    with open(outfile_path, mode=mode, encoding="utf-8") as f:
        json_record = json.dumps(patient_jsonl, cls=NpEncoder)
        f.write(json_record + "\n")

        if first_line:
            mode = "a"
            first_line = False
