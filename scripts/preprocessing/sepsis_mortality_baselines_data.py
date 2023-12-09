import os.path as op
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from lab_transformers.utils import json_lines_loader

processed_data_dir = sys.argv[1]
evaluation_data_dir = sys.argv[2]

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

# build dataframe from dictionary as rows
mortality_df = pd.DataFrame.from_dict(mortality_df, orient="index")
LOS_df = pd.DataFrame.from_dict(LOS_df, orient="index")

# Impute missing values in each dataframe with the mean of the column
mortality_df = mortality_df.fillna(mortality_df.mean())
LOS_df = LOS_df.fillna(LOS_df.mean())

mortality_df.to_csv(
    op.join(evaluation_data_dir, "sepsis_mortality_baselines_data.csv"), index=False
)
LOS_df.to_csv(
    op.join(evaluation_data_dir, "sepsis_los_baselines_data.csv"), index=False
)

print("\n COMPLETED SUCCESSFULLY \n", flush=True)
