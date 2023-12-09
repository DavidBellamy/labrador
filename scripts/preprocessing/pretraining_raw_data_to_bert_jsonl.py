"""
This script loads the MIMIC-IV labevents.csv data at data_path, and creates a JSON line for each patient that contains:
subject_id, tokens, time_deltas, hospital admission id's, and charttimes,
where time_deltas are the time in days (float) between each lab measurement.
"""

import json
from itertools import groupby
import os
import os.path as op
import sqlite3
import sys
from typing import Dict, Tuple, Union

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from statsmodels.distributions import ECDF
from tqdm import tqdm

from lab_transformers.data.tokenize_tabular_data import mimic4_eCDFer
from lab_transformers.utils import NpEncoder


class MakeJSONlines:
    def __init__(
        self,
        raw_lab_data_file_name: str,
        raw_admissions_data_file_name: str,
        data_path: str,
        output_path: str,
        random_seed: int,
        train_pct: float,
        val_pct: float,
        test_pct: float,
        min_frequency: int,
        num_bins: int = 10,
    ) -> None:
        self.raw_labfile = raw_lab_data_file_name
        self.raw_admissionsfile = raw_admissions_data_file_name
        self.data_path = data_path
        self.output_path = output_path
        self.random_seed = random_seed
        self.train_pct = train_pct
        self.val_pct = val_pct
        self.test_pct = test_pct
        self.min_frequency = min_frequency
        self.num_bins = num_bins

        # Create a controllabe RNG for data splitting
        self.rng = np.random.default_rng(self.random_seed)

        # Initialize attribute for holding the frequency ranks of the categorical vocabulary
        # This is filled in by the compute_frequency_ranks() method
        self.frequency_ranks = None

    def call(self, test_number: Union[int, None] = None) -> None:
        print("Loading raw data...\n")
        df, admissions = self.load_data()

        print("Filtering low frequency lab tests...\n")
        df = self.filter_rare_categorical(df)

        print("Merging in hadm_id's for each lab test...\n")
        df = self.merge_in_hadm_id_from_admissions(df, admissions)

        print("Computing frequency rankings of lab codes...\n")
        self.frequency_ranks = self.compute_frequency_ranks(df)

        print("Computing time deltas between labs...\n")
        df = self.compute_time_delta(df)

        print("Splitting data into train, validation, test...\n")
        patient_dict, data_dict = self.split_data(df)

        print("Transforming lab values into probabilities via the eCDF...\n")
        data_dict = self.probability_transform_values(data_dict)

        print("Converting eCDF probabilities to tokens...\n")
        codebook = self.probability_to_tokens(data_dict)

        print("Writing JSON lines to disk...\n")
        self.write_json_lines(patient_dict, data_dict, test_number, codebook)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Load labevents.csv (requires ~32Gb of memory)
        lab_data_path = os.path.join(self.data_path, self.raw_labfile)

        labevents = pd.read_csv(
            lab_data_path,
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

        # Subset to only the columns needed for json lines
        columns_needed = [
            "subject_id",
            "itemid",
            "valuenum",
            "value",
            "charttime",
            "hadm_id",
        ]
        df = labevents[columns_needed]

        # Load admissions.csv (will be used to merge in hadm_id's for each lab test)
        admissions_data_path = os.path.join(self.data_path, self.raw_admissionsfile)
        admissions = pd.read_csv(admissions_data_path)
        admissions["admittime"] = pd.to_datetime(admissions["admittime"])
        admissions["dischtime"] = pd.to_datetime(admissions["dischtime"])
        admissions = admissions[
            ["subject_id", "hadm_id", "admittime", "dischtime", "edregtime"]
        ]  # subset the necessary cols

        return df, admissions

    def filter_rare_categorical(self, raw_lab_data: pd.DataFrame) -> pd.DataFrame:
        # Filter out itemid's with insufficient frequency
        # Note: the first filter condition selects lab codes that have no numeric values but occur >= MIN_FREQUENCY times
        # the second filter condition selects lab codes that have numeric values,
        # and both the numeric values and codes occur >= MIN_FREQUENCY times

        filtered_lab_data = raw_lab_data.groupby("itemid").filter(
            lambda x: (
                len(x["itemid"]) >= self.min_frequency
                and len(x["valuenum"].dropna()) == 0
            )
            or (
                len(x["valuenum"].dropna()) >= self.min_frequency
                and len(x["itemid"]) >= self.min_frequency
            )
        )

        return filtered_lab_data

    def merge_in_hadm_id_from_admissions(
        self, df: pd.DataFrame, admissions: pd.DataFrame
    ) -> pd.DataFrame:
        # Make the db in memory (requires 90+ Gb of memory with full labevents.csv)
        conn = sqlite3.connect(":memory:")

        # write the tables
        df.to_sql("df", conn, index=False)
        admissions.to_sql("admissions", conn, index=False)

        qry = """select 
                    df.subject_id, 
                    itemid,
                    valuenum,
                    charttime,
                    df.hadm_id labs_hadm_id, 
                    admissions.hadm_id adm_hadm_id
                from df left join admissions 
                    on ((charttime between case 
                                            when edregtime is not null then min(edregtime, admittime) 
                                            else admittime end 
                                        and dischtime) 
                                    and admissions.subject_id = df.subject_id)"""

        # Perform the SQL merge/join
        df = pd.read_sql_query(qry, conn)

        # Drop rows where both hadm_id's exist but aren't equal (only ~0.01% of rows have this)
        df = df[
            ~(
                (df.labs_hadm_id != df.adm_hadm_id)
                & ~(df.labs_hadm_id.isnull())
                & ~(df.adm_hadm_id.isnull())
            )
        ]

        # Merge the two hadm_id columns together
        df["hadm_id"] = df["labs_hadm_id"].fillna(df["adm_hadm_id"])

        # Drop the labs_hadm_id and adm_hadm_id columns
        df = df.drop(["labs_hadm_id", "adm_hadm_id"], axis=1)

        return df

    def compute_frequency_ranks(self, raw_lab_data: pd.DataFrame) -> Dict[str, int]:
        # Next, we will determine the integer frequency rank of each lab code in the raw data

        # compute frequency of each unique lab code
        labcode_freqs = dict(raw_lab_data.itemid.value_counts())

        # replace frequencies of lab codes with their integer rank (ranks start at 1)
        frequency_ranks = {}
        for i, (key, _) in enumerate(labcode_freqs.items()):
            frequency_ranks[key] = i + 1

        # Save the map from MIMIC-IV lab codes to their frequency ranks (useful for getting descriptions of lab codes)
        codebook = pd.DataFrame.from_dict(frequency_ranks, orient="index").reset_index()
        codebook.columns = ["itemid", "frequency_rank"]

        d_labitems = os.path.join(self.data_path, "d_labitems.csv")
        labitem_descriptions = pd.read_csv(
            d_labitems
        )  # load descriptions of each lab code
        codebook = codebook.merge(
            labitem_descriptions, on="itemid"
        )  # merge the descriptions with the codebook

        filename = os.path.join(self.output_path, "labcode_codebook_labrador.csv")
        codebook.to_csv(filename, index=False)  # save the codebook

        return frequency_ranks

    def compute_time_delta(self, df: pd.DataFrame) -> pd.DataFrame:
        # Convert charttime Pandas datetime (for computing time deltas later)
        df["charttime"] = pd.to_datetime(df["charttime"])

        # Sort by subject_id and charttime (ascending)
        df = df.sort_values(["subject_id", "charttime"], inplace=False)

        # calculate time deltas (next time minus previous time)
        df["time_delta"] = df.charttime - df.charttime.shift(1)

        # correct rows at border between 2 patients (replace with 0)
        df.loc[(df.subject_id != df.subject_id.shift(1)), "time_delta"] = pd.Timedelta(
            "0 days"
        )

        # Convert time_delta's to decimal days (e.g. 5.35 days)
        df["time_delta"] = df["time_delta"].dt.total_seconds() / (60 * 60 * 24)

        return df

    def split_data(
        self, df: pd.DataFrame
    ) -> Tuple[Dict[str, NDArray[np.integer]], Dict[str, pd.DataFrame]]:
        # Sort patients into train/validation/test sets
        patient_list = df.subject_id.unique()

        # Shuffle the order of patients
        self.rng.shuffle(patient_list)

        train_size = int(np.floor(self.train_pct * len(patient_list)))
        val_size = int(np.ceil(self.val_pct * len(patient_list)))
        test_size = int(len(patient_list) - train_size - val_size)

        train_patients = patient_list[:train_size]
        val_patients = patient_list[train_size : train_size + val_size]
        test_patients = patient_list[train_size + val_size :]

        # Split out the training data
        train_df = df[df.subject_id.isin(train_patients)]

        # Extract the unique itemid's from the training data partition
        train_itemids = train_df.itemid.unique()

        # Split out the val/test sets if the itemid also exists in the training data
        val_df = df[
            (df.subject_id.isin(val_patients)) & (df.itemid.isin(train_itemids))
        ]
        test_df = df[
            (df.subject_id.isin(test_patients)) & (df.itemid.isin(train_itemids))
        ]

        return {
            "train_patients": train_patients,
            "val_patients": val_patients,
            "test_patients": test_patients,
        }, {"train_df": train_df, "val_df": val_df, "test_df": test_df}

    def probability_transform_values(
        self, splits: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        train_df = splits["train_df"]
        val_df = splits["val_df"]
        test_df = splits["test_df"]

        unique_itemids = train_df.itemid.unique()
        compressed_ecdf_data = {}
        for itemid in tqdm(unique_itemids, desc="Computing eCDFs"):
            lab_values = train_df[
                ~np.isnan(train_df.valuenum) & (train_df.itemid == itemid)
            ]["valuenum"].values

            if len(lab_values) == 0:
                continue

            # Calculate the empirical CDF for the current lab test
            ecdf = ECDF(lab_values)

            # Compress the eCDF to just the unique lab values (and their probabilities)
            unique_ixs = []
            cum_lengths = 0
            for _, g in groupby(ecdf.x):
                group = list(g)
                cum_lengths += len(group)
                unique_ix = cum_lengths - 1
                unique_ixs.append(unique_ix)

            # Store the resulting compressed eCDF data
            compressed_ecdf_data[f"{itemid}_x"] = ecdf.x[unique_ixs]
            compressed_ecdf_data[f"{itemid}_y"] = ecdf.y[unique_ixs]

        # Save the compressed eCDF values and probabilities
        np.savez(op.join(self.output_path, "mimic4_ecdfs.npz"), **compressed_ecdf_data)

        # Load the result back and use it to probability transform the validation and test data splits
        ecdf_data = np.load(op.join(self.output_path, "mimic4_ecdfs.npz"))
        eCDFer = mimic4_eCDFer(ecdf_data)

        # filter rows to just itemid's in the eCDF Numpy zip archive (npz)
        train_df[train_df.itemid.isin(eCDFer.itemids)] = train_df[
            train_df.itemid.isin(eCDFer.itemids)
        ].apply(eCDFer, axis=1)
        val_df = val_df[val_df.itemid.isin(eCDFer.itemids)]
        test_df = test_df[test_df.itemid.isin(eCDFer.itemids)]

        train_df.apply(
            eCDFer, axis=1
        )  # only apply if valuenum is not nan and itemid exists in eCDFs

        for itemid in train_df["itemid"].unique():
            train_df[train_df.itemid == itemid]

        # Apply the training eCDFer to data splits
        train_df["probs"] = eCDFer(train_df["itemid"], train_df["valuenum"])
        val_df["probs"] = eCDFer(val_df["itemid"], val_df["valuenum"])
        test_df["probs"] = eCDFer(test_df["itemid"], test_df["valuenum"])

        return {"train_df": train_df, "val_df": val_df, "test_df": test_df}

    def probability_to_tokens(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        # map all probability values in column `probs` to tokens
        train_df = data_dict["train_df"]

        # compute frequency of each unique lab code
        labcode_freqs = dict(train_df.itemid.value_counts())

        # replace frequencies of lab codes with their integer rank (ranks start at 1)
        frequency_ranks = {}
        for i, (key, _) in enumerate(labcode_freqs.items()):
            frequency_ranks[key] = i + 1

        unique_itemids_with_values = train_df.dropna()["itemid"].unique()
        itemids_without_values = list(
            set(train_df["itemid"].unique()).difference(set(unique_itemids_with_values))
        )
        quantiles = [i / self.num_bins for i in range(1, self.num_bins + 1)]
        quantiles.append(np.nan)
        itemids = np.repeat(unique_itemids_with_values, repeats=self.num_bins + 1)
        upper_bounds = np.tile(quantiles, len(unique_itemids_with_values))
        codebook = pd.DataFrame(data={"itemid": itemids, "valuenum": upper_bounds})

        nan_itemids = pd.DataFrame(
            data={"itemid": itemids_without_values, "valuenum": np.nan}
        )
        codebook = pd.concat([codebook, nan_itemids], axis=0)

        # merge in frequency ranks
        codebook["frequency_rank"] = codebook["itemid"].map(frequency_ranks)

        # sort rows by (code frequency rank, quantile)
        codebook = codebook.sort_values(by=["frequency_rank", "valuenum"])

        # create integer tokens starting at 1
        codebook["token"] = range(1, len(codebook) + 1)

        # Save the map from MIMIC-IV lab codes to their frequency ranks (useful for getting descriptions of lab codes)
        d_labitems = os.path.join(self.data_path, "d_labitems.csv")
        labitem_descriptions = pd.read_csv(
            d_labitems
        )  # load descriptions of each lab code
        codebook = codebook.merge(
            labitem_descriptions, on="itemid"
        )  # merge the descriptions with the codebook

        filename = os.path.join(self.output_path, "labcode_codebook_bert.csv")
        codebook.to_csv(filename, index=False)  # save the codebook

        return codebook

    def write_json_lines(
        self,
        patient_dict: Dict[str, NDArray[np.integer]],
        data_dict: Dict[str, pd.DataFrame],
        test_number: Union[int, None],
        codebook: pd.DataFrame,
    ) -> None:
        # Create the output paths for the 3 data splits
        train_jsonl_file = os.path.join(
            self.output_path, f"bert_train_patients{test_number}.jsonl"
        )
        val_jsonl_file = os.path.join(
            self.output_path, f"bert_validation_patients{test_number}.jsonl"
        )
        test_jsonl_file = os.path.join(
            self.output_path, f"bert_test_patients{test_number}.jsonl"
        )

        # Write the 3 data splits to their respective paths
        self.json_lines_writer(
            train_jsonl_file,
            patient_dict["train_patients"],
            data_dict["train_df"],
            "training",
            codebook,
        )
        self.json_lines_writer(
            val_jsonl_file,
            patient_dict["val_patients"],
            data_dict["val_df"],
            "validation",
            codebook,
        )
        self.json_lines_writer(
            test_jsonl_file,
            patient_dict["test_patients"],
            data_dict["test_df"],
            "testing",
            codebook,
        )

    def json_lines_writer(
        self,
        filepath: str,
        patient_list: NDArray[np.integer],
        df: pd.DataFrame,
        name: str,
        codebook: pd.DataFrame,
    ) -> None:
        # Generate JSON lines and write to train_set.jsonl, val_set.jsonl, and test_set.jsonl at output_path
        first_line = True
        mode = "w"

        # Make an index out of subject_id for faster subsetting of the df
        df.set_index("subject_id", inplace=True)

        for patient in tqdm(patient_list, desc=f"Writing {name} JSON lines..."):
            temp = df.loc[df.index == patient]

            # Filter out patients that only have a single lab (no bag to learn context from)
            if len(temp) < 2:
                continue  # skip this patient

            # retrieve the token for a given (itemid, valuenum) pair from `codebook`
            tokens = []
            for itemid, prob in zip(temp.itemid.values, temp.probs.values):
                if np.isnan(prob):
                    tokens.append(
                        codebook[
                            (codebook.itemid == itemid) & (np.isnan(codebook.valuenum))
                        ].token.values[0]
                    )
                else:
                    tokens.append(
                        codebook[
                            (codebook.itemid == itemid) & (prob <= codebook.valuenum)
                        ].token.min()
                    )

            # Create individual patient JSON line
            patient_jsonl = {
                "subject_id": patient,
                "token": tokens,
                "time_deltas": temp.time_delta.values.tolist(),
                "hadm_id": temp.hadm_id.values.tolist(),
                "charttime": np.datetime_as_string(temp.charttime, unit="m").tolist(),
            }

            # Write it to file
            with open(filepath, mode=mode, encoding="utf-8") as f:
                json_record = json.dumps(patient_jsonl, cls=NpEncoder)
                f.write(json_record + "\n")

                if first_line:
                    mode = "a"
                    first_line = False


if __name__ == "__main__":
    # Collect sys args for the preprocessor
    data_path = sys.argv[1]  # Path to labevents.csv
    output_path = sys.argv[2]  # Path to /data/processed
    RANDOM_SEED = int(sys.argv[3])
    train_pct = float(sys.argv[4])
    val_pct = float(sys.argv[5])
    test_pct = float(sys.argv[6])
    MIN_FREQUENCY = int(
        sys.argv[7]
    )  # The minimum frequency for lab codes (to be included in the data)
    num_bins = int(sys.argv[8])

    # Initialize the preprocessor
    preprocessor = MakeJSONlines(
        "labevents.csv",
        "admissions.csv",
        data_path,
        output_path,
        RANDOM_SEED,
        train_pct,
        val_pct,
        test_pct,
        MIN_FREQUENCY,
        num_bins,
    )

    # Perform preprocessing
    preprocessor.call()
