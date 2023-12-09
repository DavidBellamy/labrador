import json
import os.path as op
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from lab_transformers.utils import json_lines_loader, NpEncoder


def make_lab_bags_for_labrador(
    jsonl_batch: list,
    filepath: str,
    max_time_delta: float,
    min_bag_length: int = 3,
    null_threshold: float = 0.5,
) -> None:
    """Creates all unique bags of labs spanning max_time_delta (and with size min_bag_length) for the patients
    in jsonl_batch. Bags with > null_threshold missing lab values (ie. <NULL> strings) are filtered out.

    Inputs:
    > jsonl_batch: a list of JSON lines, where each line contains the 6 keys: subject_id, lab_codes, lab_values,
        time_deltas, hadm_id, and charttime.
    > filepath: a string specifying the path to the desired output jsonl file.
    > max_time_delta: a float specifying the maximum time period that a bag may span.
    > min_bag_length: a positive integer specifying the minimum length requirement for each bag.

    Returns:
    > No return value, has the side effect of writing JSON lines containing all precomputed bags for each patient to
        the file at filepath. Each JSON line has the following structure:
        {'subject_id': 123456, code_bags: [[1, 2, 3], [4, 5, 6]], value_bags: [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        'hadm_id': [101, 102], 'charttime': ["2175-12-30T17:03", "2143-08-14T05:01"]}

        The hadm_id is the hospital admission ID for each corresponding bag in code_bags and value_bags. This may have
        missingness. Similarly, 'charttime' is the moment when the labs were added to the patient's chart. When
        max_time_delta = 0, each bag only has 'charttime' value, whereas bags with larger values of max_time_delta could
        have multiple, in which case we take the minimum of all those times (i.e. the start time of the bag).
    """

    # For each patient loop over time deltas and construct bags of labs with max_time_delta width
    # Redundant subsets are filtered out
    # Only bags with min_bag_length will be included
    output_jsonl = []
    for patient in tqdm(jsonl_batch, desc="Making bags of labs"):
        # Separate out the patient's data components (reduces the time spent indexing below)
        time_deltas = patient["time_deltas"]
        lab_codes = patient["lab_codes"]
        lab_values = patient["lab_values"]
        hadm_ids = patient["hadm_id"]
        charttimes = patient["charttime"]

        bags_of_lab_indexes = (
            []
        )  # will hold the bags of indexes, which correspond to bags of codes/values
        code_bags = []  # will hold the bags of codes for the current patient
        value_bags = []  # will hold the bags of values for the current patient
        hadm_id_list = []  # will hold the hadm_id for each bag of codes/values
        charttime_list = []  # will hold the start time fore ach bag of codes/values

        end_of_list = len(patient["time_deltas"])
        for index in range(end_of_list):
            # Start a set of indexes to be returned, beginning with the current index
            index_list = [index]

            # collect indexes going rightwards until max_time_delta is surpassed or end of list is reached
            cumsum = 0
            while True:
                index += 1
                if index >= end_of_list:
                    break

                cumsum += time_deltas[index]
                if cumsum > max_time_delta:
                    break

                index_list.append(index)

            # pass if the proposed bag of lab indexes is not at least min_bag_length
            if len(index_list) < min_bag_length:
                continue

            # pass if the proposed bag contains > null_threshold missing lab values
            null_proportion = [lab_values[i] for i in index_list].count("<NULL>") / len(
                index_list
            )
            if null_proportion > null_threshold:
                continue

            # collect this proposed bag of lab indexes, only if it isn't a subset of any that came before it
            sets = {frozenset(e) for e in bags_of_lab_indexes}
            proposed_indexes = set(index_list)
            if not any(proposed_indexes <= s for s in sets):
                bags_of_lab_indexes.append(index_list)

                # Convert the bag of lab indexes into the corresponding lab codes, values, hadm_id's and charttimes
                codes = [lab_codes[i] for i in index_list]
                values = [lab_values[i] for i in index_list]
                temp_hadm_ids = [hadm_ids[i] for i in index_list]
                temp_charttimes = np.array(
                    [pd.to_datetime(charttimes[i]) for i in index_list],
                    dtype=np.datetime64,
                )
                bag_start_time = min(temp_charttimes)

                # If there were multiple hospital admission IDs for the same bag, assign 'NA' to this bag's hadm_id
                if len(set(temp_hadm_ids)) > 1:
                    hadm_id = float("nan")
                else:
                    hadm_id = temp_hadm_ids[
                        0
                    ]  # take the first hadm_id from the list, since all are the same

                code_bags.append(codes)
                value_bags.append(values)
                hadm_id_list.append(hadm_id)
                charttime_list.append(bag_start_time)

        if len(bags_of_lab_indexes) > 0:
            patient_jsonl = {
                "subject_id": patient["subject_id"],
                "code_bags": code_bags,
                "value_bags": value_bags,
                "hadm_id": hadm_id_list,
                "charttime": np.datetime_as_string(charttime_list, unit="m").tolist(),
            }
            output_jsonl.append(patient_jsonl)

    # Write JSON lines
    first_line = True
    mode = "w"
    for patient in tqdm(output_jsonl, desc=f"Writing JSON lines..."):
        # Write patient to file
        with open(filepath, mode=mode, encoding="utf-8") as f:
            json_record = json.dumps(patient, cls=NpEncoder)
            f.write(json_record + "\n")

            if first_line:
                mode = "a"
                first_line = False


if __name__ == "__main__":
    # Collect sys args
    data_path = sys.argv[1]  # Path to /data/processed
    output_path = data_path
    max_time_delta = float(sys.argv[2])
    min_bag_length = int(sys.argv[3])
    null_threshold = float(sys.argv[4])

    # Load some JSON lines data
    train_jsonl_path = op.join(data_path, "labrador_train_patients.jsonl")
    val_jsonl_path = op.join(data_path, "labrador_validation_patients.jsonl")
    test_jsonl_path = op.join(data_path, "labrador_test_patients.jsonl")

    path_list = [train_jsonl_path, val_jsonl_path, test_jsonl_path]
    out_filenames = [
        op.join(
            data_path,
            f"labrador_train_bags.jsonl",
        ),
        op.join(
            data_path,
            f"labrador_validation_bags.jsonl",
        ),
        op.join(
            data_path,
            f"labrador_test_bags.jsonl",
        ),
    ]

    for path, outfile_name in zip(path_list, out_filenames):
        jsonl = json_lines_loader(path)
        make_lab_bags_for_labrador(
            jsonl, outfile_name, max_time_delta, min_bag_length, null_threshold
        )
