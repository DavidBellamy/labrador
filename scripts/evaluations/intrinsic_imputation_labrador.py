import json
import os
import os.path as op
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from lab_transformers.models.labrador.model import Labrador
from lab_transformers.utils import json_lines_loader


def labrador_imputer(
    model: Labrador,
    data: List[Dict[str, Any]],
    codebook: pd.DataFrame,
    config: Dict,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate random missing lab values in `data` then impute them using `model`.
    Randomness is controlled by `rng`.
    `codebook` is included solely to map lab code tokens to itemids for easier
    data visualizations and interpretation of results.
    `config` is a Dict that contains the batch_size, mask_token, null_token, and pad_token.
        These are needed to generate the masked and padded bags of labs for imputation.

    Returns a DataFrame with the following columns:
    - ypred: lab value imputed by `model`
    - ytrue: true lab value
    - subject_id: patient id
    - charttime: hospital chart time of the lab test
    - itemid: MIMIC-IV lab code of the imputed lab value

    """
    # pick random patients
    patient_ixs = rng.choice(len(data), config["batch_size"])

    # pick random bag for each patient
    bag_ixs = [rng.choice(len(data[ix]["value_bags"])) for ix in patient_ixs]

    # collect patient id's
    patient_ids = [data[ix]["subject_id"] for ix in patient_ixs]

    # pick random token to mask in each bag
    mask_ixs = [
        rng.choice(len(data[p_ix]["value_bags"][b_ix]))
        for p_ix, b_ix in zip(patient_ixs, bag_ixs)
    ]

    masked_value_bags = [
        [
            config["mask_token"] if ix == m_ix else x
            for ix, x in enumerate(data[p_ix]["value_bags"][b_ix])
        ]
        for p_ix, b_ix, m_ix in zip(patient_ixs, bag_ixs, mask_ixs)
    ]

    masked_code_bags = [
        [
            config["mask_token"] if ix == m_ix else x
            for ix, x in enumerate(data[p_ix]["code_bags"][b_ix])
        ]
        for p_ix, b_ix, m_ix in zip(patient_ixs, bag_ixs, mask_ixs)
    ]

    # Replace <NULL> (i.e. missing) lab values with null token
    masked_value_bags = [
        [config["null_token"] if x == "<NULL>" else x for x in bag]
        for bag in masked_value_bags
    ]

    # Pad the masked bags
    max_bag_length = np.max([len(bag) for bag in masked_value_bags])
    padded_value_bags = tf.keras.preprocessing.sequence.pad_sequences(
        masked_value_bags,
        maxlen=max_bag_length,
        dtype="float32",
        padding="post",
        truncating="post",
        value=config["pad_token"],
    )

    padded_code_bags = tf.keras.preprocessing.sequence.pad_sequences(
        masked_code_bags,
        maxlen=max_bag_length,
        dtype="int32",
        padding="post",
        truncating="post",
        value=config["pad_token"],
    )

    ytrues = [
        data[p_ix]["value_bags"][b_ix][m_ix]
        for p_ix, b_ix, m_ix in zip(patient_ixs, bag_ixs, mask_ixs)
    ]
    ytrue_code = [
        data[p_ix]["code_bags"][b_ix][m_ix]
        for p_ix, b_ix, m_ix in zip(patient_ixs, bag_ixs, mask_ixs)
    ]
    itemids = [
        codebook[codebook.frequency_rank == code].itemid.item() for code in ytrue_code
    ]
    charttimes = [
        data[p_ix]["charttime"][b_ix] for p_ix, b_ix in zip(patient_ixs, bag_ixs)
    ]

    # Get model outputs
    outputs = model(
        {"continuous_input": padded_value_bags, "categorical_input": padded_code_bags},
        training=False,
    )

    # Filter outputs to only include the masked values
    zeros = tf.cast(tf.zeros_like(padded_value_bags), dtype=tf.bool)
    ones = tf.cast(tf.ones_like(padded_value_bags), dtype=tf.bool)
    mask = tf.where(padded_value_bags == config["mask_token"], ones, zeros)
    ypreds = tf.boolean_mask(outputs["continuous_output"], mask).numpy().squeeze()

    return pd.DataFrame(
        data={
            "ypred": ypreds,
            "ytrue": ytrues,
            "subject_id": patient_ids,
            "charttime": charttimes,
            "itemid": itemids,
        }
    )


if __name__ == "__main__":
    time_string = time.strftime("%Y%m%d-%H%M%S")
    config_path = sys.argv[1]

    with open(config_path) as f:
        config = json.load(f)

    rng = np.random.default_rng(config["random_seed"])

    test_data = json_lines_loader(config["dataset_path"])
    codebook = pd.read_csv(config["codebook_path"])

    with open(op.join(config["model_path"], "config.json")) as f:
        model_config = json.load(f)

    if config["ablation"].lower() == "true":
        model = Labrador(model_config)
    else:
        model = Labrador(model_config)
        model.load_weights(op.join(config["model_path"], "variables/variables"))

    # Generate imputations batch-wise
    result_dfs = []
    for _ in tqdm(
        range(config["num_batches"]),
        desc="Running intrinsic imputation with Labrador",
    ):
        result_dfs.append(labrador_imputer(model, test_data, codebook, config, rng))
    df = pd.concat(result_dfs).replace("<NULL>", np.nan)

    # Create results directory if it doesn't exist
    if not op.exists(config["output_directory"]):
        os.mkdir(config["output_directory"])

    if config["ablation"].lower() == "true":
        df.to_csv(
            op.join(
                config["output_directory"],
                f"intrinsic_imputation_labrador_{time_string}_ablation",
            )
            + ".csv",
            index=False,
        )
    else:
        df.to_csv(
            op.join(
                config["output_directory"],
                f"intrinsic_imputation_labrador_{time_string}",
            )
            + ".csv",
            index=False,
        )
