import json
import os
import os.path as op
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import TFBertForMaskedLM, BertConfig
from tqdm import tqdm

from lab_transformers.utils import json_lines_loader
from lab_transformers.models.bert.bert_custom_keydim import TFBertForMaskedLM


def bert_imputer_batch(
    model: TFBertForMaskedLM,
    bert_data: List[Dict[str, Any]],
    labrador_data: List[Dict[str, Any]],
    codebook: pd.DataFrame,
    config: Dict,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate random missing lab values in `bert_data` then impute them using `model`.

    `continuous_data` is used to get the true lab values for comparison. # TODO: remove this dependency by storing true lab values with BERT's data.
    Randomness is controlled by `rng`.
    `codebook` is used to map lab code tokens to MIMIC-IV itemids for easier data visualizations and interpretation of results.
    `config` is a Dict that contains the softmax_ypred_conversion_method, batch_size, mask_token, null_token, and pad_token.
        The softmax_ypred_conversion_method is used to convert BERT's output probability distribution over tokens to a single imputed value.
        These are needed to generate the masked and padded bags of labs for imputation.

    Returns a DataFrame with the following columns:
    - ypred: lab value imputed by `model`
    - ytrue: true lab value
    - subject_id: patient id
    - charttime: hospital chart time of the lab test
    - itemid: MIMIC-IV lab code of the imputed lab value
    - token: the token from BERT's vocabulary that was imputed

    """
    # pick random patients
    patient_ixs = rng.choice(len(bert_data), config["batch_size"])

    # pick random bag for each patient
    bag_ixs = [rng.choice(len(bert_data[ix]["token_bags"])) for ix in patient_ixs]

    # collect patient id's
    patient_ids = [bert_data[ix]["subject_id"] for ix in patient_ixs]

    # pick random token to mask in each bag
    mask_ixs = [
        rng.choice(
            np.min(
                [
                    len(bert_data[p_ix]["token_bags"][b_ix]),
                    model.config.max_position_embeddings,
                ]
            )
        )
        for p_ix, b_ix in zip(patient_ixs, bag_ixs)
    ]

    masked_bags = [
        [
            config["mask_token"] if ix == m_ix else x
            for ix, x in enumerate(bert_data[p_ix]["token_bags"][b_ix])
        ]
        for p_ix, b_ix, m_ix in zip(patient_ixs, bag_ixs, mask_ixs)
    ]

    # Pad the masked bags
    max_bag_length = np.max([len(bag) for bag in masked_bags])
    padded_bags = tf.keras.preprocessing.sequence.pad_sequences(
        masked_bags,
        maxlen=np.min([max_bag_length, model.config.max_position_embeddings]),
        dtype="int32",
        padding="post",
        truncating="post",
        value=config["pad_token"],
    )

    # for each token masked, look up in BERT's codebook, return all tokens for that itemid (i.e. which ix to filter softmax)
    masked_tokens = [
        bert_data[p_ix]["token_bags"][b_ix][m_ix]
        for p_ix, b_ix, m_ix in zip(patient_ixs, bag_ixs, mask_ixs)
    ]
    softmax_ixs = [
        codebook[
            codebook.itemid == codebook[codebook.token == m_token].itemid.item()
        ].token.values
        for m_token in masked_tokens
    ]

    softmax_value_map = [
        codebook[
            codebook.itemid == codebook[codebook.token == m_token].itemid.item()
        ].valuenum.values
        for m_token in masked_tokens
    ]

    logits = model(padded_bags).logits

    # process softmaxes: filter to the indexes corresponding to the itemid that was masked
    y_preds = []
    y_trues = []
    itemids = []
    ytrues_discrete = []
    charttimes = []
    pt_ids = []
    for i in range(logits.shape[0]):
        # skip tokens that never have lab values (nothing to impute)
        if np.all(np.isnan(softmax_value_map[i])):
            continue

        if config["softmax_ypred_conversion_method"].lower() == "argmax":
            # method 1: argmax the filtered softmax, return the left boundary of the interval or NULL (if applicable)
            ix = np.argmax(
                logits[i][mask_ixs[i]][softmax_ixs[i].min() : softmax_ixs[i].max() + 1]
            )
            if np.isnan(softmax_value_map[i][ix]):
                y_pred = np.nan
            elif ix == 0:  # if predicted decile is 1st decile, set y_pred to 0
                y_pred = 0
            else:
                y_pred = softmax_value_map[i][ix - 1]

        elif config["softmax_ypred_conversion_method"].lower() == "weighted_quantiles":
            # method 2: calc weighted avg of left boundary of the quantiles to get y_pred
            zeros = tf.cast(tf.zeros_like(softmax_value_map[i]), dtype=tf.bool)
            ones = tf.cast(tf.ones_like(softmax_value_map[i]), dtype=tf.bool)
            mask = tf.where(np.isnan(softmax_value_map[i]), zeros, ones)
            filtered_map = tf.boolean_mask(softmax_value_map[i], mask)
            quantile_leftboundaries = np.insert(filtered_map, 0, 0)[:-1]

            # drop corresponding index in logits[i][mask_ixs[i]
            filtered_logits = tf.boolean_mask(
                logits[i][mask_ixs[i]][softmax_ixs[i].min() : softmax_ixs[i].max() + 1],
                mask,
            )

            # renormalize remaining logits to probs
            softmax = tf.nn.softmax(filtered_logits)

            # Weighted average of the left boundaries of the quantiles
            y_pred = np.sum(quantile_leftboundaries * softmax)

        else:
            raise ValueError(
                f"softmax_ypred_conversion_method must be in ['argmax', 'weighted_quantiles'], got {config['softmax_ypred_conversion_method']}."
            )

        y_preds.append(y_pred)

        # Get y_true
        # find matching patient in continuous dataset
        pt_id = patient_ids[i]
        pt_ix = patient_ixs[i]
        bag_ix = bag_ixs[i]
        mask_ix = mask_ixs[i]
        charttime = bert_data[pt_ix]["charttime"][bag_ix]
        discrete_ytrue = bert_data[pt_ix]["token_bags"][bag_ix][mask_ix]
        ytrues_discrete.append(discrete_ytrue)
        charttimes.append(charttime)
        pt_ids.append(pt_id)

        # look up discrete y_true token in BERT's codebook to get itemid
        itemid = codebook[codebook.token == discrete_ytrue].itemid.item()
        itemids.append(itemid)

        for line in labrador_data:
            if (line["subject_id"] == pt_id) and (charttime in line["charttime"]):
                # match the chart time to the bag
                continuous_bag_ix = [
                    i for i, time in enumerate(line["charttime"]) if time == charttime
                ].pop()
                try:
                    standardized_ytrue = line["value_bags"][continuous_bag_ix][mask_ix]

                    if standardized_ytrue == "<NULL>":
                        standardized_ytrue = np.nan
                    y_trues.append(standardized_ytrue)
                    break
                except IndexError:
                    y_trues.append(np.nan)
                    break
        else:
            # no corresponding y_true found
            y_trues.append(np.nan)

    return pd.DataFrame(
        data={
            "ypred": y_preds,
            "ytrue": y_trues,
            "subject_id": pt_ids,
            "charttime": charttimes,
            "token": ytrues_discrete,
            "itemid": itemids,
        }
    )


if __name__ == "__main__":
    time_string = time.strftime("%Y%m%d-%H%M%S")
    config_path = sys.argv[1]

    with open(config_path) as f:
        config = json.load(f)

    rng = np.random.default_rng(config["random_seed"])

    bert_data = json_lines_loader(config["bert_dataset_path"])
    labrador_data = json_lines_loader(config["labrador_dataset_path"])

    codebook = pd.read_csv(config["codebook_path"])

    if config["ablation"].lower() == "true":
        with open(op.join(config["model_path"], "config.json")) as f:
            bertconfig = BertConfig.from_dict(json.load(f))

        model = TFBertForMaskedLM(config=bertconfig)

    else:
        model = TFBertForMaskedLM.from_pretrained(config["model_path"])

    result_dfs = []
    for ix in tqdm(
        range(config["num_batches"]),
        desc="Running intrinsic imputations with BERT",
    ):
        result_dfs.append(
            bert_imputer_batch(
                ix,
                model,
                bert_data,
                labrador_data,
                codebook,
                config,
                rng,
            )
        )
    df = pd.concat(result_dfs)

    # Create results directory if it doesn't exist
    if not op.exists(config["output_directory"]):
        os.mkdir(config["output_directory"])

    if config["ablation"].lower() == "true":
        df.to_csv(
            op.join(
                config["output_directory"],
                f"intrinsic_imputation_bert_{time_string}_ablation",
            )
            + ".csv",
            index=False,
        )
    else:
        df.to_csv(
            op.join(
                config["output_directory"],
                f"intrinsic_imputation_bert_{time_string}",
            )
            + ".csv",
            index=False,
        )
