from typing import Dict, Generator, Iterable, List, Tuple, Union

import numpy as np
import tensorflow as tf


def labrador_datagen(
    data: List[Dict[str, Iterable]],
    random_seed: int,
    mask_token: int,
    null_token: int,
    shuffle_patients: bool = False,
    include_metadata: bool = False,
) -> Generator[
    Union[
        Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]],
        Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor], Dict[str, List]],
    ],
    None,
    None,
]:
    """
    Data generator used in the process of creating TensorFlow Record Files for Labrador.
    Yields 1 masked bag of labs at a time in the form of dictionaries with keys.
    """
    while True:
        rng = np.random.default_rng(random_seed)
        if shuffle_patients:
            rng.shuffle(data)
        for patient in data:
            subject_id = patient["subject_id"]
            for charttime, code_bag, value_bag in zip(
                patient["charttime"], patient["code_bags"], patient["value_bags"]
            ):
                # shuffle the order of the labs within the bag (ensure perm equivariance)
                c = list(zip(code_bag, value_bag))
                rng.shuffle(c)
                code_bag, value_bag = zip(*c)

                # Replace <NULL> lab values (missing) with special null_token
                value_bag = [null_token if x == "<NULL>" else x for x in value_bag]

                # Find valid mask indexes (ie. where lab value is non-null)
                valid_indexes = np.where(np.array(value_bag) != null_token)[0]

                # pick (valid) mask index
                mask_ix = rng.choice(valid_indexes)

                # mask bag
                code_inputs = tf.convert_to_tensor(
                    [
                        mask_token if ix == mask_ix else token
                        for ix, token in enumerate(code_bag)
                    ],
                    dtype=tf.int32,
                )
                value_inputs = tf.convert_to_tensor(
                    [
                        mask_token if ix == mask_ix else token
                        for ix, token in enumerate(value_bag)
                    ],
                    dtype=tf.float32,
                )

                # replace non-targets with -1 (used in loss computation)
                code_labels = tf.convert_to_tensor(
                    [
                        -1 if x != mask_token else z
                        for x, z in zip(code_inputs, code_bag)
                    ],
                    dtype=tf.int32,
                )
                value_labels = tf.convert_to_tensor(
                    [
                        -1 if x != mask_token else z
                        for x, z in zip(value_inputs, value_bag)
                    ],
                    dtype=tf.float32,
                )

                # yield masked bags and label bags
                if include_metadata:
                    yield {
                        "categorical_input": code_inputs,
                        "continuous_input": value_inputs,
                    }, {
                        "categorical_output": code_labels,
                        "continuous_output": value_labels,
                    }, {
                        "charttime": [charttime],
                        "subject_id": [subject_id],
                    }
                else:
                    yield {
                        "categorical_input": code_inputs,
                        "continuous_input": value_inputs,
                    }, {
                        "categorical_output": code_labels,
                        "continuous_output": value_labels,
                    }
