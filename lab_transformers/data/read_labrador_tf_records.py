"""
The function get_dataset() is used to read the TFRecord data during Labrador's pre-training.

In order to read the TFRecord data, use the following snippet:

filenames = tf.io.gfile.glob(f"{tfrecords_dir}/*.tfrec")
dataset = get_dataset(filenames, batch_size, pad_token, random_seed)
for element in dataset:
    # do stuff with element
"""

from typing import Dict, List, Tuple

import tensorflow as tf


# Helper functions for reading/preparing TFRecord data
def parse_tfrecord_fn(
    serialized_example,
) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
    feature_description = {
        "continuous_input": tf.io.FixedLenFeature([], tf.string),
        "categorical_input": tf.io.FixedLenFeature([], tf.string),
        "continuous_output": tf.io.FixedLenFeature([], tf.string),
        "categorical_output": tf.io.FixedLenFeature([], tf.string),
    }
    parsed_example = tf.io.parse_single_example(serialized_example, feature_description)
    return {
        "continuous_input": tf.io.parse_tensor(
            parsed_example["continuous_input"], out_type=tf.float32
        ),
        "categorical_input": tf.io.parse_tensor(
            parsed_example["categorical_input"], out_type=tf.int32
        ),
    }, {
        "continuous_output": tf.io.parse_tensor(
            parsed_example["continuous_output"], out_type=tf.float32
        ),
        "categorical_output": tf.io.parse_tensor(
            parsed_example["categorical_output"], out_type=tf.int32
        ),
    }


def get_dataset(
    filenames: List[str],
    batch_size: int,
    pad_token: int,
    random_seed: int,
    shuffle_buffer_size: int,
) -> tf.data.TFRecordDataset:
    dataset = (
        tf.data.TFRecordDataset(filenames, num_parallel_reads=tf.data.AUTOTUNE)
        .map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(shuffle_buffer_size, seed=random_seed)
        .padded_batch(
            batch_size=batch_size,
            padding_values=(
                {"categorical_input": pad_token, "continuous_input": float(pad_token)},
                {"categorical_output": -1, "continuous_output": -1.0},
            ),
            padded_shapes=(
                {"categorical_input": [None], "continuous_input": [None]},
                {"categorical_output": [None], "continuous_output": [None]},
            ),
            drop_remainder=True,
        )
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataset
