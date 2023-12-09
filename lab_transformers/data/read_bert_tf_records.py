"""
The function get_dataset() is used to read the TFRecord data during BERT's pre-training.

In order to read the TFRecord data, use the following snippet:

filenames = tf.io.gfile.glob(f"{tfrecords_dir}/*.tfrec")
dataset = get_dataset(filenames, batch_size, pad_token, random_seed, shuffle_buffer_size)
for element in dataset:
    # do stuff with element
"""

from typing import Dict, List

import tensorflow as tf


# Helper functions for reading/preparing TFRecord data
def parse_tfrecord_fn(serialized_example) -> Dict[str, tf.Tensor]:
    feature_description = {
        "input_ids": tf.io.FixedLenFeature([], tf.string),
        "labels": tf.io.FixedLenFeature([], tf.string),
    }
    parsed_example = tf.io.parse_single_example(serialized_example, feature_description)
    return {
        "input_ids": tf.io.parse_tensor(parsed_example["input_ids"], out_type=tf.int32),
        "labels": tf.io.parse_tensor(parsed_example["labels"], out_type=tf.int32),
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
            padding_values={"input_ids": pad_token, "labels": -100},
            padded_shapes={"input_ids": [None], "labels": [None]},
            drop_remainder=True,
        )
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataset
