import os
import os.path as op
import sys

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from lab_transformers.data.labrador_datagen import labrador_datagen
from lab_transformers.utils import json_lines_loader

processed_data_dir = sys.argv[1]
random_seed = int(sys.argv[2])
mask_token = int(sys.argv[3])
null_token = int(sys.argv[4])
labrador_bags_filename = sys.argv[5]
output_tfrecords_dir = sys.argv[6]


# Helper functions for writing TFRecords
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_example(sample):
    feature = {
        "continuous_input": _bytes_feature(
            tf.io.serialize_tensor(sample[0]["continuous_input"])
        ),
        "categorical_input": _bytes_feature(
            tf.io.serialize_tensor(sample[0]["categorical_input"])
        ),
        "continuous_output": _bytes_feature(
            tf.io.serialize_tensor(sample[1]["continuous_output"])
        ),
        "categorical_output": _bytes_feature(
            tf.io.serialize_tensor(sample[1]["categorical_output"])
        ),
    }

    return tf.train.Example(
        features=tf.train.Features(feature=feature)
    ).SerializeToString()


# note: 10Mb+ and 100Mb+ per tfrecord file is best,
# and have at least 10 times as many files as there will be hosts (CPU cores?) reading data
num_samples_per_tfrecord = (
    20_000  # 20k makes about 180 x ~10Mb tfrecord files for the full train data
)
duplication_factor = 1  # produces (duplication_factor * num_bags) tfrecord samples
delete_previous_tfrecords = True

# Load training data (to be converted to TFRecords)
data_jsonl = json_lines_loader(op.join(processed_data_dir, labrador_bags_filename))
num_bags = np.sum([len(patient["code_bags"]) for patient in data_jsonl])
num_samples_to_generate = num_bags * int(duplication_factor)
num_tfrecords = num_samples_to_generate // num_samples_per_tfrecord

if num_samples_to_generate % num_samples_per_tfrecord:
    num_tfrecords += 1  # add one record if there are any remaining samples

if op.exists(output_tfrecords_dir) and delete_previous_tfrecords:
    # clear any pre-existing tfrecords in this folder
    # this avoids any possibility of overwriting some pre-existing tfrecords, while not overwriting others.
    # this may happen if a previous run of this script produced 20 tfrecord files, and a subsequent run only 10.
    # the first 10 tfrecord files will be overwritten, but the other (old) 10 will remain.
    # this will cause issues in downstream use of this data
    files = tf.io.gfile.glob(f"{output_tfrecords_dir}/*.tfrec")
    for f in files:
        os.remove(f)

# create TFRecords output folder if it doesn't exist
if not op.exists(output_tfrecords_dir):
    os.makedirs(output_tfrecords_dir)

generator_counter = 0
datagen = labrador_datagen(
    data_jsonl, random_seed, mask_token, null_token, shuffle_patients=True
)
for tfrec_num in range(num_tfrecords):
    # NOTE: TFRecordWriter overwrites the file content if it already exists

    # if current datagen doesn't have 1 full record left in it, switch to the next datagen (new random seed)
    num_generations_left = num_bags - generator_counter
    if num_generations_left < num_samples_per_tfrecord:
        generator_counter = 0
        datagen = labrador_datagen(
            data_jsonl,
            random_seed + tfrec_num,
            mask_token,
            null_token,
            shuffle_patients=True,
        )

    with tf.io.TFRecordWriter(
        output_tfrecords_dir + "/file_%.2i.tfrec" % tfrec_num
    ) as writer:
        for i, sample in tqdm(
            enumerate(datagen, start=1),
            desc=f"Writing tfrecord {tfrec_num + 1} of {num_tfrecords}",
            total=num_samples_per_tfrecord,
        ):
            example = create_example(sample)
            writer.write(example)
            generator_counter += 1

            if i % num_samples_per_tfrecord == 0:
                break
