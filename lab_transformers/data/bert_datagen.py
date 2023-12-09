from typing import Dict, Generator, Iterable, List

import numpy as np
import tensorflow as tf


def bert_datagen(
    data: List[Dict[str, Iterable]],
    random_seed: int,
    mask_token: int,
    shuffle_patients: bool = False,
) -> Generator[Dict[str, tf.Tensor], None, None]:
    """
    Data generator used in the process of creating TensorFlow Record Files for BERT.
    Yields 1 masked bag of labs at a time in the form of dictionaries with keys.
    """
    while True:
        rng = np.random.default_rng(random_seed)
        if shuffle_patients:
            rng.shuffle(data)
        for patient in data:
            for bag in patient["token_bags"]:
                # shuffle the order of the labs within the bag (ensure perm equivariance in Bert w/ pos embeds)
                bag_copy = bag.copy()
                rng.shuffle(bag_copy)

                # pick mask index
                mask_ix = rng.choice(len(bag_copy))

                # mask bag
                inputs = tf.convert_to_tensor(
                    [
                        mask_token if ix == mask_ix else int(token)
                        for ix, token in enumerate(bag_copy)
                    ],
                    dtype=tf.int32,
                )

                # replace non-targets with -100 (used in loss computation)
                labels = tf.convert_to_tensor(
                    [
                        -100 if x != mask_token else int(z)
                        for x, z in zip(inputs, bag_copy)
                    ],
                    dtype=tf.int32,
                )

                # yield masked bag and label bag
                yield {"input_ids": inputs, "labels": labels}
