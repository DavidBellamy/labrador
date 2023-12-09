from typing import Dict

import tensorflow as tf
import tensorflow.keras.layers as layers


class MLMPredictionHead(layers.Layer):
    def __init__(
        self, vocab_size: int, embedding_dim: int, continuous_head_activation: str
    ) -> None:
        super(MLMPredictionHead, self).__init__()

        self.num_classes = vocab_size  # Number unique lab codes
        self.embedding_dim = embedding_dim
        self.num_continuous = 1  # Output should only include a single (standardized) lab value at each bag position

        self.dense_categorical = layers.TimeDistributed(
            layers.Dense(units=self.embedding_dim, activation="relu")
        )
        self.dense_continuous = layers.TimeDistributed(
            layers.Dense(units=self.embedding_dim + self.num_classes, activation="relu")
        )

        self.categorical_head = layers.TimeDistributed(
            layers.Dense(units=self.num_classes)
        )
        self.categorical_head_activation = layers.Activation(
            "softmax", dtype="float32", name="categorical_output"
        )
        self.continuous_head = layers.TimeDistributed(layers.Dense(self.num_continuous))
        self.continuous_head_activation = layers.Activation(
            continuous_head_activation, dtype="float32", name="continuous_output"
        )

    def call(self, inputs: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        The forward pass of Labrador's two prediction heads.

        Inputs:
            > inputs: a Tensor with shape (batch_size x max_bag_length x embedding_dim).

        Outputs:
            > A Dict containing the model's categorical predictions and continuous predictions. The categorical
                predictions are a probability distribution over the vocabulary with shape
                (batch_size x max_bag_length x vocab_size). Continuous predictions are a Tensor with
                shape (batch_size x max_bag_length x 1), and each value is a prediction for the corresponding lab value.
        """

        cat = self.dense_categorical(inputs)
        categorical_logits = self.categorical_head(cat)
        categorical_prediction = self.categorical_head_activation(categorical_logits)
        augmented_inputs = layers.concatenate([inputs, categorical_prediction], axis=-1)
        cont = self.dense_continuous(augmented_inputs)
        continuous_prediction = self.continuous_head(cont)
        continuous_prediction = self.continuous_head_activation(continuous_prediction)
        continuous_prediction._keras_mask = None  # Keras-specific hack
        categorical_prediction._keras_mask = None

        return {
            "categorical_output": categorical_prediction,
            "continuous_output": continuous_prediction,
        }

    def get_config(self) -> Dict[str, int]:
        config = {"num_classes": self.num_classes, "embedding_dim": self.embedding_dim}

        return config

    @classmethod
    def from_config(cls, config) -> "MLMPredictionHead":
        return cls(**config)
