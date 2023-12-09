from typing import Dict, Any

import tensorflow as tf
import tensorflow.keras.layers as layers


class ContinuousEmbedding(layers.Layer):
    def __init__(
        self,
        embedding_dim: int,
        pad_token: int,
        mask_token: int,
        null_token: int,
        mask_padding: bool = True,
        **kwargs
    ) -> None:
        """
        Labrador's continuous embedding layer for lab values.
        """

        super(ContinuousEmbedding, self).__init__()

        self.embedding_dim = embedding_dim
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.null_token = null_token
        self.mask_padding = mask_padding

        # Maintain a two-element embedding table for the special mask_token and null_token embeddings
        self.special_token_embeddings = layers.Embedding(
            input_dim=2,
            output_dim=self.embedding_dim,
            mask_zero=False,
            name="special_token_embeddings",
        )

        # We do a linear projection of each lab value element-wise
        self.dense1 = layers.TimeDistributed(
            layers.Dense(units=self.embedding_dim, activation="linear")
        )

        # We use a non-linear MLP on the combined lab value + lab code embeddings
        self.dense2 = layers.TimeDistributed(
            layers.Dense(units=self.embedding_dim, activation="relu")
        )

        # We layer normalize the final output to avoid extreme values being passed to the transformer blocks
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)

    def call(
        self, x_continuous: tf.Tensor, x_categorical_embeddings: tf.Tensor
    ) -> tf.Tensor:
        """
        The forward pass of the ContinuousEmbedding layer.

        :param x_continuous: Tensor with shape (batch_size x max_bag_length).
            Contains the [0, 1] standardized lab values from the current batch.
        :param x_categorical_embeddings: Tensor with shape (batch_size x max_bag_length x embedding_dim).
            Contains the marginal embeddings of the lab codes from the current batch.

        :return: A tensor of shape (batch_size x max_bag_len x embedding_dim).
            Contains one embedding vector of size embedding_dim for each lab (value, code) pair in the batch.
        """

        # If batch size = 1
        if len(x_continuous.shape) == 1:
            x_continuous = tf.expand_dims(x_continuous, 0)

        batch_size = tf.shape(x_continuous)[0]
        max_bag_length = tf.shape(x_continuous)[1]

        # Compute padding mask so that it gets passed forward in the Transformer
        padding_mask = self.compute_mask(x_continuous, mask_padding=self.mask_padding)

        # Compute 2 boolean masks with shape (batch_size, max_bag_length, embedding_dim):
        # 1 in the position of null token and 0 elsewhere, and
        # 1 in the poition of mask tokens and 0 elsewhere.
        x_continuous = tf.expand_dims(x_continuous, -1)
        boolean_for_mask = tf.equal(
            x_continuous, tf.constant(self.mask_token, dtype=x_continuous.dtype)
        )
        boolean_for_mask = tf.repeat(boolean_for_mask, self.embedding_dim, axis=-1)
        boolean_for_null = tf.equal(
            x_continuous, tf.constant(self.null_token, dtype=x_continuous.dtype)
        )
        boolean_for_null = tf.repeat(boolean_for_null, self.embedding_dim, axis=-1)

        # Project each lab value to embedding_dim vector, x.shape = (batch_size, max_bag_length, embedding_dim)
        x = self.dense1(x_continuous)

        # Tile the null and mask embeddings into a tensor with shape (batch_size x max_bag_length x embedding_dim)
        mask_value_embedding = self.special_token_embeddings(0)
        null_value_embedding = self.special_token_embeddings(1)
        mask_embedding = tf.expand_dims(tf.expand_dims(mask_value_embedding, 0), 0)
        null_embedding = tf.expand_dims(tf.expand_dims(null_value_embedding, 0), 0)
        mask_embedding_matrix = tf.tile(mask_embedding, [batch_size, max_bag_length, 1])
        null_embedding_matrix = tf.tile(null_embedding, [batch_size, max_bag_length, 1])

        # Apply the corresponding boolean mask to the special token tensors
        mask_tensor = tf.multiply(
            mask_embedding_matrix,
            tf.cast(boolean_for_mask, mask_embedding_matrix.dtype),
        )
        null_tensor = tf.multiply(
            null_embedding_matrix,
            tf.cast(boolean_for_null, null_embedding_matrix.dtype),
        )

        # Apply the negation of both boolean tensors to x (zero out any embeddings for mask_token or null_token)
        # we will swap in the special_token_embedding for each of these next
        combined_boolean = tf.math.logical_and(
            tf.math.logical_not(boolean_for_mask), tf.math.logical_not(boolean_for_null)
        )
        x = tf.multiply(x, tf.cast(combined_boolean, x.dtype))

        # Sum the 3 tensors: 2 special token tensors and x, the zero'd-out projected value tensor
        x = tf.math.add_n([x, mask_tensor, null_tensor])

        # Aum the above result (sum of 3 tensors) and the marginal categorical embeddings
        x = x + x_categorical_embeddings

        x = self.dense2(x)
        x = self.layernorm(x)

        # Attach padding mask to x for future layer usage
        x._keras_mask = padding_mask

        return x

    def compute_mask(self, inputs: tf.Tensor, mask_padding: bool = None) -> tf.Tensor:
        if not self.mask_padding:
            return None
        return tf.not_equal(inputs, self.pad_token)

    def get_config(self) -> Dict[str, Any]:
        config = {
            "embedding_dim": self.embedding_dim,
            "pad_token": self.pad_token,
            "mask_token": self.mask_token,
            "null_token": self.null_token,
            "mask_padding": self.mask_padding,
        }
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ContinuousEmbedding":
        return cls(**config)
