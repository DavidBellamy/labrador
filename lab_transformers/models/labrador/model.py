from typing import Dict, Any

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers

from lab_transformers.models.labrador.continuous_embedding_layer import (
    ContinuousEmbedding,
)
from lab_transformers.models.labrador.prediction_heads import MLMPredictionHead


class Labrador(keras.Model):
    """
    Labrador: a BERT-style transformer model trained on a masked language model objective.
    """

    def __init__(self, params: Dict[str, Any], **kwargs) -> None:
        super(Labrador, self).__init__()

        self._params = params
        self.mask_token = params["mask_token"]
        self.pad_token = params["pad_token"]
        self.null_token = params["null_token"]
        self.vocab_size = params["vocab_size"]
        self.embedding_dim = params["embedding_dim"]
        self.transformer_heads = params["transformer_heads"]
        self.num_blocks = params["transformer_blocks"]
        self.transformer_feedforward_dim = params["transformer_feedforward_dim"]
        self.include_head = params["include_head"]
        self.continuous_head_activation = params["continuous_head_activation"]
        self.dropout_rate = params["dropout_rate"]

        self.transformer_activation_fn = layers.ReLU()

        if isinstance(self.include_head, str):
            self.include_head = self.include_head.lower() == "true"
        elif not isinstance(self.include_head, bool):
            raise ValueError(
                f"include_head must be a boolean or a string, not {type(self.include_head)}"
            )

        # Note: input_dim = vocab_size + 2 because there are 2 special categorical tokens to embed
        # (mask_token and pad_token).
        self.categorical_embedding_layer = layers.Embedding(
            input_dim=self.vocab_size + 2,
            output_dim=self.embedding_dim,
            mask_zero=True,
            name="categorical_embedding",
        )

        self.continuous_embedding_layer = ContinuousEmbedding(
            embedding_dim=self.embedding_dim,
            pad_token=self.pad_token,
            mask_token=self.mask_token,
            null_token=self.null_token,
            name="continuous_embedding",
        )

        self.projection_layer = layers.Dense(units=self.embedding_dim)

        self.blocks = []
        for block in range(self.num_blocks):
            self.blocks.append(
                TransformerBlock(
                    embed_dim=self.embedding_dim,
                    num_heads=self.transformer_heads,
                    activation=self.transformer_activation_fn,
                    feedforward_dim=self.transformer_feedforward_dim,
                    first_block=True if block == 0 else False,
                    dropout_rate=self.dropout_rate,
                    name=f"transformer_block_{block + 1}",
                )
            )

        self.head = MLMPredictionHead(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            continuous_head_activation=self.continuous_head_activation,
        )

    def call(
        self, inputs: Dict[str, tf.Tensor], training: bool = True, **kwargs
    ) -> Dict[str, tf.Tensor]:
        """
        :param inputs: Dict with 2 keys: `categorical_input` and `continuous_input`.
        :param training: Bool indicating whether to run the model with or without dropout.
        :return: Dict of categorical predictions, shape (batch_size, max_bag_length, vocab_size),
            and continuous predictions, shape (batch_size, max_bag_length, 1).
        """

        categorical_input = inputs["categorical_input"]
        continuous_input = inputs["continuous_input"]

        x_categorical = self.categorical_embedding_layer(categorical_input)
        x_continuous = self.continuous_embedding_layer(continuous_input, x_categorical)

        x = layers.concatenate([x_categorical, x_continuous])
        x = self.projection_layer(x)

        for i in range(self.num_blocks):
            x = self.blocks[i](x, training=training)

        if self.include_head:
            x = self.head(x)

        return x


class TransformerBlock(layers.Layer):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        activation: str,
        feedforward_dim: int,
        dropout_rate: float,
        first_block: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """
        The canonical transformer block from the original paper "Attention is All You Need".
        """
        super(TransformerBlock, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.activation = activation
        self.feedforward_dim = feedforward_dim
        self.first_block = first_block
        self.dropout_rate = dropout_rate

        self.att = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.embed_dim
        )
        self.ffn = keras.Sequential(
            [
                layers.Dense(self.feedforward_dim, activation="linear"),
                self.activation,
                layers.Dense(self.embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(self.dropout_rate)
        self.dropout2 = layers.Dropout(self.dropout_rate)
        self.add = layers.Add()

    def call(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(self.add([inputs, attn_output]))
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(self.add([out1, ffn_output]))

    def get_config(self) -> Dict[str, Any]:
        config = {
            "embed_dim": self.embed_dim,
            "numheads": self.num_heads,
            "activation": self.activation,
            "feedforward_dim": self.feedforward_dim,
            "first_block": self.first_block,
            "dropout_rate": self.dropout_rate,
        }

        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TransformerBlock":
        return cls(**config)
