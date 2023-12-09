from typing import Union, Dict

import numpy as np
import tensorflow as tf
from tensorflow import keras
from transformers import BertConfig

from lab_transformers.models.bert.bert_custom_keydim import TFBertForMaskedLM


class BertFinetuneWrapper(keras.Model):
    def __init__(
        self,
        base_model_path: Union[None, str],
        output_size: int,
        output_activation: str,
        dropout_rate: float,
        add_extra_dense_layer: bool,
        train_base_model: bool = False,
    ) -> None:
        """
        Wrapper for the BERT model that allows for finetuning on downstream tasks.
        """
        super(BertFinetuneWrapper, self).__init__()

        self.base_model_path = base_model_path
        self.output_size = output_size
        self.output_activation = output_activation
        self.train_base_model = train_base_model
        self.dropout_rate = dropout_rate
        self.add_extra_dense_layer = add_extra_dense_layer

        if self.base_model_path is not None:
            print(f"Loading weights from {self.base_model_path}")
            self.base_model = TFBertForMaskedLM.from_pretrained(base_model_path)
        else:
            # If no saved model is provided, random weights are used
            transformer_params = BertConfig(
                attention_probs_dropout_prob=0.1,
                hidden_act="relu",
                hidden_dropout_prob=0.1,
                hidden_size=1024,
                initializer_range=0.02,
                intermediate_size=1024,
                layer_norm_eps=1e-12,
                max_position_embeddings=90,
                num_attention_heads=4,
                num_hidden_layers=10,
                pad_token_id=0,
                position_embedding_type="absolute",
                type_vocab_size=2,
                use_cache=True,
                vocab_size=4251,
            )
            self.base_model = TFBertForMaskedLM(config=transformer_params)

        self.base_model.config.update({"output_hidden_states": True})
        self.max_seq_length = self.base_model.config.max_position_embeddings

        self.base_model.trainable = train_base_model

        self.pool = keras.layers.GlobalAveragePooling1D()
        self.dropout = keras.layers.Dropout(rate=self.dropout_rate)
        self.dense_nonmimic = keras.layers.Dense(units=14, activation="relu")
        self.extra_dense_layer = keras.layers.Dense(units=1038, activation="relu")
        self.output_layer = keras.layers.Dense(
            units=self.output_size, activation=self.output_activation
        )

    def call(self, inputs: Dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        # Truncate input sequences to max_seq_length
        input_ids = inputs["input_ids"][:, : self.max_seq_length]

        x = self.base_model(input_ids).hidden_states[-1]
        x._keras_mask = None
        x = self.pool(x, mask=tf.cast(tf.math.not_equal(input_ids, 0), tf.int32))

        if "non_mimic_features" in inputs.keys():
            non_mimic_features = self.dense_nonmimic(inputs["non_mimic_features"])
            x = tf.concat([x, non_mimic_features], axis=1)

        if self.add_extra_dense_layer:
            x = self.extra_dense_layer(x)

        x = self.dropout(x, training=training)
        outputs = self.output_layer(x)
        return outputs
