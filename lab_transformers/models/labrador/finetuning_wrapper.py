from typing import Union, Dict, Any

import tensorflow as tf
from tensorflow import keras

from lab_transformers.models.labrador.model import Labrador


class LabradorFinetuneWrapper(keras.Model):
    def __init__(
        self,
        base_model_path: Union[None, str],
        output_size: int,
        output_activation: str,
        model_params: Dict[str, Any],
        dropout_rate: float,
        add_extra_dense_layer: bool,
        train_base_model: bool = False,
    ) -> None:
        """
        A wrapper for a Labrador model that allows for finetuning on a downstream task.
        """

        super(LabradorFinetuneWrapper, self).__init__()

        self.transformer_params = model_params
        self.base_model_path = base_model_path
        self.output_size = output_size
        self.output_activation = output_activation
        self.train_base_model = train_base_model
        self.dropout_rate = dropout_rate
        self.max_seq_len = model_params["max_seq_length"]
        self.add_extra_dense_layer = add_extra_dense_layer

        self.base_model = Labrador(self.transformer_params)
        if self.base_model_path is not None:
            print(f"\n Loading weights from {self.base_model_path} \n", flush=True)
            self.base_model.load_weights(base_model_path)

        self.base_model.include_head = False
        self.base_model.trainable = train_base_model

        self.pool = keras.layers.GlobalAveragePooling1D()
        self.dense_nonmimic = keras.layers.Dense(units=14, activation="relu")
        self.dropout = keras.layers.Dropout(rate=self.dropout_rate)
        self.extra_dense_layer = keras.layers.Dense(units=1038, activation="relu")
        self.output_layer = keras.layers.Dense(
            units=self.output_size, activation=self.output_activation
        )

    def call(self, inputs: Dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        # Truncate input sequences to max_seq_length
        base_model_inputs = {
            "categorical_input": inputs["categorical_input"][:, : self.max_seq_len],
            "continuous_input": inputs["continuous_input"][:, : self.max_seq_len],
        }

        x = self.base_model(base_model_inputs, training=training)
        x._keras_mask = None
        x = self.pool(
            x,
            mask=tf.cast(
                tf.math.not_equal(base_model_inputs["categorical_input"], 0), tf.int32
            ),
        )

        if "non_mimic_features" in inputs.keys():
            non_mimic_features = self.dense_nonmimic(inputs["non_mimic_features"])
            x = tf.concat([x, non_mimic_features], axis=1)

        if self.add_extra_dense_layer:
            x = self.extra_dense_layer(x)

        x = self.dropout(x, training=training)
        outputs = self.output_layer(x)
        return outputs
