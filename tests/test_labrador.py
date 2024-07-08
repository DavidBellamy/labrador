import os
from typing import Dict
import unittest

import numpy as np
import tensorflow as tf

from lab_transformers.models.labrador.finetuning_wrapper import LabradorFinetuneWrapper
from lab_transformers.models.labrador.loss import CategoricalMLMLoss, ContinuousMLMLoss
from lab_transformers.models.labrador.model import Labrador
from lab_transformers.utils import empty_folder


class TestLabrador(unittest.TestCase):
    def setUp(self) -> None:
        self.params = {
            "mask_token": 0,
            "pad_token": 1,
            "null_token": 2,
            "vocab_size": 100,
            "embedding_dim": 32,
            "transformer_activation": "relu",
            "transformer_heads": 1,
            "transformer_blocks": 1,
            "transformer_feedforward_dim": 32,
            "include_head": True,
            "continuous_head_activation": "sigmoid",
            "dropout_rate": 0.1,
            "max_seq_length": 90,
        }

        self.model = Labrador(self.params)

        self.inputs = {
            "categorical_input": np.array([[1, 2, 3], [1, 2, 3]]),
            "continuous_input": np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]),
        }

        current_script_path = os.path.abspath(__file__)
        repo_root_directory = os.path.dirname(os.path.dirname(current_script_path))
        self.model_target_directory = os.path.join(
            repo_root_directory, "tests/test_models"
        )
        os.makedirs(self.model_target_directory, exist_ok=True)

    def tearDown(self) -> None:
        # Clear the contents of test/test_models
        empty_folder(self.model_target_directory)

    def test_labrador_categorical_embedding_layer(self):
        categorical_input = self.inputs["categorical_input"]

        x_categorical = self.model.categorical_embedding_layer(categorical_input)

        self.assertEqual(
            x_categorical.shape,
            (
                categorical_input.shape[0],
                categorical_input.shape[1],
                self.params["embedding_dim"],
            ),
        )
        self.assertIsInstance(x_categorical, tf.Tensor)

    def test_labrador_continuous_embedding_layer(self):
        continuous_input = self.inputs["continuous_input"]

        x_categorical = tf.random.uniform(
            (
                continuous_input.shape[0],
                continuous_input.shape[1],
                self.params["embedding_dim"],
            )
        )
        x_continuous = self.model.continuous_embedding_layer(
            continuous_input, x_categorical
        )

        self.assertEqual(
            x_continuous.shape,
            (
                continuous_input.shape[0],
                continuous_input.shape[1],
                self.params["embedding_dim"],
            ),
        )
        self.assertIsInstance(x_continuous, tf.Tensor)

    def test_labrador_MLM_head(self):
        x = tf.random.uniform((32, 5, self.params["embedding_dim"]))

        output = self.model.head(x)

        self.assertIsInstance(output, Dict)
        self.assertEqual(
            output["categorical_output"].shape, (32, 5, self.params["vocab_size"])
        )
        self.assertEqual(output["continuous_output"].shape, (32, 5, 1))

    def test_labrador_forward_pass_return_type(self):
        output = self.model(self.inputs)

        self.assertIsInstance(output, Dict)
        self.assertIsInstance(output["categorical_output"], tf.Tensor)
        self.assertIsInstance(output["continuous_output"], tf.Tensor)

    def test_labrador_model_returns_correct_output_shapes(self):
        categorical_input = tf.random.uniform(
            (32, 5), maxval=self.params["vocab_size"], dtype=tf.int32
        )
        continuous_input = tf.random.uniform((32, 5), dtype=tf.float32)
        inputs = {
            "categorical_input": categorical_input,
            "continuous_input": continuous_input,
        }

        output = self.model(inputs)

        self.assertEqual(
            output["categorical_output"].shape, (32, 5, self.params["vocab_size"])
        )
        self.assertEqual(output["continuous_output"].shape, (32, 5, 1))

    def test_labrador_forward_pass_return_type_without_head(self):
        self.model.include_head = False
        output = self.model(self.inputs)

        self.assertIsInstance(output, tf.Tensor)

    def test_labrador_model_returns_correct_output_shapes_without_head(self):
        self.model.include_head = False
        categorical_input = tf.random.uniform(
            (17, 5), maxval=self.params["vocab_size"], dtype=tf.int32
        )
        continuous_input = tf.random.uniform((17, 5), dtype=tf.float32)
        inputs = {
            "categorical_input": categorical_input,
            "continuous_input": continuous_input,
        }

        output = self.model(inputs)

        self.assertEqual(output.shape, (17, 5, self.params["embedding_dim"]))

    def test_labrador_loss(self):
        labels = {
            "categorical_output": np.array(
                [[-1, 2, 3, -1], [1, -1, -1, 3]]
            ),  # (2, 4)  = (batch_size, max_bag_len)
            "continuous_output": np.array([[-1, 0.25, 0.5, -1], [0.1, -1, -1, 0.4]]),
        }

        preds = {
            "categorical_output": np.array(  # (2, 4, 3) = (batch_size, max_bag_len, vocab_size)
                [
                    [
                        [0.4361403, 0.2468242, 0.31703544],
                        [0.25, 0.25, 0.75],
                        [0.4, 0.25, 0.35],
                        [0.27461904, 0.3124466, 0.41293427],
                    ],
                    [
                        [0.35, 0.42, 0.23],
                        [0.30096892, 0.19483292, 0.50419813],
                        [0.244302, 0.38868132, 0.36701667],
                        [0.33, 0.37, 0.3],
                    ],
                ]
            ),
            "continuous_output": np.array(
                [[[0.99], [0.25], [0.5], [0.11]], [[0.1], [0.22], [0.33], [0.4]]]
            ),  # (2, 4, 1) = (batch_size, max_bag_len, 1)
        }

        train_categorical_loss = CategoricalMLMLoss()(
            labels["categorical_output"], preds["categorical_output"]
        )
        train_continuous_loss = ContinuousMLMLoss()(
            labels["continuous_output"], preds["continuous_output"]
        )

        self.assertIsInstance(train_categorical_loss, tf.Tensor)
        self.assertIsInstance(train_continuous_loss, tf.Tensor)
        self.assertEqual(train_continuous_loss, 0)
        self.assertAlmostEqual(train_categorical_loss.numpy(), 1.2282637414393478)

    def test_labrador_finetuning_wrapper(self):
        model_wrapper = LabradorFinetuneWrapper(
            base_model_path=None,
            output_size=3,
            output_activation="softmax",
            model_params=self.params,
            dropout_rate=0.1,
            add_extra_dense_layer=False,
            train_base_model=True,
        )

        # Add non-mimic features to inputs to test the wrapper on them
        inputs = self.inputs | {"non_mimic_features": np.random.rand(2, 100)}

        output = model_wrapper(inputs)

        # Assert sum of each row of outputs equals 1 (softmax)
        self.assertTrue(np.allclose(np.sum(output.numpy(), axis=1), 1))

    def test_saveload_labrador_model(self):
        self.model(
            self.inputs
        )  # need to build the model with some inputs before saving

        self.model.save(self.model_target_directory)

        loaded_model = Labrador(self.params)
        loaded_model.load_weights(
            os.path.join(self.model_target_directory, "variables/variables")
        )

        # Generate predictions by the original and loaded models (given the same inputs)
        orig_preds = self.model(self.inputs, training=False)
        loaded_preds = loaded_model(self.inputs, training=False)

        # Assert that all predictions are identical
        self.assertTrue(
            np.array_equal(
                orig_preds["categorical_output"],
                loaded_preds["categorical_output"],
            )
        )
        self.assertTrue(
            np.array_equal(
                orig_preds["continuous_output"],
                loaded_preds["continuous_output"],
            )
        )


if __name__ == "__main__":
    unittest.main()
