import unittest

import numpy as np
import tensorflow as tf

from lab_transformers.models.labrador.loss import CategoricalMLMLoss, ContinuousMLMLoss


class TestCustomLoss(unittest.TestCase):
    def setUp(self) -> None:
        # Make fake "true" data with batch_size = 3, max_bag_length = 3, vocab_size = 4
        self.vocab_size = 4

        # Create the target labels, where -1 indicates that those values/indices should be ignored in loss calculation
        self.y_true = {
            "categorical_labels": tf.convert_to_tensor(
                [[2, -1, -1], [-1, 1, -1], [-1, -1, 1]]
            ),
            "continuous_labels": tf.convert_to_tensor(
                [[-1, -1, -1], [-1, -2.0, -1], [-1, -1, 0.3]]
            ),
        }

        # Make an instance of the loss classes
        self.categorical_loss = CategoricalMLMLoss()
        self.continuous_loss = ContinuousMLMLoss()
        self.loss_weights = {"categorical_output": 1.0, "continuous_output": 1.0}

    def test_call_method_computes_zero_loss_on_perfect_predictions(self):
        # Create perfect prediction output
        perfect_y_pred = {
            "categorical_output": tf.convert_to_tensor(
                [
                    [[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                    [[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]],
                    [[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]],
                ],
                dtype=tf.float32,
            ),
            "continuous_output": tf.convert_to_tensor(
                [[0, 1.1, 0.75], [-0.5, -2.0, 0], [0.1, 0.2, 0.3]]
            ),
        }

        # Compute the loss (should be 0)
        cat_loss = self.categorical_loss(
            self.y_true["categorical_labels"], perfect_y_pred["categorical_output"]
        )
        cont_loss = self.continuous_loss(
            self.y_true["continuous_labels"], perfect_y_pred["continuous_output"]
        )

        loss = (
            self.loss_weights["categorical_output"] * cat_loss
            + self.loss_weights["continuous_output"] * cont_loss
        )

        # Assert the loss is equal to 0
        self.assertTrue(np.allclose(loss, 0, atol=1.0e-6))

    def test_call_method_computes_correct_loss_on_imperfect_continuous_predictions(
        self,
    ):
        # Create y_pred with perfect categorical predictions but imperfect continuous predictions
        y_pred = {
            "categorical_output": tf.convert_to_tensor(
                [
                    [[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                    [[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]],
                    [[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]],
                ],
                dtype=tf.float32,
            ),
            "continuous_output": tf.convert_to_tensor(
                [[0.5, 0.5, 0.5], [0, 1, 0], [1, 1, 1]], dtype=tf.float32
            ),
        }

        # Compute the correct loss by hand
        mse = 4.745  # calculated by hand
        correct_loss = (
            self.loss_weights["categorical_output"] * 0
            + self.loss_weights["continuous_output"] * mse
        )

        # Call the loss function on these predictions
        cat_loss = self.categorical_loss(
            self.y_true["categorical_labels"], y_pred["categorical_output"]
        )
        cont_loss = self.continuous_loss(
            self.y_true["continuous_labels"], y_pred["continuous_output"]
        )
        loss = (
            self.loss_weights["categorical_output"] * cat_loss
            + self.loss_weights["continuous_output"] * cont_loss
        )

        # Assert the loss function returns the correct loss
        self.assertTrue(np.allclose(loss, correct_loss))

    def test_call_method_computes_correct_loss_on_imperfect_categorical_predictions(
        self,
    ):
        # Create y_pred with perfect continuous predictions but imperfect categorical predictions
        y_pred = {
            "categorical_output": tf.convert_to_tensor(
                [
                    [[0.25, 0.25, 0.25, 0.25], [1, 0, 0, 0], [0.5, 0.5, 0, 0]],
                    [[0.9, 0.05, 0.05, 0], [0.5, 0, 0, 0.5], [0.2, 0.2, 0.3, 0.3]],
                    [
                        [0.3, 0.3, 0.3, 0.1],
                        [0.1, 0.2, 0.3, 0.4],
                        [0.95, 0.02, 0.02, 0.01],
                    ],
                ],
                dtype=tf.float32,
            ),
            "continuous_output": tf.convert_to_tensor(
                [[0, 1.1, 0.75], [-0.5, -2.0, 0], [0.1, 0.2, 0.3]]
            ),
        }

        # Calculate the correct categorical loss by hand
        epsilon = tf.constant(1e-6, dtype=tf.float32)
        crossentropy = (
            -(
                tf.math.log(0.25 + epsilon)
                + tf.math.log(0.5 + epsilon)
                + tf.math.log(0.95 + epsilon)
            )
            / 3
        )
        correct_loss = (
            self.loss_weights["categorical_output"] * crossentropy
            + self.loss_weights["continuous_output"] * 0
        )

        # Call the loss function on these predictions
        cat_loss = self.categorical_loss(
            self.y_true["categorical_labels"], y_pred["categorical_output"]
        )
        cont_loss = self.continuous_loss(
            self.y_true["continuous_labels"], y_pred["continuous_output"]
        )
        loss = (
            self.loss_weights["categorical_output"] * cat_loss
            + self.loss_weights["continuous_output"] * cont_loss
        )

        # Assert the loss function returns the correct loss
        self.assertTrue(np.allclose(loss, correct_loss))

    def test_call_method_computes_correct_loss_on_imperfect_categorical_and_continuous_predictions(
        self,
    ):
        # We will use a combination of the imperfect categorical and imperfect continuous predictions from above
        y_pred = {
            "categorical_output": tf.convert_to_tensor(
                [
                    [[0.25, 0.25, 0.25, 0.25], [1, 0, 0, 0], [0.5, 0.5, 0, 0]],
                    [[0.9, 0.05, 0.05, 0], [0.5, 0, 0, 0.5], [0.2, 0.2, 0.3, 0.3]],
                    [
                        [0.3, 0.3, 0.3, 0.1],
                        [0.1, 0.2, 0.3, 0.4],
                        [0.95, 0.02, 0.02, 0.01],
                    ],
                ],
                dtype=tf.float32,
            ),
            "continuous_output": tf.convert_to_tensor(
                [[[0.5], [0.5], [0.5]], [[0], [1], [0]], [[1], [1], [1]]],
                dtype=tf.float32,
            ),
        }

        # Correct categorical loss
        epsilon = tf.constant(1e-6, dtype=tf.float32)
        crossentropy = (
            -(
                tf.math.log(0.25 + epsilon)
                + tf.math.log(0.5 + epsilon)
                + tf.math.log(0.95 + epsilon)
            )
            / 3
        )

        # Correct continuous loss
        mse = 4.745  # calculated by hand

        # Correct overall loss
        correct_loss = (
            self.loss_weights["categorical_output"] * crossentropy
            + self.loss_weights["continuous_output"] * mse
        )

        # Call the loss function on these predictions
        cat_loss = self.categorical_loss(
            self.y_true["categorical_labels"], y_pred["categorical_output"]
        )
        cont_loss = self.continuous_loss(
            self.y_true["continuous_labels"], y_pred["continuous_output"]
        )
        loss = (
            self.loss_weights["categorical_output"] * cat_loss
            + self.loss_weights["continuous_output"] * cont_loss
        )

        # Assert that the loss function returns the correct loss
        self.assertTrue(np.allclose(loss, correct_loss))

    def test_call_method_returns_0_when_nothing_is_masked(self):
        # Create a y_true that indicates nothing was masked (ie. nothing to be predicted, nothing to compute loss for)
        y_true = {
            "categorical_labels": tf.convert_to_tensor(
                [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]
            ),
            "continuous_labels": tf.convert_to_tensor(
                [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]
            ),
        }

        # Make some nonsense predictions (shouldn't affect loss since mask is 0 everywhere)
        y_pred = {
            "categorical_output": tf.convert_to_tensor(
                [
                    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                    [[0, 0, 0.0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                ],
                dtype=tf.float32,
            ),
            "continuous_output": tf.convert_to_tensor(
                [[[0], [0], [0]], [[0], [0], [0]], [[0], [0], [0]]], dtype=tf.float32
            ),
        }

        # Compute the loss
        cat_loss = self.categorical_loss(
            y_true["categorical_labels"], y_pred["categorical_output"]
        )
        cont_loss = self.continuous_loss(
            y_true["continuous_labels"], y_pred["continuous_output"]
        )
        loss = (
            self.loss_weights["categorical_output"] * cat_loss.numpy()
            + self.loss_weights["continuous_output"] * cont_loss.numpy()
        )

        self.assertEqual(loss, 0)


if __name__ == "__main__":
    unittest.main()
