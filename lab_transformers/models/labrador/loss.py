import tensorflow as tf
from tensorflow.keras.losses import (
    MeanSquaredError,
    SparseCategoricalCrossentropy,
    KLDivergence,
)


class ContinuousMLMLoss(tf.keras.losses.Loss):
    def __init__(self, *args, **kwargs) -> None:
        """
        Masked MSE for Labrador's continuous prediction head.
        """
        super(ContinuousMLMLoss, self).__init__()
        self.mse = MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_masked = tf.boolean_mask(y_true, tf.not_equal(y_true, -1))
        y_pred_masked = tf.boolean_mask(y_pred, tf.not_equal(y_true, -1))

        if tf.size(y_true_masked) == 0 and tf.size(y_pred_masked) == 0:
            return 0

        loss = self.mse(y_true_masked, y_pred_masked)
        return loss


class CategoricalMLMLoss(tf.keras.losses.Loss):
    def __init__(self, *args, **kwargs) -> None:
        """
        Masked SparseCategoricalCrossentropy for Labrador's categorical prediction head.
        """
        super(CategoricalMLMLoss, self).__init__()
        self.scce = SparseCategoricalCrossentropy(
            from_logits=False, reduction=tf.keras.losses.Reduction.NONE
        )

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_masked = tf.boolean_mask(y_true, tf.not_equal(y_true, -1))
        y_true_masked = tf.subtract(
            y_true_masked, 1
        )  # true labels are 1-indexed, need to be 0-indexed
        y_pred_masked = tf.boolean_mask(y_pred, tf.not_equal(y_true, -1))
        loss = self.scce(y_true_masked, y_pred_masked)
        return loss
