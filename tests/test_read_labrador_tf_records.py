import unittest
from unittest.mock import MagicMock, patch

import tensorflow as tf

from lab_transformers.data.read_labrador_tf_records import get_dataset


class TestRead_Labrador_TFRecords(unittest.TestCase):
    def setUp(self):
        self.mock_filepaths = ["fake_path1.tfrec", "fake_path2.tfrec"]  # Example paths
        self.filepaths = tf.io.gfile.glob(
            f"tests/test_data/fake_labrador_tfrecords/*.tfrec"
        )
        self.batch_size = 32
        self.pad_token = 0
        self.random_seed = 3141592
        self.shuffle_buffer_size = 100

    def test_get_dataset(self):
        # Arrange
        filepaths = self.filepaths
        batch_size = self.batch_size
        pad_token = self.pad_token
        random_seed = self.random_seed
        shuffle_buffer_size = self.shuffle_buffer_size

        # Act
        dataset = get_dataset(
            filepaths, batch_size, pad_token, random_seed, shuffle_buffer_size
        )
        batch = next(iter(dataset))

        # Assert
        self.assertIsInstance(batch, tuple)
        self.assertIn("continuous_input", batch[0])
        self.assertIn("categorical_input", batch[0])
        self.assertIn("continuous_output", batch[1])
        self.assertIn("categorical_output", batch[1])
        self.assertIsInstance(batch[0]["continuous_input"], tf.Tensor)
        self.assertIsInstance(batch[0]["categorical_input"], tf.Tensor)
        self.assertIsInstance(batch[1]["continuous_output"], tf.Tensor)
        self.assertIsInstance(batch[1]["categorical_output"], tf.Tensor)
        self.assertEqual(batch[0]["continuous_input"].dtype, tf.float32)
        self.assertEqual(batch[0]["categorical_input"].dtype, tf.int32)
        self.assertEqual(batch[1]["continuous_output"].dtype, tf.float32)
        self.assertEqual(batch[1]["categorical_output"].dtype, tf.int32)
        self.assertEqual(
            batch[0]["continuous_input"].shape, batch[0]["categorical_input"].shape
        )
        self.assertEqual(
            batch[0]["continuous_input"].shape, batch[1]["continuous_output"].shape
        )
        self.assertEqual(
            batch[0]["continuous_input"].shape, batch[1]["categorical_output"].shape
        )
        self.assertEqual(
            batch[0]["categorical_input"].shape, batch[1]["categorical_output"].shape
        )
        self.assertEqual(
            batch[0]["categorical_input"].shape, batch[1]["continuous_output"].shape
        )
        self.assertEqual(
            batch[1]["continuous_output"].shape, batch[1]["categorical_output"].shape
        )

    @patch("lab_transformers.data.read_labrador_tf_records.tf.data.TFRecordDataset")
    @patch("lab_transformers.data.read_labrador_tf_records.parse_tfrecord_fn")
    def test_dataset_parameters(self, mock_parse_fn, mock_tfrecord_dataset):
        # Arrange
        expected_dataset = MagicMock()
        mock_tfrecord_dataset.return_value = expected_dataset
        expected_dataset.map.return_value = expected_dataset
        expected_dataset.shuffle.return_value = expected_dataset
        expected_dataset.padded_batch.return_value = expected_dataset
        expected_dataset.prefetch.return_value = expected_dataset

        # Act
        dataset = get_dataset(
            self.mock_filepaths,
            self.batch_size,
            self.pad_token,
            self.random_seed,
            self.shuffle_buffer_size,
        )

        # Assert
        mock_tfrecord_dataset.assert_called_once_with(
            self.mock_filepaths, num_parallel_reads=tf.data.AUTOTUNE
        )
        expected_dataset.map.assert_called_once_with(
            mock_parse_fn, num_parallel_calls=tf.data.AUTOTUNE
        )
        expected_dataset.shuffle.assert_called_once_with(
            self.shuffle_buffer_size, seed=self.random_seed
        )
        expected_dataset.padded_batch.assert_called_once_with(
            batch_size=self.batch_size,
            padding_values=(
                {
                    "categorical_input": self.pad_token,
                    "continuous_input": float(self.pad_token),
                },
                {"categorical_output": -1, "continuous_output": -1.0},
            ),
            padded_shapes=(
                {"categorical_input": [None], "continuous_input": [None]},
                {"categorical_output": [None], "continuous_output": [None]},
            ),
            drop_remainder=True,
        )
        expected_dataset.prefetch.assert_called_once_with(tf.data.AUTOTUNE)
        self.assertEqual(dataset, expected_dataset)


if __name__ == "__main__":
    unittest.main()
