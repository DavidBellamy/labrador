import unittest
from unittest.mock import MagicMock, patch

import tensorflow as tf

from lab_transformers.data.read_bert_tf_records import get_dataset


class TestRead_BERT_TFRecords(unittest.TestCase):
    def setUp(self):
        self.mock_filepaths = ["fake_path1.tfrec", "fake_path2.tfrec"]  # Example paths
        self.filepaths = tf.io.gfile.glob(
            f"tests/test_data/fake_bert_tfrecords/*.tfrec"
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
        self.assertIsInstance(batch, dict)
        self.assertIn("input_ids", batch)
        self.assertIn("labels", batch)
        self.assertIsInstance(batch["input_ids"], tf.Tensor)
        self.assertIsInstance(batch["labels"], tf.Tensor)
        self.assertEqual(batch["input_ids"].dtype, tf.int32)
        self.assertEqual(batch["labels"].dtype, tf.int32)
        self.assertEqual(batch["input_ids"].shape, batch["labels"].shape)

    @patch("lab_transformers.data.read_bert_tf_records.tf.data.TFRecordDataset")
    @patch("lab_transformers.data.read_bert_tf_records.parse_tfrecord_fn")
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
            padding_values={"input_ids": self.pad_token, "labels": -100},
            padded_shapes={"input_ids": [None], "labels": [None]},
            drop_remainder=True,
        )
        expected_dataset.prefetch.assert_called_once_with(tf.data.AUTOTUNE)
        self.assertEqual(dataset, expected_dataset)


if __name__ == "__main__":
    unittest.main()
