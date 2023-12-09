import numpy as np
import pandas as pd
import unittest

from lab_transformers.data.tokenize_tabular_data import (
    make_bert_inputs,
    map_lab_values_to_eCDF_values,
    make_labrador_inputs,
    mimic4_eCDFer,
)


class TestTokenizeTabularData(unittest.TestCase):
    def setUp(self):
        # Load eCDF map
        self.ecdf_data = np.load("data/processed/mimic4_ecdfs.npz")
        self.mapper = mimic4_eCDFer(self.ecdf_data)

    def test_mimic4_ecdfer_call(self):
        # Test __call__ method
        itemids = [50862, 50863, 50864]
        lab_values = [0.5, 1.1, 3]
        null_token = np.nan
        output = self.mapper(itemids, lab_values, null_token)
        self.assertIsInstance(output, np.ndarray)
        self.assertEqual(output.dtype, np.float_)

    def test_mimic4_ecdfer_with_nan_values(self):
        # Test that nan values are mapped to null_token
        itemids = [50862, 50863, 50864]
        lab_values = [0.5, 1.1, np.nan]
        null_token = np.nan
        output = self.mapper(itemids, lab_values, null_token)
        self.assertTrue(np.isnan(output[-1]))

    def test_map_lab_values_to_eCDF_values(self):
        # Test map_lab_values_to_eCDF_values function
        ecdf_data = np.load("data/processed/mimic4_ecdfs.npz")
        df = pd.DataFrame(
            {
                "50862": [0.5, 1.1, 3],
                "50863": [1.2, 2.3, 4.5],
                "50864": [2.5, 3.7, 6.8],
            }
        )

        # Call the function
        output_df = map_lab_values_to_eCDF_values(df, ecdf_data)

        # Assert
        self.assertIsInstance(output_df, pd.DataFrame)
        self.assertEqual(output_df.columns.tolist(), df.columns.tolist())
        self.assertEqual(output_df.shape, df.shape)
        self.assertTrue((output_df >= 0).all().all())
        self.assertTrue((output_df <= 1).all().all())

    def test_make_bert_inputs(self):
        # Set up test data
        self.df = pd.DataFrame(
            {
                "50862": [0.5, 0.6, 0.3],
                "50863": [0.2, 0.3, 0.5],
                "label": [0.0, 1.0, 0.0],
            }
        )
        self.label_col = "label"
        self.codebook = pd.DataFrame(
            {"itemid": [50862, 50863], "valuenum": [1.0, 3.0], "token": ["1", "2"]}
        )
        self.mask_null = False
        self.mask_token = None

        # Call the function
        model_inputs, labels, non_mimic_feature_values = make_bert_inputs(
            self.df, self.label_col, self.codebook, self.mask_null, self.mask_token
        )

        # Assert model_inputs
        self.assertIsInstance(model_inputs, dict)
        self.assertIn("input_ids", model_inputs)
        self.assertIsInstance(model_inputs["input_ids"], np.ndarray)
        self.assertEqual(model_inputs["input_ids"].shape, (3, 2))
        self.assertEqual(model_inputs["input_ids"].dtype, np.int32)

        # Assert labels
        self.assertIsInstance(labels, np.ndarray)
        self.assertEqual(labels.shape, (3, 1))
        self.assertEqual(labels.dtype, np.float64)

        # Assert non_mimic_feature_values
        self.assertEqual(non_mimic_feature_values, None)

    def test_make_bert_inputs_invalid_lab_values(self):
        # Set up test data
        self.df = pd.DataFrame(
            {
                "50862": [
                    0.5,
                    1.1,
                    0.3,
                ],  # lab values should be post-eCDF transformation, lying on [0, 1]
                "50863": [0.2, 0.3, 0.5],
                "label": [0.0, 1.0, 0.0],
            }
        )
        self.label_col = "label"
        self.codebook = pd.DataFrame(
            {"itemid": [50862, 50863], "valuenum": [1.0, 3.0], "token": ["1", "2"]}
        )
        self.mask_null = False
        self.mask_token = None

        # Call the function and expect an AssertionError
        with self.assertRaises(AssertionError):
            make_bert_inputs(
                self.df, self.label_col, self.codebook, self.mask_null, self.mask_token
            )

    def test_make_labrador_inputs(self):
        # Set up test data
        df = pd.DataFrame(
            {"50862": [0.5, 0.9, 1], "50863": [0.2, 1, 1], "label": [0.0, 1.0, 0.0]}
        )
        label_col = "label"
        codebook = pd.DataFrame(
            {"itemid": [50862, 50863], "frequency_rank": [1, 2], "token": ["1", "2"]}
        )
        null_token = -1

        # Call the function
        model_inputs, labels, non_mimic_feature_values = make_labrador_inputs(
            df, label_col, codebook, null_token
        )

        # Assert model_inputs
        self.assertIsInstance(model_inputs, dict)
        self.assertIn("categorical_input", model_inputs)
        self.assertIn("continuous_input", model_inputs)
        self.assertIsInstance(model_inputs["categorical_input"], np.ndarray)
        self.assertIsInstance(model_inputs["continuous_input"], np.ndarray)
        self.assertEqual(model_inputs["categorical_input"].shape, (3, 2))
        self.assertEqual(model_inputs["continuous_input"].shape, (3, 2))
        self.assertEqual(model_inputs["categorical_input"].dtype, np.int64)
        self.assertEqual(model_inputs["continuous_input"].dtype, np.float64)

        # Assert labels
        self.assertIsInstance(labels, np.ndarray)
        self.assertEqual(labels.shape, (3, 1))
        self.assertEqual(labels.dtype, np.float64)

        # Assert non_mimic_feature_values
        self.assertEqual(non_mimic_feature_values, None)


if __name__ == "__main__":
    unittest.main()
