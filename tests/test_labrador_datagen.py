import numpy as np
import unittest

from lab_transformers.data.labrador_datagen import labrador_datagen


class TestLabradorDatagen(unittest.TestCase):
    def setUp(self) -> None:
        # Make some test data
        self.data = [
            {
                "subject_id": "1",
                "charttime": ["2022-01-01", "2022-01-02"],
                "code_bags": [[1, 2, 3], [4, 5, 6]],
                "value_bags": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            },
            {
                "subject_id": "2",
                "charttime": ["2022-01-03", "2022-01-04"],
                "code_bags": [[7, 8, 9], [10, 11, 12]],
                "value_bags": [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            },
        ]

        # Set up the generator's initialization params
        self.params = {
            "random_seed": 3141592,
            "mask_token": 999,
            "null_token": 1000,
            "shuffle_patients": False,
            "include_metadata": False,
        }

    def test_labrador_datagen_output(self):
        # Instantiate the generator
        gen = labrador_datagen(self.data, **self.params)

        # Generate a batch from the generator
        batch = next(gen)

        # Unpack the batch
        inputs, targets = batch

        # Assert the output types
        self.assertIsInstance(inputs, dict)
        self.assertIsInstance(targets, dict)

        # Assert the input keys
        self.assertIn("categorical_input", inputs)
        self.assertIn("continuous_input", inputs)

        # Assert the target keys
        self.assertIn("categorical_output", targets)
        self.assertIn("continuous_output", targets)

        # Assert the input shapes
        self.assertEqual(inputs["categorical_input"].shape, (3,))
        self.assertEqual(inputs["continuous_input"].shape, (3,))

        # Assert the target shapes
        self.assertEqual(targets["categorical_output"].shape, (3,))
        self.assertEqual(targets["continuous_output"].shape, (3,))

    def test_labrador_datagen_masking(self):
        # Instantiate the generator
        gen = labrador_datagen(self.data, **self.params)

        # Generate a batch from the generator
        batch = next(gen)

        # Unpack the batch
        inputs, targets = batch

        # Get the mask index
        mask_ix = np.where(inputs["categorical_input"] == self.params["mask_token"])[0][
            0
        ]

        # Assert the masking
        self.assertEqual(inputs["continuous_input"][mask_ix], self.params["mask_token"])
        self.assertNotEqual(
            targets["categorical_output"][mask_ix], self.params["mask_token"]
        )
        self.assertNotEqual(
            targets["continuous_output"][mask_ix], self.params["mask_token"]
        )

    def test_reproducibility(self):
        # Assume: Create two instances with the same parameters. These should generate identical batches of data
        # because they share the same random_seed parameter
        data1 = labrador_datagen(self.data, **self.params)
        data2 = labrador_datagen(self.data, **self.params)

        # Action: Generate a batch from each generator
        inputs1, targets1 = next(data1)
        inputs2, targets2 = next(data2)

        # Assert: the batches generated by each instance of the data generator are identical
        self.assertIsNone(
            np.testing.assert_array_equal(
                targets1["categorical_output"], targets2["categorical_output"]
            )
        )
        self.assertIsNone(
            np.testing.assert_allclose(
                targets1["continuous_output"], targets2["continuous_output"]
            )
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                inputs1["categorical_input"], inputs2["categorical_input"]
            )
        )
        self.assertIsNone(
            np.testing.assert_allclose(
                inputs1["continuous_input"], inputs2["continuous_input"]
            )
        )


if __name__ == "__main__":
    unittest.main()
