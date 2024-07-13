import os
import unittest

import numpy as np
import tensorflow as tf
from transformers import BertConfig, TFBertForMaskedLM
from transformers.modeling_tf_outputs import TFMaskedLMOutput

from lab_transformers.models.bert.finetuning_wrapper import BertFinetuneWrapper

# from lab_transformers.models.bert.model_custom_keydim import TFBertForMaskedLM
from lab_transformers.utils import empty_folder


class TestBERT(unittest.TestCase):
    def setUp(self) -> None:
        self.params = BertConfig(
            attention_probs_dropout_prob=0.1,
            hidden_act="relu",
            hidden_dropout_prob=0.1,
            hidden_size=32,
            initializer_range=0.02,
            intermediate_size=64,
            layer_norm_eps=1e-12,
            max_position_embeddings=90,
            model_type="bert",
            num_attention_heads=1,
            num_hidden_layers=2,
            pad_token_id=0,
            position_embedding_type="absolute",
            transformers_version="4.24.0",
            type_vocab_size=2,
            use_cache=True,
            vocab_size=100,
        )

        self.model = TFBertForMaskedLM(config=self.params)
        self.inputs = {
            "input_ids": np.random.choice(
                self.params.vocab_size, (1, self.params.max_position_embeddings)
            )
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

    def test_bert_forward_pass(self):
        output = self.model(self.inputs)

        self.assertIsInstance(output, TFMaskedLMOutput)
        self.assertIn("logits", output)
        self.assertEqual(
            output.logits.shape,
            (1, self.params.max_position_embeddings, self.params.vocab_size),
        )

    def test_bert_loss(self):
        labels = np.array([[-100, 1, 2, -100], [0, -100, -100, 2]])
        output_logits = np.array(
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
        )

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )

        # Clip negative labels to zero here to avoid NaNs and errors - those positions will get masked later anyway
        unmasked_lm_losses = loss_fn(y_true=tf.nn.relu(labels), y_pred=output_logits)

        # make sure only labels that are not equal to -100
        # are taken into account for the loss computation
        lm_loss_mask = tf.cast(labels != -100, dtype=unmasked_lm_losses.dtype)
        masked_lm_losses = unmasked_lm_losses * lm_loss_mask
        reduced_masked_lm_loss = tf.reduce_sum(masked_lm_losses) / tf.reduce_sum(
            lm_loss_mask
        )
        self.assertEqual(reduced_masked_lm_loss, 1.148901016653798)

    def test_bert_finetuning_wrapper(self):
        model_wrapper = BertFinetuneWrapper(
            base_model_path=None,
            output_size=3,
            output_activation="softmax",
            dropout_rate=0.1,
            add_extra_dense_layer=False,
            train_base_model=False,
        )

        # Add non-mimic features to inputs to test the wrapper on them
        inputs = self.inputs | {"non_mimic_features": np.random.rand(1, 100)}

        output = model_wrapper(inputs)

        # Assert sum of each row of outputs equals 1 (softmax)
        self.assertTrue(np.allclose(np.sum(output.numpy(), axis=1), 1))

    def test_saveload_bert_model(self):
        self.model(self.inputs, training=False)  # Initialize the model
        self.model.save_pretrained(self.model_target_directory)

        loaded_model = TFBertForMaskedLM.from_pretrained(self.model_target_directory)

        # Generate predictions by the original and loaded models (given the same inputs)
        orig_preds = self.model(self.inputs, training=False)
        loaded_preds = loaded_model(self.inputs, training=False)

        # Assert that all predictions are identical
        self.assertTrue(
            np.array_equal(
                orig_preds["logits"],
                loaded_preds["logits"],
            )
        )
        self.assertTrue(np.array_equal(orig_preds["logits"], loaded_preds["logits"]))


if __name__ == "__main__":
    unittest.main()
