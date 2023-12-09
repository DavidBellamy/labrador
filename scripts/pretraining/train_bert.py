import os
import os.path as op
import sys
import time

import numpy as np
import tensorflow as tf
from transformers import BertConfig, TFBertForMaskedLM
from tqdm import tqdm
import wandb

from lab_transformers.data.read_bert_tf_records import get_dataset

# from lab_transformers.models.bert.model_custom_keydim import TFBertForMaskedLM

data_path = sys.argv[1]
random_seed = int(sys.argv[2])
pad_token = int(sys.argv[3])
vocab_size = int(sys.argv[4])
embed_dim = int(sys.argv[5])

time_string = time.strftime("%Y%m%d-%H%M%S")
use_wandb = True

# Set configuration
system_config = {
    "processed_data_path": data_path,
    "random_seed": random_seed,
    "wandb_project_name": "bert_pretraining",
    "wandb_run_name": "run_from_ckpt",
}

data_config = {
    "tfdata_shuffle_buffer_size": 2_560,
    "max_seq_len": 90,
    "tfrecords_dir_train": "data_full/bert_tfrecords_train",
    "tfrecords_dir_val": "data_full/bert_tfrecords_val",
}

train_config = {
    "steps_per_epoch": (20_000 * 204) // 256,
    "num_train_epochs": 100,
    "learning_rate": 1e-5,
    "batch_size": 256,
    "model_save_batch_frequency": 14_000,  # save the model every n batches during training
    "model_checkpoint_directory_name": f"bert194M_{time_string}",
    "validation_steps": ((20_000 * 29) // 256) // 2,
    "validation_step_frequency": 3_500,
}  # perform validation every n training batches

model_config = {
    "vocab_size": vocab_size + 2,
    "hidden_size": embed_dim,
    "num_hidden_layers": 10,
    "num_attention_heads": 4,
    "intermediate_size": 1024,
    "hidden_act": "relu",
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "max_position_embeddings": data_config["max_seq_len"],
}

config = {
    "data_config": data_config,
    "train_config": train_config,
    "model_config": model_config,
    "system_config": system_config,
}

if use_wandb:
    wandb.login(key=os.environ["wandb_key"])
    wandb.init(
        project=config["system_config"]["wandb_project_name"],
        settings=wandb.Settings(start_method="thread"),
        config=config,
        name=config["system_config"]["wandb_run_name"],
    )

# Read TFRecord data
train_filenames = tf.io.gfile.glob(
    op.join(config["data_config"]["tfrecords_dir_train"], "*.tfrec")
)

val_filenames = tf.io.gfile.glob(
    op.join(config["data_config"]["tfrecords_dir_val"], "*.tfrec")
)
train_dataset = get_dataset(
    train_filenames,
    config["train_config"]["batch_size"],
    pad_token,
    random_seed,
    config["data_config"]["tfdata_shuffle_buffer_size"],
)
val_dataset = get_dataset(
    train_filenames,
    config["train_config"]["batch_size"],
    pad_token,
    random_seed,
    config["data_config"]["tfdata_shuffle_buffer_size"],
)

# Instantiate BERT
bertconfig = BertConfig(**config["model_config"])
model = TFBertForMaskedLM(config=bertconfig)

# Create an optimizer
optimizer = tf.keras.optimizers.Adam(
    learning_rate=config["train_config"]["learning_rate"]
)

# Create a loss function
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction=tf.keras.losses.Reduction.NONE
)


@tf.function(reduce_retracing=True)
def train_step_exec(x, y):
    with tf.GradientTape() as tape:
        outputs = model(x, training=True)

        # Clip negative labels to zero here to avoid NaNs and errors - those positions will get masked later anyway
        unmasked_lm_losses = loss_fn(y_true=tf.nn.relu(y), y_pred=outputs["logits"])
        # make sure only labels that are not equal to -100
        # are taken into account for the loss computation
        lm_loss_mask = tf.cast(y != -100, dtype=unmasked_lm_losses.dtype)
        masked_lm_losses = unmasked_lm_losses * lm_loss_mask
        reduced_masked_lm_loss = tf.reduce_sum(masked_lm_losses) / tf.reduce_sum(
            lm_loss_mask
        )

    gradients = tape.gradient(reduced_masked_lm_loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    return reduced_masked_lm_loss


@tf.function(reduce_retracing=True)
def val_step_exec(x, y):
    outputs = model(x, training=False)

    # Clip negative labels to zero here to avoid NaNs and errors - those positions will get masked later anyway
    unmasked_lm_losses = loss_fn(y_true=tf.nn.relu(y), y_pred=outputs["logits"])
    # make sure only labels that are not equal to -100
    # are taken into account for the loss computation
    lm_loss_mask = tf.cast(y != -100, dtype=unmasked_lm_losses.dtype)
    masked_lm_losses = unmasked_lm_losses * lm_loss_mask
    reduced_masked_lm_loss = tf.reduce_sum(masked_lm_losses) / tf.reduce_sum(
        lm_loss_mask
    )

    return reduced_masked_lm_loss


steps_per_epoch = config["train_config"]["steps_per_epoch"]
validation_step_frequency = config["train_config"]["validation_step_frequency"]
validation_steps = config["train_config"]["validation_steps"]
for epoch in range(config["train_config"]["num_train_epochs"]):
    print("\nStart of epoch %d" % (epoch,))
    for train_step, batch in tqdm(enumerate(train_dataset, start=1)):
        if train_step % config["train_config"]["model_save_batch_frequency"] == 0:
            model.save_pretrained(
                op.join(
                    "model_weights",
                    f"{config['train_config']['model_checkpoint_directory_name']}_epoch{epoch}",
                )
            )

        if train_step % steps_per_epoch == 0:
            break

        if train_step % validation_step_frequency == 0:
            loss_list = []
            for val_step, val_batch in enumerate(val_dataset, start=1):
                if val_step % validation_steps == 0:
                    if use_wandb:
                        wandb.log(
                            {"val_loss": np.mean(loss_list), "val_step": val_step}
                        )

                    break

                if (
                    val_batch["input_ids"].shape[1]
                    > config["data_config"]["max_seq_len"]
                ):  # skip batches that have seqlen > max_seq_len (avoid OOM)
                    continue

                val_loss = val_step_exec(val_batch["input_ids"], val_batch["labels"])
                loss_list.append(val_loss)

        if (
            batch["input_ids"].shape[1] > config["data_config"]["max_seq_len"]
        ):  # skip batches that have seqlen > max_seq_len (avoid OOM)
            continue

        train_loss = train_step_exec(batch["input_ids"], batch["labels"])

        if use_wandb:
            wandb.log({"train_loss": train_loss.numpy(), "train_step": train_step})
