import os
import os.path as op
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
import wandb

from lab_transformers.data.read_labrador_tf_records import get_dataset
from lab_transformers.models.labrador.loss import CategoricalMLMLoss, ContinuousMLMLoss
from lab_transformers.models.labrador.model import Labrador

# Parse arguments
random_seed = int(sys.argv[1])
mask_token = int(sys.argv[2])
null_token = int(sys.argv[3])
pad_token = int(sys.argv[4])
vocab_size = int(sys.argv[5])
embed_dim = int(sys.argv[6])

use_wandb = True

# Set configuration
system_config = {
    "random_seed": random_seed,
    "wandb_project_name": "labrador_pretraining",
    "wandb_run_name": "run2",
    "use_mixed_precision": False,
}

data_config = {
    "tfdata_shuffle_buffer_size": 2_560,
    "max_seq_len": 90,
    "tfrecords_dir_train": "data_full/labrador_tfrecords_train",
    "tfrecords_dir_val": "data_full/labrador_tfrecords_val",
}

time_string = time.strftime("%Y%m%d-%H%M%S")
train_config = {
    "steps_per_epoch": (20_000 * 182) // 256,
    "num_train_epochs": 100,
    "learning_rate": 1e-5,
    "batch_size": 256,
    "model_save_batch_frequency": 14_000,  # save the model every n batches during training
    "model_checkpoint_directory_name": f"labrador_{time_string}",
    "validation_steps": ((20_000 * 27) // 256) // 2,
    "validation_step_frequency": 3_500,
}  # perform validation every n training batches

model_config = {
    "mask_token": mask_token,
    "null_token": null_token,
    "pad_token": pad_token,
    "vocab_size": vocab_size,
    "embedding_dim": embed_dim,
    "transformer_activation": "relu",
    "transformer_heads": 4,
    "transformer_blocks": 10,
    "transformer_feedforward_dim": 1024,
    "include_head": True,
    "continuous_head_activation": "sigmoid",
    "categorical_loss_fn": CategoricalMLMLoss(),
    "continuous_loss_fn": ContinuousMLMLoss(),
    "loss_weights": {"categorical_output": 1.0, "continuous_output": 1.0},
    "dropout_rate": 0.1,
}

config = {
    "data_config": data_config,
    "train_config": train_config,
    "model_config": model_config,
    "system_config": system_config,
}

if config["system_config"]["use_mixed_precision"]:
    mixed_precision.set_global_policy("mixed_float16")

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

# Instantiate the transformer model
model = Labrador(config["model_config"])

# Create an optimizer
optimizer = tf.keras.optimizers.Adam(
    learning_rate=config["train_config"]["learning_rate"]
)
optimizer = mixed_precision.LossScaleOptimizer(optimizer)


@tf.function
def train_step_exec(x, y):
    with tf.GradientTape() as tape:
        outputs = model(x, training=True)

        train_categorical_loss = categorical_loss_fn(
            y["categorical_output"], outputs["categorical_output"]
        )
        train_continuous_loss = continuous_loss_fn(
            y["continuous_output"], outputs["continuous_output"]
        )
        train_combined_loss = train_categorical_loss + train_continuous_loss
        scaled_loss = optimizer.get_scaled_loss(train_combined_loss)

    scaled_gradients = tape.gradient(scaled_loss, model.trainable_weights)
    gradients = optimizer.get_unscaled_gradients(scaled_gradients)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    return train_categorical_loss, train_continuous_loss, train_combined_loss


@tf.function
def val_step_exec(x, y):
    outputs = model(x, training=False)
    val_categorical_loss = categorical_loss_fn(
        y["categorical_output"], outputs["categorical_output"]
    )
    val_continuous_loss = continuous_loss_fn(
        y["continuous_output"], outputs["continuous_output"]
    )
    val_combined_loss = val_categorical_loss + val_continuous_loss
    return val_categorical_loss, val_continuous_loss, val_combined_loss


steps_per_epoch = config["train_config"]["steps_per_epoch"]
validation_step_frequency = config["train_config"]["validation_step_frequency"]
validation_steps = config["train_config"]["validation_steps"]
categorical_loss_fn = config["model_config"]["categorical_loss_fn"]
continuous_loss_fn = config["model_config"]["continuous_loss_fn"]
for epoch in range(config["train_config"]["num_train_epochs"]):
    print("\nStart of epoch %d" % (epoch,))
    for train_step, (x_batch_train, y_batch_train) in enumerate(train_dataset, start=1):
        if train_step % config["train_config"]["model_save_batch_frequency"] == 0:
            model.save(
                op.join(
                    "model_weights",
                    f"{config['train_config']['model_checkpoint_directory_name']}_epoch{epoch}",
                )
            )

        if train_step % steps_per_epoch == 0:
            break

        if train_step % validation_step_frequency == 0:
            cat_loss, cont_loss, comb_loss = [], [], []
            for val_step, (x_batch_val, y_batch_val) in enumerate(val_dataset, start=1):
                if val_step % validation_steps == 0:
                    if use_wandb:
                        wandb.log(
                            {
                                "val_loss": np.mean(comb_loss),
                                "val_continuous_output_loss": np.mean(cont_loss),
                                "val_categorical_output_loss": np.mean(cat_loss),
                                "val_step": val_step,
                            }
                        )

                    break

                if (
                    x_batch_val["categorical_input"].shape[1]
                    > config["data_config"]["max_seq_len"]
                ):
                    continue
                categorical_loss, continuous_loss, combined_loss = val_step_exec(
                    x_batch_val, y_batch_val
                )
                cat_loss.append(categorical_loss)
                cont_loss.append(continuous_loss)
                comb_loss.append(combined_loss)

        if (
            x_batch_train["categorical_input"].shape[1]
            > config["data_config"]["max_seq_len"]
        ):
            continue
        categorical_loss, continuous_loss, combined_loss = train_step_exec(
            x_batch_train, y_batch_train
        )

        if use_wandb:
            wandb.log(
                {
                    "loss": combined_loss.numpy(),
                    "continuous_output_loss": continuous_loss.numpy(),
                    "categorical_output_loss": categorical_loss.numpy(),
                    "train_step": train_step,
                }
            )
