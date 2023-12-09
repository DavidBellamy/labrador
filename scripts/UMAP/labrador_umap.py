import json
import os
import os.path as op
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import umap

from lab_transformers.models.labrador.model import Labrador
from lab_transformers.utils import gen_combinations, json_lines_loader

# Make tf quiet
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

config_path = sys.argv[1]

with open(config_path) as f:
    config = json.load(f)

# set random seed
rng = np.random.default_rng(config["random_seed"])

# Load Labrador
with open(op.join(config["saved_model_path"], "config.json")) as f:
    saved_config = json.load(f)

model = Labrador(saved_config)
model.load_weights(op.join(config["saved_model_path"], "variables/variables"))
model.include_head = False

# Load data
df = json_lines_loader(config["data_path"])
num_bags = config["num_bags_of_labs_to_umap"]

# Sample approximately one (lab code, lab value) pair per bag of labs in the data
embedding_data = {
    "code_bags": [],
    "value_bags": [],
    "subject_id": [],
    "hadm_id": [],
    "charttime": [],
}
num_labs = 0
while num_labs < num_bags:
    # Randomly sample a patient and a bag of labs
    patient_ix = rng.choice(len(df))
    bag_ix = rng.choice(len(df[patient_ix]["code_bags"]))
    lab_code_bag = df[patient_ix]["code_bags"][bag_ix]
    lab_value_bag = df[patient_ix]["value_bags"][bag_ix]

    # Collect data needed for UMAP reduction / visualizations
    embedding_data["code_bags"].append(lab_code_bag)
    embedding_data["value_bags"].append(lab_value_bag)
    embedding_data["subject_id"].append(df[patient_ix]["subject_id"])
    embedding_data["hadm_id"].append(df[patient_ix]["hadm_id"][bag_ix])
    embedding_data["charttime"].append(df[patient_ix]["charttime"][bag_ix])

    # Remove the bag from the patient
    df[patient_ix]["code_bags"].pop(bag_ix)

    # If that patient has no more bags of labs, remove the patient
    if len(df[patient_ix]["code_bags"]) == 0:
        df.pop(patient_ix)

    num_labs += len(lab_code_bag)

# Replace <NULL> strings with Labrador's <null_token>
embedding_data["value_bags"] = [
    [v if v != "<NULL>" else saved_config["null_token"] for v in bag]
    for bag in embedding_data["value_bags"]
]

# convert bags of labs into batches of Labrador inputs
model_inputs = dict()
batch_size = config["forward_pass_batch_size"]
num_batches = int(np.ceil(len(embedding_data["code_bags"]) / batch_size))
umap_data = {
    "embeddings": [],
    "subject_id": [],
    "charttime": [],
    "hadm_id": [],
    "lab_code": [],
    "lab_value": [],
}

print("\nCreating batches of Labrador embeddings\n")
for i in tqdm(range(num_batches)):
    # Get the batch
    start_ix = i * batch_size
    stop_ix = min((i + 1) * batch_size, len(embedding_data["code_bags"]))
    batch = {k: embedding_data[k][start_ix:stop_ix] for k in embedding_data.keys()}

    # Pad the bags
    max_bag_len = max([len(bag) for bag in batch["code_bags"]])
    model_inputs["categorical_input"] = tf.keras.preprocessing.sequence.pad_sequences(
        batch["code_bags"], padding="post", maxlen=max_bag_len, value=0, dtype="int32"
    )
    model_inputs["continuous_input"] = tf.keras.preprocessing.sequence.pad_sequences(
        batch["value_bags"],
        padding="post",
        maxlen=max_bag_len,
        value=0,
        dtype="float32",
    )

    # Get the model's embeddings for the current batch
    embeddings = model.predict(model_inputs)

    # Mask the embeddings from padding
    mask = np.not_equal(model_inputs["categorical_input"], saved_config["pad_token"])
    umap_data["embeddings"].append(embeddings[mask])
    umap_data["subject_id"].append(
        np.tile(batch["subject_id"], (max_bag_len, 1)).T[mask]
    )
    umap_data["charttime"].append(np.tile(batch["charttime"], (max_bag_len, 1)).T[mask])
    umap_data["hadm_id"].append(np.tile(batch["hadm_id"], (max_bag_len, 1)).T[mask])
    umap_data["lab_code"].append(model_inputs["categorical_input"][mask])
    umap_data["lab_value"].append(model_inputs["continuous_input"][mask])

umap_data = {k: np.concatenate(umap_data[k]) for k in umap_data.keys()}

for hps in gen_combinations(config["umap_parameters"]):
    # Intialize the UMAP reducer
    reducer = umap.UMAP(
        n_neighbors=hps["umap_n_neighbors"],
        min_dist=hps["umap_min_dist"],
        random_state=config["random_seed"],
        verbose=True,
    )

    # Fit UMAP on the embedding columns
    print("\nFitting UMAP reducer\n")
    start = time.time()
    umap_dimreduced_embeddings = reducer.fit_transform(umap_data["embeddings"])
    end = time.time()
    print(f"\nUMAP fit complete in {end - start}\n", file=sys.stderr)

    # Create Pandas dataframe out of UMAP-reduced embeddings, subject_id, hadm_id, and charttime
    umap_df = pd.DataFrame(umap_dimreduced_embeddings, columns=["x", "y"])
    umap_df["subject_id"] = umap_data["subject_id"]
    umap_df["hadm_id"] = umap_data["hadm_id"]
    umap_df["charttime"] = umap_data["charttime"]
    umap_df["lab_code"] = umap_data["lab_code"]
    umap_df["lab_value"] = umap_data["lab_value"]

    # Save the dataframe
    time_string = time.strftime("%Y%m%d-%H%M%S")
    umap_df.to_csv(
        op.join(
            config["outfile_dir"],
            f"{config['outfile_name']}_"
            f"{time_string}_"
            f"nb{hps['umap_n_neighbors']}_"
            f"md{hps['umap_min_dist']}.csv",
        ),
        index=False,
    )
