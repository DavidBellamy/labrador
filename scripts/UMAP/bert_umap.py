import json
import os
import os.path as op
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import TFBertForMaskedLM
from tqdm import tqdm
import umap

# from lab_transformers.models.bert.model_custom_keydim import TFBertForMaskedLM
from lab_transformers.utils import gen_combinations, json_lines_loader

# Make tf quiet
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

config_path = sys.argv[1]

with open(config_path) as f:
    config = json.load(f)

# set random seed
rng = np.random.default_rng(config["random_seed"])

# Load BERT
model = TFBertForMaskedLM.from_pretrained(config["saved_model_path"])
model.config.update({"output_hidden_states": True})

# Load data
df = json_lines_loader(config["data_path"])
num_bags = config["num_bags_of_labs_to_umap"]

# Sample approximately one (lab code, lab value) pair per bag of labs in the data
embedding_data = {"token_bags": [], "subject_id": [], "hadm_id": [], "charttime": []}
num_labs = 0
while num_labs < num_bags:
    # Randomly sample a patient and a bag of labs
    patient_ix = rng.choice(len(df))
    bag_ix = rng.choice(len(df[patient_ix]["token_bags"]))
    token_bag = df[patient_ix]["token_bags"][bag_ix]

    # Collect data needed for UMAP reduction / visualizations
    embedding_data["token_bags"].append(token_bag)
    embedding_data["subject_id"].append(df[patient_ix]["subject_id"])
    embedding_data["hadm_id"].append(df[patient_ix]["hadm_id"][bag_ix])
    embedding_data["charttime"].append(df[patient_ix]["charttime"][bag_ix])

    # Remove the bag from the patient
    df[patient_ix]["token_bags"].pop(bag_ix)

    # If that patient has no more bags of labs, remove the patient
    if len(df[patient_ix]["token_bags"]) == 0:
        df.pop(patient_ix)

    num_labs += len(token_bag)

# convert bags of labs into batches of BERT inputs
model_inputs = dict()
batch_size = config["forward_pass_batch_size"]
num_batches = int(np.ceil(len(embedding_data["token_bags"]) / batch_size))
umap_data = {
    "embeddings": [],
    "subject_id": [],
    "charttime": [],
    "hadm_id": [],
    "token": [],
}

print("\nCreating batches of BERT embeddings\n")
for i in tqdm(range(num_batches)):
    # Get the batch
    start_ix = i * batch_size
    stop_ix = min((i + 1) * batch_size, len(embedding_data["token_bags"]))
    batch = {k: embedding_data[k][start_ix:stop_ix] for k in embedding_data.keys()}

    # Pad the bags
    max_bag_len = 90
    model_inputs["input_ids"] = tf.keras.preprocessing.sequence.pad_sequences(
        batch["token_bags"], padding="post", maxlen=max_bag_len, value=0, dtype="int32"
    )

    # Get the model's embeddings for the current batch
    embeddings = model.predict(model_inputs)["hidden_states"][-1]

    # Mask the embeddings from padding
    mask = np.not_equal(model_inputs["input_ids"], 0)
    umap_data["embeddings"].append(embeddings[mask])
    umap_data["subject_id"].append(
        np.tile(batch["subject_id"], (max_bag_len, 1)).T[mask]
    )
    umap_data["charttime"].append(np.tile(batch["charttime"], (max_bag_len, 1)).T[mask])
    umap_data["hadm_id"].append(np.tile(batch["hadm_id"], (max_bag_len, 1)).T[mask])
    umap_data["token"].append(model_inputs["input_ids"][mask])

print("\nConcatenating batch-wise data\n")
umap_data = {k: np.concatenate(umap_data[k]) for k in umap_data.keys()}

# Create output directory if it doesn't exist
if not os.path.exists(config["outfile_dir"]):
    os.makedirs(config["outfile_dir"])

for hps in gen_combinations(config["umap_parameters"]):
    # Intialize the UMAP reducer
    print("\nInitializing UMAP reducer\n")
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
    print(f"\nUMAP fit complete in {end - start}\n")

    # Create Pandas dataframe out of UMAP-reduced embeddings, subject_id, hadm_id, and charttime
    umap_df = pd.DataFrame(umap_dimreduced_embeddings, columns=["x", "y"])
    umap_df["subject_id"] = umap_data["subject_id"]
    umap_df["hadm_id"] = umap_data["hadm_id"]
    umap_df["charttime"] = umap_data["charttime"]
    umap_df["token"] = umap_data["token"]

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
