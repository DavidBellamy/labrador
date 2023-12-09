import json
import os
import os.path as op
import sys
import time

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold
from tensorflow.keras.utils import set_random_seed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy, Reduction
from tqdm import tqdm
import wandb

from lab_transformers.models.bert.finetuning_wrapper import BertFinetuneWrapper
from lab_transformers.utils import gen_combinations

time_string = time.strftime("%Y%m%d-%H%M%S")
wandb.login(key=os.environ["wandb_key"])

config_path = sys.argv[1]
fraction_of_data_to_use = float(sys.argv[2])

# Load the job's config
with open(config_path) as f:
    config = json.load(f)

os.environ["WANDB_MODE"] = config["system_config"]["wandb_mode"]

# Add fraction to the config
config["system_config"]["fraction_of_data_to_use"] = fraction_of_data_to_use

# Set tf/keras & numpy random seeds
set_random_seed(config["system_config"]["random_seed"])
rng = np.random.default_rng(config["system_config"]["random_seed"])

transformer_dir = config.get("system_config", {}).get("transformer_dir", None)
transformer_path = (
    op.join(config["system_config"]["saved_model_path"], transformer_dir)
    if transformer_dir is not None
    else None
)

# Load evaluation data
model_inputs = np.load(
    op.join(
        config["system_config"]["dataset_dir"], config["system_config"]["dataset_name"]
    )
)

num_classes = len(np.unique(model_inputs["label"]))
scce = SparseCategoricalCrossentropy(
    from_logits=False, reduction=Reduction.SUM_OVER_BATCH_SIZE
)
results = []
use_wandb = config["system_config"]["use_wandb"].lower() == "true"
for rep in range(config["train_config"]["num_reps"]):
    # Separate a random 10% of the data for testing
    test_idx = rng.choice(
        model_inputs["label"].shape[0],
        size=int(model_inputs["label"].shape[0] * 0.1),
        replace=False,
    )
    X_test = {k: model_inputs[k][test_idx] for k in model_inputs.keys()}
    y_test = model_inputs["label"][test_idx]

    # Use the remaining 90% of the data for the evaluation
    mask = np.ones(model_inputs["label"].shape[0], dtype=bool)
    mask[test_idx] = False
    X_dev = {k: model_inputs[k][mask] for k in model_inputs.keys()}
    y_dev = model_inputs["label"][mask]

    # Separate a random fraction of the data for model selection
    train_ix = rng.choice(
        y_dev.shape[0],
        size=int(fraction_of_data_to_use * y_dev.shape[0]),
        replace=False,
    )
    X_train = {k: X_dev[k][train_ix] for k in X_dev.keys()}
    y_train = y_dev[train_ix]

    best_ce = None
    best_hps = None
    num_hp_combinations = len(list(gen_combinations(config["train_config"])))
    for j, train_config_i in enumerate(gen_combinations(config["train_config"])):
        if (
            config["train_config"]["skip_cross_validation"].lower() == "true"
        ) and num_hp_combinations == 1:
            best_hps = train_config_i
            break
        elif (
            config["train_config"]["skip_cross_validation"].lower() == "true"
        ) and num_hp_combinations > 1:
            raise ValueError(
                "skip_cross_validation is True but there are multiple hyperparameter combinations to "
                "evaluate. Set skip_cross_validation to False or remove the hyperparameter combinations."
            )

        transformer_crossentropies = []
        kf = KFold(
            n_splits=config["train_config"]["num_k_folds"],
            shuffle=True,
            random_state=config["system_config"]["random_seed"],
        )
        for i, (ktrain_index, kval_index) in tqdm(enumerate(kf.split(y_train))):
            X_ktrain = {k: X_train[k][ktrain_index] for k in X_train.keys()}
            X_kval = {k: X_train[k][kval_index] for k in X_train.keys()}
            y_ktrain, y_kval = y_train[ktrain_index], y_train[kval_index]

            # Standardize the continuous non-mimic-lab features in X
            scaler = preprocessing.StandardScaler().fit(
                X_ktrain["non_mimic_features_continuous"]
            )
            X_ktrain["non_mimic_features_continuous"] = scaler.transform(
                X_ktrain["non_mimic_features_continuous"]
            )

            # Combine discrete and continuous non-mimic features
            X_ktrain["non_mimic_features"] = np.concatenate(
                [
                    X_ktrain["non_mimic_features_discrete"],
                    X_ktrain["non_mimic_features_continuous"],
                ],
                axis=1,
            )

            # Fit and evaluate the model
            if config.get("train_config", {}).get("real_ensembling_samples", False):
                model = [
                    BertFinetuneWrapper(
                        base_model_path=transformer_path,
                        output_size=num_classes,
                        output_activation=config["train_config"]["output_activation"],
                        dropout_rate=train_config_i["dropout_rate"],
                        add_extra_dense_layer=train_config_i["add_extra_dense_layer"],
                        train_base_model=config["train_config"][
                            "train_base_model"
                        ].lower()
                        == "true",
                    )
                    for _ in range(config["train_config"]["real_ensembling_samples"])
                ]

            else:
                model = [
                    BertFinetuneWrapper(
                        base_model_path=transformer_path,
                        output_size=num_classes,
                        output_activation=config["train_config"]["output_activation"],
                        dropout_rate=train_config_i["dropout_rate"],
                        add_extra_dense_layer=train_config_i["add_extra_dense_layer"],
                        train_base_model=config["train_config"][
                            "train_base_model"
                        ].lower()
                        == "true",
                    )
                ]

            for m in model:
                m.compile(
                    loss=config["train_config"]["loss_function"],
                    optimizer=Adam(learning_rate=train_config_i["learning_rate"]),
                    run_eagerly=config["system_config"]["run_eagerly"].lower()
                    == "true",
                )

            for ix, m in enumerate(model):
                m.fit(
                    X_ktrain,
                    y_ktrain,
                    epochs=train_config_i["num_epochs"],
                    batch_size=train_config_i["batch_size"],
                    verbose=True,
                )

            # Calculate categorical cross-entropy  (compare to y_kval)
            X_kval["non_mimic_features_continuous"] = scaler.transform(
                X_kval["non_mimic_features_continuous"]
            )
            X_kval["non_mimic_features"] = np.concatenate(
                [
                    X_kval["non_mimic_features_discrete"],
                    X_kval["non_mimic_features_continuous"],
                ],
                axis=1,
            )

            if config.get("train_config", {}).get(
                "monte_carlo_dropout_ensembling_samples", False
            ):
                pred_samples = np.array(
                    [
                        m.call(X_kval, training=True)
                        for m in model
                        for _ in range(
                            config["train_config"][
                                "monte_carlo_dropout_ensembling_samples"
                            ]
                        )
                    ]
                )
                y_kpred = np.mean(pred_samples, axis=0)
            else:
                y_kpred = [m.call(X_kval) for m in model]
                y_kpred = np.mean(y_kpred, axis=0)

            categorical_ce = scce(y_kval, y_kpred).numpy()
            transformer_crossentropies.append(categorical_ce)

        if use_wandb:
            run_name = (
                f"config_loss_fraction{fraction_of_data_to_use:.2f}_rep{rep}_config{j}"
            )
            run = wandb.init(
                project=config["system_config"]["wandb_project_name"],
                config=config | train_config_i | {"rep_num": rep, "config_num": j},
                name=run_name,
            )
            wandb.log({"cross_entropy_per_config": np.mean(transformer_crossentropies)})
            run.finish()

        if best_ce is None or np.mean(transformer_crossentropies) < best_ce:
            best_ce = np.mean(transformer_crossentropies)
            best_hps = train_config_i

    # Fit the model with the best hyperparameters on the full training set
    if config.get("train_config", {}).get("real_ensembling_samples", False):
        model = [
            BertFinetuneWrapper(
                base_model_path=transformer_path,
                output_size=num_classes,
                output_activation=config["train_config"]["output_activation"],
                dropout_rate=best_hps["dropout_rate"],
                add_extra_dense_layer=best_hps["add_extra_dense_layer"],
                train_base_model=config["train_config"]["train_base_model"].lower()
                == "true",
            )
            for _ in range(config["train_config"]["real_ensembling_samples"])
        ]

    else:
        model = [
            BertFinetuneWrapper(
                base_model_path=transformer_path,
                output_size=num_classes,
                output_activation=config["train_config"]["output_activation"],
                dropout_rate=best_hps["dropout_rate"],
                add_extra_dense_layer=best_hps["add_extra_dense_layer"],
                train_base_model=config["train_config"]["train_base_model"].lower()
                == "true",
            )
        ]

    for m in model:
        m.compile(
            loss=config["train_config"]["loss_function"],
            optimizer=Adam(learning_rate=best_hps["learning_rate"]),
            run_eagerly=config["system_config"]["run_eagerly"].lower() == "true",
        )

    scaler = preprocessing.StandardScaler().fit(
        X_train["non_mimic_features_continuous"]
    )
    X_train_scaled = {k: X_train[k] for k in X_train.keys()}
    X_train_scaled["non_mimic_features_continuous"] = scaler.transform(
        X_train["non_mimic_features_continuous"]
    )
    X_train_scaled["non_mimic_features"] = np.concatenate(
        [
            X_train_scaled["non_mimic_features_discrete"],
            X_train_scaled["non_mimic_features_continuous"],
        ],
        axis=1,
    )

    for ix, m in enumerate(model):
        m.fit(
            X_train_scaled,
            y_train,
            epochs=best_hps["num_epochs"],
            batch_size=best_hps["batch_size"],
            verbose=True,
        )

    X_test_scaled = {k: X_test[k] for k in X_test.keys()}
    X_test_scaled["non_mimic_features_continuous"] = scaler.transform(
        X_test["non_mimic_features_continuous"]
    )
    X_test_scaled["non_mimic_features"] = np.concatenate(
        [
            X_test_scaled["non_mimic_features_discrete"],
            X_test_scaled["non_mimic_features_continuous"],
        ],
        axis=1,
    )

    if config.get("train_config", {}).get(
        "monte_carlo_dropout_ensembling_samples", False
    ):
        pred_samples = np.array(
            [
                m.call(X_test_scaled, training=True)
                for m in model
                for _ in range(
                    config["train_config"]["monte_carlo_dropout_ensembling_samples"]
                )
            ]
        )
        y_pred = np.mean(pred_samples, axis=0)
    else:
        y_pred = [m.call(X_test_scaled) for m in model]
        y_pred = np.mean(y_pred, axis=0)

    ce = scce(y_test, y_pred).numpy()

    # Save y_test and y_pred to csv
    test_prediction_df = pd.concat(
        [
            pd.DataFrame({i: y_pred[:, i] for i in range(y_pred.shape[1])}),
            pd.DataFrame({"y_test": y_test.squeeze()}),
        ],
        axis=1,
    )
    test_prediction_df.to_csv(
        op.join(
            config["system_config"]["results_path"],
            config["outfile_name_root"]
            + f"_{time_string}_{fraction_of_data_to_use}_test_predictions_rep{rep}.csv",
        ),
        index=False,
    )

    if use_wandb:
        run_name = f"besthp_test_loss_fraction{fraction_of_data_to_use:.2f}_rep{rep}"
        run = wandb.init(
            project=config["system_config"]["wandb_project_name"],
            config=config | best_hps | {"rep_num": rep},
            name=run_name,
        )
        wandb.log({"test_ce": ce, "rep": rep})
        run.finish()

    if transformer_dir is not None:
        results.extend(
            [
                {
                    "rep": rep,
                    "fraction": fraction_of_data_to_use,
                    "method": transformer_dir,
                    "metric": "cross_entropy",
                    "value": ce,
                }
            ]
        )
    else:
        results.extend(
            [
                {
                    "rep": rep,
                    "fraction": fraction_of_data_to_use,
                    "method": "bert_ablation",
                    "metric": "cross_entropy",
                    "value": ce,
                }
            ]
        )

# Save results
df = pd.DataFrame(results)

# Create results directory if it doesn't exist
if not op.exists(config["system_config"]["results_path"]):
    os.mkdir(config["system_config"]["results_path"])

df.to_csv(
    op.join(
        config["system_config"]["results_path"],
        config["outfile_name_root"] + f"_{time_string}_{fraction_of_data_to_use}",
    )
    + ".csv",
    index=False,
)
