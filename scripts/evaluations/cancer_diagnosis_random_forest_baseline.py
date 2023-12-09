import os
import os.path as op
import sys
import time

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from tensorflow import one_hot
from tensorflow.keras.losses import SparseCategoricalCrossentropy, Reduction
from tensorflow_addons.metrics import F1Score
from tqdm import tqdm

from lab_transformers.utils import gen_combinations

time_string = time.strftime("%Y%m%d-%H%M%S")
outfile_name = f"cancer_diagnosis_random_forest_baseline_{time_string}"
dataset_name = "cancer_diagnosis_baselines_data.csv"
label_column = "Disease"

# Argument Parsing
results_dir = sys.argv[1]
results_path = op.join(results_dir, "cancer_diagnosis/")
dataset_path = op.join(sys.argv[2], dataset_name)
k_inner = int(sys.argv[3])
k_outer = int(sys.argv[4])
random_seed = int(sys.argv[5])
num_subsamples = int(sys.argv[6])
rng = np.random.default_rng(random_seed)

randomforest_config = {
    "n_estimators": [30, 100, 200],
    "max_depth": [3, 5, 10],
    "max_features": [None, "sqrt"],
}

# Load dataset and separate feature/label columns
df = pd.read_csv(dataset_path)
feature_cols = list(df.columns)
feature_cols.remove(label_column)
X, y = df[feature_cols].to_numpy(), df[label_column].to_numpy()

# Gather quantities for data standardization/transformation in-the-loop
num_classes = len(np.unique(y))
categorical_variables = [
    "FamilyHistory",
    "BetelNut",
    "Drinking",
    "Smoking",
    "Diabetes",
    "Hypertension",
    "gender",
    "51487",
    "Disease",
]
continuous_col_ixs = [
    df.columns.get_loc(col) for col in df.columns if col not in categorical_variables
]

scce = SparseCategoricalCrossentropy(
    from_logits=False, reduction=Reduction.SUM_OVER_BATCH_SIZE
)
f1_score = F1Score(num_classes=num_classes, average="micro", threshold=None)
results = []
for rep in range(k_outer):
    # Separate a random 10% of the data for testing
    test_ix = rng.choice(X.shape[0], size=int(0.1 * X.shape[0]), replace=False)
    X_test, y_test = X[test_ix], y[test_ix]

    # Use the remaining 90% of the data for the evaluation
    mask = np.ones(X.shape[0], dtype=bool)
    mask[test_ix] = False
    X_dev, y_dev = X[mask], y[mask]

    for fraction in tqdm(np.arange(1, num_subsamples + 1) / num_subsamples):
        train_ix = rng.choice(
            X_dev.shape[0], size=int(fraction * X_dev.shape[0]), replace=False
        )
        X_train, y_train = X_dev[train_ix], y_dev[train_ix]

        # Random Forest baseline (with hyperparameter tuning / model selection)
        best_ce = None
        best_rf_hps = None
        for rf_config in gen_combinations(randomforest_config):
            rf_crossentropies = []
            kf = KFold(n_splits=k_inner, shuffle=True, random_state=random_seed)
            for i, (ktrain_index, kval_index) in enumerate(kf.split(X_train)):
                X_ktrain, X_kval = X_train[ktrain_index], X_train[kval_index]
                y_ktrain, y_kval = y_train[ktrain_index], y_train[kval_index]

                # Standardize continuous features in X
                # Get column indices of continuous features
                scaler = preprocessing.StandardScaler().fit(
                    X_ktrain[:, continuous_col_ixs]
                )
                X_ktrain[:, continuous_col_ixs] = scaler.transform(
                    X_ktrain[:, continuous_col_ixs]
                )

                rf = RandomForestClassifier(**rf_config, random_state=random_seed)
                rf.fit(X_ktrain, y_ktrain)

                # Calculate categorical cross-entropy  (compare to y_kval)
                X_kval[:, continuous_col_ixs] = scaler.transform(
                    X_kval[:, continuous_col_ixs]
                )
                y_kpred = rf.predict_proba(X_kval)
                categorical_ce = scce(y_kval, y_kpred).numpy()
                rf_crossentropies.append(categorical_ce)

            if best_ce is None or np.mean(rf_crossentropies) < best_ce:
                best_ce = np.mean(rf_crossentropies)
                best_rf_hps = rf_config

        rf = RandomForestClassifier(**best_rf_hps, random_state=random_seed)
        scaler = preprocessing.StandardScaler().fit(X_train[:, continuous_col_ixs])
        X_train_scaled = X_train.copy()
        X_train_scaled[:, continuous_col_ixs] = scaler.transform(
            X_train[:, continuous_col_ixs]
        )
        rf.fit(X_train_scaled, y_train)
        X_test_scaled = X_test.copy()
        X_test_scaled[:, continuous_col_ixs] = scaler.transform(
            X_test[:, continuous_col_ixs]
        )
        y_pred = rf.predict_proba(X_test_scaled)
        rf_ce = scce(y_test, y_pred).numpy()
        rf_f1 = f1_score(one_hot(y_test, num_classes), y_pred).numpy()

        results.extend(
            [
                {
                    "rep": rep,
                    "fraction": fraction,
                    "method": "random_forest",
                    "metric": "cross_entropy",
                    "value": rf_ce,
                },
                {
                    "rep": rep,
                    "fraction": fraction,
                    "method": "random_forest",
                    "metric": "f1",
                    "value": rf_f1,
                },
            ]
        )

# Save results
df = pd.DataFrame(results)

# Create results directory if it doesn't exist
if not op.exists(results_path):
    os.mkdir(results_path)

df.to_csv(op.join(results_path, outfile_name) + ".csv", index=False)
