import os
import os.path as op
import sys
import time

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold
from tensorflow.keras.losses import BinaryCrossentropy, Reduction
from tqdm import tqdm
from xgboost import XGBClassifier

from lab_transformers.utils import gen_combinations

time_string = time.strftime("%Y%m%d-%H%M%S")
outfile_name = f"sepsis_mortality_xgboost_baseline_{time_string}"
dataset_name = "sepsis_mortality_baselines_data.csv"
label_column = "mortality_indicator"

# Argument Parsing
results_dir = sys.argv[1]
results_path = op.join(results_dir, "sepsis_mortality/")
dataset_path = op.join(sys.argv[2], dataset_name)
k_inner = int(sys.argv[3])
k_outer = int(sys.argv[4])
random_seed = int(sys.argv[5])
num_subsamples = int(sys.argv[6])
rng = np.random.default_rng(random_seed)

xgboost_config = {
    "n_estimators": [30, 100, 200],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "max_depth": [3, 5],
}

# Load dataset and separate feature/label columns
df = pd.read_csv(dataset_path)
feature_cols = list(df.columns)
feature_cols.remove(label_column)
X, y = df[feature_cols].to_numpy(), df[label_column].to_numpy()
num_classes = len(np.unique(y))

bce = BinaryCrossentropy(from_logits=False, reduction=Reduction.SUM_OVER_BATCH_SIZE)
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

        # XGBoost baseline (with hyperparameter tuning / model selection)
        best_ce = None
        best_xgb_hps = None
        for xgb_config in gen_combinations(xgboost_config):
            xgb_crossentropies = []
            kf = KFold(n_splits=k_inner, shuffle=True, random_state=random_seed)
            for i, (ktrain_index, kval_index) in enumerate(kf.split(X_train)):
                X_ktrain, X_kval = X_train[ktrain_index], X_train[kval_index]
                y_ktrain, y_kval = y_train[ktrain_index], y_train[kval_index]

                # Standardize continuous features in X
                # Get column indices of continuous features
                scaler = preprocessing.StandardScaler().fit(X_ktrain)
                X_ktrain = scaler.transform(X_ktrain)

                xgb = XGBClassifier(**xgb_config, random_state=random_seed)
                xgb.fit(X_ktrain, y_ktrain)

                # Calculate binary cross-entropy  (compare to y_kval)
                X_kval = scaler.transform(X_kval)
                y_kpred = xgb.predict_proba(X_kval)
                binary_ce = bce(y_kval, y_kpred[:, -1]).numpy()
                xgb_crossentropies.append(binary_ce)

            if best_ce is None or np.mean(xgb_crossentropies) < best_ce:
                best_ce = np.mean(xgb_crossentropies)
                best_xgb_hps = xgb_config

        xgb = XGBClassifier(**best_xgb_hps, random_state=random_seed)
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_scaled = X_train.copy()
        X_train_scaled = scaler.transform(X_train)
        xgb.fit(X_train_scaled, y_train)
        X_test_scaled = X_test.copy()
        X_test_scaled = scaler.transform(X_test)
        y_pred = xgb.predict_proba(X_test_scaled)
        xgb_ce = bce(y_test, y_pred[:, -1]).numpy()

        results.append(
            {
                "rep": rep,
                "fraction": fraction,
                "method": "xgboost",
                "metric": "cross_entropy",
                "value": xgb_ce,
            }
        )

# Save results
df = pd.DataFrame(results)

# Create results directory if it doesn't exist
if not op.exists(results_path):
    os.mkdir(results_path)

df.to_csv(op.join(results_path, outfile_name) + ".csv", index=False)
