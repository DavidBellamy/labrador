import os
import os.path as op
import sys
import time

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from tqdm import tqdm
from xgboost import XGBRegressor

from lab_transformers.utils import gen_combinations

time_string = time.strftime("%Y%m%d-%H%M%S")
outfile_name = f"drinks_per_day_baselines_{time_string}"
dataset_name = "drinks_per_day.csv"
label_column = "drinks"

# Argument Parsing
results_dir = sys.argv[1]
results_path = op.join(results_dir, "drinks_per_day/")
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

# Fill any missing data with column mean
df = df.fillna(df.mean())
X, y = df[feature_cols].to_numpy(), df[label_column].to_numpy()

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

        # Linear Regression baseline
        lr = LinearRegression()
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_scaled = X_train.copy()
        X_train_scaled = scaler.transform(X_train)
        lr.fit(X_train_scaled, y_train)
        X_test_scaled = scaler.transform(X_test)
        lr_mse = np.mean((y_test - lr.predict(X_test_scaled)) ** 2)

        # Random Forest baseline (with hyperparameter tuning)
        best_mse = None
        best_rf_hps = None
        for rf_config in gen_combinations(randomforest_config):
            rf_mses = []
            kf = KFold(n_splits=k_inner, shuffle=True, random_state=random_seed)
            for i, (ktrain_index, kval_index) in enumerate(kf.split(X_train)):
                X_ktrain, X_kval = X_train[ktrain_index], X_train[kval_index]
                y_ktrain, y_kval = y_train[ktrain_index], y_train[kval_index]

                # Standardize continuous features in X
                scaler = preprocessing.StandardScaler().fit(X_ktrain)
                X_ktrain = scaler.transform(X_ktrain)

                rf = RandomForestRegressor(**rf_config, random_state=random_seed)
                rf.fit(X_ktrain, y_ktrain)

                X_kval = scaler.transform(X_kval)
                rf_mses.append(np.mean((y_kval - rf.predict(X_kval)) ** 2))

            if best_mse is None or np.mean(rf_mses) < best_mse:
                best_mse = np.mean(rf_mses)
                best_rf_hps = rf_config

        rf = RandomForestRegressor(**best_rf_hps, random_state=random_seed)
        scaler = preprocessing.StandardScaler().fit(X_train)
        rf.fit(X_train_scaled, y_train)
        rf_mse = np.mean((y_test - rf.predict(X_test_scaled)) ** 2)

        # XGBoost baseline (with hyperparameter tuning)
        best_mse = None
        best_xgb_hps = None
        for xgb_config in gen_combinations(xgboost_config):
            xgb_mses = []
            kf = KFold(n_splits=k_inner, shuffle=True, random_state=random_seed)
            for i, (ktrain_index, kval_index) in enumerate(kf.split(X_train)):
                X_ktrain, X_kval = X_train[ktrain_index], X_train[kval_index]
                y_ktrain, y_kval = y_train[ktrain_index], y_train[kval_index]

                # Standardize continuous features in X
                scaler = preprocessing.StandardScaler().fit(X_ktrain)
                X_ktrain = scaler.transform(X_ktrain)

                xgb = XGBRegressor(**xgb_config, random_state=random_seed)
                xgb.fit(X_ktrain, y_ktrain)

                X_kval = scaler.transform(X_kval)
                xgb_mses.append(np.mean((y_kval - xgb.predict(X_kval)) ** 2))

            if best_mse is None or np.mean(xgb_mses) < best_mse:
                best_mse = np.mean(xgb_mses)
                best_xgb_hps = xgb_config

        xgb = XGBRegressor(**best_xgb_hps, random_state=random_seed)
        xgb.fit(X_train_scaled, y_train)
        xgb_mse = np.mean((y_test - xgb.predict(X_test_scaled)) ** 2)

        results.extend(
            [
                {
                    "rep": rep,
                    "fraction": fraction,
                    "method": "linear_regression",
                    "mse": lr_mse,
                },
                {
                    "rep": rep,
                    "fraction": fraction,
                    "method": "random_forest",
                    "mse": rf_mse,
                },
                {"rep": rep, "fraction": fraction, "method": "xgboost", "mse": xgb_mse},
            ]
        )

# Save results
df = pd.DataFrame(results)

# Create results directory if it doesn't exist
if not op.exists(results_path):
    os.mkdir(results_path)

df.to_csv(op.join(results_path, outfile_name) + ".csv", index=False)
