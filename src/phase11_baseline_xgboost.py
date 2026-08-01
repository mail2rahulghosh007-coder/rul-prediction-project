# src/phase8_baseline.py

import numpy as np
import pandas as pd
import json
import mlflow
import dagshub
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

dagshub.init(repo_owner='mail2rahulghosh007-coder', repo_name='rul_prediction_project', mlflow=True)
mlflow.set_experiment("RUL_Prediction_CMAPSS_FD001")

N_ESTIMATORS = 200
MAX_DEPTH = 6
LEARNING_RATE = 0.05

mlflow.start_run(run_name="XGBoost_baseline")
mlflow.set_tag("model_type", "XGBoost")
mlflow.log_param("n_estimators", N_ESTIMATORS)
mlflow.log_param("max_depth", MAX_DEPTH)
mlflow.log_param("learning_rate", LEARNING_RATE)

# Load the same sliding window training data used for the deep learning model
X_train = np.load('data/X_train.npy')
y_train = np.load('data/y_train.npy')

# XGBoost needs 2D input, so flatten each 30x17 window into a single row of 510 values
X_train_flat = X_train.reshape(X_train.shape[0], -1)
print("Flattened X_train shape:", X_train_flat.shape)

# Train a simple XGBoost regressor
baseline_model = XGBRegressor(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    learning_rate=LEARNING_RATE,
    random_state=42
)
baseline_model.fit(X_train_flat, y_train)
print("XGBoost training complete")

# Load the test set (saved by phase7_evaluate.py)
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')

X_test_flat = X_test.reshape(X_test.shape[0], -1)
print("Flattened X_test shape:", X_test_flat.shape)

# Predict using the baseline model
baseline_predictions = baseline_model.predict(X_test_flat)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, baseline_predictions))
mae = mean_absolute_error(y_test, baseline_predictions)

print(f"\nXGBoost Baseline — Test RMSE: {rmse:.2f} cycles")
print(f"XGBoost Baseline — Test MAE: {mae:.2f} cycles")

mlflow.log_metric("test_rmse", rmse)
mlflow.log_metric("test_mae", mae)
baseline_model.save_model('models/xgboost_model.json')
mlflow.log_artifact('models/xgboost_model.json')
mlflow.end_run()
print("Logged run to MLflow")

# ---- Load the other models' real, current metrics for a full comparison ----
def load_metrics(path, label):
    try:
        with open(path, 'r') as f:
            m = json.load(f)
        return m['rmse'], m['mae']
    except FileNotFoundError:
        print(f"\nWarning: {path} not found. Run the corresponding evaluate script first.")
        return None, None

cnn_rmse, cnn_mae = load_metrics('models/cnn_lstm_metrics.json', 'CNN-LSTM')
transformer_rmse, transformer_mae = load_metrics('models/transformer_metrics.json', 'Transformer')

print("\n--- Final Comparison ---")
if cnn_rmse is not None:
    print(f"CNN-LSTM     -> RMSE: {cnn_rmse}, MAE: {cnn_mae}")
else:
    print("CNN-LSTM     -> (not available, run phase7_evaluate.py first)")
if transformer_rmse is not None:
    print(f"Transformer  -> RMSE: {transformer_rmse}, MAE: {transformer_mae}")
else:
    print("Transformer  -> (not available, run phase11_evaluate_transformer.py first)")
print(f"XGBoost      -> RMSE: {rmse:.2f}, MAE: {mae:.2f}")