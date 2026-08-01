# src/phase11_evaluate_transformer.py
# Evaluates the trained Transformer on the test set, mirroring
# phase7_evaluate.py's structure exactly so results are directly comparable.

import pandas as pd
import numpy as np
import torch
import joblib
import json
from phase8_transformer_model import TransformerRULModel

WINDOW_SIZE = 30
MAX_RUL = 125

column_names = ['engine_id', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3']
column_names += [f'sensor_{i}' for i in range(1, 22)]

test_df = pd.read_csv('data/test_FD001.txt', sep='\s+', header=None)
test_df = test_df.iloc[:, :26]
test_df.columns = column_names

true_rul = pd.read_csv('data/RUL_FD001.txt', header=None)
true_rul.columns = ['RUL']

constant_sensors = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10',
                     'sensor_16', 'sensor_18', 'sensor_19']
test_df.drop(columns=constant_sensors, inplace=True)

sensor_cols = [c for c in test_df.columns if 'sensor' in c]
op_setting_cols = ['op_setting_1', 'op_setting_2', 'op_setting_3']
feature_cols = op_setting_cols + sensor_cols

scaler = joblib.load('models/scaler.pkl')
test_df[feature_cols] = scaler.transform(test_df[feature_cols])

# Reuse the exact same X_test.npy / y_test.npy saved by phase7_evaluate.py
# (same windows, same true RUL) -- guarantees a fair, matched comparison.
X_test = np.load('data/X_test.npy')
true_rul_capped = np.load('data/y_test.npy')

num_features = X_test.shape[2]
model = TransformerRULModel(num_features, WINDOW_SIZE)
model.load_state_dict(torch.load('models/transformer_model.pth'))
model.eval()

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
with torch.no_grad():
    predictions = model(X_test_tensor).numpy().flatten()

rmse = np.sqrt(np.mean((predictions - true_rul_capped) ** 2))
mae = np.mean(np.abs(predictions - true_rul_capped))

print(f"\nTransformer Test RMSE: {rmse:.2f} cycles")
print(f"Transformer Test MAE: {mae:.2f} cycles")

comparison = pd.DataFrame({
    'engine_id': test_df['engine_id'].unique(),
    'predicted_RUL': predictions.round(1),
    'actual_RUL': true_rul_capped
})
print("\nSample predictions vs actual (first 10 engines):")
print(comparison.head(10))

metrics = {
    "model": "Transformer",
    "rmse": round(float(rmse), 2),
    "mae": round(float(mae), 2)
}
with open('models/transformer_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print("\nSaved metrics to models/transformer_metrics.json")