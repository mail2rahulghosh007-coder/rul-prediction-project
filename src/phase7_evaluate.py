# src/phase7_evaluate.py

import pandas as pd
import numpy as np
import torch
import joblib
from phase5_model import CNN_LSTM_Model

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

X_test = []
for engine_id in test_df['engine_id'].unique():
    engine_data = test_df[test_df['engine_id'] == engine_id].reset_index(drop=True)

    if len(engine_data) < WINDOW_SIZE:
        padding_needed = WINDOW_SIZE - len(engine_data)
        first_row = engine_data.iloc[0:1]
        padding = pd.concat([first_row] * padding_needed, ignore_index=True)
        engine_data = pd.concat([padding, engine_data], ignore_index=True)

    last_window = engine_data.loc[len(engine_data)-WINDOW_SIZE:, feature_cols].values
    X_test.append(last_window)

X_test = np.array(X_test)
print("X_test shape:", X_test.shape)

np.save('data/X_test.npy', X_test)

num_features = X_test.shape[2]
model = CNN_LSTM_Model(num_features, WINDOW_SIZE)
model.load_state_dict(torch.load('models/cnn_lstm_model.pth'))
model.eval()

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
with torch.no_grad():
    predictions = model(X_test_tensor).numpy().flatten()

true_rul_capped = true_rul['RUL'].clip(upper=MAX_RUL).values

np.save('data/y_test.npy', true_rul_capped)

rmse = np.sqrt(np.mean((predictions - true_rul_capped) ** 2))
mae = np.mean(np.abs(predictions - true_rul_capped))

print(f"\nTest RMSE: {rmse:.2f} cycles")
print(f"Test MAE: {mae:.2f} cycles")

comparison = pd.DataFrame({
    'engine_id': test_df['engine_id'].unique(),
    'predicted_RUL': predictions.round(1),
    'actual_RUL': true_rul_capped
})
print("\nSample predictions vs actual (first 10 engines):")
print(comparison.head(10))

# ---- Save metrics so other scripts (e.g. phase8_baseline.py) can read the
# real, current numbers instead of anyone hardcoding a stale value ----
import json
metrics = {
    "model": "CNN_LSTM",
    "rmse": round(float(rmse), 2),
    "mae": round(float(mae), 2)
}
with open('models/cnn_lstm_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print("\nSaved metrics to models/cnn_lstm_metrics.json")