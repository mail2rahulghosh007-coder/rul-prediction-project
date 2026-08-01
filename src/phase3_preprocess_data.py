# src/phase3_preprocess.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load data with RUL from Phase 2
train_df = pd.read_csv('data/train_with_rul.csv')

# These sensors stay constant throughout the dataset — they carry no useful signal
constant_sensors = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10',
                     'sensor_16', 'sensor_18', 'sensor_19']

train_df.drop(columns=constant_sensors, inplace=True)

# Columns that will actually be fed into the model
sensor_cols = [c for c in train_df.columns if 'sensor' in c]
op_setting_cols = ['op_setting_1', 'op_setting_2', 'op_setting_3']
feature_cols = op_setting_cols + sensor_cols

print("Number of usable sensors:", len(sensor_cols))
print("Total features (op_settings + sensors):", len(feature_cols))
print("Feature columns:", feature_cols)

# Normalize all feature columns to a 0-1 range
scaler = MinMaxScaler()
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])

# Sanity check — values should now be between 0 and 1
print("\nAfter normalization (first 5 rows):")
print(train_df[feature_cols].head())

# Save the scaler itself — we'll need the EXACT same scaling for test data later
joblib.dump(scaler, 'models/scaler.pkl')

# Save the preprocessed data
train_df.to_csv('data/train_preprocessed.csv', index=False)
print("\nSaved to data/train_preprocessed.csv")
print("Saved scaler to models/scaler.pkl")