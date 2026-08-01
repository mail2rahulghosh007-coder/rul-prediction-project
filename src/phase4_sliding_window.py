# src/phase4_sliding_window.py

import pandas as pd
import numpy as np

# Load preprocessed data from Phase 3
train_df = pd.read_csv('data/train_preprocessed.csv')

# Same feature columns as before
sensor_cols = [c for c in train_df.columns if 'sensor' in c]
op_setting_cols = ['op_setting_1', 'op_setting_2', 'op_setting_3']
feature_cols = op_setting_cols + sensor_cols

WINDOW_SIZE = 30  # how many cycles we look back at, per prediction

def create_sequences(df, feature_cols, window_size):
    X_sequences = []   # will hold all the windows (matrices)
    y_labels = []       # will hold the RUL label for each window

    # Process one engine at a time
    for engine_id in df['engine_id'].unique():
        engine_data = df[df['engine_id'] == engine_id].reset_index(drop=True)
        num_cycles = len(engine_data)

        # Skip engines that have fewer cycles than our window size
        if num_cycles < window_size:
            continue

        # Slide the window across this engine's life, one cycle at a time
        for start in range(num_cycles - window_size + 1):
            end = start + window_size

            # The window itself: window_size rows x len(feature_cols) columns
            window = engine_data.loc[start:end-1, feature_cols].values

            # The label: RUL at the LAST cycle of this window
            label = engine_data.loc[end-1, 'RUL']

            X_sequences.append(window)
            y_labels.append(label)

    return np.array(X_sequences), np.array(y_labels)

X, y = create_sequences(train_df, feature_cols, WINDOW_SIZE)

print("X shape:", X.shape)   # (num_windows, window_size, num_features)
print("y shape:", y.shape)   # (num_windows,)

print("\nExample — first window's shape:", X[0].shape)
print("Example — first window's label (RUL):", y[0])

# Save as numpy arrays for the training phase
np.save('data/X_train.npy', X)
np.save('data/y_train.npy', y)
print("\nSaved X_train.npy and y_train.npy")