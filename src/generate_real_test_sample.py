# src/generate_real_test_sample.py
# Pulls a REAL 30-cycle window from an actual test engine (with a known true
# RUL) and formats it as JSON, ready to paste into the FastAPI /docs UI.
# This gives a meaningful sanity check -- unlike synthetic data, we can
# compare the API's prediction against the actual known answer.

import pandas as pd
import json
import sys

ENGINE_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 1  # change engine via CLI arg
WINDOW_SIZE = 30

column_names = ['engine_id', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3']
column_names += [f'sensor_{i}' for i in range(1, 22)]

test_df = pd.read_csv('data/test_FD001.txt', sep='\s+', header=None)
test_df = test_df.iloc[:, :26]
test_df.columns = column_names

true_rul = pd.read_csv('data/RUL_FD001.txt', header=None)
true_rul.columns = ['RUL']
# RUL_FD001.txt is ordered by engine_id (1st row = engine 1, 2nd row = engine 2, ...)
true_rul_for_engine = true_rul.iloc[ENGINE_ID - 1]['RUL']
true_rul_capped = min(true_rul_for_engine, 125)

FEATURE_COLUMNS = [
    'op_setting_1', 'op_setting_2', 'op_setting_3',
    'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 'sensor_9',
    'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
    'sensor_17', 'sensor_20', 'sensor_21'
]

engine_data = test_df[test_df['engine_id'] == ENGINE_ID].reset_index(drop=True)
num_cycles = len(engine_data)

if num_cycles < WINDOW_SIZE:
    # pad with first row repeated, same logic as phase7_evaluate.py
    padding_needed = WINDOW_SIZE - num_cycles
    first_row = engine_data.iloc[0:1]
    padding = pd.concat([first_row] * padding_needed, ignore_index=True)
    engine_data = pd.concat([padding, engine_data], ignore_index=True)

last_window = engine_data.tail(WINDOW_SIZE)[FEATURE_COLUMNS]

cycles = last_window.to_dict(orient='records')
payload = {'cycles': cycles}

with open('real_test_sample.json', 'w') as f:
    json.dump(payload, f, indent=2)

print(f"Engine {ENGINE_ID} — actual total cycles in test set: {num_cycles}")
print(f"TRUE RUL (capped at 125): {true_rul_capped}")
print(f"Saved 30-cycle input to real_test_sample.json")
print(f"\nAfter calling /predict with this file, compare the response's")
print(f"'predicted_rul' against the TRUE RUL above ({true_rul_capped}).")