# src/phase1_load_data.py

import pandas as pd

# Define column names since the raw file has no header row
column_names = ['engine_id', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3']
sensor_names = [f'sensor_{i}' for i in range(1, 22)]  # sensor_1 to sensor_21
column_names += sensor_names

# Read the raw text file
# sep='\s+' means columns are separated by one or more spaces
# header=None means the file has no header row, we assign column names manually
train_df = pd.read_csv('data/train_FD001.txt', sep='\s+', header=None)

# Drop any extra trailing columns, keep only the first 26
train_df = train_df.iloc[:, :26]

# Assign proper column names
train_df.columns = column_names

# Sanity checks
print("Data shape:", train_df.shape)
print("\nFirst 5 rows:")
print(train_df.head())
print("\nTotal unique engines:", train_df['engine_id'].nunique())
print("\nCycle range per engine (first 5 engines):")
print(train_df.groupby('engine_id')['cycle'].max().head())

# Save as clean CSV for use in later phases
train_df.to_csv('data/train_cleaned.csv', index=False)
print("\nSaved cleaned data to data/train_cleaned.csv")