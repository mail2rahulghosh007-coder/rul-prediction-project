# src/phase2_create_rul.py

import pandas as pd

# Load the cleaned data from Phase 1
train_df = pd.read_csv('data/train_cleaned.csv')

def add_rul(df, max_rul=125):
    # Find the max cycle (i.e. failure point) for each engine
    max_cycle = df.groupby('engine_id')['cycle'].max().reset_index()
    max_cycle.columns = ['engine_id', 'max_cycle']

    # Merge the max_cycle info back into the main dataframe
    df = df.merge(max_cycle, on='engine_id', how='left')

    # RUL = how many cycles remain before failure
    df['RUL'] = df['max_cycle'] - df['cycle']

    # Cap RUL at max_rul, since early-life sensor data carries no degradation signal
    df['RUL'] = df['RUL'].clip(upper=max_rul)

    # Drop the helper column, we don't need it anymore
    df.drop(columns=['max_cycle'], inplace=True)
    return df

train_df = add_rul(train_df)

# Sanity checks
print("Shape after adding RUL:", train_df.shape)
print("\nEngine 1 — first 5 rows (RUL should be capped at 125):")
print(train_df[train_df['engine_id'] == 1][['engine_id', 'cycle', 'RUL']].head())

print("\nEngine 1 — last 5 rows (RUL should approach 0):")
print(train_df[train_df['engine_id'] == 1][['engine_id', 'cycle', 'RUL']].tail())

# Save this version with RUL included
train_df.to_csv('data/train_with_rul.csv', index=False)
print("\nSaved to data/train_with_rul.csv")