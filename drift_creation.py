import pandas as pd
import numpy as np
import random

# Load the dataset
df = pd.read_csv("Dataset.csv", low_memory=False)

# Split into reference (70%) and current (30%)
split_index = int(len(df) * 0.7)
reference_data = df.iloc[:split_index].copy()
current_data = df.iloc[split_index:].copy()

# Select numeric columns only
numeric_cols = reference_data.select_dtypes(include=[np.number]).columns.tolist()

# Randomly select half of them to apply drift
random.seed(42)
random.shuffle(numeric_cols)
drift_cols = numeric_cols[:len(numeric_cols) // 2]

# Simulate drift: add noise to current_data for those columns
for col in drift_cols:
    if col in current_data.columns:
        # Shift mean and increase std
        original_mean = current_data[col].mean()
        original_std = current_data[col].std()
        noise = np.random.normal(loc=original_mean + original_std, scale=original_std * 1.5, size=current_data.shape[0])
        current_data[col] = noise

# Combine the original reference and drifted current data
drifted_df = pd.concat([reference_data, current_data], axis=0).reset_index(drop=True)

# Save the modified dataset
drifted_df.to_csv("Dataset_with_drift.csv", index=False)
print("✅ Drifted dataset saved as Dataset_with_drift.csv")