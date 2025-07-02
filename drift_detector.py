import pandas as pd
import json
import numpy as np
import os
from drift_detect import detect_drift  # Make sure this is in your working directory

# Load data
df = pd.read_csv("Data/Dataset_with_drift.csv", low_memory=False)
df = df[df["Default"].notnull()]

# Split
ref, cur = np.split(df, [int(0.7 * len(df))])

# Detect drift
drift_report = detect_drift(ref, cur)

# Save report
with open("drift_report.json", "w") as f:
    json.dump(drift_report, f, indent=4)

# Determine drift
drift_detected = any(v.get("drift_detected", False) for v in drift_report.values())

# Always write flag
with open("drift_detected.txt", "w") as f:
    if drift_detected:
        f.write("drift detected\n")
    else:
        f.write("no drift\n")


def population_stability_index(expected, actual, buckets=10):
    def scale(x): return (x - x.min()) / (x.max() - x.min())
    expected_pct = np.histogram(scale(expected), bins=buckets)[0] / len(expected)
    actual_pct = np.histogram(scale(actual), bins=buckets)[0] / len(actual)
    return np.sum((expected_pct - actual_pct) * np.log((expected_pct + 1e-8) / (actual_pct + 1e-8)))