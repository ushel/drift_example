import pandas as pd
import numpy as np

def calculate_psi(expected, actual, buckets=10):
    def scale(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))

    expected = scale(expected)
    actual = scale(actual)

    breakpoints = np.linspace(0, 1, buckets + 1)
    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    psi = np.sum((expected_pct - actual_pct) * np.log((expected_pct + 1e-8) / (actual_pct + 1e-8)))
    return psi

# Load predicted and actual values
pred = pd.read_csv("Data/outputs/predicted.csv")["Default_Probability"]
actual = pd.read_csv("Data/outputs/actual.csv")["Default"]

# Calculate PSI
psi_score = calculate_psi(expected=actual, actual=pred)
print(f"PSI Score: {psi_score:.4f}")

# Save drift flag
with open("psi_drift_detected.txt", "w") as f:
    if psi_score > 0.2:
        f.write("PSI drift detected\n")
    else:
        f.write("No PSI drift\n")
