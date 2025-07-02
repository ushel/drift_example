# drift_detection_evidently.py
# evidently uses Kolmogorov–Smirnov (KS) Test to detect drift but only for numerical features and CHi-squared for categorical features
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Load reference and drifted datasets
reference_data = pd.read_csv("Dataset.csv")
current_data = pd.read_csv("Dataset_with_drift.csv")

# Optional: align columns if needed
common_columns = reference_data.columns.intersection(current_data.columns)
reference_data = reference_data[common_columns]
current_data = current_data[common_columns]

# Create Evidently Report
report = Report(metrics=[DataDriftPreset()])

# Run drift detection
report.run(reference_data=reference_data, current_data=current_data)

# Save the report as HTML
report.save_html("evidently_drift_report.html")
print("Drift report saved as evidently_drift_report.html")
