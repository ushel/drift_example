import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Load the dataset
df = pd.read_csv("Dataset.csv")

# Split into reference (first 70%) and current (last 30%)
split_index = int(len(df) * 0.7)
reference_data = df.iloc[:split_index]
current_data = df.iloc[split_index:]

# Create and run Evidently drift report
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference_data, current_data=current_data)

# Save as HTML
report.save_html("drift_report.html")
