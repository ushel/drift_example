from prometheus_client import start_http_server, Gauge
import time
import json

# Prometheus metric
drift_gauge = Gauge("data_drift_detected", "1 if any feature drift detected, else 0")

DRIFT_FILE = "D:\Projects\Drift_detection\drift_example\drift_report.json"

def read_drift_status():
    try:
        with open(DRIFT_FILE, "r") as f:
            data = json.load(f)
            # Check if any feature has drift_detected: true
            for feature, stats in data.items():
                if stats.get("drift_detected", False):
                    return 1
            return 0
    except Exception as e:
        print(f"Error reading drift file: {e}")
        return 0

if __name__ == "__main__":
    print("Starting Prometheus drift exporter at http://localhost:8000/metrics")
    start_http_server(8000)
    while True:
        drift_status = read_drift_status()
        drift_gauge.set(drift_status)
        time.sleep(30)
