import os
import subprocess

MODEL_PATH = "Data/outputs/model.pkl"

def should_train():
    # Case 1: Drift detected
    if os.path.exists("drift_detected.txt"):
        with open("drift_detected.txt", "r") as f:
            if "drift detected" in f.read():
                print("Drift detected. Proceeding to train.")
                return True

    # Case 2: No model exists yet
    if not os.path.exists(MODEL_PATH):
        print("No model found. Proceeding to train.")
        return True

    print("No drift and model already exists. Skipping training.")
    return False

if should_train():
    subprocess.run(["python", "main.py"], check=True)
