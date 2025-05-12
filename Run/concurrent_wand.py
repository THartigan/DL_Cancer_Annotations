
import sys
import os
script_path = os.path.dirname(os.path.abspath(__file__))
crime_dir = os.path.abspath(os.path.join(script_path, ".."))
sys.path.insert(0, crime_dir)
import time

import subprocess
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--sweep_key", type=str, required=True, help="W&B Sweep Key")

run_args = parser.parse_args()
sweep_key = run_args.sweep_key
processes = []
max_concurrent = 4

for i in range(0, max_concurrent):
    # Wait until less than max_concurrent processes are going
    process = subprocess.Popen([
        "wandb", "agent", f"tjh200-university-of-cambridge/TROPHY/{sweep_key}"
    ])
    processes.append(process)
    print(f"Launched process {i}")
    time.sleep(15)

print("All scripts run successfully")