import sys
import os
script_path = os.path.dirname(os.path.abspath(__file__))
crime_dir = os.path.abspath(os.path.join(script_path, ".."))
sys.path.insert(0, crime_dir)
import time

import subprocess

processes = []
max_concurrent = 4
num_required_runs = 6

for i in range(0, num_required_runs):
    # Wait until less than max_concurrent processes are going
    while len(processes) >= max_concurrent:
        # Check if any processes have completed
        for p in processes:
            if p.poll() is not None:
                processes.remove(p)
        time.sleep(1) 

    process = subprocess.Popen(["python", f"{crime_dir}/Run/optimiser_run.py", "--process_id", str(i)])
    processes.append(process)
    print(f"Launched process {i}")
    time.sleep(15)

for p in processes:
    p.wait()

print("All scripts run successfully")