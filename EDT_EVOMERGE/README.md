# EDT - EvoMerge

## How to Use

This project is a research prototype for distributed post-training and merging of LLMs using **EDT** strategy. The code is designed for modularity and flexibility, setting it up requires a few manual steps to configure your environment. Follow these instructions carefully to get started.

---

### 1. Configure Machine IPs

Edit the `machines.json` file to include the IP addresses of all the machines in your cluster:

```json
[
  "192.168.0.1",
  "192.168.0.2",
  "192.168.0.3"
]
```

> ⚠️ **Important:** All machines **must share a common disk** (e.g., NFS mount, virtual shared disk). We use shared disk to simulate communication between nodes as a research prototype.

---

### 2. Configure Training Hyperparameters

Open `evolution.json` and configure your training settings:

* `BASE_DIR`: **Must be on the shared disk** accessible by all machines.
* Set other hyperparameters as needed (e.g., population size, mutation rate, etc.).

Example:

```json
{
  "BASE_DIR": "/mnt/shared/edt_run",
  "population_size": 8,
  ...
}
```

---

### 3. Launch Distributed Training

Use the script `launch_to_machines.py` to distribute the launch command across all specified machines.

```bash
python launch_to_machines.py
```

> **Note:** This script sends launch commands to each machine via SSH and assumes you have SSH access configured (e.g., using keys) and Python/venv environments prepared.

---

### 4. Understand the Launch Process

To customize or debug the worker launch process, inspect the following files:

* `edt.py`: Defines the core EDT implementation logic.
* `launch_to_machines.py`: Contains the commands dispatched to remote machines.

  * **Check here** how the worker servers are launched.
  * The launch command may need to activate a virtual environment (`venv`) or set environment variables depending on your machine setup.

> This project does not enforce a specific environment setup (e.g., Docker, conda, venv), so adjust commands to fit your configuration.