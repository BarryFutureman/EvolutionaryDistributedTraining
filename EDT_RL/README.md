# EDT - RL

## How to Use

This project is a research prototype for distributed post-training and merging of LLMs using **EDT** strategy. The code is designed for modularity and flexibility, setting it up requires a few manual steps to configure your environment. Follow these instructions carefully to get started.

---
Here’s a polished **How to Use** section for the README, adapted for the **EDT-RL** (Evolutionary Diversity Training for Reinforcement Learning) repo. It mirrors the structure and style of the previous one while reflecting the differences you mentioned:

---

## How to Use

This repo contains an experimental implementation of **EDT-RL**, a distributed reinforcement learning method using evolutionary reward diversity. The setup is similar to the EDT language model pretraining project, but tailored for RL in a platform-fighting game environment.

Follow these steps to get started:

---

### 1. Configure Your Machines

Edit the `machines.json` file with the IP addresses of the machines you want to use:

```json
[
  "192.168.1.10",
  "192.168.1.11"
]
```

> ⚠️ **Note:** All machines **must share a common disk** (e.g., via NFS or a virtual shared drive). This is required because the agents share checkpoints and logs via common file paths.

---

### 2. Set Hyperparameters

* Open `edt.py` or `pdt.py` and set:

  * Reward functions and DNA initial configuration.
  * PPO hyperparameters (e.g., learning rate, number of steps, entropy bonus).
  * Training loop parameters (e.g., number of generations, population size, mutation rate).

This file defines the core evolutionary loop and logic for training, evaluation, selection, crossover, and mutation.

---

### 3. Launch Training on Distributed Machines

To distribute the training job across machines, run:

```bash
python launch_to_machines.py
```

> This script dispatches commands to remote machines over SSH. You’ll need to make sure:
>
> * A Python environment is prepared on each machine.
> * You review `launch_to_machines.py` to modify the worker launch commands as needed (especially for activating your virtual environment).

---

### 4. Run the scripts

Run `edt.py` or `pbt.py` on the main machine.