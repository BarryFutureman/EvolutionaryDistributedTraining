import json
import requests
import time
import os
import random
from transformers import AutoModel, AutoConfig, AutoTokenizer
import shutil
from tqdm import tqdm
import sys
import gc
import wandb

# ANSI escape codes for colors
PURPLE = "\033[95m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

with open("evolution.json", "r") as config_file:
    EVOLUTION = json.load(config_file)
with open("machines.json", "r") as machines_file:
    MACHINES = json.load(machines_file)

PROJECT_NAME = EVOLUTION["PROJECT_NAME"]
NUM_WORKERS = len(MACHINES)
POPULATION_SIZE = NUM_WORKERS
MUTATION_PROBABILITY = EVOLUTION.get("MUTATION_PROBABILITY", 0.5)
MUTATION_STRENGTH = EVOLUTION.get("MUTATION_STRENGTH", 1)
ELITISM = EVOLUTION.get("ELITISM", 0)

INITIAL_POPULATION = EVOLUTION["INITIAL_POPULATION"]
if len(INITIAL_POPULATION) < POPULATION_SIZE:
    filled_population = INITIAL_POPULATION + random.choices(INITIAL_POPULATION,
                                                            k=POPULATION_SIZE - len(INITIAL_POPULATION))
else:
    filled_population = INITIAL_POPULATION[:POPULATION_SIZE]

MAX_GENERATIONS = EVOLUTION["MAX_GENERATIONS"]
START_GENERATION = 0


def wait_for_workers():
    statuses = {}
    while True:
        all_finished = True
        for machine in MACHINES:
            url = f"http://{machine}:7899/"
            try:
                response = requests.get(url)
                response.raise_for_status()
                data = response.json()
                status = data["status"]
                statuses[machine] = status
                if status != "SUCCESS" and status != "FAILED":
                    all_finished = False
                elif status == "FAILED":
                    print(f"One of the machines failed: {statuses}")
                    quit()
            except requests.exceptions.RequestException as e:
                print(f"Error GETTING status from {machine}: {e}")
                all_finished = False

        print("Current statuses:", statuses)
        if all_finished:
            print("All workers done.")
            break
        time.sleep(5)


# ============ INITIALIZATION =============
generation_index = START_GENERATION

for machine, model_path in zip(MACHINES, filled_population):
    full_command = (
        f". ../myenv/bin/activate && "
        f"pip uninstall -y tensorflow"
    )
    payload = {"command": full_command}
    try:
        url = f"http://{machine}:7899/"
        response = requests.post(url, json=payload)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error sending command '{full_command}' to {machine}: {e}")
wait_for_workers()

generation_index = START_GENERATION
if generation_index == 0:
    for machine, model_path in zip(MACHINES, filled_population):
        full_command = (
            f". ../myenv/bin/activate && "
            f"python3.11 init_weights.py "
            f"--model_path {model_path} "
            f"--output_path Gen{generation_index:04d} "
            f"--cache_dir {os.getcwd()}/cache "
        )
        payload = {"command": full_command}
        try:
            url = f"http://{machine}:7899/"
            response = requests.post(url, json=payload)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error sending command '{full_command}' to {machine}: {e}")
    wait_for_workers()

# ====== WANDB SETUP ======
wandb.login(key="a2eae9bb4b212c39378ce78b9850053f54b9eb3b")
wandb.init(project=PROJECT_NAME)

# ====================== EVOLUTION BEGINS ======================
pbar = tqdm(total=MAX_GENERATIONS, desc="Evolution Progress", initial=generation_index)
while True:
    current_directory = os.getcwd()
    for machine in MACHINES:
        path = f"{current_directory}/{machine}/Gen{generation_index - 1:04d}"
        if not os.path.exists(path) and generation_index - 1 >= 0:
            raise NotImplementedError(f"Wrong generation index, {path} does not exist")

    # ============ EVAL =============
    fixed_seed = random.randint(0, 1000000)
    for machine in MACHINES:
        model_dir = f"{current_directory}/{machine}/Gen{generation_index:04d}"
        with open(f"{model_dir}/genome.json", 'r') as f:
            genome_data = json.load(f)

        full_command = (
            f". ../myenv/bin/activate && "
            f"python3.11 eval.py "
            f"--model_path {model_dir} "
            f"--genome_path {model_dir}/genome.json "
            f"--seed {fixed_seed} "
            f"--batch_size {EVOLUTION['EVAL_BATCH_SIZE']} "
            f"--cache_dir {current_directory}/cache "
        )
        payload = {"command": full_command}
        try:
            url = f"http://{machine}:7899/"
            response = requests.post(url, json=payload)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error sending command '{full_command}' to {machine}: {e}")

    # Wait for eval scripts to finish
    wait_for_workers()

    model_fitness = {}
    for machine in MACHINES:
        model_dir = f"{machine}/Gen{generation_index:04d}"
        with open(f"{model_dir}/genome.json", 'r') as f:
            genome_data = json.load(f)
        model_fitness[model_dir] = genome_data["fitness"]
    all_genomes = []
    for path_key, fitness_score in model_fitness.items():
        genome_file = os.path.join(path_key, "genome.json")
        with open(genome_file, "r") as gf:
            genome_data = json.load(gf)
        genome_data["fitness"] = fitness_score
        genome_data["model_path"] = path_key
        all_genomes.append(genome_data)

    # --- WANDB LOGGING ---
    fitness_values = [g["fitness"] for g in all_genomes]
    avg_fitness = sum(fitness_values) / len(fitness_values) if fitness_values else 0
    top_fitness = max(fitness_values) if fitness_values else 0

    # Compute best and average DNA
    log_dict = {
        "generation": generation_index,
        "avg_fitness": avg_fitness,
        "top_fitness": top_fitness,
    }

    if all_genomes:
        dnas = [g["dna"] for g in all_genomes]
        best_genome = max(all_genomes, key=lambda g: g["fitness"])
        best_dna = best_genome["dna"]
        avg_dna = [float(sum(x)) / len(x) for x in zip(*dnas)]

        # Log each DNA value individually for better visualization
        log_dict.update({
            f"avg_dna_{i}": val for i, val in enumerate(avg_dna)
        })
        log_dict.update({
            f"best_dna_{i}": val for i, val in enumerate(best_dna)
        })

    wandb.log(log_dict)


    # ============ SELECTION ============
    def rank_based_selection(genomes, num_pairs):
        # Sort genomes by fitness in descending order
        sorted_genomes = sorted(genomes, key=lambda g: g["fitness"], reverse=True)
        n = len(sorted_genomes)

        # Calculate rank-based probabilities (higher rank = higher probability)
        # Using linear ranking: P(rank) = (2*rank)/(n*(n+1))
        rank_probs = [(2 * (n - i)) / (n * (n + 1)) for i in range(n)]

        pairs = []
        selected_pairs_set = set()

        for _ in range(num_pairs):
            attempts = 0
            while attempts < 1000:
                # Select two different genomes based on rank probabilities
                p1 = random.choices(sorted_genomes, weights=rank_probs, k=1)[0]
                p2 = random.choices(sorted_genomes, weights=rank_probs, k=1)[0]

                # Create pair identifier that is order-independent
                pair_id = tuple(sorted([p1["model_path"], p2["model_path"]]))

                if p1 != p2 and pair_id not in selected_pairs_set:
                    selected_pairs_set.add(pair_id)
                    pairs.append((p1, p2))
                    break

                attempts += 1
            else:
                print(f"{YELLOW}Warning: Could not find unique pair after 1000 attempts{RESET}")
                # Fall back to using any different pair
                while True:
                    p1 = random.choices(sorted_genomes, weights=rank_probs, k=1)[0]
                    p2 = random.choices(sorted_genomes, weights=rank_probs, k=1)[0]
                    if p1 != p2:
                        pairs.append((p1, p2))
                        break
        return pairs


    n = POPULATION_SIZE - ELITISM  # number of pairs for crossover (non-elites)

    if len(MACHINES) == 1:
        # Only one genome, pair with itself
        selected_pairs = [(all_genomes[0], all_genomes[0])]
        print(f"{PURPLE}Only one machine: skipping selection, self-pairing for crossover.{RESET}")
    else:
        selected_pairs = rank_based_selection(all_genomes, n)
        print(f"{PURPLE}Selected pairs for crossover (with fitness):{RESET}")

        # Compute ranks for all_genomes (sorted by fitness descending)
        sorted_genomes = sorted(all_genomes, key=lambda g: g["fitness"], reverse=True)
        model_path_to_rank = {g["model_path"]: idx + 1 for idx, g in enumerate(sorted_genomes)}

        for idx, (g1, g2) in enumerate(selected_pairs):
            print(
                f"  Pair {idx + 1}: "
                f"Rank {model_path_to_rank[g1['model_path']]} (fitness={g1['fitness']}) "
                f"<-> "
                f"Rank {model_path_to_rank[g2['model_path']]} (fitness={g2['fitness']})"
            )

        # Elitism: add (elite, elite) pairs for top ELITISM genomes
        elite_pairs = []
        for elite_genome in sorted_genomes[:ELITISM]:
            elite_pairs.append((elite_genome, elite_genome))
        selected_pairs.extend(elite_pairs)

    # ============ CROSSOVER ============
    for machine, (g1, g2) in zip(MACHINES, selected_pairs):
        output_path = f"{current_directory}/{machine}/Gen{generation_index + 1:04d}"

        full_command = (
            f". ../myenv/bin/activate && "
            f"python3.11 crossover.py "
            f"--model1_path {current_directory}/{g1['model_path']} "
            f"--model2_path {current_directory}/{g2['model_path']} "
            f"--output_path {output_path} "
        )
        payload = {"command": full_command}
        try:
            url = f"http://{machine}:7899/"
            response = requests.post(url, json=payload)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error sending command '{full_command}' to {machine}: {e}")

    wait_for_workers()

    # ============ MUTATION ============
    # Set dna_mutated for a MUTATION_PROBABILITY ratio of the population
    num_to_mutate = max(1, int(round(MUTATION_PROBABILITY * len(MACHINES))))
    machine_dirs = [
        (machine, f"{current_directory}/{machine}/Gen{generation_index + 1:04d}")
        for machine in MACHINES
    ]
    random.shuffle(machine_dirs)
    mutate_set = set(machine for machine, _ in machine_dirs[:num_to_mutate])
    for machine, model_dir in machine_dirs:
        genome_path = f"{model_dir}/genome.json"
        with open(genome_path, 'r') as f:
            genome_data = json.load(f)
        genome_data["dna_mutated"] = machine in mutate_set
        with open(genome_path, 'w') as f:
            json.dump(genome_data, f, indent=4)

    for machine in MACHINES:
        model_dir = f"{current_directory}/{machine}/Gen{generation_index + 1:04d}"
        with open(f"{model_dir}/genome.json", 'r') as f:
            genome_data = json.load(f)

        mutation_strength = MUTATION_STRENGTH if genome_data["dna_mutated"] else 0
        full_command = (
            f". ../myenv/bin/activate && "
            f"python3.11 mutation.py "
            f"--model_path {model_dir} "
            f"--mutation_strength {mutation_strength} "
            f"--cache_dir {current_directory}/cache "
        )
        payload = {"command": full_command}
        try:
            url = f"http://{machine}:7899/"
            response = requests.post(url, json=payload)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error sending command '{full_command}' to {machine}: {e}")

    wait_for_workers()

    # ============ WRAP UP EVOLUTION STEP ============
    if generation_index >= MAX_GENERATIONS:
        print(f"{GREEN}Reached maximum generations ({MAX_GENERATIONS}). Stopping.{RESET}")
        pbar.close()
        break

    # --- Cleanup previous generation's safetensors ---
    if all_genomes:  # Ensure there are genomes to process
        # Find the genome with the highest fitness from the current generation
        best_genome = max(all_genomes, key=lambda g: g['fitness'])
        best_model_dir = best_genome['model_path']
        print(
            f"{YELLOW}Best model directory for Gen {generation_index}: {best_model_dir} (Fitness: {best_genome['fitness']}){RESET}")

        # Iterate through all genomes of the current generation
        for genome in all_genomes:
            current_model_dir = genome['model_path']
            # Skip the best model's directory
            if current_model_dir == best_model_dir:
                print(f"{GREEN}Keeping safetensors for best model: {current_model_dir}{RESET}")
                continue

            # Construct the path to the safetensors file (assuming standard name)
            # Adjust "model.safetensors" if your models save with a different name
            safetensors_file_path = os.path.join(current_model_dir, "model.safetensors")

            # Check if the file exists and delete it
            if os.path.exists(safetensors_file_path):
                try:
                    os.remove(safetensors_file_path)
                    print(f"{RED}Deleted safetensors: {safetensors_file_path}{RESET}")
                except OSError as e:
                    print(f"{RED}Error deleting file {safetensors_file_path}: {e}{RESET}")
            else:
                # It might be normal for the file not to exist if it was already cleaned up
                # or if the model saving failed, but we print a notice just in case.
                print(
                    f"{YELLOW}Safetensors file not found (already deleted or failed save?): {safetensors_file_path}{RESET}")
    else:
        print(f"{YELLOW}No genomes found for generation {generation_index}, skipping safetensors cleanup.{RESET}")

    # --- Optional: Explicit Garbage Collection ---
    # Helps free up memory, especially after deleting large files
    print(f"{CYAN}Running garbage collection...{RESET}")
    gc.collect()
    print(f"{CYAN}Garbage collection finished.{RESET}")

    pbar.update(1)
    generation_index += 1