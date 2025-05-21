import json
import requests
import time
import os
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import shutil
from tqdm import tqdm
import sys
import gc
import wandb
import torch
from concurrent.futures import ThreadPoolExecutor
import torch.optim as optim

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
if generation_index == 0:
    for machine, model_path in zip(MACHINES, filled_population):
        full_command = (
            f"python init_weights.py "
            f"--model_path {model_path} "
            f"--output_path {machine}/Gen{generation_index:04d} "
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
wandb.login()
wandb.init(project=PROJECT_NAME)

# ====================== EVOLUTION BEGINS ======================
pbar = tqdm(total=MAX_GENERATIONS, desc="Evolution Progress", initial=generation_index)

# --- Outer optimizer state (for momentum) ---
outer_optimizer = None
base_model = None

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
        full_command = (
            f"python eval.py "
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

    log_dict = {
        "generation": generation_index,
        "avg_fitness": avg_fitness,
        "top_fitness": top_fitness,
    }

    # Optionally log DNA stats if present
    # if all_genomes:
    #     dnas = [g["dna"] for g in all_genomes]
    #     best_genome = max(all_genomes, key=lambda g: g["fitness"])
    #     best_dna = best_genome["dna"]
    #     avg_dna = [float(sum(x)) / len(x) for x in zip(*dnas)]
    #     log_dict.update({
    #         f"avg_dna_{i}": val for i, val in enumerate(avg_dna)
    #     })
    #     log_dict.update({
    #         f"best_dna_{i}": val for i, val in enumerate(best_dna)
    #     })

    wandb.log(log_dict)

    # ============ REMOVE SELECTION & CROSSOVER ============
    # Instead of selection/crossover, just copy each model to the next generation
    for machine in MACHINES:
        src_dir = f"{current_directory}/{machine}/Gen{generation_index:04d}"
        dst_dir = f"{current_directory}/{machine}/Gen{generation_index + 1:04d}"
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)
        print(f"{CYAN}Copied {src_dir} -> {dst_dir}{RESET}")

    # ============ MUTATION ============
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
            f"python mutation.py "
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

    # ============ OUTER STEP (DiLoCo style, SGD with momentum, nesterov=True) ============
    print(f"{CYAN}Starting outer step (averaging weight updates across models, SGD with momentum)...{RESET}")

    prev_gen_dirs = [f"{current_directory}/{machine}/Gen{generation_index:04d}" for machine in MACHINES]
    curr_gen_dirs = [f"{current_directory}/{machine}/Gen{generation_index + 1:04d}" for machine in MACHINES]

    if base_model is not None:
        del base_model
        torch.cuda.empty_cache()
        gc.collect()
    base_model = AutoModelForCausalLM.from_pretrained(prev_gen_dirs[0], device_map="cpu")
    trained_models = [
        AutoModelForCausalLM.from_pretrained(curr_dir, device_map="cpu")
        for curr_dir in curr_gen_dirs
    ]
    num_models = len(trained_models)

    # Step 1: Compute and accumulate parameter differences (delta = trained - base)
    delta_accumulator = []
    for base_param in base_model.parameters():
        delta_accumulator.append(torch.zeros_like(base_param.data))

    for trained_model in trained_models:
        for i, (base_param, trained_param) in enumerate(zip(base_model.parameters(), trained_model.parameters())):
            delta = trained_param.data - base_param.data
            delta_accumulator[i] += delta / num_models  # average delta

    # Step 2: Assign these deltas as gradients to base model (negative for SGD step toward trained)
    for param, avg_delta in zip(base_model.parameters(), delta_accumulator):
        param.grad = -avg_delta

    # Step 3: Apply SGD update using pseudo-gradients
    outer_lr = EVOLUTION.get("OUTER_LR", 0.7)
    outer_momentum = EVOLUTION.get("OUTER_MOMENTUM", 0.9)
    outer_nesterov = EVOLUTION.get("OUTER_NESTEROV", True)
    current_base_model_params = list(base_model.parameters())

    if outer_optimizer is None:
        outer_optimizer = optim.SGD(
            current_base_model_params,
            lr=outer_lr,
            momentum=outer_momentum,
            nesterov=outer_nesterov
        )
    else:
        optimizer_is_stale = True
        if outer_optimizer.param_groups and outer_optimizer.param_groups[0]['params']:
            optimizer_params = outer_optimizer.param_groups[0]['params']
            if len(optimizer_params) == len(current_base_model_params) and \
               all(p_opt is p_model for p_opt, p_model in zip(optimizer_params, current_base_model_params)):
                optimizer_is_stale = False

        if optimizer_is_stale:
            new_optimizer = optim.SGD(
                current_base_model_params,
                lr=outer_lr,
                momentum=outer_momentum,
                nesterov=outer_nesterov
            )
            new_optimizer.load_state_dict(outer_optimizer.state_dict())
            outer_optimizer = new_optimizer
        else:
            for group in outer_optimizer.param_groups:
                group['lr'] = outer_lr
                group['momentum'] = outer_momentum
                group['nesterov'] = outer_nesterov

    outer_optimizer.step()
    outer_optimizer.zero_grad()

    # Save the updated base model to the next generation for all workers
    next_gen_dirs = [f"{current_directory}/{machine}/Gen{generation_index + 1:04d}" for machine in MACHINES]
    tokenizer = AutoTokenizer.from_pretrained(prev_gen_dirs[0])

    for source, target in zip(prev_gen_dirs, curr_gen_dirs):
        for fname in ["optimizer.pt", "scheduler.pt"]:
            p1_file = os.path.join(source, fname)
            out_file = os.path.join(target, fname)
            if os.path.exists(p1_file):
                shutil.copy(p1_file, out_file)

    def save_model_to_dir(target_dir):
        base_model.save_pretrained(target_dir)
        tokenizer.save_pretrained(target_dir)
        print(f"{GREEN}Saved outer-stepped model to {target_dir}{RESET}")

    with ThreadPoolExecutor(max_workers=len(next_gen_dirs)) as executor:
        executor.map(save_model_to_dir, next_gen_dirs)

    for m in trained_models:
        del m
    torch.cuda.empty_cache()
    gc.collect()

    print(f"{CYAN}Outer step complete. Proceeding to safetensors cleanup...{RESET}")

    # ============ WRAP UP EVOLUTION STEP ============
    if generation_index >= MAX_GENERATIONS:
        print(f"{GREEN}Reached maximum generations ({MAX_GENERATIONS}). Stopping.{RESET}")
        pbar.close()
        break

    # --- Cleanup previous generation's safetensors ---
    if all_genomes:
        best_genome = max(all_genomes, key=lambda g: g['fitness'])
        best_model_dir = best_genome['model_path']
        print(
            f"{YELLOW}Best model directory for Gen {generation_index}: {best_model_dir} (Fitness: {best_genome['fitness']}){RESET}")

        for genome in all_genomes:
            current_model_dir = genome['model_path']
            if current_model_dir == best_model_dir:
                print(f"{GREEN}Keeping safetensors for best model: {current_model_dir}{RESET}")
                continue
            safetensors_file_path = os.path.join(current_model_dir, "model.safetensors")
            if os.path.exists(safetensors_file_path):
                try:
                    os.remove(safetensors_file_path)
                    print(f"{RED}Deleted safetensors: {safetensors_file_path}{RESET}")
                except OSError as e:
                    print(f"{RED}Error deleting file {safetensors_file_path}: {e}{RESET}")
            else:
                print(
                    f"{YELLOW}Safetensors file not found (already deleted or failed save?): {safetensors_file_path}{RESET}")
    else:
        print(f"{YELLOW}No genomes found for generation {generation_index}, skipping safetensors cleanup.{RESET}")

    print(f"{CYAN}Running garbage collection...{RESET}")
    gc.collect()
    print(f"{CYAN}Garbage collection finished.{RESET}")

    pbar.update(1)
    generation_index += 1
