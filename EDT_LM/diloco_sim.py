import json
import time
import os
import random
from transformers import AutoModel, AutoConfig, AutoTokenizer
import shutil
from tqdm import tqdm
import sys
import gc
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from concurrent.futures import ThreadPoolExecutor
import torch.optim as optim
import subprocess  # For local command execution

# ANSI escape codes for colors
PURPLE = "\033[95m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

with open("evolution.json", "r") as config_file:
    EVOLUTION = json.load(config_file)

# --- Simulate multiple machines by duplicating the train folder ---
NUM_MACHINES = EVOLUTION.get("NUM_MACHINES", 1)
MACHINES = [f"train_sim_{i}" for i in range(NUM_MACHINES)]

# Duplicate the train folder for each simulated machine if not already present
base_train_dir = os.path.join(os.getcwd(), "train")
for machine in MACHINES:
    machine_dir = os.path.join(os.getcwd(), machine)
    if os.path.exists(base_train_dir):
        if os.path.exists(machine_dir):
            shutil.rmtree(machine_dir)
            os.mkdir(machine_dir)
        shutil.copytree(base_train_dir, machine_dir, dirs_exist_ok=True)
    else:
        NotImplementedError()

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

def run_local_command(command, machine="train_sim_0"):
    """Run a shell command locally in the machine's directory and print output/errors."""
    train_dir = os.path.join(os.getcwd(), machine)
    print(f"Running command in {train_dir}: {command}")
    result = subprocess.run(command, shell=True, cwd=train_dir)
    if result.returncode != 0:
        print(f"{RED}Command failed: {command}{RESET}")
        quit()
    return result.returncode

def wait_for_workers():
    # No remote workers, so just pass
    pass

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
        # Run locally in the simulated machine folder
        run_local_command(full_command, machine=machine)
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
        # Run locally in the simulated machine folder
        run_local_command(full_command, machine=machine)

    # Wait for eval scripts to finish (no-op)
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

    # if all_genomes:
    #     dnas = [g["dna"] for g in all_genomes]
    #     best_genome = max(all_genomes, key=lambda g: g["fitness"])
    #     best_dna = best_genome["dna"]
    #     avg_dna = [float(sum(x)) / len(x) for x in zip(*dnas)]
    #
    #     # Log each DNA value individually for better visualization
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
            f"python mutation.py "
            f"--model_path {model_dir} "
            f"--mutation_strength {mutation_strength} "
            f"--cache_dir {current_directory}/cache "
        )
        # Run locally in the simulated machine folder
        run_local_command(full_command, machine=machine)

    wait_for_workers()

    # ============ OUTER STEP (DiLoCo style, SGD with momentum, nesterov=True) ============

    print(f"{CYAN}Starting outer step (averaging weight updates across models, SGD with momentum)...{RESET}")

    prev_gen_dirs = [f"{current_directory}/{machine}/Gen{generation_index:04d}" for machine in MACHINES]
    curr_gen_dirs = [f"{current_directory}/{machine}/Gen{generation_index + 1:04d}" for machine in MACHINES]

    # Load base model (from previous generation, e.g., first worker)
    if base_model is not None:
        del base_model
        torch.cuda.empty_cache()
        gc.collect()
    base_model = AutoModelForCausalLM.from_pretrained(prev_gen_dirs[0], device_map="cpu")
    # Load all trained models (after mutation)
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
    outer_lr = EVOLUTION.get("OUTER_LR", 1.0)
    outer_momentum = EVOLUTION.get("OUTER_MOMENTUM", 0.0)
    outer_nesterov = EVOLUTION.get("OUTER_NESTEROV", False) # Default to True if not specified
    # # Step 3: Apply SGD update using pseudo-gradients
    # outer_lr = EVOLUTION.get("OUTER_LR", 0.7)
    # outer_momentum = EVOLUTION.get("OUTER_MOMENTUM", 0.9)
    # outer_nesterov = EVOLUTION.get("OUTER_NESTEROV", True) # Default to True if not specified

    current_base_model_params = list(base_model.parameters())

    if outer_optimizer is None:
        # print(f"{YELLOW}Initializing outer_optimizer for the first time.{RESET}")
        outer_optimizer = optim.SGD(
            current_base_model_params,
            lr=outer_lr,
            momentum=outer_momentum,
            nesterov=outer_nesterov
        )
    else:
        # Check if the optimizer's parameters are stale (i.e., from a previous base_model instance)
        optimizer_is_stale = True # Assume stale unless proven otherwise
        if outer_optimizer.param_groups and outer_optimizer.param_groups[0]['params']:
            optimizer_params = outer_optimizer.param_groups[0]['params']
            if len(optimizer_params) == len(current_base_model_params) and \
               all(p_opt is p_model for p_opt, p_model in zip(optimizer_params, current_base_model_params)):
                optimizer_is_stale = False
        
        if optimizer_is_stale:
            # print(f"{YELLOW}Outer_optimizer is stale. Re-initializing with new model params and transferring state.{RESET}")
            # Optimizer is stale (points to old model's parameters).
            # Create a new optimizer for the current base_model's parameters.
            new_optimizer = optim.SGD(
                current_base_model_params,
                lr=outer_lr,
                momentum=outer_momentum,
                nesterov=outer_nesterov
            )
            # Transfer state from the old optimizer to the new one.
            # This handles momentum buffers.
            new_optimizer.load_state_dict(outer_optimizer.state_dict())
            outer_optimizer = new_optimizer
        else:
            # print(f"{GREEN}Outer_optimizer is up-to-date with model parameters. Updating hyperparameters.{RESET}")
            # Optimizer is still valid for the current base_model parameters (e.g., if base_model wasn't reloaded)
            # Update hyperparameters in case they changed in EVOLUTION.json
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
        # Carry over .pt files from parents if they exist
        for fname in ["optimizer.pt", "scheduler.pt"]:
            p1_file = os.path.join(source, fname)
            out_file = os.path.join(target, fname)
            # Prefer p1's .pt file if both exist
            if os.path.exists(p1_file):
                shutil.copy(p1_file, out_file)

    def save_model_to_dir(target_dir):
        base_model.save_pretrained(target_dir)
        tokenizer.save_pretrained(target_dir)

        print(f"{GREEN}Saved outer-stepped model to {target_dir}{RESET}")

    with ThreadPoolExecutor(max_workers=len(next_gen_dirs)) as executor:
        executor.map(save_model_to_dir, next_gen_dirs)

    # Cleanup
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
            safetensors_file_path = os.path.join(current_model_dir, "model.safetensors")

            # Check if the file exists and delete it
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