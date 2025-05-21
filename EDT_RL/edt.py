import json
import requests
import time
import os
import random
from crossover import crossover
import shutil
from tqdm import tqdm  # Add this import
import sys

# ANSI escape codes for colors
PURPLE = "\033[95m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

MACHINES_FILE = "machines.json"
PROJECT_NAME = "EDT_RL"
#                       /PROJECT_NAME
#                           /base_code + sub_folder4each_machine
#                               /gen_folders + train_scripts


START_GENERATION = 0
NUM_AGENTS = 2  # Number of agents to randomly select from the pool
FIXED_FRAMES_PER_GEN = 1400000  # Adjust this number as needed

# Add new constants at the top with other constants
MAX_GENERATIONS = 40  # Total number of generations to run
INITIAL_MUTATION_RATIO = 0.4  # Start with 40% of population being mutated
FINAL_MUTATION_RATIO = 0.01  # End with 1% of population being mutated
INITIAL_MUTATION_STRENGTH = 0.5  # Initially mutate 50% of the reward DNA
FINAL_MUTATION_STRENGTH = 0.01  # End with mutating 10% of the reward DNA
STABLE_STEPS_START = 2  # Number of generations without mutation
STABLE_STEPS_END = 4

# Load machine ips
with open(MACHINES_FILE, "r") as f:
    MACHINES = json.load(f)

MODELS = [
    f"SLOT_{m}" for m in MACHINES
]

# Remove the repeated part from COMMANDS
BASE_COMMAND = f". ../myenv/bin/activate && python3.11 ppo_vectors.py --num_envs {64} " \
               f"--frames_per_batch {64 * 32} --mini_batch_size {16 * 32} --ppo_epochs 4"
COMMANDS = [
    "--context_length 32 --hidden_size 32 --num_layers 4 --clip_epsilon 0.2 --entropy_coef 0.1" for m in MACHINES
]

MAX_REWARD_GENE_INDEX = 18
REWARDS = [
    # [0, 0, 17, 2, 11, 16],
    # [0, 0, 17, 2, 12, 15],
    # [0, 0, 18, 2, 12, 15],
    # [17, 14, 18, 2, 12, 16],
    # [17, 17, 18, 2, 12, 16],
    [13, 17, 18, 2, 12, 16],
]
# Ensure the length of REWARDS matches the length of MACHINES
while len(REWARDS) != len(MACHINES):
    new_reward_dna = random.choice(REWARDS).copy()
    REWARDS.append(new_reward_dna)

assert len(COMMANDS) == len(MACHINES) == len(MODELS) == len(REWARDS)

generation_index = START_GENERATION
if generation_index == 0:
    for model_name, machine, partial_cmd, reward_dna in zip(MODELS, MACHINES, COMMANDS, REWARDS):
        initial_model_dir = f"{machine}/Gen{generation_index:04d}/{model_name}"
        os.makedirs(initial_model_dir, exist_ok=True)

        env_info = {
            'env_name': model_name,
            'reward_dna': reward_dna,
            'agents': []  # No agents for initial generation
        }

        with open(f"{initial_model_dir}/genome.json", 'w') as f:
            json.dump({"env": env_info}, f, indent=4)
    generation_index += 1

# Redirect standard output to a file
log_file = open("log.txt", "w")
sys.stdout = log_file

# Now read speed from speed.json and run the real commands
models_pool = []
models_pool_previous_generations = []
current_directory = os.getcwd()
if generation_index > 1:
    for model_name, machine in zip(MODELS, MACHINES):
        models_pool.append(f"{current_directory}/{machine}/Gen{generation_index - 1:04d}/{model_name}/Policy")

pbar = tqdm(total=MAX_GENERATIONS, desc="Evolution Progress", initial=generation_index)
while True:
    current_directory = os.getcwd()
    for machine in MACHINES:
        path = f"{current_directory}/{machine}/Gen{generation_index - 1:04d}"
        if not os.path.exists(path) and generation_index - 1 >= 0:
            raise NotImplementedError(f"Wrong generation index, {path} does not exist")

    # Loop through all machines to run the ppo script
    for model_name, machine, partial_cmd, reward_dna in zip(MODELS, MACHINES, COMMANDS, REWARDS):
        # Real command with wandb
        full_command = (
            f"{BASE_COMMAND} "
            f"--logger_backend wandb --project_name {PROJECT_NAME} "
            f"{partial_cmd} "
            f"--total_frames {FIXED_FRAMES_PER_GEN} "
            f"--gen {generation_index} "
            f"--model_name {model_name}"
        )
        payload = {"command": full_command}
        try:
            url = f"http://{machine}:7899/"
            response = requests.post(url, json=payload)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            print(f"{CYAN}Command '{full_command}' sent to {machine}. Response: {response.json()}{RESET}")
        except requests.exceptions.RequestException as e:
            print(f"{RED}Error sending COMMAND {full_command} to {machine}: {e}{RESET}")

    # Wait for all scripts to finish
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
            print("All servers finished.")
            break
        time.sleep(5)

    # Send eval command to all servers
    for machine in MACHINES:
        payload = {"command": ". ../myenv/bin/activate && python3.11 eval.py"}
        try:
            url = f"http://{machine}:7899/"
            response = requests.post(url, json=payload)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error sending eval command to {machine}: {e}")

    # Wait for eval scripts to finish
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
            print("Evaluation run finished.")
            break
        time.sleep(5)

    # Increment generation index, next we go into selection and crossover
    generation_index += 1
    # Copy the last generation folder to the current generation folder
    for model_name, machine in zip(MODELS, MACHINES):
        src_dir = f"{current_directory}/{machine}/Gen{generation_index - 1:04d}"
        dest_dir = f"{current_directory}/{machine}/Gen{generation_index:04d}"
        if os.path.exists(src_dir):
            shutil.copytree(src_dir, dest_dir)

    # Gather fitness from each machine's eval_result.json
    model_fitness = {}
    for machine in MACHINES:
        result_json = os.path.join(current_directory, machine, "eval", "eval_result.json")
        if os.path.exists(result_json):
            with open(result_json, "r") as f:
                data = json.load(f)
                for model_path, rank_val in data.items():
                    key = model_path.lstrip("../").rstrip("/Policy")
                    model_fitness[key] = model_fitness.get(key, 0) + rank_val

    # Load genome data, add fitness and model_path
    all_genomes = []
    for path_key, fitness_score in model_fitness.items():
        genome_file = os.path.join(path_key, "genome.json")
        with open(genome_file, "r") as gf:
            genome_data = json.load(gf)
        genome_data["fitness"] = fitness_score
        genome_data["model_path"] = path_key
        all_genomes.append(genome_data)


    def spin_roulette(genomes, sum_fitness, scale):
        pick = random.random() * sum_fitness
        running = 0
        for g in genomes:
            running += g["fitness"] ** scale
            if running >= pick:
                return g
        return genomes[-1]


    def roulette_wheel_selection(genomes, num_pairs, scale):
        sum_fit = sum(g["fitness"] ** scale for g in genomes)
        pairs = []
        for _ in range(num_pairs):
            p1 = spin_roulette(genomes, sum_fit, scale)
            p2 = spin_roulette(genomes, sum_fit, scale)
            while p2 == p1:
                p2 = spin_roulette(genomes, sum_fit, scale)
            pairs.append((p1, p2))
        return pairs


    def rank_based_selection(genomes, num_pairs):
        # Sort genomes based on fitness in descending order
        sorted_genomes = sorted(genomes, key=lambda g: g["fitness"], reverse=True)

        # Assign ranks to genomes
        ranks = list(range(1, len(sorted_genomes) + 1))

        # Calculate selection probabilities based on ranks
        total_ranks = sum(ranks)
        selection_probs = [rank / total_ranks for rank in ranks]

        pairs = []
        for _ in range(num_pairs):
            p1 = random.choices(sorted_genomes, weights=selection_probs, k=1)[0]
            p2 = random.choices(sorted_genomes, weights=selection_probs, k=1)[0]
            while p2 == p1:
                p2 = random.choices(sorted_genomes, weights=selection_probs, k=1)[0]
            pairs.append((p1, p2))
        return pairs


    # Calculate the scaling factor based on generation progress
    progress = min(generation_index / MAX_GENERATIONS, 1.0)
    scale = 0.1 + 2.4 * progress  # Scale ranges from 0.1 to 2.5

    n = len(all_genomes)  # number of pairs
    selected_pairs = roulette_wheel_selection(all_genomes, n, scale)

    models_pool = []
    models_pool_previous_generations = []
    current_directory = os.getcwd()
    for model_name, machine in zip(MODELS, MACHINES):
        models_pool.append(f"{current_directory}/{machine}/Gen{generation_index:04d}/{model_name}/Policy")
    # # Randomly select 4 generation numbers from 0 to current generation - 1
    # if generation_index > 0:
    #     selected_gens = random.sample(range(max(0, generation_index)), min(2, generation_index))
    #     for gen in selected_gens:
    #         for model_name, machine in zip(MODELS, MACHINES):
    #             model_path = f"{current_directory}/{machine}/Gen{gen:04d}/{model_name}/Policy"
    #             if os.path.exists(model_path):
    #                 models_pool_previous_generations.append(model_path)

    # Perform crossover to generate the next generation
    for i, (g1, g2) in enumerate(selected_pairs):
        output_path = f"{current_directory}/{MACHINES[i % len(MACHINES)]}/Gen{generation_index:04d}/{MODELS[i]}"
        os.makedirs(output_path, exist_ok=True)

        result_genome = crossover(g1, g2, output_path)

        selected_agents = random.sample(models_pool, min(NUM_AGENTS, len(models_pool)))  # Randomly select N agents
        # selected_agents += random.sample(models_pool_previous_generations,
        #                                  min(NUM_AGENTS, len(models_pool_previous_generations)))
        result_genome["env"]["agents"] = selected_agents

        # Save result_genome as genome.json
        with open(os.path.join(output_path, "genome.json"), "w") as f:
            json.dump(result_genome, f, indent=4)

    # Replace the mutation section with this new code
    # Calculate current mutation parameters based on generation progress
    progress = min(generation_index / MAX_GENERATIONS, 1.0)
    current_mutation_ratio = INITIAL_MUTATION_RATIO - (INITIAL_MUTATION_RATIO - FINAL_MUTATION_RATIO) * progress
    current_mutation_strength = INITIAL_MUTATION_STRENGTH - (
            INITIAL_MUTATION_STRENGTH - FINAL_MUTATION_STRENGTH) * progress

    if STABLE_STEPS_START < generation_index < MAX_GENERATIONS - STABLE_STEPS_END:
        num_to_mutate = int(current_mutation_ratio * len(selected_pairs))
        num_to_mutate = max(1, num_to_mutate)  # Ensure at least one mutation
        mutate_indices = random.sample(range(len(selected_pairs)), num_to_mutate)

        for idx in mutate_indices:
            output_path = f"{current_directory}/{MACHINES[idx % len(MACHINES)]}/Gen{generation_index:04d}/{MODELS[idx]}"
            genome_file = os.path.join(output_path, "genome.json")

            with open(genome_file, "r") as f:
                genome_data = json.load(f)

            reward_dna = genome_data["env"]["reward_dna"].copy()
            reward_dna_original = genome_data["env"]["reward_dna"].copy()
            num_mutations = max(1, int(len(reward_dna) * current_mutation_strength))
            mutation_indices = random.sample(range(len(reward_dna)), num_mutations)

            for mutation_idx in mutation_indices:
                reward_dna[mutation_idx] = random.randint(0, MAX_REWARD_GENE_INDEX)

            genome_data["env"]["reward_dna_before_mutation"] = reward_dna_original
            genome_data["env"]["reward_dna"] = reward_dna

            with open(genome_file, "w") as f:
                json.dump(genome_data, f, indent=4)

    # Add generation check before incrementing
    if generation_index >= MAX_GENERATIONS:
        print(f"{GREEN}Reached maximum generations ({MAX_GENERATIONS}). Stopping.{RESET}")
        pbar.close()
        break

    # Update progress bar before generation increment
    pbar.update(2)
    generation_index += 1

# Close the log file at the end of the script
log_file.close()

