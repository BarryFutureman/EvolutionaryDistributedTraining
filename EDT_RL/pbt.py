import json
import requests
import time
import os
import random
import shutil
from tqdm import tqdm
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
PBT_EXPLOIT_FRACTION = 0.2 # Fraction of population to exploit/explore

# Add new constants at the top with other constants
MAX_GENERATIONS = 40  # Total number of generations to run
INITIAL_MUTATION_RATIO = 0.0  # Start with 40% of population being mutated
FINAL_MUTATION_RATIO = 0  # End with 1% of population being mutated
INITIAL_MUTATION_STRENGTH = 0  # Initially mutate 50% of the reward DNA
FINAL_MUTATION_STRENGTH = 0  # End with mutating 10% of the reward DNA
STABLE_STEPS_START = 40  # Number of generations without mutation
STABLE_STEPS_END = 4

# Load machine ips
with open(MACHINES_FILE, "r") as f:
    MACHINES = json.load(f)

MODELS = [
    f"SLOT_{m}" for m in MACHINES
]

# Remove the repeated part from COMMANDS
BASE_COMMAND = f". /student/jian1034/Desktop/RL_Contest/Iv4/myenv/bin/activate && python3.11 ppo_vectors.py --num_envs {64} " \
               f"--frames_per_batch {64 * 32} --mini_batch_size {16 * 32} --ppo_epochs 4"
COMMANDS = [
    "--context_length 32 --hidden_size 32 --num_layers 4 --clip_epsilon 0.2 --entropy_coef 0.1" for m in MACHINES
]

MAX_REWARD_GENE_INDEX = 18
REWARDS = [
    [0, 0, 17, 2, 11, 16],
    [0, 0, 17, 2, 12, 15],
    [0, 0, 18, 2, 12, 15],
    [17, 14, 18, 2, 12, 16],
    [9, 17, 18, 2, 12, 16],
    [13, 17, 18, 2, 12, 16],
    [13, 17, 18, 2, 12, 7],
    [8, 17, 18, 2, 12, 16],
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
        payload = {"command": ". /student/jian1034/Desktop/RL_Contest/Iv4/myenv/bin/activate && python3.11 eval.py"}
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
        genome_file = os.path.join(current_directory, path_key, "genome.json") # Ensure path_key is relative to current_directory if needed
        # If path_key is already absolute, current_directory might not be needed.
        # Assuming path_key from eval_result.json is like "machine_X/GenYYYY/SLOT_machine_X"
        # So, os.path.join(current_directory, path_key, ...) is correct.
        if not os.path.exists(genome_file):
            # Fallback if current_directory was already part of path_key (e.g. if path_key was absolute)
            genome_file_alt = os.path.join(path_key, "genome.json")
            if os.path.exists(genome_file_alt):
                genome_file = genome_file_alt
            else:
                print(f"{RED}Genome file not found at {genome_file} or {genome_file_alt}{RESET}")
                continue # Skip this entry if genome file is missing

        with open(genome_file, "r") as gf:
            genome_data = json.load(gf)
        genome_data["fitness"] = fitness_score
        # Ensure model_path in all_genomes is consistently relative (e.g., "machine_X/GenYYYY/SLOT_machine_X")
        # The key from model_fitness (path_key) is already in this format.
        genome_data["model_path"] = path_key
        all_genomes.append(genome_data)

    # Sort genomes by fitness in descending order (best first)
    all_genomes.sort(key=lambda g: g["fitness"], reverse=True)

    # PBT Implementation
    num_population = len(MODELS)
    num_exploit_candidates = max(1, int(num_population * PBT_EXPLOIT_FRACTION)) if num_population > 0 else 0

    # Create a mapping from the original model path (from prev gen) to its rank
    # model_path in all_genomes is like "m3/Gen0001/SLOT_m3"
    model_ranks = {genome['model_path']: rank for rank, genome in enumerate(all_genomes)}

    next_gen_genome_infos = [] 

    for k in range(num_population): # Iterate through each slot for the new generation
        target_machine = MACHINES[k]
        target_model_name = MODELS[k]
        
        # Path of the agent that ran in this slot in the *previous* generation
        # generation_index is for the *new* generation. Prev gen is generation_index-1.
        lookup_path_prev_gen = f"{target_machine}/Gen{generation_index-1:04d}/{target_model_name}"

        try:
            agent_genome_prev_gen = next(g for g in all_genomes if g['model_path'] == lookup_path_prev_gen)
        except StopIteration:
            print(f"{RED}Error: Could not find previous generation genome for slot {lookup_path_prev_gen} in all_genomes.{RESET}")
            # This is a critical error, means all_genomes might be incomplete or paths don't match.
            # For now, we'll let it raise an error or handle by creating a default/random agent.
            # As a placeholder, re-initialize if not found (not ideal for PBT)
            print(f"{YELLOW}Initializing agent {target_model_name} with random DNA due to missing previous genome.{RESET}")
            parent_reward_dna = random.choice(REWARDS).copy() # Fallback
            # Model to load from would be None, ppo_script handles fresh start
            model_to_load_from = None # Indicates fresh start for this agent
            # Add to next_gen_genome_infos with placeholder values
            output_path_next_gen = f"{current_directory}/{target_machine}/Gen{generation_index:04d}/{target_model_name}"
            os.makedirs(output_path_next_gen, exist_ok=True)
            next_gen_genome_infos.append({
                "output_path": output_path_next_gen,
                "reward_dna": parent_reward_dna,
                "model_to_load_from": model_to_load_from,
                "target_model_name": target_model_name,
                "was_fallback": True
            })
            continue # Skip PBT logic for this agent

        rank = model_ranks[agent_genome_prev_gen['model_path']] 

        parent_reward_dna = agent_genome_prev_gen['env']['reward_dna'].copy()
        model_to_load_from = agent_genome_prev_gen['model_path'] + "/Policy" 

        if rank >= num_population - num_exploit_candidates and num_exploit_candidates > 0: # Agent is in bottom fraction
            if all_genomes and len(all_genomes) >= num_exploit_candidates : # Ensure there are top performers to choose from
                top_performers_genomes = all_genomes[:num_exploit_candidates]
                chosen_parent_genome = random.choice(top_performers_genomes)
                
                parent_reward_dna = chosen_parent_genome['env']['reward_dna'].copy()
                model_to_load_from = chosen_parent_genome['model_path'] + "/Policy"
            else:
                print(f"{YELLOW}Warning: Not enough top performers to copy from for agent {target_model_name}. Continuing with own lineage.{RESET}")


        output_path_next_gen = f"{current_directory}/{target_machine}/Gen{generation_index:04d}/{target_model_name}"
        os.makedirs(output_path_next_gen, exist_ok=True)
        
        next_gen_genome_infos.append({
            "output_path": output_path_next_gen,
            "reward_dna": parent_reward_dna, 
            "model_to_load_from": model_to_load_from,
            "target_model_name": target_model_name
        })

    # Mutation (PBT Explore step)
    progress = min(generation_index / MAX_GENERATIONS, 1.0) 
    current_mutation_ratio = INITIAL_MUTATION_RATIO - (INITIAL_MUTATION_RATIO - FINAL_MUTATION_RATIO) * progress
    current_mutation_strength = INITIAL_MUTATION_STRENGTH - (
            INITIAL_MUTATION_STRENGTH - FINAL_MUTATION_STRENGTH) * progress

    if STABLE_STEPS_START <= generation_index < MAX_GENERATIONS - STABLE_STEPS_END: # Apply mutation within these generation bounds
        num_to_mutate_overall = int(current_mutation_ratio * num_population)
        num_to_mutate_overall = max(1, num_to_mutate_overall) if num_population > 0 and current_mutation_ratio > 0 else 0 # Ensure at least one if ratio > 0
        
        indices_to_consider_mutation = list(range(num_population))
        mutate_indices_selected = random.sample(indices_to_consider_mutation, num_to_mutate_overall)

        for idx in indices_to_consider_mutation:
            agent_info = next_gen_genome_infos[idx]
            if agent_info.get("was_fallback", False): # Skip mutation for fallback agents if they were just initialized
                agent_info["mutated_in_this_step"] = False
                continue

            original_dna_for_mutation = agent_info["reward_dna"].copy()
            
            if idx in mutate_indices_selected:
                reward_dna_to_mutate = agent_info["reward_dna"]
                
                num_gene_mutations = max(1, int(len(reward_dna_to_mutate) * current_mutation_strength))
                if len(reward_dna_to_mutate) > 0 : # Ensure reward_dna is not empty
                    gene_mutation_indices = random.sample(range(len(reward_dna_to_mutate)), min(num_gene_mutations, len(reward_dna_to_mutate)))

                    for gene_idx in gene_mutation_indices:
                        reward_dna_to_mutate[gene_idx] = random.randint(0, MAX_REWARD_GENE_INDEX)
                
                    agent_info["reward_dna_before_mutation"] = original_dna_for_mutation
                    agent_info["mutated_in_this_step"] = True
                else:
                    agent_info["mutated_in_this_step"] = False # Cannot mutate empty DNA
            else:
                agent_info["mutated_in_this_step"] = False
    else: # Not in mutation window, mark all as not mutated
        for agent_info in next_gen_genome_infos:
            agent_info["mutated_in_this_step"] = False


    # Write new genome.json files and prepare models_pool for 'agents' field
    models_pool_next_gen = [] 
    for agent_info in next_gen_genome_infos:
        output_path = agent_info["output_path"]
        
        env_details = {
            'env_name': agent_info["target_model_name"],
            'reward_dna': agent_info["reward_dna"], 
            'agents': [], 
            'load_from_path': agent_info["model_to_load_from"]
        }
        if agent_info.get("mutated_in_this_step", False):
            env_details['reward_dna_before_mutation'] = agent_info["reward_dna_before_mutation"]

        genome_content_to_save = {
            "env": env_details,
            "fitness": "tbd", 
            "model_path": output_path.replace(f"{current_directory}/", "", 1) # Store relative path
        }
        
        with open(os.path.join(output_path, "genome.json"), "w") as f:
            json.dump(genome_content_to_save, f, indent=4)
            
        models_pool_next_gen.append(f"{output_path}/Policy")

    # Populate 'agents' field in all new genome.json files
    for agent_info in next_gen_genome_infos:
        output_path = agent_info["output_path"]
        genome_file_path = os.path.join(output_path, "genome.json")
        
        with open(genome_file_path, "r") as f:
            genome_data = json.load(f)
            
        if models_pool_next_gen: # Ensure pool is not empty
            num_opponents_to_select = min(NUM_AGENTS, len(models_pool_next_gen))
            selected_opponent_agents = random.sample(models_pool_next_gen, num_opponents_to_select)
            # Store agent paths relative to the project root for portability
            relative_opponent_agents = [p.replace(f"{current_directory}/", "", 1) for p in selected_opponent_agents]
            genome_data['env']['agents'] = relative_opponent_agents
        else:
            genome_data['env']['agents'] = []

        with open(genome_file_path, "w") as f:
            json.dump(genome_data, f, indent=4)

    # Add generation check before continuing
    if generation_index >= MAX_GENERATIONS:
        print(f"{GREEN}Reached maximum generations ({MAX_GENERATIONS}). Stopping.{RESET}")
        pbar.close()
        break

    # Update progress bar
    pbar.update(1)
    # generation_index is already incremented for the next generation. No need to increment again here.

# Close the log file at the end of the script
log_file.close()

