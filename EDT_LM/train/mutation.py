from mutation_datasets import cosmopedia  # changed from smol_smol_talk
import sft

from datasets import concatenate_datasets
import random
import argparse
import json
import time

# Define datasets and their max_shards
DATASET_CONFIGS = [
    {"module": cosmopedia, "max_shards": 4},  # changed from smol_smol_talk
]

DNA_SIZE = 3


def generate_dna():
    return [0 for _ in range(DNA_SIZE)]


def load_mutation_data(dna_seq, tokenizer, cache_dir, batch_multiplier=1):
    """
    dna_seq: list of N integers, where N = sum of max_shards for all datasets
    Returns: (combined_dataset, selected_dataset_names)
    """
    dataset_map = []
    for cfg in DATASET_CONFIGS:
        ds_module = cfg["module"]
        max_shards = cfg["max_shards"]
        for shard_index in range(max_shards):
            seed = int(time.time() * 1000) % (2**32)
            dataset_map.append((ds_module, seed, shard_index, max_shards))
    top_indices = [0]
    datasets = []
    selected_dataset_names = []
    for idx in top_indices:
        ds_module, seed, shard_index, max_shards = dataset_map[idx]
        shard_index = 4
        datasets.append(
            ds_module.load_data(
                tokenizer=tokenizer,
                seed=seed,
                cache_dir=cache_dir,
                subset_index=shard_index,
                num_examples=100 * batch_multiplier
            )
        )
        # Save dataset name as "module_name:shard_index"
        selected_dataset_names.append(f"{ds_module.__name__}:{shard_index}of{max_shards}")
    combined_dataset = concatenate_datasets(datasets)
    return combined_dataset, selected_dataset_names


def run_mutation(model_path, dna_seq, cache_dir, output_path=None):
    if output_path is None:
        output_path = model_path
    # Unpack optimizer and scheduler from prepare_models
    model, tokenizer, optimizer, scheduler = sft.prepare_models(model_path, cache_dir=cache_dir)
    
    # Load the dataset and get selected dataset names
    processed_dataset, selected_dataset_names = load_mutation_data(dna_seq, tokenizer, cache_dir, batch_multiplier=1)
    
    # Run training, passing optimizer and scheduler
    sft.run_training(model, output_path, tokenizer, processed_dataset, f"{output_path}", batch_multiplier=1, optimizer=optimizer, scheduler=scheduler)

    # Save updated genome and selected datasets to genome.json
    genome_path = f"{model_path}/genome.json"
    with open(f"{model_path}/genome.json", 'r') as f:
        genome_data = json.load(f)
    genome_data["dna"] = dna_seq
    genome_data["selected_datasets"] = selected_dataset_names
    genome_data["model_path"] = model_path
    genome_data["mutation_path"] = output_path
    with open(genome_path, 'w') as f:
        json.dump(genome_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="../cache")
    parser.add_argument("--mutation_strength", type=int, default=0)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()
    
    model_path = args.model_path
    output_path = args.output_path if args.output_path is not None else model_path
    
    with open(f"{model_path}/genome.json", 'r') as f:
        genome_data = json.load(f)
    dna = genome_data.get('dna', generate_dna())

    # # Apply mutation if mutation_strength > 0
    # if args.mutation_strength > 0:
    #     n = min(args.mutation_strength, len(dna))
    #     mutate_indices = random.sample(range(len(dna)), n)
    #     for idx in mutate_indices:
    #         dna[idx] = random.randint(0, 3)
    n = min(1, len(dna))
    mutate_indices = random.sample(range(len(dna)), n)
    for idx in mutate_indices:
        dna[idx] = random.randint(0, 3)
    
    run_mutation(model_path, dna, cache_dir=args.cache_dir, output_path=output_path)
