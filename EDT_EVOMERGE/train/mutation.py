from mutation_datasets import r1_distill, opus_instruct, smoltalk
from sft import prepare_models, run_training

from datasets import concatenate_datasets
import random
import argparse
import json
import time

# List of dataset modules
DATASETS = [
    r1_distill,
    opus_instruct,
    smoltalk,
    # Add more datasets here as needed
]
TOP_K = 1

DNA_SIZE = len(DATASETS)

def generate_dna():
    return [random.randint(1, 10) for _ in range(DNA_SIZE)]

def load_mutation_data(dna_seq, tokenizer, cache_dir, batch_size=16):
    """
    dna_seq: list of N integers, where N = number of datasets
    Returns: (combined_dataset, selected_dataset_names)
    """
    top_indices = sorted(range(len(dna_seq)), key=lambda i: dna_seq[i], reverse=True)[:TOP_K]
    datasets = []
    selected_dataset_names = []
    for idx in top_indices:
        ds_module = DATASETS[idx]
        seed = int(time.time() * 1000) % (2**32)
        datasets.append(
            ds_module.load_data(
                tokenizer=tokenizer,
                seed=seed,
                cache_dir=cache_dir,
                num_examples=100,
                batch_size=batch_size
            )
        )
        selected_dataset_names.append(f"{ds_module.__name__}")
    combined_dataset = concatenate_datasets(datasets)
    return combined_dataset, selected_dataset_names

def run_mutation(model_path, dna_seq, cache_dir, batch_size=16):
    model, tokenizer = prepare_models(model_path, cache_dir=cache_dir)
    
    # Load the dataset and get selected dataset names
    processed_dataset, selected_dataset_names = load_mutation_data(dna_seq, tokenizer, cache_dir, batch_size=batch_size)
    
    # Run training
    run_training(model, model_path, tokenizer, processed_dataset, f"{model_path}", cache_dir=cache_dir, batch_size=batch_size)

    # Save updated genome and selected datasets to genome.json
    genome_path = f"{model_path}/genome.json"
    with open(genome_path, 'r') as f:
        genome_data = json.load(f)
    genome_data["dna"] = dna_seq
    genome_data["selected_datasets"] = selected_dataset_names
    with open(genome_path, 'w') as f:
        json.dump(genome_data, f, indent=4)

def mutation_main(model_path, mutation_strength=0, cache_dir="../cache", batch_size=16):
    with open(f"{model_path}/genome.json", 'r') as f:
        genome_data = json.load(f)
    dna = genome_data["dna"]

    # Apply mutation if mutation_strength > 0
    if mutation_strength > 0:
        dna = [
            random.randint(-mutation_strength, mutation_strength)
            for d in dna
        ]

    run_mutation(model_path, dna, cache_dir=cache_dir, batch_size=batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="../cache")
    parser.add_argument("--mutation_strength", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    
    mutation_main(
        model_path=args.model_path,
        mutation_strength=args.mutation_strength,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size
    )
