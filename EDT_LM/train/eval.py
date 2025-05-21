import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import argparse
import json

import sft

# --- Argparse for configurable parameters ---
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-0.6B", help="Model name or path")
parser.add_argument("--genome_path", type=str, default="genome.json")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
parser.add_argument("--cache_dir", type=str, default="cache", help="Cache directory for datasets and models")
args = parser.parse_args()

MODEL_NAME = args.model_path
SEED = 69
BATCH_SIZE = args.batch_size
CACHE_DIR = args.cache_dir

random.seed(SEED)
torch.manual_seed(SEED)

model, tokenizer, _, _ = sft.prepare_models(MODEL_NAME, cache_dir=CACHE_DIR)

# Load validation data
from mutation_datasets import cosmopedia

val_dataset = cosmopedia.load_data(
    tokenizer=tokenizer,
    seed=SEED,
    subset_index=4,
    cache_dir=CACHE_DIR,
    num_examples=40,
    split="val"
)

# Evaluate before training
print("Evaluating...")
eval_loss = sft.evaluate_model(model, tokenizer, val_dataset, eval_batch_size=BATCH_SIZE)

# --- Write fitness to genome.json ---
try:
    with open(args.genome_path, "r") as f:
        genome = json.load(f)
except Exception:
    genome = {}

genome["fitness"] = 1 / max(eval_loss, 0.04)

with open(args.genome_path, "w") as f:
    json.dump(genome, f, indent=4)
