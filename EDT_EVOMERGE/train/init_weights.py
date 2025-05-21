import json
import os
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import shutil
import argparse
from mutation import generate_dna
import gc
import torch

def init_weights_main(model_path, output_path, cache_dir="../cache"):
    output_dir = f"{output_path}"
    os.makedirs(output_dir, exist_ok=True)

    genome_data = {
        "model_path": output_path,
        "fitness": 0.0,
        "dna": generate_dna(),
    }

    with open(f"{output_dir}/genome.json", 'w') as f:
        json.dump(genome_data, f, indent=4)

    if os.path.exists(model_path):
        shutil.copytree(model_path, f"{output_dir}", dirs_exist_ok=True)
    else:
        print(f"Model path {model_path} does not exist. Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)

        model.save_pretrained(f"{output_dir}")
        tokenizer.save_pretrained(f"{output_dir}")
        
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="../cache")
    args = parser.parse_args()
    init_weights_main(args.model_path, args.output_path, args.cache_dir)
