import json
import os

import torch
from transformers import AutoModel, AutoConfig, AutoTokenizer, LlamaForCausalLM, LlamaConfig
import shutil
import argparse
import mutation
import random
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="../cache")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic model weights")
    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    output_dir = f"../{args.output_path}"
    os.makedirs(output_dir, exist_ok=True)

    genome_data = {
        "model_path": args.output_path,
        "fitness": 0.0,
        "dna": mutation.generate_dna(),
    }

    with open(f"{output_dir}/genome.json", 'w') as f:
        json.dump(genome_data, f, indent=4)

    model_path = args.model_path
    if os.path.exists(model_path):
        shutil.copytree(model_path, f"{output_dir}", dirs_exist_ok=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, cache_dir=args.cache_dir)
        config = LlamaConfig(
            vocab_size=49152,
            hidden_size=64,
            intermediate_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=1,
            hidden_act="silu",
            max_position_embeddings=2048,
            initializer_range=0.041666666666666664,
            rms_norm_eps=1e-5,
            use_cache=True,
            pad_token_id=None,
            bos_token_id=1,
            eos_token_id=2,
            pretraining_tp=1,
            tie_word_embeddings=False,
            rope_theta=10000.0,
            rope_scaling=None,
            attention_bias=False,
            attention_dropout=0.0,
            mlp_bias=False,
            head_dim=32,
            torch_dtype="bfloat16"
        )
        model = LlamaForCausalLM(config=config).to(torch.bfloat16)

        print("Model size: ", model.num_parameters())

        model.save_pretrained(f"{output_dir}")
        tokenizer.save_pretrained(f"{output_dir}")
