import torch
import json
import shutil
import os
import gc
import threading

from typing import Dict
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch.optim as optim


def slerp(t, v0, v1, DOT_THRESHOLD=0.9995, eps=1e-8):
    """Perform SLERP (Spherical Linear Interpolation) between two tensors."""
    is_torch = False
    if not isinstance(v0, np.ndarray):
        is_torch = True
        v0 = v0.detach().cpu().float().numpy()
    if not isinstance(v1, np.ndarray):
        is_torch = True
        v1 = v1.detach().cpu().float().numpy()

    v0_copy = np.copy(v0)
    v1_copy = np.copy(v1)

    v0 = normalize(v0, eps)
    v1 = normalize(v1, eps)

    dot = np.sum(v0 * v1)

    if np.abs(dot) > DOT_THRESHOLD:
        res = lerp(t, v0_copy, v1_copy)
        return maybe_torch(res, is_torch)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)

    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    res = s0 * v0_copy + s1 * v1_copy

    return maybe_torch(res, is_torch)


def lerp(t, v0, v1):
    return (1 - t) * v0 + t * v1


def maybe_torch(v, is_torch):
    if is_torch:
        return torch.from_numpy(v)
    return v


def normalize(v, eps):
    norm_v = np.linalg.norm(v)
    if norm_v > eps:
        v = v / norm_v
    return v


def load_model_from_path(folder_path: str):
    config = AutoConfig.from_pretrained(folder_path, trust_remote_code=True, cache_dir="cache")
    model = AutoModelForCausalLM.from_pretrained(folder_path, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True, cache_dir="cache")
    return model


def interpolate_t(layer_idx, num_layers, t_curve):
    """Interpolate t value for the given layer index based on the t_curve."""
    if layer_idx < 0:
        return t_curve[0]
    if layer_idx >= num_layers - 1:
        return t_curve[-1]
    position = layer_idx / (num_layers - 1) * (len(t_curve) - 1)
    lower_idx = int(position)
    upper_idx = min(lower_idx + 1, len(t_curve) - 1)
    lower_t = t_curve[lower_idx]
    upper_t = t_curve[upper_idx]
    return lerp(position - lower_idx, lower_t, upper_t)


class LazyTensorLoader:
    def __init__(self, model, device="cpu"):
        self.model = model
        self.state_dict = None
        self.lock = threading.Lock()
        self.device = device

    def get_tensor(self, key):
        with self.lock:
            if self.state_dict is None:
                self.state_dict = self.model.state_dict()
            return self.state_dict[key].to(self.device)

    def flush(self):
        with self.lock:
            self.state_dict = None


def run_slerp_merge_from_config(
    merge_config_dict: Dict, model_1, model_2, config_1, config_2, merge_output_path, base_model, device=None
):
    num_layers = min(config_1.num_hidden_layers, config_2.num_hidden_layers)
    param_t = {param["filter"]: param["value"] for param in merge_config_dict["parameters"]["t"] if "filter" in param}
    global_t = next((param["value"] for param in merge_config_dict["parameters"]["t"] if "filter" not in param), 0.5)
    model_merged = base_model

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    loader_1 = LazyTensorLoader(model_1, device=device)
    loader_2 = LazyTensorLoader(model_2, device=device)
    merged_state_dict = {}

    keys = list(model_1.state_dict().keys())

    for key in tqdm(keys, desc="SLERP merging (layer by layer)"):
        if "layer" in key:
            layer_idx = int(key.split(".")[1])
            if layer_idx >= num_layers:
                continue
            if "self_attn" in key and "self_attn" in param_t:
                t = interpolate_t(layer_idx, num_layers, param_t["self_attn"])
            elif "mlp" in key and "mlp" in param_t:
                t = interpolate_t(layer_idx, num_layers, param_t["mlp"])
            else:
                t = global_t
        else:
            t = global_t
        tensor_1 = loader_1.get_tensor(key)
        tensor_2 = loader_2.get_tensor(key)
        merged_state_dict[key] = slerp(t, tensor_1, tensor_2).cpu()

    loader_1.flush()
    loader_2.flush()
    del loader_1, loader_2
    gc.collect()
    model_merged.model.load_state_dict(merged_state_dict)
    model_merged.save_pretrained(merge_output_path)
    torch.cuda.empty_cache()
    print("SLERP merging complete! Model saved at:", merge_output_path)
    return merge_output_path


def run_linear_merge_5050(model_1, model_2, config_1, config_2, merge_output_path):
    """Linear merge with t=0.5 for all parameters."""
    model_merged = AutoModelForCausalLM.from_config(model_1.config)
    state_dict_1 = model_1.state_dict()
    state_dict_2 = model_2.state_dict()
    merged_state_dict = {}

    for key in tqdm(state_dict_1.keys(), desc="Linear 50-50 merging"):
        merged_state_dict[key] = lerp(0.5, state_dict_1[key], state_dict_2[key])

    model_merged.load_state_dict(merged_state_dict)
    # model_merged.save_pretrained(merge_output_path)

    return model_merged


def run_sgd(model_1, model_2, base_model, output_path, model1_path, model2_path, lr=0.7, momentum=0.0, nesterov=False):
    params1 = list(model_1.parameters())
    params2 = list(model_2.parameters())
    base_params = list(base_model.parameters())
    num_models = 2

    delta_accumulator = []
    for base_param in base_params:
        delta_accumulator.append(torch.zeros_like(base_param.data))

    for i, (base_param, p1, p2) in enumerate(zip(base_params, params1, params2)):
        delta = ((p1.data - base_param.data) + (p2.data - base_param.data)) / num_models
        delta_accumulator[i] += delta

    for param, avg_delta in zip(base_params, delta_accumulator):
        param.grad = -avg_delta

    optim_path1 = os.path.join(model1_path, "outer_optim.pt")
    optim_path2 = os.path.join(model2_path, "outer_optim.pt")
    optimizer = optim.SGD(base_params, lr=lr, momentum=momentum, nesterov=nesterov)

    merged_optim_state = None
    if os.path.exists(optim_path1) and os.path.exists(optim_path2):
        print(f"Merging outer optimizers: {optim_path1} and {optim_path2}")
        state1 = torch.load(optim_path1, map_location="cpu")
        state2 = torch.load(optim_path2, map_location="cpu")
        merged_optim_state = {}
        # Merge state dicts by averaging tensors for matching keys
        for k in state1.keys():
            if k in state2:
                if isinstance(state1[k], dict) and isinstance(state2[k], dict):
                    merged_optim_state[k] = {}
                    for sk in state1[k]:
                        if sk in state2[k]:
                            v1, v2 = state1[k][sk], state2[k][sk]
                            if torch.is_tensor(v1) and torch.is_tensor(v2):
                                merged_optim_state[k][sk] = (v1 + v2) / 2
                            else:
                                merged_optim_state[k][sk] = v1  # fallback to first
                        else:
                            merged_optim_state[k][sk] = state1[k][sk]
                    for sk in state2[k]:
                        if sk not in merged_optim_state[k]:
                            merged_optim_state[k][sk] = state2[k][sk]
                elif torch.is_tensor(state1[k]) and torch.is_tensor(state2[k]):
                    merged_optim_state[k] = (state1[k] + state2[k]) / 2
                else:
                    merged_optim_state[k] = state1[k]
            else:
                merged_optim_state[k] = state1[k]
        for k in state2.keys():
            if k not in merged_optim_state:
                merged_optim_state[k] = state2[k]
        optimizer.load_state_dict(merged_optim_state)
    elif os.path.exists(optim_path1):
        print(f"Loading outer optimizer: {optim_path1}")
        optimizer.load_state_dict(torch.load(optim_path1, map_location="cpu"))
    elif os.path.exists(optim_path2):
        print(f"Loading outer optimizer: {optim_path2}")
        optimizer.load_state_dict(torch.load(optim_path2, map_location="cpu"))
    elif not "0000" in optim_path1:
        raise NotImplementedError(f"What, no {optim_path1} or {optim_path2}?")

    optimizer.step()
    optimizer.zero_grad()
    optim_out_path = os.path.join(output_path, "outer_optim.pt")
    torch.save(optimizer.state_dict(), optim_out_path)

    base_model.save_pretrained(output_path)
    print("SGD merge complete! Model saved at:", output_path)
    torch.cuda.empty_cache()
    return output_path


def crossover_main(model1_path, model2_path, output_path):
    # Load genome.json for each parent
    p1_genome_path = os.path.join(model1_path, "genome.json")
    p2_genome_path = os.path.join(model2_path, "genome.json")
    with open(p1_genome_path, "r") as f:
        p1_genome = json.load(f)
    with open(p2_genome_path, "r") as f:
        p2_genome = json.load(f)

    model1_base_path = model1_path
    model2_base_path = model2_path
    model1_path = p1_genome.get("mutation_path")
    model2_path = p2_genome.get("mutation_path")

    # Load current models (after mutation)
    model_1 = load_model_from_path(model1_path)
    model_2 = load_model_from_path(model2_path)
    model1_base = load_model_from_path(model1_base_path)
    model2_base = load_model_from_path(model2_base_path)

    base_model = run_linear_merge_5050(model1_base, model2_base, config_1=None, config_2=None, merge_output_path=output_path)

    tokenizer = AutoTokenizer.from_pretrained(model1_path, trust_remote_code=True, cache_dir="cache")
    tokenizer.save_pretrained(output_path)

    # Carry over .pt files from parents if they exist
    for fname in ["optimizer.pt", "scheduler.pt"]:
        p1_file = os.path.join(model1_path, fname)
        p2_file = os.path.join(model2_path, fname)
        out_file = os.path.join(output_path, fname)
        if os.path.exists(p1_file):
            shutil.copy(p1_file, out_file)
        elif os.path.exists(p2_file):
            shutil.copy(p2_file, out_file)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Save genome config
    with open(os.path.join(output_path, "genome.json"), 'w') as genome_file:
        p1_genome.pop("p1", None)
        p1_genome.pop("p2", None)
        p2_genome.pop("p1", None)
        p2_genome.pop("p2", None)

        # --- DNA-based hyperparameter selection ---
        dna = uniform_dna_crossover(p1_genome["dna"], p2_genome["dna"])
        lr_settings = [0.7, 0.8, 0.9, 1.0]
        momentum_settings = [0.7, 0.9, 0.99, 2.00]
        nesterov_settings = [False, False, True, True]
        idx_lr = dna[0] if len(dna) > 0 else 2
        idx_momentum = dna[1] if len(dna) > 1 else 2
        idx_nesterov = dna[2] if len(dna) > 2 else 2
        lr = lr_settings[0]
        momentum = momentum_settings[1]
        nesterov = nesterov_settings[-1]
        # -----------------------------------------

        genome_data = {
            "fitness": 0.0,
            "model_path": output_path,
            "dna": dna,
            "p1": p1_genome,
            "p2": p2_genome,
        }
        json.dump(genome_data, genome_file, indent=4)

    # Run SGD merge (momentum and lr can be parameterized as needed)
    run_sgd(
        model_1, model_2, base_model, output_path, model1_base_path, model2_base_path,
        lr=lr, momentum=momentum, nesterov=nesterov
    )

    print("Done!")
    del model_1, model_2, model1_base
    gc.collect()
    torch.cuda.empty_cache()


def uniform_dna_crossover(dna1, dna2):
    """Perform uniform crossover between two lists of integers."""
    assert len(dna1) == len(dna2), "DNA lengths must be the same for uniform crossover."
    return [dna1[i] if np.random.rand() > 0.5 else dna2[i] for i in range(len(dna1))]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SGD merge two Hugging Face models using base models from genome.")
    parser.add_argument("--model1_path", type=str, required=True)
    parser.add_argument("--model2_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="crossover_result", required=False)
    args = parser.parse_args()

    crossover_main(args.model1_path, args.model2_path, args.output_path)
