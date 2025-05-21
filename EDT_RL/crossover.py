import torch
import json
import shutil
import os

from typing import Dict
import numpy as np
from transformers import AutoModel, AutoConfig


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


def load_model_from_folder(folder_path: str):
    """Load a Hugging Face transformer model from a folder."""
    config = AutoConfig.from_pretrained(folder_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(folder_path, config=config, trust_remote_code=True)
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


def run_slerp_merge_from_config(merge_config_dict: Dict, merge_output_path: str):
    """Run SLERP-based merging based on a MergeKit-style configuration."""

    slices = merge_config_dict['slices'][0]['sources']
    model_1_path, model_2_path = slices[0]['model'], slices[1]['model']

    model_1 = load_model_from_folder(model_1_path)
    model_2 = load_model_from_folder(model_2_path)

    config_1 = AutoConfig.from_pretrained(model_1_path, trust_remote_code=True)
    config_2 = AutoConfig.from_pretrained(model_2_path, trust_remote_code=True)

    num_layers = min(config_1.num_hidden_layers, config_2.num_hidden_layers)

    # Extract interpolation parameters from the config
    param_t = {param["filter"]: param["value"] for param in merge_config_dict["parameters"]["t"] if "filter" in param}
    global_t = next((param["value"] for param in merge_config_dict["parameters"]["t"] if "filter" not in param), 0.5)

    model_merged = AutoModel.from_config(model_1.config, trust_remote_code=True)  # Create an empty model

    state_dict_1 = model_1.state_dict()
    state_dict_2 = model_2.state_dict()
    merged_state_dict = {}

    for key in state_dict_1.keys():
        if "layer" in key:
            layer_idx = int(key.split(".")[1])  # Extract layer index
            if layer_idx >= num_layers:
                continue

            if "self_attn" in key and "self_attn" in param_t:
                t = interpolate_t(layer_idx, num_layers, param_t["self_attn"])
            elif "mlp" in key and "mlp" in param_t:
                t = interpolate_t(layer_idx, num_layers, param_t["mlp"])
            else:
                t = global_t  # Use global interpolation value if not specified

        else:
            t = global_t  # Use global interpolation for non-layer parameters

        slerp_result = slerp(t, state_dict_1[key], state_dict_2[key])
        merged_state_dict[key] = slerp_result

    model_merged.load_state_dict(merged_state_dict)

    # Save merged model
    model_merged.save_pretrained(merge_output_path)

    torch.cuda.empty_cache()
    print("SLERP merging complete! Model saved at:", merge_output_path)

    return merge_output_path


def run_slerp_merge(p1_folder, p2_folder, output_path):
    with open(f"{p1_folder}/config.json", 'r') as f:
        p1_config = json.load(f)
    with open(f"{p2_folder}/config.json", 'r') as f:
        p2_config = json.load(f)

    num_layers = min(p1_config['num_hidden_layers'], p2_config['num_hidden_layers'])

    self_attn_t_curve = [0, 0.5, 0.3, 0.7, 1]
    mlp_t_curve = [1, 0.5, 0.7, 0.3, 0]

    merge_config_dict = {'slices': [
        {'sources': [{'model': p1_folder, 'layer_range': [0, num_layers]},
                     {'model': p2_folder, 'layer_range': [0, num_layers]}]}],
        'merge_method': 'slerp',
        'base_model': p1_folder,
        'parameters': {'t': [
            {'filter': 'self_attn', 'value': self_attn_t_curve},
            {'filter': 'mlp', 'value': mlp_t_curve},
            {'value': 0.5}
        ]},
        'dtype': 'float32',
        'tokenizer_source': None}

    run_slerp_merge_from_config(merge_config_dict, output_path)
    torch.cuda.empty_cache()
    print("Done!")


def uniform_crossover(dna1, dna2):
    """Perform uniform crossover between two lists of integers."""
    assert len(dna1) == len(dna2), "DNA lengths must be the same for uniform crossover."
    return [dna1[i] if np.random.rand() > 0.5 else dna2[i] for i in range(len(dna1))]


def crossover(g1, g2, output_path):
    g1_policy_path = os.path.join(g1["model_path"], "Policy")
    g2_policy_path = os.path.join(g2["model_path"], "Policy")
    g1_value_path = os.path.join(g1["model_path"], "Value")
    g2_value_path = os.path.join(g2["model_path"], "Value")

    policy_output_path = os.path.join(output_path, "Policy")
    value_output_path = os.path.join(output_path, "Value")

    run_slerp_merge(g1_policy_path, g2_policy_path, policy_output_path)
    run_slerp_merge(g1_value_path, g2_value_path, value_output_path)

    reward_dna_1 = g1["env"]["reward_dna"]
    reward_dna_2 = g2["env"]["reward_dna"]

    new_reward_dna = uniform_crossover(reward_dna_1, reward_dna_2)

    genome_data = {
        "model_path": output_path,
        "env": {
            'env_name': g1["env"]["env_name"],
            'reward_dna': new_reward_dna,
            'agents': []
        },
        "p1": g1,
        "p2": g2
    }

    return genome_data



