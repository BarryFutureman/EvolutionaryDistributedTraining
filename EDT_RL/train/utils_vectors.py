from __future__ import annotations

import gym.envs
import gymnasium
import torch.nn
import torch.optim
from tensordict.nn import TensorDictModule
from torchrl.data.tensor_specs import CategoricalBox
from torchrl.envs import (
    CatFrames,
    DoubleToFloat,
    EndOfLifeTransform,
    EnvCreator,
    ExplorationType,
    GrayScale,
    GymEnv,
    GymWrapper,
    set_gym_backend,
    NoopResetEnv,
    ParallelEnv,
    RenameTransform,
    Resize,
    RewardSum,
    SignTransform,
    StepCounter,
    ToTensorImage,
    TransformedEnv,
    VecNorm,
    UnsqueezeTransform
)
from torchrl.modules import (
    ActorValueOperator,
    ConvNet,
    MLP,
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
)
from model import TransformerModule
import os
from env.wb_env_wrapper import WBGymWrapper, SelfPlayWarehouseBrawl
from env.wb_env import RewardManager, RandomAgent, ConstantAgent, BasedAgent
from reward_functions import dna_to_reward_functions
import random
import shutil
import json


# ====================================================================
# Environment utils
# --------------------------------------------------------------------


class OpponentsCfg:
    def __init__(self, agents: list[str] = None):
        if agents is None:
            agents = []
        from agent import TransformerAgent
        self.agent_to_path = {}
        self.opponents = self.agent_to_path
        self.agents_pool = []
        for agent_path in agents:
            loaded_agent = TransformerAgent(agent_path)
            self.agents_pool.append(loaded_agent)
            self.agent_to_path[loaded_agent] = agent_path

        if not agents:
            base_agents = [ConstantAgent()]
            for base_agent in base_agents:
                self.agents_pool.append(base_agent)
                self.agent_to_path[base_agent] = str(base_agent)

    def validate_probabilities(self) -> None:
        pass

    def process(self) -> None:
        pass

    def on_env_reset(self):
        # Select a randoma agent
        agent = random.choice(self.agents_pool)
        agent_name = self.agent_to_path[agent]

        # If self-play is selected, return the trained model
        print(f'>>> Self play opponent selected {agent_name}')
        opponent = agent

        opponent.get_env_info(self.env)
        return opponent


def make_base_env(**kwargs):
    # Reward manager
    if not kwargs.get("reward_dna"):
        reward_manager = RewardManager()
    else:
        reward_manager = RewardManager(*dna_to_reward_functions(kwargs.get("reward_dna")))

    agents = kwargs.get("agents")
    opponent_cfg = OpponentsCfg(agents)

    with set_gym_backend("gymnasium"):
        env = WBGymWrapper(env=SelfPlayWarehouseBrawl(
            reward_manager=reward_manager,
            opponent_cfg=opponent_cfg
        ), categorical_action_encoding=True)
        env = TransformedEnv(env)
        return env


def make_parallel_env(env_name, num_envs, device, env_kwargs=None, context_length=4, is_test=False):
    if env_kwargs is None:
        env_kwargs = {}
    env = ParallelEnv(
        num_envs,
        EnvCreator(lambda: make_base_env(**env_kwargs)),
        serial_for_single=True,
        device=device,
    )
    env = TransformedEnv(env)
    env.append_transform(UnsqueezeTransform(dim=-2, in_keys=["observation"], out_keys=["observation"]))  # Add new dimension
    env.append_transform(CatFrames(N=context_length, dim=-2, in_keys=["observation"], out_keys=["observation"]))  # Adjust dim for CatFrames
    env.append_transform(RewardSum())
    env.append_transform(StepCounter(max_steps=4500))
    if not is_test:
        env.append_transform(SignTransform(in_keys=["reward"]))

    return env


# ====================================================================
# Model utils
# --------------------------------------------------------------------

def make_ppo_modules(proof_environment, device, model_dir=None, model_kwargs=None):
    # Define input shape
    input_shape = proof_environment.observation_spec["observation"].shape

    # Define distribution class and kwargs
    if isinstance(proof_environment.action_spec.space, CategoricalBox):
        num_outputs = proof_environment.action_spec.space.n
        distribution_class = torch.distributions.Categorical
        distribution_kwargs = {}
    else:  # is ContinuousBox
        num_outputs = proof_environment.action_spec.shape
        distribution_class = TanhNormal
        distribution_kwargs = {
            "low": proof_environment.action_spec.space.low.to(device),
            "high": proof_environment.action_spec.space.high.to(device),
        }

    # Define input keys
    in_keys = ["observation"]

    # Define the transformer models for policy and value networks
    policy_transformer_model = TransformerModule(
        in_features=int(input_shape[-1]),
        out_features=num_outputs,
        device=device,
        load_path=model_dir + "/Policy" if model_dir else None,
        **model_kwargs
    )

    value_transformer_model = TransformerModule(
        in_features=int(input_shape[-1]),
        out_features=1,
        device=device,
        load_path=model_dir + "/Value" if model_dir else None,
        **model_kwargs
    )

    # Define policy module using transformer
    policy_module = TensorDictModule(
        module=policy_transformer_model,
        in_keys=in_keys,
        out_keys=["logits"],
    )

    # Add probabilistic sampling of the actions
    policy_module = ProbabilisticActor(
        policy_module,
        in_keys=["logits"],
        spec=proof_environment.full_action_spec.to(device),
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    # Define value module using transformer
    value_module = ValueOperator(
        module=value_transformer_model,
        in_keys=in_keys,
    )

    # Print model parameter sizes in billions
    total_params = sum(p.numel() for p in policy_transformer_model.parameters()) + \
                   sum(p.numel() for p in value_transformer_model.parameters())
    print(f"Total Model Parameters: {total_params / 1e6:.2f} Million")
    print("Model dtype:", next(policy_transformer_model.parameters()).dtype)
    return policy_module, value_module


def load_ppo_models(proof_environment, device, model_name, save_dir, model_kwargs):
    save_dir_full = save_dir + "/" + model_name if os.path.isdir(save_dir + "/" + model_name) else None
    if os.path.isdir(save_dir_full + "/Policy") and os.path.isdir(save_dir_full + "/Value"):
        pass
    else:
        save_dir_full = None

    policy_module, value_module = make_ppo_modules(
        proof_environment,
        device=device,
        model_dir=save_dir_full,
        model_kwargs=model_kwargs
    )

    # if save_dir_full:
    #     saved_optimizer_state_dict = torch.load(f'{save_dir_full}/optimizer.pt')
    # else:
    #     saved_optimizer_state_dict = None
    saved_optimizer_state_dict = None

    return policy_module, value_module, saved_optimizer_state_dict


def save_ppo_models(actor, critic, optimizer, model_name, save_dir):
    """Saves the actor and critic models, optimizer state, and the model architecture script."""
    save_dir_full = save_dir + "/" + model_name

    policy_module = actor.module
    actor_model = policy_module[0].model
    value_module = critic.module
    critic_model = value_module.model

    actor_save_path = save_dir_full + "/Policy"
    critic_save_path = save_dir_full + "/Value"
    actor_model.save_pretrained(actor_save_path)
    critic_model.save_pretrained(critic_save_path)

    # Copy model architecture script
    modeling_ivy_path = "modeling_ivy.py"  # Assuming it's in the same directory
    shutil.copy(modeling_ivy_path, os.path.join(actor_save_path, "modeling_ivy.py"))
    shutil.copy(modeling_ivy_path, os.path.join(critic_save_path, "modeling_ivy.py"))

    # Update config.json
    def update_config_json(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        config["auto_map"] = {
            "AutoConfig": "modeling_ivy.IvyConfig",
            "AutoModel": "modeling_ivy.Ivy4RL",
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    update_config_json(os.path.join(actor_save_path, "config.json"))
    update_config_json(os.path.join(critic_save_path, "config.json"))

    # # Save optimizer state
    # torch.save(optimizer.state_dict(), f"{save_dir_full}/optimizer.pt")


def make_ppo_models(env_name, device, model_name, save_dir, model_kwargs):
    proof_environment = make_parallel_env(env_name, 1, device=device)
    policy_module, value_module, saved_optimizer_state_dict = load_ppo_models(
        proof_environment,
        device=device,
        model_name=model_name,
        save_dir=save_dir,
        model_kwargs=model_kwargs,
    )

    return policy_module, value_module, saved_optimizer_state_dict


# ====================================================================
# Evaluation utils
# --------------------------------------------------------------------

def eval_model(actor, test_env, num_episodes=3):
    test_rewards = []
    for _ in range(num_episodes):
        td_test = test_env.rollout(
            policy=actor,
            auto_reset=True,
            auto_cast_to_device=True,
            break_when_any_done=True,
            max_steps=10_000_000,
        )
        reward = td_test["next", "episode_reward"][td_test["next", "done"]]
        test_rewards.append(reward.cpu())
    del td_test
    return torch.cat(test_rewards, 0).mean()


if __name__ == '__main__':
    env = make_parallel_env("Hello-v0", num_envs=1, device="cpu")
    print(env.observation_spec)
    quit()
    print(env.transform)  # Print the transform to check the order
