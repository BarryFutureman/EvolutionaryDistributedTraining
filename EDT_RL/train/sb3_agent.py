from env.wb_env import Agent, SelfPlayWarehouseBrawl
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Optional, Union
from model import TransformerModule
import numpy as np
import torch


class TransformerPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, *args, **kwargs):
        super(TransformerPolicy, self).__init__(observation_space, action_space, *args, **kwargs)
        self.transformer = TransformerModule(
            in_features=observation_space.shape[0],
            out_features=1024,
            device="cpu"
        )
        self.memory = torch.zeros((1, 4, observation_space.shape[0]), device="cpu")

        # Define the action mapping
        self.directions = [
            [0, 0, 0, 0],  # No movement
            [1, 0, 0, 0],  # W
            [0, 1, 0, 0],  # A
            [0, 0, 1, 0],  # S
            [0, 0, 0, 1],  # D
            [1, 1, 0, 0],  # W + A
            [1, 0, 0, 1],  # W + D
            [0, 1, 1, 0],  # A + S
            [0, 0, 1, 1],  # S + D
        ]

        self.actions = [
            [0, 0, 0, 0, 0, 0],  # No action
            [1, 0, 0, 0, 0, 0],  # Jump
            [0, 1, 0, 0, 0, 0],  # Light Attack
            [0, 0, 1, 0, 0, 0],  # Heavy Attack
            [0, 0, 0, 1, 0, 0],  # Pickup/Throw
            [0, 0, 0, 0, 1, 0],  # Dash/Dodge
            [0, 0, 0, 0, 0, 1],  # Taunt
        ]

    def predict(self, obs, state=None, episode_start=None, deterministic=False):
        obs = torch.from_numpy(obs).to(self.transformer.model.device).to(self.transformer.model.dtype)
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(dim=0)

        print(obs.shape, "again", len(obs.shape))

        # Update memory
        self.memory = torch.cat((self.memory[:, 1:], obs.unsqueeze(1)), dim=1)

        model_output = self.transformer(self.memory)
        print(model_output.shape)
        action_index = torch.argmax(model_output, dim=-1)
        print(action_index)

        # probs = torch.softmax(model_output, dim=-1)
        # action_index = torch.multinomial(probs, 1)
        # print(action_index, "softmax multinomial", torch.argmax(model_output, dim=-1), "argmax!!")

        action_out = self._discrete_to_box(int(action_index))

        return action_out, {}

    def _discrete_to_box(self, action_int):
        num_actions = len(self.actions)

        direction_idx = action_int // num_actions
        action_idx = action_int % num_actions

        binary_action = self.directions[direction_idx] + self.actions[action_idx]

        return np.array(binary_action, dtype=np.float32)

    

class TransformerAgent(Agent):
    def __init__(
            self,
            file_path: str,
            # example_argument = 0,
    ):
        super().__init__(file_path)

    def _initialize(self) -> None:
        self.model = PPO(TransformerPolicy, SelfPlayWarehouseBrawl(), verbose=0, device="cpu")
        self.model.policy.transformer.load_model_weights(self.file_path)

        self.model.policy.eval()

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action

