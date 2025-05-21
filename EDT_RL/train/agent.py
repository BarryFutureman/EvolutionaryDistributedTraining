from env.wb_env import Agent, SelfPlayWarehouseBrawl
from typing import Optional, Union
from utils_vectors import ProbabilisticActor, TensorDictModule, TransformerModule, make_parallel_env
import numpy as np
import torch


class TransformerAgent(Agent):
    def __init__(
            self,
            file_path: str,
            # example_argument = 0,
    ):
        super().__init__(file_path)

        self.device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = None

        proof_environment = make_parallel_env("yeah", 1, device=self.device)
        self.full_act_spec = proof_environment.full_action_spec
        del proof_environment

        # Define the action mapping
        self.directions = [
            [1, 0, 0, 0],  # W
            [0, 1, 0, 0],  # A
            [0, 0, 1, 0],  # S
            [0, 0, 0, 1],  # D
            [1, 1, 0, 0],  # W + A
            [1, 0, 0, 1],  # W + D
            [0, 1, 1, 0],  # A + S
            [0, 0, 1, 1],  # S + D
        ]

        # ["W", "A", "S", "D", "space", 'h', 'l', 'j', 'k', 'g']

        self.actions = [
            [0, 0, 0, 0, 0, 0],  # No action
            [1, 0, 0, 0, 0, 0],  # Jump
            # [0, 1, 0, 0, 0, 0],  # Pickup/Throw
            [0, 0, 1, 0, 0, 0],  # Dodge
            [0, 0, 0, 1, 0, 0],  # Light Attack
            [0, 0, 0, 0, 1, 0],  # Heavy Attack
            # [0, 0, 0, 0, 0, 1],  # Taunt (Doesn't do anything?)
        ]

        self.transformer = None

    def _initialize(self) -> None:
        if self.transformer is None:
            # save_dir = "/".join(self.file_path.split("/")[:-2])
            # model_name = self.file_path.split("/")[-2]

            # Define the transformer models for policy and value networks
            policy_transformer_model = TransformerModule(
                in_features=30,
                out_features=40,
                device=str(self.device),
                load_path=self.file_path,
            )

            # Define policy module using transformer
            policy_module = TensorDictModule(
                module=policy_transformer_model,
                in_keys=["observation"],
                out_keys=["logits"],
            )


            policy_module = ProbabilisticActor(
                policy_module,
                in_keys=["logits"],
                spec=self.full_act_spec.to(self.device),
                distribution_class=torch.distributions.Categorical,
                distribution_kwargs={},
                return_log_prob=True,
            )

            self.transformer = policy_module
            self.transformer.eval()
    
        policy_module = self.transformer.module
        transformer_model = policy_module[0].model

        self.memory = torch.zeros((1, transformer_model.config.max_position_embeddings, 30), device=self.device)

    def _discrete_to_box(self, action_int):
        num_actions = len(self.actions)

        direction_idx = action_int // num_actions
        action_idx = action_int % num_actions

        binary_action = self.directions[direction_idx] + self.actions[action_idx]

        return np.array(binary_action, dtype=np.float32)

    def predict(self, obs):
        # print(np.round(obs, 2).tolist()[:15], np.round(obs, 2).tolist()[15:])

        policy_module = self.transformer.module
        transformer_model = policy_module[0].model

        obs = torch.from_numpy(obs).to(self.device).to(transformer_model.dtype)
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(dim=0)

        # Update memory
        self.memory = torch.cat((self.memory[:, 1:], obs.unsqueeze(1)), dim=1)

        with torch.no_grad():
            action_index = self.transformer(self.memory)[1]

        action_out = self._discrete_to_box(int(action_index))

        # print(action_out)

        return action_out
