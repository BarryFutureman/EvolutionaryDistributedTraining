from torch import nn
from typing import Callable, Dict, List, Sequence, Tuple, Type, Union
import torch
from modeling_ivy import Ivy4RL, IvyConfig


class TransformerModule(nn.Module):
    def __init__(
            self,
            in_features: int | None = None,
            out_features: int | torch.Size = None,
            device: str | None = None,
            load_path: str | None = None,
            hidden_size: int = 32,
            intermediate_size: int = 192,
            num_layers: int = 4,
            context_length: int = 4,
    ):
        super().__init__()
        self.device = device
        if load_path:
            self.load_model_weights(load_path)
        else:
            print("\033[36m[Initializing model]\033[39m")
            self.config = IvyConfig(
                obs_size=int(in_features),
                out_size=int(out_features),  # If don't cast to int we get "not json serializable"
                hidden_size=hidden_size,
                intermediate_size=hidden_size*4,
                num_hidden_layers=num_layers,
                num_attention_heads=4,
                num_key_value_heads=2,
                hidden_act="silu",
                max_position_embeddings=context_length,
                initializer_range=0.02,
                rms_norm_eps=1e-6,
                use_cache=True,
                tie_word_embeddings=False,
                rope_theta=10000.0,
                rope_scaling=None,
                use_sliding_window=False,
                sliding_window=4096,
                max_window_layers=21,
                attention_dropout=0.0,
            )
            self.model = Ivy4RL(
                self.config
            ).to(self.device)

    def load_model_weights(self, load_path):
        print(f"\033[36m[Loading model from {load_path}]\033[39m")
        self.model = Ivy4RL.from_pretrained(load_path).to(self.device)
        self.config = self.model.config

        # Ensure all parameters are set to require gradients
        for param in self.model.parameters():
            param.requires_grad = True

        # Set model to training mode
        self.model.train()

    def forward(self, *inputs: Tuple[torch.Tensor]) -> torch.Tensor:
        if len(inputs) > 1:
            raise NotImplementedError()

        inputs_vectors = inputs[0]

        *batch, T, D = inputs_vectors.shape

        if len(batch) > 1:
            inputs_vectors = inputs_vectors.flatten(0, len(batch) - 1)

        # # Step 1: Reverse the time dimension using torch.flip
        # reversed_inputs = torch.flip(inputs_vectors, dims=[1])
        #
        # # Step 2: Slice every 4th frame from the reversed tensor
        # backward_sampled = reversed_inputs[:, ::4, :]
        #
        # # Step 3: Reverse back to restore chronological order (optional if needed)
        # inputs_vectors = torch.flip(backward_sampled, dims=[1])

        # Keep the last 64 frames and slice the rest
        if T > 64:
            inputs_vectors = torch.cat((inputs_vectors[:, :-128:16, :], inputs_vectors[:, -128:, :]), dim=1)

        logits = self.model(
            inputs_embeds=inputs_vectors,
            use_cache=False,
        )

        if len(batch) > 1:
            logits = logits.unflatten(0, batch)

        return logits


if __name__ == '__main__':
    # t_m = TransformerModule(in_features=30, out_features=40, hidden_size=128, num_layers=18, context_length=1024)
    # t_m = TransformerModule(in_features=30, out_features=40, hidden_size=96, num_layers=8, context_length=1024)
    t_m = TransformerModule(in_features=30, out_features=40, hidden_size=64, intermediate_size=192, num_layers=8, context_length=1024)
    print(t_m)
    print(f"{t_m.model.num_parameters()  / 1e6:.2f} Million Parameters")
