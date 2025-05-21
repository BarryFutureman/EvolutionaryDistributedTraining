from __future__ import annotations

import warnings
import os
import argparse
import json

import torch.optim
import tqdm
import wandb

from tensordict import TensorDict

from torchrl._utils import timeit
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value.advantages import GAE
from torchrl.record.loggers import generate_exp_name, get_logger
from utils_vectors import eval_model, make_parallel_env, make_ppo_models, save_ppo_models


def parse_args():
    parser = argparse.ArgumentParser(description="PPO Training")
    # Model
    parser.add_argument("--model_name", type=str, default="IVY")
    parser.add_argument("--gen", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--context_length", type=int, default=16)
    # Environment
    parser.add_argument("--num_envs", type=int, default=2)
    # Collector
    parser.add_argument("--frames_per_batch", type=int, default=2048)
    parser.add_argument("--total_frames", type=int, default=40000)
    # Logger
    parser.add_argument("--logger_backend", type=str, default="no_wandb")
    parser.add_argument("--project_name", type=str, default="MixRLRun01")
    # Optimizer
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--anneal_lr", type=bool, default=True)
    parser.add_argument("--optim_device", type=str, default=None)
    # Loss
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--mini_batch_size", type=int, default=1024)
    parser.add_argument("--ppo_epochs", type=int, default=6)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_epsilon", type=float, default=0.3)
    parser.add_argument("--anneal_clip_epsilon", type=bool, default=False)
    # anneal_clip_epsilon is hard to recover from after resetting for continue training
    parser.add_argument("--critic_coef", type=float, default=1.0)
    parser.add_argument("--entropy_coef", type=float, default=0.02)
    parser.add_argument("--loss_critic_type", type=str, default="l2")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set device
    device = args.optim_device if args.optim_device else "cuda:0"
    print("CUDA:", torch.cuda.is_available())
    device = torch.device(device)
    print("Device: ", device)

    from transformers.utils import is_flash_attn_2_available
    print(f"flash_attn_2_available: {is_flash_attn_2_available()}")

    # Settings
    total_frames = args.total_frames
    frames_per_batch = args.frames_per_batch
    mini_batch_size = args.mini_batch_size

    # Register gym environment
    load_dir = f"Gen{args.gen-1:04d}"
    genome_json_path = os.path.join(load_dir, f"{args.model_name}/genome.json")
    with open(genome_json_path, 'r') as f:
        genome_json = json.load(f)
    env_kwargs = genome_json["env"]
    env_name = env_kwargs["env_name"]

    # Create models
    save_dir = f"Gen{args.gen:04d}"
    model_kwargs = {"hidden_size": args.hidden_size, 
                    "num_layers": args.num_layers, 
                    "context_length": args.context_length}
    actor, critic, saved_optimizer_state_dict = \
        make_ppo_models(env_name, device=device,
                        model_name=args.model_name, save_dir=load_dir, model_kwargs=model_kwargs)

    # Create collector
    collector = SyncDataCollector(
        create_env_fn=make_parallel_env(env_name, args.num_envs,
                                        device=device, context_length=args.context_length,
                                        env_kwargs=env_kwargs),
        policy=actor,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
        max_frames_per_traj=-1,
    )

    # Create data buffer
    sampler = SamplerWithoutReplacement()
    data_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(
            frames_per_batch, device=device
        ),
        sampler=sampler,
        batch_size=mini_batch_size,
    )

    # Create loss and adv modules
    adv_module = GAE(
        gamma=args.gamma,
        lmbda=args.gae_lambda,
        value_network=critic,
        average_gae=False,
        device=device,
        vectorized=True,
    )
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=args.clip_epsilon,
        loss_critic_type=args.loss_critic_type,
        entropy_coef=args.entropy_coef,
        critic_coef=args.critic_coef,
        normalize_advantage=True,
    )

    # Create optimizers
    def group_optimizers(*optimizers: torch.optim.Optimizer) -> torch.optim.Optimizer:
        """Groups multiple optimizers into a single one.
        All optimizers are expected to have the same type.
        """
        cls = None
        params = []
        for optimizer in optimizers:
            if optimizer is None:
                continue
            if cls is None:
                cls = type(optimizer)
            if cls is not type(optimizer):
                raise ValueError("Cannot group optimizers of different type.")
            params.extend(optimizer.param_groups)
        return cls(params)
    actor_optim = torch.optim.Adam(
        actor.parameters(), lr=torch.tensor(args.lr, device=device), eps=args.eps
    )
    critic_optim = torch.optim.Adam(
        critic.parameters(), lr=torch.tensor(args.lr, device=device), eps=args.eps
    )
    optim = group_optimizers(actor_optim, critic_optim)
    del actor_optim, critic_optim

    if saved_optimizer_state_dict and False:
        optim.load_state_dict(saved_optimizer_state_dict)
        print("\033[36m[Loaded saved_optimizer_state_dict]\033[39m")
    else:
        print("\033[36m[Initialized optimizer]\033[39m")

    # Create logger
    logger = None
    if args.logger_backend == "wandb":
        group_name = f"[{args.model_name}]"
        exp_name = generate_exp_name("PPO", f"gen{args.gen:04d}_{args.model_name}")
        logger = get_logger(
            args.logger_backend,
            logger_name="ppo",
            experiment_name=exp_name,
            wandb_kwargs={
                "config": vars(args),
                "project": args.project_name,
                "group": group_name,
            },
        )

    # Main loop
    collected_frames = 0
    num_network_updates = torch.zeros((), dtype=torch.int64, device=device)
    pbar = tqdm.tqdm(total=total_frames)
    tracked_speed = pbar.format_dict["rate"]
    num_mini_batches = frames_per_batch // mini_batch_size
    total_network_updates = (
        (total_frames // frames_per_batch) * args.ppo_epochs * num_mini_batches
    )

    def update(batch, num_network_updates):
        optim.zero_grad(set_to_none=True)

        # Linearly decrease the learning rate and clip epsilon
        alpha = torch.ones((), device=device)
        if args.anneal_lr:
            alpha = 1 - (num_network_updates / total_network_updates)
            for group in optim.param_groups:
                group["lr"] = args.lr * alpha
        if args.anneal_clip_epsilon:
            loss_module.clip_epsilon.copy_(args.clip_epsilon * alpha)
        num_network_updates = num_network_updates + 1

        # Forward pass PPO loss
        loss = loss_module(batch)
        loss_sum = loss["loss_critic"] + loss["loss_objective"] + loss["loss_entropy"]
        # Backward pass
        loss_sum.backward()
        torch.nn.utils.clip_grad_norm_(
            loss_module.parameters(), max_norm=args.max_grad_norm
        )

        # Update the networks
        optim.step()
        return loss.detach().set("alpha", alpha), num_network_updates

    # extract cfg variables
    cfg_loss_ppo_epochs = args.ppo_epochs
    losses = TensorDict(batch_size=[cfg_loss_ppo_epochs, num_mini_batches])

    def save_checkpoint(actor, critic, checkpoint_dir, step):
        save_ppo_models(actor, critic, model_name=args.model_name, optimizer=optim,
                        save_dir=f"{checkpoint_dir}/{args.model_name}_step={step}")
        print(f"Checkpoint saved at step {step}")

    collector_iter = iter(collector)
    total_iter = len(collector)
    for i in range(total_iter):
        # timeit.printevery(1000, total_iter, erase=True)

        with timeit("collecting"):
            try:
                data = next(collector_iter)
            except StopIteration:
                break

        metrics_to_log = {}
        frames_in_batch = data.numel()
        collected_frames += frames_in_batch
        pbar.update(frames_in_batch)
        if (pbar.n / pbar.total) * 100 < 80:
            tracked_speed = pbar.format_dict["rate"]

        # Get training rewards and episode lengths
        episode_rewards = data["next", "episode_reward"][data["next", "done"]]
        if len(episode_rewards) > 0:
            episode_length = data["next", "step_count"][data["next", "done"]]
            metrics_to_log.update(
                {
                    "train/reward": episode_rewards.mean().item(),
                    "train/episode_length": episode_length.sum().item()
                    / len(episode_length),
                }
            )

        with timeit("training"):
            for j in range(cfg_loss_ppo_epochs):

                # Compute GAE
                with torch.no_grad(), timeit("adv"):
                    data = adv_module(data)
                with timeit("rb - extend"):
                    # Update the data buffer
                    data_reshape = data.reshape(-1)
                    data_buffer.extend(data_reshape)

                for k, batch in enumerate(data_buffer):
                    with timeit("update"):
                        loss, num_network_updates = update(
                            batch, num_network_updates=num_network_updates
                        )
                    loss = loss.clone()
                    num_network_updates = num_network_updates.clone()
                    losses[j, k] = loss.select(
                        "loss_critic", "loss_entropy", "loss_objective"
                    )

        # Get training losses and times
        losses_mean = losses.apply(lambda x: x.float().mean(), batch_size=[])
        for key, value in losses_mean.items():
            metrics_to_log.update({f"train/{key}": value.item()})
        metrics_to_log.update(
            {
                "train/lr": loss["alpha"] * args.lr,
                "train/clip_epsilon": loss["alpha"] * args.clip_epsilon,
            }
        )

        if logger:
            metrics_to_log.update(timeit.todict())
            metrics_to_log["time/speed"] = pbar.format_dict["rate"]
            for key, value in metrics_to_log.items():
                logger.log_scalar(key, value, collected_frames)

        collector.update_policy_weights_()

    collector.shutdown()

    print(f"Saving models to {save_dir}/{args.model_name}")
    save_ppo_models(actor, critic, model_name=args.model_name, optimizer=optim, save_dir=save_dir)
    print("Models saved successfully.")
    
    # Save genome.json with all settings in args
    genome_json = vars(args).copy()
    genome_json["env"] = env_kwargs
    genome_json["speed"] = tracked_speed
    with open(f"{save_dir}/{args.model_name}/genome.json", 'w') as f:
        json.dump(genome_json, f, indent=4)
    print("genome.json saved successfully.")


if __name__ == "__main__":
    main()
