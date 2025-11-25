"""
PPO Training Script - SINGLE EXPERIMENT (FIXED)
================================================

This script runs a single experiment with 1 configuration and 1 seed.
FIXED VERSION - Corrected critical training issues:
1. Clip ratio increased to 0.2 (from 0.1) - standard PPO value
2. GAE Lambda increased to 0.95 (from 0.90) - better credit assignment
3. Seed only used for initial reset, not episode resets - ensures diversity
4. Frame stacking (4 frames) added for temporal information (ball velocity, paddle movement)

CONFIGURATION:
============================================
HYPERPARAMETERS:
- Learning Rate: 1e-4
- Entropy Coefficient: 0.01
- Clip Ratio: 0.2 (FIXED - was 0.1)
- GAE Lambda: 0.95 (FIXED - was 0.90)
- Gamma (Discount): 0.99
- Epochs per Update: 5
- Total Episodes: 20000
- Batch Size: 128
- Rollout Steps: 2048
- Seed: 0 (only for initial environment setup)
- Frame Stack: 4 frames (ADDED - provides temporal information)

TOTAL RUNS: 1
"""

import json
import ale_py
import os
import numpy as np
from datetime import datetime
import time

import torch
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, FrameStackObservation
from ppo_pong import PPOTrainer


# ============================================================================
# CONFIGURATION
# ============================================================================


def create_config() -> dict:
    """Create the configuration dictionary for single experiment."""
    return {
        "algorithm": "PPO",
        "environment": "ALE/Pong-v5",
        "training": {
            "total_episodes": 20000,
            "rollout_steps": 2048,
            "epochs_per_update": 5,
            "batch_size": 128,
        },
        "network": {
            "hidden_sizes": [512],
            "activation": "relu",
            "shared_extractor": True,
        },
        "hyperparameters": {
            "learning_rate": 1e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,  # FIXED: was 0.90, now 0.95 for better credit assignment
            "clip_ratio": 0.2,   # FIXED: was 0.1, now 0.2 (standard PPO value)
            "entropy_coeff": 0.01,
            "value_coeff": 0.5,
            "max_grad_norm": 0.5,
        },
        "exploration": {"type": "entropy", "entropy_schedule": "constant"},
    }


# ============================================================================
# SINGLE EXPERIMENT RUNNER
# ============================================================================


def run_single_experiment(output_dir: str = "results_single_20000_new_6action_4frame", seed: int = 0):
    """Run one training experiment."""
    
    # Create output directories
    logs_dir = os.path.join(output_dir, "logs")
    configs_dir = os.path.join(output_dir, "configs")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(configs_dir, exist_ok=True)
    
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    config = create_config()
    config_name = f"ppo_lr1e4_entropy0.01_single"

    print("\n" + "=" * 70)
    print("PPO PONG - SINGLE EXPERIMENT")
    print("=" * 70)
    print(f"\nConfiguration: {config_name}")
    print(f"Learning Rate: {config['hyperparameters']['learning_rate']:.0e}")
    print(f"Entropy Coeff: {config['hyperparameters']['entropy_coeff']:.2f}")
    print(f"Clip Ratio: {config['hyperparameters']['clip_ratio']:.2f}")
    print(f"GAE Lambda: {config['hyperparameters']['gae_lambda']:.2f}")
    print(f"Total Episodes: {config['training']['total_episodes']}")
    print(f"Seed: {seed}")
    print("=" * 70 + "\n")

    # Save config
    config_path = os.path.join(configs_dir, f"{config_name}.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to: {config_path}")

    # Create environment
    env = gym.make(config["environment"], render_mode=None)
    env = GrayscaleObservation(env)
    env = FrameStackObservation(env, stack_size=4)  # Stack 4 frames for temporal information

    # Record start time
    start_time = time.time()
    print(f"\nStarting training at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Train
    trainer = PPOTrainer(env, config, seed=seed)
    metrics = trainer.train(
        total_episodes=config["training"]["total_episodes"], log_interval=100
    )

    # Compute elapsed time
    end_time = time.time()
    gpu_time_seconds = end_time - start_time
    gpu_time_minutes = gpu_time_seconds / 60.0

    env.close()

    print(f"\n{'='*70}")
    print(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {gpu_time_minutes:.2f} minutes ({gpu_time_seconds:.1f} seconds)")
    print(f"{'='*70}\n")

    # Save model
    model_path = os.path.join(logs_dir, f"{config_name}_seed{seed}_model.pt")
    trainer.save_model(model_path)
    print(f"Model saved to: {model_path}")

    # Save episode-level CSV
    import csv

    csv_path = os.path.join(logs_dir, f"{config_name}_seed{seed}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "episode_length"])
        for ep, (reward, length) in enumerate(
            zip(metrics["episode_rewards"], metrics["episode_lengths"])
        ):
            writer.writerow([ep, reward, length])
    print(f"Episode data saved to: {csv_path}")

    # Save update-level metrics
    if "entropy_history" in metrics:
        updates_csv_path = os.path.join(
            logs_dir, f"{config_name}_seed{seed}_updates.csv"
        )
        with open(updates_csv_path, "w", newline="") as f_updates:
            writer_u = csv.writer(f_updates)
            writer_u.writerow(
                [
                    "update_idx",
                    "episodes",
                    "entropy_mean",
                    "clip_fraction_mean",
                    "policy_loss_mean",
                    "value_loss_mean",
                ]
            )
            for idx, (episodes_done, ent, clip_frac, pl, vl) in enumerate(
                zip(
                    metrics["update_episodes"],
                    metrics["entropy_history"],
                    metrics["clip_fraction_history"],
                    metrics["policy_loss_history"],
                    metrics["value_loss_history"],
                )
            ):
                writer_u.writerow([idx, episodes_done, ent, clip_frac, pl, vl])
        print(f"Update data saved to: {updates_csv_path}")

    # Print summary statistics
    final_reward = np.mean(metrics["episode_rewards"][-100:])
    first_100_reward = np.mean(metrics["episode_rewards"][:100])
    
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Total Episodes: {len(metrics['episode_rewards'])}")
    print(f"First 100 Episodes Avg Reward: {first_100_reward:.2f}")
    print(f"Last 100 Episodes Avg Reward: {final_reward:.2f}")
    print(f"Improvement: {final_reward - first_100_reward:+.2f}")
    print(f"Best Episode Reward: {max(metrics['episode_rewards']):.2f}")
    print(f"Worst Episode Reward: {min(metrics['episode_rewards']):.2f}")
    print("=" * 70 + "\n")

    # Save summary
    summary_path = os.path.join(logs_dir, "experiment_summary.txt")
    with open(summary_path, "w") as f:
        f.write("PPO PONG - SINGLE EXPERIMENT SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Configuration: {config_name}\n")
        f.write(f"Seed: {seed}\n\n")
        f.write("HYPERPARAMETERS:\n")
        f.write(f"  Learning Rate: {config['hyperparameters']['learning_rate']:.0e}\n")
        f.write(f"  Entropy Coeff: {config['hyperparameters']['entropy_coeff']:.2f}\n")
        f.write(f"  Clip Ratio: {config['hyperparameters']['clip_ratio']:.2f}\n")
        f.write(f"  GAE Lambda: {config['hyperparameters']['gae_lambda']:.2f}\n")
        f.write(f"  Gamma: {config['hyperparameters']['gamma']:.2f}\n")
        f.write(f"  Epochs per Update: {config['training']['epochs_per_update']}\n")
        f.write(f"  Batch Size: {config['training']['batch_size']}\n")
        f.write(f"  Rollout Steps: {config['training']['rollout_steps']}\n\n")
        f.write("RESULTS:\n")
        f.write(f"  Total Episodes: {len(metrics['episode_rewards'])}\n")
        f.write(f"  First 100 Episodes Avg: {first_100_reward:.2f}\n")
        f.write(f"  Last 100 Episodes Avg: {final_reward:.2f}\n")
        f.write(f"  Improvement: {final_reward - first_100_reward:+.2f}\n")
        f.write(f"  Training Time: {gpu_time_minutes:.2f} minutes\n")

    print(f"Summary saved to: {summary_path}\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    run_single_experiment(output_dir="results_single", seed=0)
