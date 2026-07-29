#!/usr/bin/env python3
"""
train_rl.py — Vollständiges RL-Training mit W&B + OpenPipe + Checkpointing
===========================================================================
Trainiert Q-Learning auf GridWorld-Varianten und Policy Gradient.
Mit W&B Experiment Tracking, OpenPipe Fine-Tuning Logging, und Checkpointing.

Usage:
    python train_rl.py                    # Standard-Training
    python train_rl.py --env cliff        # Cliff Walking
    python train_rl.py --env obstacles    # GridWorld mit Hindernissen
    python train_rl.py --algo pg          # Nur Policy Gradient
    python train_rl.py --episodes 2000    # Mehr Episoden
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path

from rl_agent import (
    GridWorld, QLearning, PolicyGradient, DQNAgent,
    setup_tracking, get_sweep_config, OpenPipeLogger, WandBLogger,
    log_model_artifact, log_predictions_table,
    WANDB_AVAILABLE, OPENPIPE_AVAILABLE, TORCH_AVAILABLE
)

# ── Konfiguration ──────────────────────────────────────────────
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)


def train_q_learning(env: GridWorld, config: dict, wandb_run=None,
                     op_logger=None) -> QLearning:
    """Trainiert Q-Learning auf der gegebenen Umgebung."""
    print(f"\n{'='*60}")
    print(f"  Q-Learning: {config.get('env_name', 'GridWorld')}")
    print(f"{'='*60}")

    agent = QLearning(
        env,
        lr=config.get("lr", 0.1),
        gamma=config.get("gamma", 0.99),
        epsilon=config.get("epsilon_start", 0.3),
        wandb_run=wandb_run,
    )

    episodes = config.get("episodes", 500)
    rewards = agent.train(episodes=episodes)

    # Ergebnisse
    avg_last_10 = np.mean(rewards[-10:])
    avg_last_100 = np.mean(rewards[-100:])
    print(f"\n  Episoden: {episodes}")
    print(f"  Avg Reward (last 10):  {avg_last_10:.3f}")
    print(f"  Avg Reward (last 100): {avg_last_100:.3f}")

    # Policy visualisieren
    arrows = {0: "↑", 1: "→", 2: "↓", 3: "←", -1: "🎯"}
    policy = agent.get_policy()
    print("\n  Gelernte Policy:")
    for r in range(env.size):
        row = "  " + " ".join(arrows[policy[r, c]] for c in range(env.size))
        print(row)

    # Checkpoint speichern
    ckpt_path = CHECKPOINT_DIR / f"q_learning_{config.get('env_name', 'grid')}.npz"
    agent.save_checkpoint(str(ckpt_path))

    # W&B Artifact loggen
    if wandb_run:
        log_model_artifact(wandb_run, str(ckpt_path),
                          f"q-learning-{config.get('env_name', 'grid')}",
                          metadata={"algorithm": "q-learning",
                                    "avg_reward_10": float(avg_last_10),
                                    "avg_reward_100": float(avg_last_100)})

    # OpenPipe Logging
    if op_logger:
        op_logger.log_episode({
            "algorithm": "q-learning",
            "env": config.get("env_name", "gridworld"),
            "episodes": episodes,
            "final_avg_reward_10": float(avg_last_10),
            "final_avg_reward_100": float(avg_last_100),
            "lr": config.get("lr", 0.1),
            "gamma": config.get("gamma", 0.99),
        })

    return agent


def train_policy_gradient(config: dict, wandb_run=None,
                          op_logger=None) -> PolicyGradient:
    """Trainiert Policy Gradient (REINFORCE)."""
    print(f"\n{'='*60}")
    print(f"  Policy Gradient (REINFORCE)")
    print(f"{'='*60}")

    pg = PolicyGradient(
        state_dim=config.get("state_dim", 4),
        action_dim=config.get("action_dim", 2),
        lr=config.get("lr", 0.01),
        gamma=config.get("gamma", 0.99),
        wandb_run=wandb_run,
    )

    episodes = config.get("episodes", 200)
    total_rewards = []

    for ep in range(episodes):
        state = np.random.randn(config.get("state_dim", 4)) * 0.1
        episode = []
        for _ in range(config.get("max_steps", 100)):
            action, _ = pg.sample_action(state)
            reward = 1.0 if action == 0 else -0.1
            episode.append((state, action, reward))
            state = np.random.randn(config.get("state_dim", 4)) * 0.1

        pg.update(episode)
        ep_reward = sum(r for _, _, r in episode)
        total_rewards.append(ep_reward)

        if wandb_run and ep % 10 == 0:
            wandb_run.log({
                "policy_gradient/episode": ep,
                "policy_gradient/reward": ep_reward,
                "policy_gradient/avg_reward_50": np.mean(total_rewards[-50:])
            })

        if op_logger:
            op_logger.log_episode({
                "episode": ep,
                "algorithm": "reinforce",
                "reward": ep_reward,
                "actions": [a for _, a, _ in episode],
                "states_shape": episode[0][0].shape[0] if episode else 0
            })

    avg_last_10 = np.mean(total_rewards[-10:])
    print(f"\n  Episoden: {episodes}")
    print(f"  Avg Reward (last 10): {avg_last_10:.1f}")

    # Checkpoint
    ckpt_path = CHECKPOINT_DIR / "policy_gradient.npz"
    pg.save_checkpoint(str(ckpt_path))

    # W&B Artifact loggen
    if wandb_run:
        log_model_artifact(wandb_run, str(ckpt_path),
                          "policy-gradient-reinforce",
                          metadata={"algorithm": "reinforce",
                                    "avg_reward_10": float(avg_last_10)})

    return pg


def train_dqn(env: GridWorld, config: dict, wandb_run=None,
              op_logger=None):
    """Trainiert DQN (PyTorch) auf der gegebenen Umgebung."""
    if not TORCH_AVAILABLE:
        print("❌ PyTorch nicht installiert — DQN übersprungen")
        return None

    print(f"\n{'='*60}")
    print(f"  DQN (PyTorch): {config.get('env_name', 'GridWorld')}")
    print(f"{'='*60}")

    dqn = DQNAgent(
        state_dim=2,  # (row, col) normalisiert
        action_dim=4,
        lr=config.get("lr", 0.001),
        gamma=config.get("gamma", 0.99),
        epsilon_start=config.get("epsilon_start", 1.0),
        epsilon_decay=config.get("epsilon_decay", 0.995),
        memory_size=config.get("memory_size", 5000),
        batch_size=config.get("batch_size", 32),
        target_update=config.get("target_update", 10),
        hidden_dim=config.get("hidden_dim", 64),
        wandb_run=wandb_run,
    )

    episodes = config.get("episodes", 500)
    rewards = dqn.train(env, episodes=episodes)

    avg_last_10 = np.mean(rewards[-10:])
    avg_last_100 = np.mean(rewards[-100:])
    print(f"\n  Episoden: {episodes}")
    print(f"  Avg Reward (last 10):  {avg_last_10:.3f}")
    print(f"  Avg Reward (last 100): {avg_last_100:.3f}")

    # DQN Policy visualisieren
    arrows = {0: "↑", 1: "→", 2: "↓", 3: "←"}
    print("\n  Gelernte DQN-Policy:")
    for r in range(env.size):
        row = "  "
        for c in range(env.size):
            if (r, c) == env.goal:
                row += " 🎯 "
            else:
                import torch
                state_vec = np.array([r / env.size, c / env.size], dtype=np.float32)
                with torch.no_grad():
                    q_vals = dqn.policy_net(
                        torch.FloatTensor(state_vec).unsqueeze(0)
                    )
                    best = q_vals.argmax(dim=1).item()
                row += f" {arrows[best]} "
        print(row)

    # Checkpoint
    ckpt_path = CHECKPOINT_DIR / f"dqn_{config.get('env_name', 'grid')}.pt"
    dqn.save_checkpoint(str(ckpt_path))

    # W&B Artifact loggen
    if wandb_run:
        log_model_artifact(wandb_run, str(ckpt_path),
                          f"dqn-{config.get('env_name', 'grid')}",
                          metadata={"algorithm": "dqn",
                                    "avg_reward_10": float(avg_last_10),
                                    "avg_reward_100": float(avg_last_100)})

    if op_logger:
        op_logger.log_episode({
            "algorithm": "dqn",
            "env": config.get("env_name", "gridworld"),
            "episodes": episodes,
            "final_avg_reward_10": float(avg_last_10),
            "final_avg_reward_100": float(avg_last_100),
            "lr": config.get("lr", 0.001),
            "gamma": config.get("gamma", 0.99),
        })

    return dqn


def main():
    parser = argparse.ArgumentParser(description="RL Training Pipeline")
    parser.add_argument("--env", choices=["standard", "cliff", "obstacles"],
                        default="standard", help="GridWorld-Variante")
    parser.add_argument("--algo", choices=["ql", "pg", "dqn", "all"],
                        default="all", help="Algorithmus")
    parser.add_argument("--episodes", type=int, default=500,
                        help="Anzahl Episoden")
    parser.add_argument("--size", type=int, default=4,
                        help="Grid-Größe")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Learning Rate")
    parser.add_argument("--no-wandb", action="store_true",
                        help="W&B deaktivieren")
    parser.add_argument("--no-openpipe", action="store_true",
                        help="OpenPipe deaktivieren")
    args = parser.parse_args()

    # ── W&B + OpenPipe Setup ──────────────────────────────────
    config = {
        "algorithm": args.algo,
        "env": args.env,
        "grid_size": args.size,
        "lr": args.lr,
        "gamma": 0.99,
        "epsilon_start": 0.3,
        "episodes": args.episodes,
    }

    wandb_run, op_client = setup_tracking(
        project="rl-agent-training",
        config=config,
        use_wandb=not args.no_wandb,
        use_openpipe=not args.no_openpipe,
    )

    op_logger = OpenPipeLogger() if not args.no_openpipe else None

    # ── Environment Setup ──────────────────────────────────────
    if args.env == "cliff":
        env = GridWorld(size=args.size, cliff=True)
        config["env_name"] = "cliff_walking"
    elif args.env == "obstacles":
        obstacles = [(1, 1), (2, 2), (1, 3)] if args.size >= 4 else [(1, 1)]
        env = GridWorld(size=args.size, obstacles=obstacles)
        config["env_name"] = "gridworld_obstacles"
    else:
        env = GridWorld(size=args.size)
        config["env_name"] = f"gridworld_{args.size}x{args.size}"

    # ── Training ───────────────────────────────────────────────
    if args.algo in ("ql", "all"):
        train_q_learning(env, config, wandb_run, op_logger)

    if args.algo in ("pg", "all"):
        pg_config = {**config, "state_dim": 4, "action_dim": 2,
                     "lr": 0.01, "max_steps": 100}
        train_policy_gradient(pg_config, wandb_run, op_logger)

    if args.algo in ("dqn", "all"):
        dqn_config = {**config, "lr": 0.001, "epsilon_start": 1.0,
                      "epsilon_decay": 0.995, "memory_size": 5000,
                      "batch_size": 32, "target_update": 10, "hidden_dim": 64}
        train_dqn(env, dqn_config, wandb_run, op_logger)

    # ── OpenPipe Export ────────────────────────────────────────
    if op_logger:
        op_logger.export_jsonl("rl_training_data.jsonl")

    # ── Sweep Info ─────────────────────────────────────────────
    if WANDB_AVAILABLE:
        print(f"\n📊 W&B Sweep-Konfiguration:")
        sweep_cfg = get_sweep_config()
        print(f"   Methode: {sweep_cfg['method']}")
        print(f"   Parameter: {list(sweep_cfg['parameters'].keys())}")

    # ── Cleanup ────────────────────────────────────────────────
    if wandb_run:
        wandb_run.finish()

    print(f"\n✅ Training abgeschlossen!")
    print(f"   📁 Checkpoints: {CHECKPOINT_DIR}/")
    print(f"   📊 W&B Run: check wandb dashboard")
    print(f"   📦 OpenPipe Data: rl_training_data.jsonl")


if __name__ == "__main__":
    main()
