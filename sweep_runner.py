#!/usr/bin/env python3
"""
sweep_runner.py — W&B Hyperparameter Sweep Runner
==================================================
Führt W&B Sweeps für Q-Learning Hyperparameter-Optimierung aus.

Usage:
    # Lokal (offline) testen:
    python sweep_runner.py --count 5 --offline

    # Auf W&B Cloud (braucht API-Key):
    python sweep_runner.py --count 20

    # Nur bestimmte Parameter testen:
    python sweep_runner.py --count 10 --env cliff --size 6
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path

from rl_agent import (
    GridWorld, QLearning, PolicyGradient,
    setup_tracking, get_sweep_config, OpenPipeLogger,
    WANDB_AVAILABLE
)

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)


def _make_env(env_type: str, size: int) -> GridWorld:
    """Erstellt eine GridWorld-Umgebung basierend auf Typ und Größe."""
    if env_type == "cliff":
        return GridWorld(size=size, cliff=True)
    elif env_type == "obstacles":
        obstacles = [(1, 1), (2, 2), (1, 3)] if size >= 4 else [(1, 1)]
        return GridWorld(size=size, obstacles=obstacles)
    else:
        return GridWorld(size=size)


def train_sweep():
    """Wird von wandb.agent() pro Sweep-Run aufgerufen."""
    import wandb

    run = wandb.init()

    # ── Environment ──────────────────────────────────────────
    env_type = wandb.config.get("env", "standard")
    size = wandb.config.get("size", 4)
    env = _make_env(env_type, size)

    # ── Agent ─────────────────────────────────────────────────
    agent = QLearning(
        env,
        lr=wandb.config.learning_rate,
        gamma=wandb.config.gamma,
        epsilon=wandb.config.epsilon_start,
        wandb_run=run,
    )

    # ── Training ──────────────────────────────────────────────
    episodes = wandb.config.get("episodes", 500)
    rewards = agent.train(episodes=episodes)

    # ── Final Metrics ─────────────────────────────────────────
    avg_last_10 = np.mean(rewards[-10:])
    avg_last_100 = np.mean(rewards[-100:])

    wandb.log({
        "final/avg_reward_10": avg_last_10,
        "final/avg_reward_100": avg_last_100,
        "final/max_reward": max(rewards),
        "final/min_reward": min(rewards),
    })

    # ── Checkpoint ────────────────────────────────────────────
    ckpt_path = CHECKPOINT_DIR / f"sweep_{run.id}.npz"
    agent.save_checkpoint(str(ckpt_path))

    run.finish()


def _run_local_grid_search(sweep_config: dict, args):
    """Führt einen lokalen Grid-Search ohne W&B Cloud durch."""
    import itertools

    lr_values = [0.01, 0.05, 0.1, 0.2, 0.5]
    gamma_values = [0.9, 0.95, 0.99]
    eps_values = [0.1, 0.3, 0.5]

    print(f"\n🔍 Lokaler Grid-Search: {len(lr_values)}×{len(gamma_values)}×{len(eps_values)} = {len(lr_values)*len(gamma_values)*len(eps_values)} Kombinationen")
    print(f"   Env: {args.env} ({args.size}×{args.size}), Episoden: {args.episodes}")
    print()

    best_reward = -float("inf")
    best_config = None
    results = []

    for lr, gamma, eps in itertools.product(lr_values, gamma_values, eps_values):
        env = _make_env(args.env, args.size)

        agent = QLearning(env, lr=lr, gamma=gamma, epsilon=eps)
        rewards = agent.train(episodes=args.episodes)
        avg_last_100 = np.mean(rewards[-100:])

        results.append({
            "lr": lr, "gamma": gamma, "epsilon": eps,
            "avg_reward_100": avg_last_100,
            "final_reward": rewards[-1],
        })

        status = "🏆" if avg_last_100 > best_reward else "  "
        print(f"   {status} lr={lr:.3f} gamma={gamma:.3f} eps={eps:.2f} → avg100={avg_last_100:.4f}")

        if avg_last_100 > best_reward:
            best_reward = avg_last_100
            best_config = {"lr": lr, "gamma": gamma, "epsilon": eps}

    # ── Summary ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"🏆 Beste Konfiguration:")
    print(f"   lr={best_config['lr']}, gamma={best_config['gamma']}, epsilon={best_config['epsilon']}")
    print(f"   Avg Reward (100): {best_reward:.4f}")
    print(f"{'='*60}")

    # ── Save best model ──────────────────────────────────────
    env = _make_env(args.env, args.size)

    best_agent = QLearning(env, **best_config)
    best_agent.train(episodes=args.episodes)
    ckpt_path = CHECKPOINT_DIR / f"best_{args.env}_{args.size}x{args.size}.npz"
    best_agent.save_checkpoint(str(ckpt_path))

    # ── Save results ─────────────────────────────────────────
    import json
    results_path = CHECKPOINT_DIR / f"grid_search_{args.env}.json"
    with open(results_path, 'w') as f:
        json.dump({
            "best_config": best_config,
            "best_reward": best_reward,
            "results": sorted(results, key=lambda x: x["avg_reward_100"], reverse=True)
        }, f, indent=2)
    print(f"📊 Ergebnisse gespeichert → {results_path}")


def main():
    parser = argparse.ArgumentParser(description="W&B Sweep Runner")
    parser.add_argument("--count", type=int, default=10,
                        help="Anzahl Sweep-Runs")
    parser.add_argument("--env", choices=["standard", "cliff", "obstacles"],
                        default="standard", help="GridWorld-Variante")
    parser.add_argument("--size", type=int, default=4,
                        help="Grid-Größe")
    parser.add_argument("--episodes", type=int, default=500,
                        help="Episoden pro Run")
    parser.add_argument("--offline", action="store_true",
                        help="Offline-Modus (kein W&B Cloud)")
    parser.add_argument("--sweep-id", type=str, default=None,
                        help="Existierende Sweep-ID (statt neuem Sweep)")
    args = parser.parse_args()

    if not WANDB_AVAILABLE:
        print("❌ W&B nicht installiert. Installiere mit: pip install wandb")
        sys.exit(1)

    import wandb

    # ── Sweep Configuration ───────────────────────────────────
    sweep_config = get_sweep_config()

    # Erweitere mit Environment-Parametern
    sweep_config["parameters"]["env"] = {"value": args.env}
    sweep_config["parameters"]["size"] = {"value": args.size}
    sweep_config["parameters"]["episodes"] = {"value": args.episodes}

    # ── Create or Resume Sweep ────────────────────────────────
    if args.sweep_id:
        sweep_id = args.sweep_id
        print(f"📊 Resuming Sweep: {sweep_id}")
    else:
        try:
            sweep_id = wandb.sweep(
                sweep_config,
                project="rl-agent-training"
            )
            print(f"📊 New Sweep created: {sweep_id}")
        except Exception as e:
            if "No API key" in str(e) or "login" in str(e).lower():
                print("⚠️  Kein W&B API-Key — führe lokalen Grid-Search durch")
                sweep_id = None
                _run_local_grid_search(sweep_config, args)
                return
            raise

    print(f"   Project: rl-agent-training")
    print(f"   Method: {sweep_config['method']}")
    print(f"   Metric: {sweep_config['metric']['name']} ({sweep_config['metric']['goal']})")
    print(f"   Parameters: {list(sweep_config['parameters'].keys())}")
    print(f"   Runs: {args.count}")
    print(f"   Env: {args.env} ({args.size}×{args.size})")
    print(f"   Episodes: {args.episodes}")

    # ── Run Sweep Agent ───────────────────────────────────────
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"

    wandb.agent(sweep_id, function=train_sweep, count=args.count)

    print(f"\n✅ Sweep abgeschlossen!")
    print(f"   Sweep ID: {sweep_id}")
    print(f"   Checkpoints: {CHECKPOINT_DIR}/")


if __name__ == "__main__":
    main()
