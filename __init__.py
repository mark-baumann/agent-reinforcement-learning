"""
agenten-verstaerkungslernen — Agent Reinforcement Learning
===========================================================
Grundlagen des Reinforcement Learning für KI-Agenten.
Q-Learning, Policy Gradient, DQN mit W&B + OpenPipe.

Verwendung:
    from agenten_verstaerkungslernen import (
        GridWorld, QLearning, PolicyGradient, DQNAgent,
        WandBLogger, OpenPipeLogger, ExperimentTracker,
        setup_tracking, get_sweep_config,
    )
"""

from rl_agent import (
    GridWorld,
    QLearning,
    PolicyGradient,
    WandBLogger,
    OpenPipeLogger,
    setup_tracking,
    get_sweep_config,
    log_model_artifact,
    log_predictions_table,
    WANDB_AVAILABLE,
    OPENPIPE_AVAILABLE,
    TORCH_AVAILABLE,
)

# DQN nur wenn PyTorch verfügbar
if TORCH_AVAILABLE:
    from rl_agent import DQNAgent, DQNNetwork

__version__ = "1.1.0"
__all__ = [
    "GridWorld",
    "QLearning",
    "PolicyGradient",
    "WandBLogger",
    "OpenPipeLogger",
    "ExperimentTracker",
    "setup_tracking",
    "get_sweep_config",
    "log_model_artifact",
    "log_predictions_table",
    "WANDB_AVAILABLE",
    "OPENPIPE_AVAILABLE",
    "TORCH_AVAILABLE",
]
