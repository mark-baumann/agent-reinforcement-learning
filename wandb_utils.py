"""
W&B Experiment Tracking für Agenten-Verstärkungslernen
======================================================
Re-Export-Wrapper um rl_agent.WandBLogger.
Bietet Abwärtskompatibilität mit dem alten WandBTracker-Namen.

Usage:
    from wandb_utils import WandBTracker
    tracker = WandBTracker(project="rl-agent-training", config={...})
    tracker.log_episode("q_learning", episode=1, reward=0.5, steps=10)
    tracker.finish()
"""

from rl_agent import WandBLogger, WANDB_AVAILABLE

# Alias für Abwärtskompatibilität
WandBTracker = WandBLogger

__all__ = ["WandBTracker", "WandBLogger", "WANDB_AVAILABLE"]
