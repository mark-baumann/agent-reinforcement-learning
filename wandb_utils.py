"""
W&B Experiment Tracking für Agenten-Verstärkungslernen
======================================================
Integriert Weights & Biases in das RL-Training.
Loggt Episoden-Metriken, Modell-Checkpoints und Hyperparameter.

Usage:
    from wandb_utils import WandBTracker
    tracker = WandBTracker(project="rl-agent-training", config={...})
    tracker.log_episode("q_learning", episode=1, reward=0.5, steps=10)
    tracker.finish()
"""

import os
import time
from typing import Optional, List

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class WandBTracker:
    """
    Gekapselter W&B-Tracker für Agenten-Verstärkungslernen.

    Features:
    - Automatischer Offline-Modus wenn kein API-Key
    - Git-Commit-Logging im Online-Modus
    - Konsistente Metrik-Namen für Q-Learning, Policy Gradient, DQN
    - Modell-Artifact-Logging
    - Tabellen-Logging für Vorhersagen
    """

    def __init__(
        self,
        project: str = "rl-agent-training",
        config: Optional[dict] = None,
        tags: Optional[list] = None,
        group: Optional[str] = None,
        job_type: str = "train",
        notes: Optional[str] = None,
        offline: bool = False,
    ):
        self.project = project
        self.run = None
        self._start_time = time.time()

        if WANDB_AVAILABLE:
            try:
                mode = "offline" if offline or not os.environ.get("WANDB_API_KEY") else "online"
                self.run = wandb.init(
                    project=project,
                    config=config or {},
                    mode=mode,
                    tags=tags or ["rl", "reinforcement-learning"],
                    group=group,
                    job_type=job_type,
                    notes=notes,
                    dir="wandb_runs",
                )
                if mode == "online":
                    try:
                        import subprocess
                        git_commit = subprocess.check_output(
                            ["git", "rev-parse", "--short", "HEAD"],
                            stderr=subprocess.DEVNULL,
                        ).decode().strip()
                        self.log({"git_commit": git_commit})
                    except Exception:
                        pass
                print(f"📊 W&B initialisiert (mode={mode}, project={project})")
            except Exception as e:
                print(f"⚠️  W&B-Init fehlgeschlagen: {e}")

    def log(self, metrics: dict, step: Optional[int] = None):
        """Loggt Metriken zu W&B."""
        if self.run:
            self.run.log(metrics, step=step)

    # ── Domain-spezifische Log-Methoden ──────────────────────

    def log_episode(self, prefix: str, episode: int, reward: float,
                    steps: Optional[int] = None, epsilon: Optional[float] = None,
                    loss: Optional[float] = None, extra: Optional[dict] = None):
        """Loggt eine Episode mit konsistenten Metrik-Namen."""
        metrics = {
            f"{prefix}/episode": episode,
            f"{prefix}/reward": reward,
        }
        if steps is not None:
            metrics[f"{prefix}/steps"] = steps
        if epsilon is not None:
            metrics[f"{prefix}/epsilon"] = epsilon
        if loss is not None:
            metrics[f"{prefix}/loss"] = loss
        if extra:
            metrics.update({f"{prefix}/{k}": v for k, v in extra.items()})
        self.log(metrics)

    def log_model(self, checkpoint_path: str, model_name: str,
                  metadata: dict = None, aliases: list = None):
        """Loggt ein Modell als W&B Artifact."""
        if not self.run:
            return
        artifact = wandb.Artifact(
            name=model_name,
            type="model",
            metadata=metadata or {},
        )
        artifact.add_file(checkpoint_path)
        self.run.log_artifact(artifact, aliases=aliases or ["latest"])
        print(f"📦 W&B Artifact geloggt: {model_name} → {checkpoint_path}")

    def log_table(self, name: str, columns: list, data: list):
        """Loggt eine Tabelle ins W&B Dashboard."""
        if not self.run:
            return
        table = wandb.Table(columns=columns)
        for row in data:
            table.add_data(*row)
        self.run.log({name: table})

    def log_sweep_config(self, algo: str = "q_learning"):
        """Loggt die Sweep-Konfiguration für den aktuellen Run."""
        if not self.run:
            return
        # Importiere get_sweep_config aus rl_agent (vermeidet Zirkelimport)
        try:
            from rl_agent import get_sweep_config
            sweep_cfg = get_sweep_config(algo)
            self.run.log({"sweep/method": sweep_cfg["method"]})
            for k, v in sweep_cfg["parameters"].items():
                self.run.log({f"sweep/param/{k}": str(v)})
        except ImportError:
            pass

    def finish(self):
        """Beendet den W&B-Run. Sicher bei mehrfachem Aufruf."""
        elapsed = time.time() - self._start_time
        if self.run:
            self.log({"total_time_seconds": elapsed})
            self.run.finish()
            self.run = None

    @property
    def is_active(self) -> bool:
        return self.run is not None
