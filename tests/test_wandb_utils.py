"""
Tests für wandb_utils.py — W&B Experiment Tracking für Agenten-Verstärkungslernen.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wandb_utils import WandBTracker, WANDB_AVAILABLE


class TestWandBTracker:
    """Tests für WandBTracker."""

    def test_initialization_offline(self):
        """Tracker sollte im Offline-Modus initialisieren."""
        tracker = WandBTracker(
            project="test-rl-agent",
            config={"lr": 0.1, "gamma": 0.99},
            tags=["test"],
            group="test-group",
            job_type="test",
            notes="Test-Run",
            offline=True,
        )
        if WANDB_AVAILABLE:
            assert tracker.is_active
            assert tracker.run is not None
        else:
            assert not tracker.is_active
        tracker.finish()

    def test_log_metrics(self):
        """Metriken sollten ohne Fehler geloggt werden."""
        tracker = WandBTracker(project="test-rl-agent", offline=True)
        if tracker.is_active:
            tracker.log({"accuracy": 0.95, "loss": 0.05})
        tracker.finish()

    def test_log_episode(self):
        """Episoden-Metriken sollten ohne Fehler geloggt werden."""
        tracker = WandBTracker(project="test-rl-agent", offline=True)
        if tracker.is_active:
            tracker.log_episode("q_learning", episode=1, reward=0.5, steps=10, epsilon=0.3)
            tracker.log_episode("policy_gradient", episode=1, reward=10.0, loss=0.01)
            tracker.log_episode("dqn", episode=1, reward=0.8, steps=15, epsilon=0.5, loss=0.02,
                               extra={"avg_reward_100": 0.75})
        tracker.finish()

    def test_log_table(self):
        """Tabellen sollten ohne Fehler geloggt werden."""
        tracker = WandBTracker(project="test-rl-agent", offline=True)
        if tracker.is_active:
            tracker.log_table("test_table", ["col1", "col2"], [["a", 1], ["b", 2]])
        tracker.finish()

    def test_log_sweep_config(self):
        """Sweep-Konfiguration sollte geloggt werden."""
        tracker = WandBTracker(project="test-rl-agent", offline=True)
        if tracker.is_active:
            tracker.log_sweep_config("q_learning")
        tracker.finish()

    def test_finish_cleans_up(self):
        """finish() sollte den Run beenden und doppeltes finish() sollte safe sein."""
        tracker = WandBTracker(project="test-rl-agent", offline=True)
        tracker.finish()
        tracker.finish()  # Doppeltes finish() sollte safe sein
        assert not tracker.is_active

    def test_log_model_no_file(self, tmp_path):
        """log_model() mit existierender Datei sollte funktionieren."""
        tracker = WandBTracker(project="test-rl-agent", offline=True)
        # Erstelle eine Dummy-Datei
        dummy_file = tmp_path / "dummy_model.pt"
        dummy_file.write_text("dummy model content")
        if tracker.is_active:
            tracker.log_model(str(dummy_file), "test-model",
                             metadata={"algo": "q-learning"})
        tracker.finish()
