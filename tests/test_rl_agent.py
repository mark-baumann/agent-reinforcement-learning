"""
Tests für rl_agent.py — GridWorld, QLearning, PolicyGradient, DQN, W&B, OpenPipe.
"""
import sys
import os
import numpy as np
import pytest

# Ensure the module is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_agent import (
    GridWorld, QLearning, PolicyGradient, OpenPipeLogger, WandBLogger,
    setup_tracking, get_sweep_config, log_model_artifact,
    log_predictions_table,
    WANDB_AVAILABLE, OPENPIPE_AVAILABLE, TORCH_AVAILABLE
)


# ═══════════════════════════════════════════════════════════════
# GridWorld Tests
# ═══════════════════════════════════════════════════════════════

class TestGridWorld:
    def test_initialization(self):
        env = GridWorld(4)
        assert env.size == 4
        assert env.pos == (0, 0)
        assert env.goal == (3, 3)

    def test_reset(self):
        env = GridWorld(4)
        env.pos = (2, 2)
        assert env.reset() == (0, 0)

    def test_step_moves_agent(self):
        env = GridWorld(4)
        pos, reward, done = env.step(1)  # right
        assert pos == (0, 1)
        assert reward == -0.01
        assert not done

    def test_step_reaches_goal(self):
        env = GridWorld(4)
        env.pos = (3, 2)
        pos, reward, done = env.step(1)  # right → goal
        assert pos == (3, 3)
        assert reward == 1.0
        assert done

    def test_step_boundary(self):
        env = GridWorld(4)
        pos, reward, done = env.step(0)  # up from (0,0)
        assert pos == (0, 0)  # stays in bounds

    def test_obstacles(self):
        env = GridWorld(4, obstacles=[(1, 1)])
        env.pos = (0, 1)
        pos, reward, done = env.step(2)  # down into obstacle
        assert pos == (0, 1)  # stays in place
        assert reward == -0.1

    def test_cliff(self):
        env = GridWorld(4, cliff=True)
        env.pos = (2, 1)
        pos, reward, done = env.step(2)  # down onto cliff
        assert pos == (0, 0)  # reset to start
        assert reward == -1.0
        assert not done

    def test_custom_start_goal(self):
        env = GridWorld(5, start=(1, 1), goal=(3, 3))
        assert env.reset() == (1, 1)
        assert env.goal == (3, 3)


# ═══════════════════════════════════════════════════════════════
# QLearning Tests
# ═══════════════════════════════════════════════════════════════

class TestQLearning:
    def test_initialization(self):
        env = GridWorld(4)
        agent = QLearning(env, lr=0.1, gamma=0.99, epsilon=0.3)
        assert agent.lr == 0.1
        assert agent.gamma == 0.99
        assert agent.epsilon == 0.3

    def test_choose_action_returns_valid_action(self):
        env = GridWorld(4)
        agent = QLearning(env)
        action = agent.choose_action((0, 0))
        assert action in [0, 1, 2, 3]

    def test_train_improves_reward(self):
        env = GridWorld(4)
        agent = QLearning(env, lr=0.5, gamma=0.99, epsilon=0.3)
        rewards = agent.train(episodes=200)
        # Early episodes should have lower reward than later ones
        assert np.mean(rewards[:50]) < np.mean(rewards[-50:])

    def test_train_reaches_goal(self):
        env = GridWorld(4)
        agent = QLearning(env, lr=0.5, gamma=0.99, epsilon=0.3)
        rewards = agent.train(episodes=300)
        # Final average should be positive (agent learns to reach goal)
        assert np.mean(rewards[-50:]) > 0.5

    def test_get_policy_returns_grid(self):
        env = GridWorld(4)
        agent = QLearning(env)
        agent.train(episodes=100)
        policy = agent.get_policy()
        assert policy.shape == (4, 4)
        assert policy[3, 3] == -1  # goal marker

    def test_epsilon_decays(self):
        env = GridWorld(4)
        agent = QLearning(env, epsilon=0.5)
        initial_eps = agent.epsilon
        agent.train(episodes=100)
        assert agent.epsilon < initial_eps

    def test_save_load_checkpoint(self, tmp_path):
        env = GridWorld(4)
        agent = QLearning(env, lr=0.5, gamma=0.99, epsilon=0.3)
        agent.train(episodes=50)

        ckpt_path = tmp_path / "test_ckpt.npz"
        agent.save_checkpoint(str(ckpt_path))

        agent2 = QLearning(env, lr=0.1, gamma=0.9, epsilon=0.9)
        agent2.load_checkpoint(str(ckpt_path))

        assert agent2.lr == 0.5
        assert agent2.gamma == 0.99
        assert agent2.epsilon == pytest.approx(agent.epsilon, abs=0.01)


# ═══════════════════════════════════════════════════════════════
# PolicyGradient Tests
# ═══════════════════════════════════════════════════════════════

class TestPolicyGradient:
    def test_initialization(self):
        pg = PolicyGradient(state_dim=4, action_dim=2, lr=0.01, gamma=0.99)
        assert pg.W1.shape == (4, 32)
        assert pg.W2.shape == (32, 2)

    def test_forward_returns_probabilities(self):
        pg = PolicyGradient(state_dim=4, action_dim=2)
        state = np.random.randn(4)
        probs = pg.forward(state)
        assert len(probs) == 2
        assert abs(np.sum(probs) - 1.0) < 1e-6
        assert np.all(probs >= 0)

    def test_sample_action_returns_valid_action(self):
        pg = PolicyGradient(state_dim=4, action_dim=2)
        state = np.random.randn(4)
        action, probs = pg.sample_action(state)
        assert action in [0, 1]
        assert len(probs) == 2

    def test_update_changes_weights(self):
        pg = PolicyGradient(state_dim=4, action_dim=2, lr=0.1)
        W1_before = pg.W1.copy()
        episode = [(np.random.randn(4), 0, 1.0) for _ in range(10)]
        pg.update(episode)
        assert not np.array_equal(W1_before, pg.W1)

    def test_save_load_checkpoint(self, tmp_path):
        pg = PolicyGradient(state_dim=4, action_dim=2, lr=0.01, gamma=0.99)
        ckpt_path = tmp_path / "pg_ckpt.npz"
        pg.save_checkpoint(str(ckpt_path))

        pg2 = PolicyGradient(state_dim=4, action_dim=2, lr=0.1, gamma=0.9)
        pg2.load_checkpoint(str(ckpt_path))

        assert np.array_equal(pg.W1, pg2.W1)
        assert pg2.lr == 0.01
        assert pg2.gamma == 0.99


# ═══════════════════════════════════════════════════════════════
# DQN Tests (nur wenn PyTorch verfügbar)
# ═══════════════════════════════════════════════════════════════

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch nicht installiert")
class TestDQN:
    def test_initialization(self):
        from rl_agent import DQNAgent
        agent = DQNAgent(state_dim=2, action_dim=4, lr=0.001)
        assert agent.state_dim == 2
        assert agent.action_dim == 4
        assert agent.epsilon == 1.0

    def test_select_action_returns_valid_action(self):
        from rl_agent import DQNAgent
        agent = DQNAgent(state_dim=2, action_dim=4)
        state = np.array([0.5, 0.5], dtype=np.float32)
        action = agent.select_action(state)
        assert action in [0, 1, 2, 3]

    def test_select_action_evaluate_mode(self):
        from rl_agent import DQNAgent
        agent = DQNAgent(state_dim=2, action_dim=4)
        state = np.array([0.5, 0.5], dtype=np.float32)
        action = agent.select_action(state, evaluate=True)
        assert action in [0, 1, 2, 3]

    def test_push_and_update(self):
        from rl_agent import DQNAgent
        agent = DQNAgent(state_dim=2, action_dim=4, batch_size=4)
        # Fill memory with enough samples
        for _ in range(10):
            agent.push(np.array([0.5, 0.5]), 1, 0.5, np.array([0.6, 0.5]), False)
        loss = agent.update()
        assert loss is not None
        assert loss > 0

    def test_train_on_gridworld(self):
        from rl_agent import DQNAgent
        env = GridWorld(4)
        agent = DQNAgent(state_dim=2, action_dim=4, lr=0.01,
                        epsilon_start=0.5, epsilon_decay=0.99,
                        memory_size=1000, batch_size=16,
                        target_update=5, hidden_dim=32)
        rewards = agent.train(env, episodes=150)
        assert len(rewards) == 150
        # Agent sollte lernen — letzte 20 Episoden sollten positive Rewards haben
        assert np.mean(rewards[-20:]) > 0.0

    def test_save_load_checkpoint(self, tmp_path):
        from rl_agent import DQNAgent
        agent = DQNAgent(state_dim=2, action_dim=4)
        ckpt_path = tmp_path / "dqn_ckpt.pt"
        agent.save_checkpoint(str(ckpt_path))

        agent2 = DQNAgent(state_dim=2, action_dim=4)
        agent2.load_checkpoint(str(ckpt_path))
        assert agent2.epsilon == agent.epsilon


# ═══════════════════════════════════════════════════════════════
# W&B Integration Tests
# ═══════════════════════════════════════════════════════════════

class TestWandBIntegration:
    def test_setup_tracking_offline(self):
        """W&B sollte im Offline-Modus initialisieren (kein API-Key)."""
        run, op_client = setup_tracking(
            project="test-project",
            config={"lr": 0.1},
            use_wandb=True,
            use_openpipe=False,
        )
        if WANDB_AVAILABLE:
            assert run is not None
            run.finish()
        else:
            assert run is None

    def test_setup_tracking_with_tags_and_group(self):
        """W&B mit Tags und Gruppe initialisieren."""
        run, _ = setup_tracking(
            project="test-project",
            config={"lr": 0.1},
            use_wandb=True,
            use_openpipe=False,
            tags=["test", "unit-test"],
            group="test-group",
            job_type="test",
        )
        if WANDB_AVAILABLE:
            assert run is not None
            run.finish()

    def test_get_sweep_config(self):
        config = get_sweep_config("q_learning")
        assert config["method"] == "bayes"
        assert "learning_rate" in config["parameters"]
        assert "gamma" in config["parameters"]

    def test_get_sweep_config_dqn(self):
        config = get_sweep_config("dqn")
        assert config["method"] == "bayes"
        assert "batch_size" in config["parameters"]
        assert "hidden_dim" in config["parameters"]

    def test_get_sweep_config_pg(self):
        config = get_sweep_config("policy_gradient")
        assert config["method"] == "bayes"
        assert "hidden_dim" in config["parameters"]


class TestWandBLogger:
    """Tests für die neue WandBLogger-Klasse."""

    def test_initialization_offline(self):
        """WandBLogger sollte im Offline-Modus initialisieren."""
        logger = WandBLogger(
            project="test-project",
            config={"lr": 0.1},
            tags=["test"],
            group="test-group",
            job_type="test",
            notes="Test-Run",
            offline=True,
        )
        if WANDB_AVAILABLE:
            assert logger.is_active
            assert logger.run is not None
        else:
            assert not logger.is_active
        logger.finish()

    def test_log_metrics(self):
        """Metriken sollten ohne Fehler geloggt werden."""
        logger = WandBLogger(project="test-project", offline=True)
        if logger.is_active:
            logger.log({"test_metric": 1.0})
            logger.log_episode("ql", episode=1, reward=0.5, steps=10, epsilon=0.3)
        logger.finish()

    def test_log_table(self):
        """Tabellen sollten ohne Fehler geloggt werden."""
        logger = WandBLogger(project="test-project", offline=True)
        if logger.is_active:
            logger.log_table("test_table", ["col1", "col2"], [["a", 1], ["b", 2]])
        logger.finish()

    def test_log_sweep_config(self):
        """Sweep-Konfiguration sollte geloggt werden."""
        logger = WandBLogger(project="test-project", offline=True)
        if logger.is_active:
            logger.log_sweep_config("q_learning")
        logger.finish()

    def test_finish_cleans_up(self):
        """finish() sollte den Run beenden."""
        logger = WandBLogger(project="test-project", offline=True)
        logger.finish()
        # Nach finish() sollte kein Fehler auftreten
        logger.finish()  # Doppeltes finish() sollte safe sein


# ═══════════════════════════════════════════════════════════════
# OpenPipe Tests
# ═══════════════════════════════════════════════════════════════

class TestOpenPipe:
    def test_logger_initialization(self):
        logger = OpenPipeLogger()
        assert logger.episodes == []

    def test_log_episode(self):
        logger = OpenPipeLogger()
        logger.log_episode({"episode": 1, "reward": 0.5})
        assert len(logger.episodes) == 1
        assert logger.episodes[0]["reward"] == 0.5

    def test_get_training_data(self):
        logger = OpenPipeLogger()
        logger.log_episode({"episode": 1})
        logger.log_episode({"episode": 2})
        data = logger.get_training_data()
        assert len(data) == 2

    def test_export_jsonl(self, tmp_path):
        logger = OpenPipeLogger()
        logger.log_episode({"episode": 1, "reward": 0.5})
        logger.log_episode({"episode": 2, "reward": 0.8})

        export_path = tmp_path / "test_data.jsonl"
        logger.export_jsonl(str(export_path))

        assert export_path.exists()
        with open(export_path) as f:
            lines = f.readlines()
        assert len(lines) == 2

    def test_export_jsonl_empty(self, tmp_path):
        logger = OpenPipeLogger()
        export_path = tmp_path / "empty.jsonl"
        logger.export_jsonl(str(export_path))
        assert export_path.exists()
        with open(export_path) as f:
            assert f.read() == ""

    def test_get_statistics(self):
        """get_statistics() sollte korrekte Statistiken liefern."""
        logger = OpenPipeLogger()
        logger.log_episode({"episode": 1, "reward": 0.5, "algorithm": "q-learning"})
        logger.log_episode({"episode": 2, "reward": 0.8, "algorithm": "q-learning"})
        logger.log_episode({"episode": 3, "reward": 0.3, "algorithm": "reinforce"})

        stats = logger.get_statistics()
        assert stats["total_episodes"] == 3
        assert stats["total_reward"] == 1.6
        assert abs(stats["avg_reward"] - 0.5333) < 0.01
        assert "q-learning" in stats["algorithms"]
        assert "reinforce" in stats["algorithms"]

    def test_get_statistics_empty(self):
        """get_statistics() sollte mit leeren Daten umgehen."""
        logger = OpenPipeLogger()
        stats = logger.get_statistics()
        assert stats["total_episodes"] == 0
        assert stats["avg_reward"] == 0.0

    def test_export_jsonl_with_compress(self, tmp_path):
        """Export mit Kompression sollte .jsonl und .jsonl.gz erstellen."""
        logger = OpenPipeLogger()
        logger.log_episode({"episode": 1, "reward": 0.5})
        logger.log_episode({"episode": 2, "reward": 0.8})

        export_path = tmp_path / "test_data.jsonl"
        logger.export_jsonl(str(export_path), compress=True)

        assert export_path.exists()
        gz_path = tmp_path / "test_data.jsonl.gz"
        assert gz_path.exists()

    def test_upload_to_openpipe_no_client(self):
        """upload_to_openpipe() sollte False zurückgeben ohne Client."""
        logger = OpenPipeLogger()  # Kein API-Key → kein Client
        result = logger.upload_to_openpipe()
        assert result is False


# ═══════════════════════════════════════════════════════════════
# ExperimentTracker Tests
# ═══════════════════════════════════════════════════════════════

class TestExperimentTracker:
    """Tests für den einheitlichen ExperimentTracker (W&B + OpenPipe)."""

    def test_initialization(self):
        """ExperimentTracker sollte ohne Fehler initialisieren."""
        from rl_agent import ExperimentTracker
        tracker = ExperimentTracker(
            project="test-project",
            config={"lr": 0.1},
            tags=["test"],
            group="test-group",
            job_type="test",
            notes="Test-Run",
            offline=True,
        )
        assert tracker.project == "test-project"
        assert tracker.config == {"lr": 0.1}
        tracker.finish()

    def test_log_episode(self):
        """log_episode() sollte W&B und OpenPipe bedienen."""
        from rl_agent import ExperimentTracker
        tracker = ExperimentTracker(
            project="test-project",
            offline=True,
        )
        tracker.log_episode("q_learning", episode=1, reward=0.5,
                           steps=10, epsilon=0.3, loss=0.01,
                           extra={"avg_reward_100": 0.45})
        # OpenPipe sollte die Episode gespeichert haben
        stats = tracker.get_statistics()
        assert stats["total_episodes"] == 1
        assert stats["total_reward"] == 0.5
        tracker.finish()

    def test_log_model(self, tmp_path):
        """log_model() sollte ohne Fehler funktionieren."""
        from rl_agent import ExperimentTracker
        dummy_file = tmp_path / "dummy_model.npz"
        dummy_file.write_text("dummy")
        tracker = ExperimentTracker(project="test-project", offline=True)
        tracker.log_model(str(dummy_file), "test-model",
                         metadata={"algo": "q-learning"})
        tracker.finish()

    def test_log_table(self):
        """log_table() sollte ohne Fehler funktionieren."""
        from rl_agent import ExperimentTracker
        tracker = ExperimentTracker(project="test-project", offline=True)
        tracker.log_table("test_table", ["col1", "col2"],
                         [["a", 1], ["b", 2]])
        tracker.finish()

    def test_export_openpipe(self, tmp_path):
        """export_openpipe() sollte JSONL-Datei erstellen."""
        from rl_agent import ExperimentTracker
        tracker = ExperimentTracker(project="test-project", offline=True)
        tracker.log_episode("ql", episode=1, reward=0.5)
        tracker.log_episode("ql", episode=2, reward=0.8)

        export_path = tmp_path / "test_export.jsonl"
        tracker.export_openpipe(str(export_path))
        assert export_path.exists()
        with open(export_path) as f:
            lines = f.readlines()
        assert len(lines) == 2
        tracker.finish()

    def test_get_statistics(self):
        """get_statistics() sollte korrekte Statistiken liefern."""
        from rl_agent import ExperimentTracker
        tracker = ExperimentTracker(project="test-project", offline=True)
        tracker.log_episode("ql", episode=1, reward=0.5)
        tracker.log_episode("pg", episode=2, reward=1.0)
        stats = tracker.get_statistics()
        assert stats["total_episodes"] == 2
        assert stats["total_reward"] == 1.5
        assert abs(stats["avg_reward"] - 0.75) < 0.01
        tracker.finish()

    def test_finish_cleans_up(self):
        """finish() sollte W&B-Run beenden und OpenPipe exportieren."""
        from rl_agent import ExperimentTracker
        tracker = ExperimentTracker(project="test-project", offline=True)
        tracker.log_episode("ql", episode=1, reward=0.5)
        tracker.finish()
        # Doppeltes finish() sollte safe sein
        tracker.finish()

    def test_is_active(self):
        """is_active sollte korrekt den Status melden."""
        from rl_agent import ExperimentTracker
        tracker = ExperimentTracker(project="test-project", offline=True)
        if WANDB_AVAILABLE:
            assert tracker.is_active
        else:
            assert not tracker.is_active
        tracker.finish()
        assert not tracker.is_active

    def test_no_wandb_no_openpipe(self):
        """Tracker sollte auch ohne W&B und OpenPipe funktionieren."""
        from rl_agent import ExperimentTracker
        tracker = ExperimentTracker(
            project="test-project",
            use_wandb=False,
            use_openpipe=False,
        )
        tracker.log_episode("ql", episode=1, reward=0.5)
        stats = tracker.get_statistics()
        assert stats["total_episodes"] == 0
        tracker.finish()


# ═══════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════

class TestIntegration:
    def test_full_q_learning_pipeline(self):
        """End-to-End: GridWorld → QLearning → Checkpoint."""
        env = GridWorld(4)
        agent = QLearning(env, lr=0.5, gamma=0.99, epsilon=0.3)
        rewards = agent.train(episodes=100)
        assert len(rewards) == 100
        policy = agent.get_policy()
        assert policy.shape == (4, 4)

    def test_full_pg_pipeline(self):
        """End-to-End: Policy Gradient Training."""
        pg = PolicyGradient(state_dim=4, action_dim=2, lr=0.1)
        for _ in range(20):
            state = np.random.randn(4)
            episode = [(state, 0, 1.0) for _ in range(10)]
            pg.update(episode)
        # Should still produce valid probabilities
        probs = pg.forward(np.random.randn(4))
        assert abs(np.sum(probs) - 1.0) < 1e-6

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch nicht installiert")
    def test_full_dqn_pipeline(self):
        """End-to-End: DQN auf GridWorld."""
        from rl_agent import DQNAgent
        env = GridWorld(4)
        agent = DQNAgent(state_dim=2, action_dim=4, lr=0.01,
                        epsilon_start=0.5, epsilon_decay=0.99,
                        memory_size=500, batch_size=16,
                        target_update=5, hidden_dim=32)
        rewards = agent.train(env, episodes=50)
        assert len(rewards) == 50
