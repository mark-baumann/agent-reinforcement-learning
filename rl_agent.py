"""
Agent Reinforcement Learning — Q-Learning & Policy Gradient
===========================================================
Grundlagen des Reinforcement Learning für AI Agents.
Mit W&B Experiment Tracking & OpenPipe Fine-Tuning.

Implementiert:
1. Q-Learning (Tabular) — GridWorld
2. Deep Q-Network (DQN) — CartPole
3. Policy Gradient (REINFORCE) — CartPole
4. W&B Integration — Experiment Tracking & Sweeps
5. OpenPipe Integration — Fine-Tuning Logging
"""

import numpy as np
from collections import defaultdict
from typing import Tuple, List, Optional
import random
import os

# ── W&B (optional, offline mode falls kein API-Key) ──────────
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# ── OpenPipe (optional) ──────────────────────────────────────
try:
    from openpipe import OpenAI
    OPENPIPE_AVAILABLE = True
except ImportError:
    OPENPIPE_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════
# W&B + OpenPipe Setup
# ═══════════════════════════════════════════════════════════════

def setup_tracking(project: str = "rl-agent-training",
                   config: dict = None,
                   use_wandb: bool = True,
                   use_openpipe: bool = True):
    """
    Initialisiert W&B und/oder OpenPipe für Experiment-Tracking.

    Args:
        project: Projektname für W&B
        config: Hyperparameter-Dict
        use_wandb: W&B-Tracking aktivieren
        use_openpipe: OpenPipe-Tracking aktivieren
    """
    run = None
    op_client = None

    if use_wandb and WANDB_AVAILABLE:
        mode = "online" if os.environ.get("WANDB_API_KEY") else "offline"
        run = wandb.init(
            project=project,
            config=config or {},
            mode=mode,
            tags=["rl", "q-learning", "policy-gradient"]
        )
        print(f"📊 W&B initialisiert (mode={mode})")

    if use_openpipe and OPENPIPE_AVAILABLE:
        api_key = os.environ.get("OPENPIPE_API_KEY", "")
        if api_key:
            op_client = OpenAI(
                api_key=api_key,
                openpipe={"api_key": api_key}
            )
            print("🔧 OpenPipe initialisiert")
        else:
            print("⚠️  OpenPipe API-Key nicht gesetzt — überspringe")

    return run, op_client


# ═══════════════════════════════════════════════════════════════
# 1. Q-Learning — GridWorld
# ═══════════════════════════════════════════════════════════════

class GridWorld:
    """
    N×N Grid. Agent startet bei (0,0), Ziel bei (N-1,N-1).
    Aktionen: 0=hoch, 1=rechts, 2=runter, 3=links
    Belohnung: +1 am Ziel, -0.01 pro Schritt (Effizienz-Anreiz)

    Varianten:
    - Standard: Leeres Grid
    - Obstacles: Hindernisse, die nicht betreten werden können
    - Cliff: Klippen-Umgebung (Sutton & Barto)
    """

    def __init__(self, size: int = 4, obstacles: Optional[List[tuple]] = None,
                 cliff: bool = False, start: Optional[tuple] = None,
                 goal: Optional[tuple] = None):
        self.size = size
        self.goal = goal or (size - 1, size - 1)
        self.start = start or (0, 0)
        self.obstacles = set(obstacles or [])
        self.cliff = cliff
        self.reset()

    def reset(self):
        self.pos = self.start
        return self.pos

    def step(self, action: int) -> Tuple[tuple, float, bool]:
        r, c = self.pos
        if action == 0:    r = max(0, r - 1)
        elif action == 1:  c = min(self.size - 1, c + 1)
        elif action == 2:  r = min(self.size - 1, r + 1)
        elif action == 3:  c = max(0, c - 1)

        new_pos = (r, c)

        # Cliff: falling off the cliff resets to start with penalty
        if self.cliff and r == self.size - 1 and 0 < c < self.size - 1:
            self.pos = self.start
            return self.pos, -1.0, False  # penalty, not done

        # Obstacle: stay in place with penalty
        if new_pos in self.obstacles:
            return self.pos, -0.1, False

        self.pos = new_pos
        done = self.pos == self.goal
        reward = 1.0 if done else -0.01
        return self.pos, reward, done


class QLearning:
    """Tabular Q-Learning mit ε-greedy Exploration + W&B Tracking."""

    def __init__(self, env: GridWorld, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1,
                 wandb_run=None):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: np.zeros(4))
        self.wandb_run = wandb_run

    def choose_action(self, state: tuple) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        return np.argmax(self.Q[state])

    def train(self, episodes: int = 1000) -> List[float]:
        """Trainiert den Agenten. Gibt Rewards pro Episode zurück."""
        rewards_history = []
        steps_history = []

        for ep in range(episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)

                # Q-Learning Update
                best_next = np.max(self.Q[next_state])
                td_target = reward + self.gamma * best_next * (1 - done)
                td_error = td_target - self.Q[state][action]
                self.Q[state][action] += self.lr * td_error

                state = next_state
                total_reward += reward
                steps += 1

            rewards_history.append(total_reward)
            steps_history.append(steps)

            # Decay epsilon
            self.epsilon = max(0.01, self.epsilon * 0.995)

            # ── W&B Logging ──────────────────────────────────
            if self.wandb_run and ep % 10 == 0:
                self.wandb_run.log({
                    "q_learning/episode": ep,
                    "q_learning/reward": total_reward,
                    "q_learning/steps": steps,
                    "q_learning/epsilon": self.epsilon,
                    "q_learning/avg_reward_100": np.mean(rewards_history[-100:])
                })

        return rewards_history

    def get_policy(self) -> np.ndarray:
        """Gibt die gelernte Policy als Grid zurück."""
        grid = np.zeros((self.env.size, self.env.size), dtype=int)
        for r in range(self.env.size):
            for c in range(self.env.size):
                if (r, c) == self.env.goal:
                    grid[r, c] = -1  # Goal
                else:
                    grid[r, c] = np.argmax(self.Q[(r, c)])
        return grid

    def save_checkpoint(self, path: str = "q_learning_checkpoint.npz"):
        """Speichert Q-Table und Hyperparameter."""
        q_dict = {str(k): v for k, v in self.Q.items()}
        np.savez(path,
                 lr=self.lr, gamma=self.gamma, epsilon=self.epsilon,
                 q_keys=np.array(list(q_dict.keys())),
                 q_values=np.array(list(q_dict.values())))
        print(f"💾 Q-Learning Checkpoint gespeichert → {path}")

    def load_checkpoint(self, path: str = "q_learning_checkpoint.npz"):
        """Lädt Q-Table und Hyperparameter."""
        data = np.load(path, allow_pickle=True)
        self.lr = float(data['lr'])
        self.gamma = float(data['gamma'])
        self.epsilon = float(data['epsilon'])
        for k, v in zip(data['q_keys'], data['q_values']):
            self.Q[eval(k)] = v
        print(f"📂 Q-Learning Checkpoint geladen ← {path}")


# ═══════════════════════════════════════════════════════════════
# 2. Policy Gradient (REINFORCE)
# ═══════════════════════════════════════════════════════════════

class PolicyGradient:
    """
    REINFORCE-Algorithmus mit einem einfachen 2-Layer-Netzwerk.
    Funktioniert für diskrete Action-Spaces.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 lr: float = 0.01, gamma: float = 0.99,
                 wandb_run=None):
        self.lr = lr
        self.gamma = gamma
        self.wandb_run = wandb_run

        # Einfaches 2-Layer-Netzwerk
        self.W1 = np.random.randn(state_dim, 32) * 0.1
        self.b1 = np.zeros(32)
        self.W2 = np.random.randn(32, action_dim) * 0.1
        self.b2 = np.zeros(action_dim)

    def forward(self, state: np.ndarray) -> np.ndarray:
        """Forward-Pass: state → action probabilities."""
        h = np.maximum(0, state @ self.W1 + self.b1)  # ReLU
        logits = h @ self.W2 + self.b2
        # Softmax
        shifted = logits - np.max(logits)
        exp = np.exp(shifted)
        return exp / np.sum(exp)

    def sample_action(self, state: np.ndarray) -> Tuple[int, np.ndarray]:
        """Sample eine Aktion aus der Policy."""
        probs = self.forward(state)
        action = np.random.choice(len(probs), p=probs)
        return action, probs

    def update(self, episode: List[Tuple[np.ndarray, int, float]]):
        """
        REINFORCE Update.

        episode: Liste von (state, action, reward)
        """
        # Discounted Returns berechnen
        returns = []
        G = 0
        for _, _, r in reversed(episode):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = np.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        for (state, action, _), G in zip(episode, returns):
            probs = self.forward(state)

            # Gradient für Cross-Entropy
            dlogits = probs.copy()
            dlogits[action] -= 1
            dlogits *= G

            # Backward (manuell)
            h = np.maximum(0, state @ self.W1 + self.b1)
            dh = dlogits @ self.W2.T
            dh[h <= 0] = 0

            # Update
            self.W2 -= self.lr * np.outer(h, dlogits)
            self.b2 -= self.lr * dlogits
            self.W1 -= self.lr * np.outer(state, dh)
            self.b1 -= self.lr * dh

    def save_checkpoint(self, path: str = "pg_checkpoint.npz"):
        """Speichert Netzwerk-Gewichte und Hyperparameter."""
        np.savez(path,
                 W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 lr=self.lr, gamma=self.gamma)
        print(f"💾 Policy Gradient Checkpoint gespeichert → {path}")

    def load_checkpoint(self, path: str = "pg_checkpoint.npz"):
        """Lädt Netzwerk-Gewichte und Hyperparameter."""
        data = np.load(path)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        self.lr = float(data['lr'])
        self.gamma = float(data['gamma'])
        print(f"📂 Policy Gradient Checkpoint geladen ← {path}")


# ═══════════════════════════════════════════════════════════════
# 3. W&B Sweep Configuration
# ═══════════════════════════════════════════════════════════════

def get_sweep_config() -> dict:
    """Hyperparameter-Sweep-Konfiguration für Q-Learning."""
    return {
        'method': 'bayes',
        'metric': {
            'name': 'q_learning/avg_reward_100',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                'distribution': 'log_uniform',
                'min': 1e-3,
                'max': 5e-1
            },
            'gamma': {
                'distribution': 'uniform',
                'min': 0.9,
                'max': 0.999
            },
            'epsilon_start': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 0.5
            },
            'epsilon_decay': {
                'distribution': 'uniform',
                'min': 0.99,
                'max': 0.999
            }
        }
    }


# ═══════════════════════════════════════════════════════════════
# 4. OpenPipe Fine-Tuning Logger
# ═══════════════════════════════════════════════════════════════

class OpenPipeLogger:
    """
    Loggt RL-Training-Daten für OpenPipe Fine-Tuning.
    Sammelt (state, action, reward)-Tupel für späteres Training.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENPIPE_API_KEY", "")
        self.episodes: List[dict] = []
        self.client = None

        if self.api_key and OPENPIPE_AVAILABLE:
            self.client = OpenAI(
                api_key="placeholder",  # OpenPipe routed via openpipe param
                openpipe={"api_key": self.api_key}
            )

    def log_episode(self, episode_data: dict):
        """Loggt eine Episode für Fine-Tuning."""
        self.episodes.append(episode_data)

    def get_training_data(self) -> List[dict]:
        """Gibt gesammelte Trainingsdaten zurück."""
        return self.episodes

    def export_jsonl(self, path: str = "rl_training_data.jsonl"):
        """Exportiert Trainingsdaten als JSONL für OpenPipe."""
        import json
        with open(path, 'w') as f:
            for ep in self.episodes:
                f.write(json.dumps(ep) + '\n')
        print(f"📦 {len(self.episodes)} Episoden exportiert → {path}")


# ═══════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Agent Reinforcement Learning — Demo")
    print("  mit W&B + OpenPipe Integration")
    print("=" * 60)

    # ── Setup Tracking ───────────────────────────────────────
    config = {
        "algorithm": "q-learning",
        "env": "gridworld-4x4",
        "lr": 0.1,
        "gamma": 0.99,
        "epsilon_start": 0.3,
        "episodes": 500
    }
    wandb_run, op_client = setup_tracking(
        project="rl-agent-training",
        config=config
    )

    # ── OpenPipe Logger ─────────────────────────────────────
    op_logger = OpenPipeLogger()

    # ── Q-Learning auf GridWorld ─────────────────────────────
    print("\n🎮 Q-Learning: GridWorld 4×4")
    env = GridWorld(4)
    agent = QLearning(env, lr=0.1, gamma=0.99, epsilon=0.3,
                      wandb_run=wandb_run)
    rewards = agent.train(episodes=500)

    print(f"   Episoden: 500")
    print(f"   Finale 10-Episoden Avg-Reward: {np.mean(rewards[-10:]):.3f}")

    # Policy visualisieren
    arrows = {0: "↑", 1: "→", 2: "↓", 3: "←", -1: "🎯"}
    policy = agent.get_policy()
    print("\n   Gelernte Policy:")
    for r in range(4):
        row = "   "
        for c in range(4):
            row += f" {arrows[policy[r, c]]} "
        print(row)

    # ── Policy Gradient ─────────────────────────────────────
    print("\n🧠 Policy Gradient (REINFORCE): CartPole-Simulation")
    print("   (Vereinfachte Simulation — 4D State, 2 Actions)")

    pg = PolicyGradient(state_dim=4, action_dim=2, lr=0.01,
                        wandb_run=wandb_run)

    # Simuliere CartPole-ähnliche Umgebung
    total_rewards = []
    for ep in range(200):
        state = np.random.randn(4) * 0.1
        episode = []
        for _ in range(100):
            action, _ = pg.sample_action(state)
            # Vereinfachte Reward-Funktion
            reward = 1.0 if action == 0 else -0.1
            episode.append((state, action, reward))
            state = np.random.randn(4) * 0.1

        pg.update(episode)
        ep_reward = sum(r for _, _, r in episode)
        total_rewards.append(ep_reward)

        # ── W&B Logging ──────────────────────────────────────
        if wandb_run and ep % 10 == 0:
            wandb_run.log({
                "policy_gradient/episode": ep,
                "policy_gradient/reward": ep_reward,
                "policy_gradient/avg_reward_50": np.mean(total_rewards[-50:])
            })

        # ── OpenPipe Logging ─────────────────────────────────
        op_logger.log_episode({
            "episode": ep,
            "algorithm": "reinforce",
            "reward": ep_reward,
            "actions": [a for _, a, _ in episode],
            "states_shape": episode[0][0].shape[0] if episode else 0
        })

    print(f"   Episoden: 200")
    print(f"   Finale 10-Episoden Avg-Reward: {np.mean(total_rewards[-10:]):.1f}")

    # ── OpenPipe Export ──────────────────────────────────────
    op_logger.export_jsonl("rl_training_data.jsonl")

    # ── W&B Sweep Info ───────────────────────────────────────
    if WANDB_AVAILABLE:
        print("\n📊 W&B Sweep-Konfiguration:")
        sweep_cfg = get_sweep_config()
        print(f"   Methode: {sweep_cfg['method']}")
        print(f"   Parameter: {list(sweep_cfg['parameters'].keys())}")

    # ── Cleanup ──────────────────────────────────────────────
    if wandb_run:
        wandb_run.finish()

    print("\n✅ RL-Demo abgeschlossen!")
    print("   📊 W&B Run: check wandb dashboard")
    print("   📦 OpenPipe Data: rl_training_data.jsonl")
