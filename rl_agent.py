"""
Agent Reinforcement Learning — Q-Learning, DQN & Policy Gradient
=================================================================
Grundlagen des Reinforcement Learning für AI Agents.
Mit W&B Experiment Tracking & OpenPipe Fine-Tuning.

Implementiert:
1. Q-Learning (Tabular) — GridWorld
2. Deep Q-Network (DQN) — PyTorch, Double DQN
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
                   use_openpipe: bool = True,
                   tags: list = None,
                   group: str = None,
                   job_type: str = "train"):
    """
    Initialisiert W&B und/oder OpenPipe für Experiment-Tracking.

    Args:
        project: Projektname für W&B
        config: Hyperparameter-Dict
        use_wandb: W&B-Tracking aktivieren
        use_openpipe: OpenPipe-Tracking aktivieren
        tags: Tags für W&B-Run
        group: Gruppe für W&B-Run (z.B. "q-learning-experiments")
        job_type: Job-Typ (train, eval, sweep)
    """
    run = None
    op_client = None

    if use_wandb and WANDB_AVAILABLE:
        mode = "online" if os.environ.get("WANDB_API_KEY") else "offline"
        run = wandb.init(
            project=project,
            config=config or {},
            mode=mode,
            tags=tags or ["rl", "q-learning", "policy-gradient"],
            group=group,
            job_type=job_type,
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


def log_model_artifact(wandb_run, checkpoint_path: str, model_name: str,
                       metadata: dict = None, aliases: list = None):
    """Loggt ein Modell-Checkpoint als W&B Artifact."""
    if not wandb_run or not WANDB_AVAILABLE:
        return
    import wandb as wb
    artifact = wb.Artifact(
        name=model_name,
        type="model",
        metadata=metadata or {},
    )
    artifact.add_file(checkpoint_path)
    wandb_run.log_artifact(artifact, aliases=aliases or ["latest"])
    print(f"📦 W&B Artifact geloggt: {model_name} → {checkpoint_path}")


def log_predictions_table(wandb_run, states: list, actions: list,
                          rewards: list, table_name: str = "predictions"):
    """Loggt eine Tabelle mit Vorhersagen für W&B Dashboard."""
    if not wandb_run or not WANDB_AVAILABLE:
        return
    import wandb as wb
    table = wb.Table(columns=["state", "action", "reward"])
    for s, a, r in zip(states, actions, rewards):
        table.add_data(str(s), a, r)
    wandb_run.log({table_name: table})


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
        import ast
        data = np.load(path, allow_pickle=True)
        self.lr = float(data['lr'])
        self.gamma = float(data['gamma'])
        self.epsilon = float(data['epsilon'])
        for k, v in zip(data['q_keys'], data['q_values']):
            self.Q[ast.literal_eval(k)] = v
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
# 3. Deep Q-Network (DQN) — PyTorch
# ═══════════════════════════════════════════════════════════════

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from collections import deque
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None  # type: ignore
    optim = None  # type: ignore
    torch = None  # type: ignore


if TORCH_AVAILABLE:

    class DQNNetwork(nn.Module):
        """Einfaches MLP für Q-Wert-Approximation."""

        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
            )

        def forward(self, x):
            return self.net(x)


    class DQNAgent:
        """
        Deep Q-Network mit Experience Replay, Target Network, Double DQN.

        Funktioniert mit GridWorld (state = (row, col) → normalisiert) und
        anderen Umgebungen mit flachem State-Vektor.
        """

        def __init__(self, state_dim: int, action_dim: int,
                     lr: float = 0.001, gamma: float = 0.99,
                     epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                     epsilon_decay: float = 0.995,
                     memory_size: int = 10_000, batch_size: int = 64,
                     target_update: int = 10, hidden_dim: int = 128,
                     wandb_run=None, device: str = "cpu"):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.gamma = gamma
            self.epsilon = epsilon_start
            self.epsilon_end = epsilon_end
            self.epsilon_decay = epsilon_decay
            self.batch_size = batch_size
            self.target_update = target_update
            self.wandb_run = wandb_run
            self.device = torch.device(device)

            self.policy_net = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
            self.target_net = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()

            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
            self.memory = deque(maxlen=memory_size)
            self.loss_fn = nn.MSELoss()

        def select_action(self, state, evaluate: bool = False) -> int:
            """Epsilon-greedy Action-Selection.

            Args:
                state: Current state (numpy array or tuple).
                evaluate: If True, always pick greedy action (no exploration).

            Returns:
                Selected action index.
            """
            if evaluate or random.random() > self.epsilon:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_values = self.policy_net(state_t)
                    return q_values.argmax(dim=1).item()
            return random.randrange(self.action_dim)

        def push(self, state, action, reward, next_state, done):
            """Store a transition in the replay buffer.

            Args:
                state: Current state before the action.
                action: Action taken.
                reward: Reward received.
                next_state: Resulting state after the action.
                done: Whether the episode terminated.
            """
            self.memory.append((state, action, reward, next_state, done))

        def update(self):
            """Perform one Double DQN training step.

            Samples a batch from replay memory and updates the policy network
            using the Double DQN target computation.

            Returns:
                Loss value (float) if enough samples in memory, else None.
            """
            if len(self.memory) < self.batch_size:
                return None

            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(np.array(states)).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)

            q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_actions = self.policy_net(next_states).argmax(dim=1)
                next_q_values = self.target_net(next_states).gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)
                target_q = rewards + self.gamma * next_q_values * (1 - dones)

            loss = self.loss_fn(q_values, target_q)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()

            return loss.item()

        def update_target(self):
            """Target-Network auf Policy-Network synchronisieren."""
            self.target_net.load_state_dict(self.policy_net.state_dict())

        def decay_epsilon(self):
            """Epsilon exponentiell reduzieren."""
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        def train(self, env, episodes: int = 500, log_interval: int = 10):
            """
            Vollständiges DQN-Training auf einer Umgebung.

            Args:
                env: Umgebung mit .reset() → state, .step(action) → (next_state, reward, done)
                episodes: Anzahl Episoden
                log_interval: W&B-Logging-Intervall

            Returns:
                Liste von Episode-Rewards
            """
            rewards_history = []
            losses_history = []

            for ep in range(1, episodes + 1):
                state = env.reset()
                episode_reward = 0.0
                episode_losses = []
                steps = 0
                done = False

                while not done:
                    action = self.select_action(state)
                    next_state, reward, done = env.step(action)

                    self.push(state, action, reward, next_state, done)
                    state = next_state
                    episode_reward += reward
                    steps += 1

                    loss = self.update()
                    if loss is not None:
                        episode_losses.append(loss)

                rewards_history.append(episode_reward)
                avg_loss = np.mean(episode_losses) if episode_losses else 0.0
                losses_history.append(avg_loss)

                self.decay_epsilon()

                if ep % self.target_update == 0:
                    self.update_target()

                if self.wandb_run and ep % log_interval == 0:
                    self.wandb_run.log({
                        "dqn/episode": ep,
                        "dqn/reward": episode_reward,
                        "dqn/avg_reward_100": np.mean(rewards_history[-100:]),
                        "dqn/epsilon": self.epsilon,
                        "dqn/loss": avg_loss,
                        "dqn/steps": steps,
                    })

            return rewards_history

        def save_checkpoint(self, path: str = "dqn_checkpoint.pt"):
            """Speichert Modell, Optimizer und Hyperparameter."""
            torch.save({
                'policy_net': self.policy_net.state_dict(),
                'target_net': self.target_net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'gamma': self.gamma,
            }, path)
            print(f"💾 DQN Checkpoint gespeichert → {path}")

        def load_checkpoint(self, path: str = "dqn_checkpoint.pt"):
            """Lädt Modell, Optimizer und Hyperparameter."""
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            self.gamma = checkpoint['gamma']
            print(f"📂 DQN Checkpoint geladen ← {path}")


# ═══════════════════════════════════════════════════════════════
# 4. W&B Sweep Configurations
# ═══════════════════════════════════════════════════════════════

def get_sweep_config(algo: str = "q_learning") -> dict:
    """Hyperparameter-Sweep-Konfiguration für Q-Learning oder DQN."""
    if algo == "dqn":
        return {
            'method': 'bayes',
            'metric': {
                'name': 'dqn/avg_reward_100',
                'goal': 'maximize'
            },
            'parameters': {
                'learning_rate': {
                    'distribution': 'log_uniform',
                    'min': 1e-4,
                    'max': 1e-2
                },
                'gamma': {
                    'distribution': 'uniform',
                    'min': 0.9,
                    'max': 0.999
                },
                'batch_size': {
                    'values': [32, 64, 128]
                },
                'hidden_dim': {
                    'values': [64, 128, 256]
                },
                'target_update': {
                    'values': [5, 10, 20]
                }
            }
        }
    elif algo == "policy_gradient":
        return {
            'method': 'bayes',
            'metric': {
                'name': 'policy_gradient/avg_reward_50',
                'goal': 'maximize'
            },
            'parameters': {
                'learning_rate': {
                    'distribution': 'log_uniform',
                    'min': 1e-4,
                    'max': 1e-1
                },
                'gamma': {
                    'distribution': 'uniform',
                    'min': 0.9,
                    'max': 0.999
                },
                'hidden_dim': {
                    'values': [16, 32, 64]
                }
            }
        }
    else:
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

    # ── DQN (PyTorch) auf GridWorld ──────────────────────────
    if TORCH_AVAILABLE:
        print("\n🤖 DQN (PyTorch): GridWorld 4×4")
        dqn_env = GridWorld(4)
        dqn_agent = DQNAgent(
            state_dim=2, action_dim=4,
            lr=0.001, gamma=0.99,
            epsilon_start=1.0, epsilon_decay=0.995,
            memory_size=5000, batch_size=32,
            target_update=10, hidden_dim=64,
            wandb_run=wandb_run,
        )
        dqn_rewards = dqn_agent.train(dqn_env, episodes=300)
        print(f"   Episoden: 300")
        print(f"   Finale 10-Episoden Avg-Reward: {np.mean(dqn_rewards[-10:]):.3f}")

        # DQN Policy visualisieren
        print("\n   Gelernte DQN-Policy:")
        for r in range(4):
            row = "   "
            for c in range(4):
                if (r, c) == dqn_env.goal:
                    row += " 🎯 "
                else:
                    state_vec = np.array([r / 4, c / 4], dtype=np.float32)
                    with torch.no_grad():
                        q_vals = dqn_agent.policy_net(
                            torch.FloatTensor(state_vec).unsqueeze(0)
                        )
                        best = q_vals.argmax(dim=1).item()
                    row += f" {arrows[best]} "
            print(row)

        dqn_agent.save_checkpoint("checkpoints/dqn_gridworld_4x4.pt")
    else:
        print("\n⚠️  PyTorch nicht installiert — DQN übersprungen")

    # ── OpenPipe Export ──────────────────────────────────────
    op_logger.export_jsonl("rl_training_data.jsonl")

    # ── W&B Sweep Info ───────────────────────────────────────
    if WANDB_AVAILABLE:
        print("\n📊 W&B Sweep-Konfigurationen:")
        for algo_name in ["q_learning", "dqn", "policy_gradient"]:
            sweep_cfg = get_sweep_config(algo_name)
            print(f"   {algo_name}: method={sweep_cfg['method']}, "
                  f"params={list(sweep_cfg['parameters'].keys())}")

    # ── Cleanup ──────────────────────────────────────────────
    if wandb_run:
        wandb_run.finish()

    print("\n✅ RL-Demo abgeschlossen!")
    print("   📊 W&B Run: check wandb dashboard")
    print("   📦 OpenPipe Data: rl_training_data.jsonl")
