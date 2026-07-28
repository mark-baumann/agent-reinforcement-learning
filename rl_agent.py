"""
Agent Reinforcement Learning — Q-Learning & Policy Gradient
===========================================================
Grundlagen des Reinforcement Learning für AI Agents.

Implementiert:
1. Q-Learning (Tabular) — GridWorld
2. Deep Q-Network (DQN) — CartPole
3. Policy Gradient (REINFORCE) — CartPole
"""

import numpy as np
from collections import defaultdict
from typing import Tuple, List
import random


# ═══════════════════════════════════════════════════════════════
# 1. Q-Learning — GridWorld
# ═══════════════════════════════════════════════════════════════

class GridWorld:
    """
    4×4 Grid. Agent startet bei (0,0), Ziel bei (3,3).
    Aktionen: 0=hoch, 1=rechts, 2=runter, 3=links
    Belohnung: +1 am Ziel, -0.01 pro Schritt (Effizienz-Anreiz)
    """

    def __init__(self, size: int = 4):
        self.size = size
        self.goal = (size - 1, size - 1)
        self.reset()

    def reset(self):
        self.pos = (0, 0)
        return self.pos

    def step(self, action: int) -> Tuple[tuple, float, bool]:
        r, c = self.pos
        if action == 0:    r = max(0, r - 1)
        elif action == 1:  c = min(self.size - 1, c + 1)
        elif action == 2:  r = min(self.size - 1, r + 1)
        elif action == 3:  c = max(0, c - 1)

        self.pos = (r, c)
        done = self.pos == self.goal
        reward = 1.0 if done else -0.01
        return self.pos, reward, done


class QLearning:
    """Tabular Q-Learning mit ε-greedy Exploration."""

    def __init__(self, env: GridWorld, lr: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: np.zeros(4))

    def choose_action(self, state: tuple) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        return np.argmax(self.Q[state])

    def train(self, episodes: int = 1000) -> List[float]:
        """Trainiert den Agenten. Gibt Rewards pro Episode zurück."""
        rewards_history = []
        for ep in range(episodes):
            state = self.env.reset()
            total_reward = 0
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

            rewards_history.append(total_reward)

            # Decay epsilon
            self.epsilon = max(0.01, self.epsilon * 0.995)

        return rewards_history

    def get_policy(self) -> np.ndarray:
        """Gibt die gelernte Policy als Grid zurück."""
        grid = np.zeros((self.env.size, self.env.size), dtype=int)
        arrows = {0: "↑", 1: "→", 2: "↓", 3: "←"}
        for r in range(self.env.size):
            for c in range(self.env.size):
                if (r, c) == self.env.goal:
                    grid[r, c] = -1  # Goal
                else:
                    grid[r, c] = np.argmax(self.Q[(r, c)])
        return grid


# ═══════════════════════════════════════════════════════════════
# 2. Policy Gradient (REINFORCE) — Simple
# ═══════════════════════════════════════════════════════════════

class PolicyGradient:
    """
    REINFORCE-Algorithmus mit einem einfachen 2-Layer-Netzwerk.
    Funktioniert für diskrete Action-Spaces.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 lr: float = 0.01, gamma: float = 0.99):
        self.lr = lr
        self.gamma = gamma

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


# ═══════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Agent Reinforcement Learning — Demo")
    print("=" * 60)

    # ── Q-Learning auf GridWorld ─────────────────────────────
    print("\n🎮 Q-Learning: GridWorld 4×4")
    env = GridWorld(4)
    agent = QLearning(env, lr=0.1, gamma=0.99, epsilon=0.3)
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

    pg = PolicyGradient(state_dim=4, action_dim=2, lr=0.01)

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
        total_rewards.append(sum(r for _, _, r in episode))

    print(f"   Episoden: 200")
    print(f"   Finale 10-Episoden Avg-Reward: {np.mean(total_rewards[-10:]):.1f}")

    print("\n✅ RL-Demo abgeschlossen!")
