"""
Q-Function Pretraining Study — Inspiriert von arXiv:2607.27203
===============================================================
"Do You Really Need to Pretrain Q-Functions for Online RL Fine-Tuning?"
Perry Dong, Ron Polonsky, Dorsa Sadigh, Chelsea Finn (2026)

Forschungsfrage: Bei gegebenem vortrainiertem Policy-Netzwerk — 
sollte die Q-Funktion ebenfalls auf Offline-Daten vortrainiert werden,
oder reicht eine zufällige Initialisierung?

Dieses Modul implementiert einen experimentellen Vergleich:
1. Pretrained Q: Q-Netzwerk wird auf Offline-Demonstrationen vortrainiert
2. Random Q: Q-Netzwerk wird zufällig initialisiert
3. Beide werden dann online fine-getuned und verglichen
"""

import numpy as np
from collections import deque
from typing import List, Tuple, Optional, Dict
import random
import json
from pathlib import Path
from dataclasses import dataclass, field

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class OfflineDataset:
    """Container für Offline-Demonstrationsdaten."""
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray

    def __len__(self):
        return len(self.states)

    def sample_batch(self, batch_size: int):
        indices = np.random.choice(len(self), batch_size, replace=False)
        return (self.states[indices], self.actions[indices],
                self.rewards[indices], self.next_states[indices], self.dones[indices])


@dataclass
class ExperimentResult:
    """Ergebnisse eines Experiments."""
    name: str
    rewards: List[float] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def summary(self) -> Dict:
        if not self.rewards:
            return {}
        arr = np.array(self.rewards)
        window = np.convolve(arr, np.ones(50)/50, mode='valid')
        threshold = 0.9 * np.max(window) if len(window) > 0 and np.max(window) > 0 else 0
        conv_ep = 0
        for i, v in enumerate(window):
            if v >= threshold:
                conv_ep = i
                break
        return {
            "name": self.name,
            "episodes": len(arr),
            "final_avg_10": float(np.mean(arr[-10:])),
            "final_avg_100": float(np.mean(arr[-100:])),
            "max_reward": float(np.max(arr)),
            "convergence_episode": conv_ep,
        }


if TORCH_AVAILABLE:

    class QNetwork(nn.Module):
        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
            )
        def forward(self, x):
            return self.net(x)


    class QFunctionPretraining:
        """Framework zum Vergleich von vortrainierten vs. zufälligen Q-Funktionen."""

        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128,
                     lr: float = 0.001, gamma: float = 0.99, device: str = "cpu"):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.hidden_dim = hidden_dim
            self.lr = lr
            self.gamma = gamma
            self.device = torch.device(device)

        def create_fresh_network(self) -> QNetwork:
            return QNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)

        def pretrain_q_function(self, network: QNetwork, offline_data: OfflineDataset,
                                epochs: int = 50, batch_size: int = 64, verbose: bool = True) -> List[float]:
            optimizer = optim.Adam(network.parameters(), lr=self.lr)
            loss_fn = nn.MSELoss()
            losses = []
            for epoch in range(epochs):
                epoch_losses = []
                n_batches = max(1, len(offline_data) // batch_size)
                for _ in range(n_batches):
                    states, actions, rewards, next_states, dones = offline_data.sample_batch(batch_size)
                    states_t = torch.FloatTensor(states).to(self.device)
                    actions_t = torch.LongTensor(actions).to(self.device)
                    rewards_t = torch.FloatTensor(rewards).to(self.device)
                    next_states_t = torch.FloatTensor(next_states).to(self.device)
                    dones_t = torch.FloatTensor(dones).to(self.device)
                    q_values = network(states_t)
                    q_current = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        q_next = network(next_states_t)
                        q_target = rewards_t + self.gamma * q_next.max(1)[0] * (1 - dones_t)
                    loss = loss_fn(q_current, q_target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_losses.append(loss.item())
                avg_loss = np.mean(epoch_losses)
                losses.append(avg_loss)
                if verbose and epoch % 10 == 0:
                    print(f"  Pretraining Epoch {epoch}/{epochs} — Loss: {avg_loss:.6f}")
            if verbose:
                print(f"  Pretraining abgeschlossen — Final Loss: {losses[-1]:.6f}")
            return losses

        def run_comparison_experiment(self, env, offline_data: OfflineDataset,
                                      online_episodes: int = 500, pretrain_epochs: int = 50,
                                      batch_size: int = 64, epsilon_start: float = 1.0,
                                      epsilon_end: float = 0.01, epsilon_decay: float = 0.995,
                                      target_update: int = 10, memory_size: int = 10000,
                                      verbose: bool = True):
            print("=" * 70)
            print("  Q-Function Pretraining Study — arXiv:2607.27203")
            print("=" * 70)

            print("\nPhase 1: Q-Funktion auf Offline-Daten vortrainieren...")
            pretrained_q = self.create_fresh_network()
            pretrain_losses = self.pretrain_q_function(pretrained_q, offline_data, pretrain_epochs, batch_size, verbose)

            print(f"\nPhase 2: Pretrained Q online fine-tunen ({online_episodes} Episoden)...")
            pretrained_result = self._online_training(env, pretrained_q, online_episodes, batch_size,
                                                      epsilon_start, epsilon_end, epsilon_decay,
                                                      target_update, memory_size, "Pretrained Q", verbose)
            pretrained_result.metadata["pretrain_losses"] = pretrain_losses
            pretrained_result.metadata["pretrain_epochs"] = pretrain_epochs

            print(f"\nPhase 3: Zufällig initialisierte Q-Funktion trainieren ({online_episodes} Episoden)...")
            random_q = self.create_fresh_network()
            random_result = self._online_training(env, random_q, online_episodes, batch_size,
                                                  epsilon_start, epsilon_end, epsilon_decay,
                                                  target_update, memory_size, "Random Q", verbose)

            self._print_comparison(pretrained_result, random_result)
            return pretrained_result, random_result

        def _online_training(self, env, q_network: QNetwork, episodes: int, batch_size: int,
                             epsilon_start: float, epsilon_end: float, epsilon_decay: float,
                             target_update: int, memory_size: int, label: str, verbose: bool) -> ExperimentResult:
            target_net = QNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
            target_net.load_state_dict(q_network.state_dict())
            target_net.eval()
            optimizer = optim.Adam(q_network.parameters(), lr=self.lr)
            loss_fn = nn.MSELoss()
            memory = deque(maxlen=memory_size)
            epsilon = epsilon_start
            result = ExperimentResult(name=label)

            for ep in range(episodes):
                state = env.reset()
                if isinstance(state, tuple):
                    state = np.array(state, dtype=np.float32)
                else:
                    state = np.array([state], dtype=np.float32)
                total_reward = 0
                steps = 0
                done = False
                while not done:
                    if random.random() < epsilon:
                        action = random.randrange(self.action_dim)
                    else:
                        with torch.no_grad():
                            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                            action = q_network(state_t).argmax(dim=1).item()
                    next_state, reward, done = env.step(action)
                    if isinstance(next_state, tuple):
                        next_state = np.array(next_state, dtype=np.float32)
                    else:
                        next_state = np.array([next_state], dtype=np.float32)
                    memory.append((state, action, reward, next_state, float(done)))
                    state = next_state
                    total_reward += reward
                    steps += 1
                    if len(memory) >= batch_size:
                        batch = random.sample(memory, batch_size)
                        s, a, r, ns, d = zip(*batch)
                        s_t = torch.FloatTensor(np.array(s)).to(self.device)
                        a_t = torch.LongTensor(a).to(self.device)
                        r_t = torch.FloatTensor(r).to(self.device)
                        ns_t = torch.FloatTensor(np.array(ns)).to(self.device)
                        d_t = torch.FloatTensor(d).to(self.device)
                        q_vals = q_network(s_t)
                        q_cur = q_vals.gather(1, a_t.unsqueeze(1)).squeeze(1)
                        with torch.no_grad():
                            next_a = q_network(ns_t).argmax(1)
                            q_next_t = target_net(ns_t)
                            q_tgt = r_t + self.gamma * q_next_t.gather(1, next_a.unsqueeze(1)).squeeze(1) * (1 - d_t)
                        loss = loss_fn(q_cur, q_tgt)
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(q_network.parameters(), 10.0)
                        optimizer.step()
                        result.losses.append(loss.item())
                result.rewards.append(total_reward)
                epsilon = max(epsilon_end, epsilon * epsilon_decay)
                if ep % target_update == 0:
                    target_net.load_state_dict(q_network.state_dict())
                if verbose and ep % 50 == 0:
                    avg_r = np.mean(result.rewards[-50:]) if len(result.rewards) >= 50 else np.mean(result.rewards)
                    print(f"  [{label}] Ep {ep:4d}/{episodes} | eps={epsilon:.3f} | AvgR(50)={avg_r:.3f}")
            return result

        def _print_comparison(self, pretrained: ExperimentResult, random_init: ExperimentResult):
            ps = pretrained.summary()
            rs = random_init.summary()
            print("\n" + "=" * 70)
            print("  ERGEBNISSE: Pretrained Q vs. Random Q")
            print("=" * 70)
            print(f"  {'Metrik':<35} {'Pretrained Q':>15} {'Random Q':>15}")
            print(f"  {'-'*35} {'-'*15} {'-'*15}")
            print(f"  {'Final Avg Reward (10 Ep)':<35} {ps.get('final_avg_10', 0):>15.3f} {rs.get('final_avg_10', 0):>15.3f}")
            print(f"  {'Final Avg Reward (100 Ep)':<35} {ps.get('final_avg_100', 0):>15.3f} {rs.get('final_avg_100', 0):>15.3f}")
            print(f"  {'Max Reward':<35} {ps.get('max_reward', 0):>15.3f} {rs.get('max_reward', 0):>15.3f}")
            print(f"  {'Konvergenz-Episode (90%)':<35} {ps.get('convergence_episode', 0):>15d} {rs.get('convergence_episode', 0):>15d}")
            pret_50 = np.mean(pretrained.rewards[:50]) if len(pretrained.rewards) >= 50 else 0
            rand_50 = np.mean(random_init.rewards[:50]) if len(random_init.rewards) >= 50 else 0
            print(f"  {'Avg Reward (erste 50 Ep)':<35} {pret_50:>15.3f} {rand_50:>15.3f}")
            if rs.get('final_avg_100', 0) > 0:
                imp = (ps.get('final_avg_100', 0) - rs.get('final_avg_100', 0)) / abs(rs.get('final_avg_100', 0)) * 100
                print(f"  {'Verbesserung durch Pretraining':<35} {imp:>14.1f}%")
            print("=" * 70)
            print("\n  FAZIT:")
            if pret_50 > rand_50 * 1.1:
                print("  Pretraining verbessert die Sample Efficiency signifikant.")
            elif ps.get('final_avg_100', 0) > rs.get('final_avg_100', 0) * 1.05:
                print("  Pretraining führt zu besserer finaler Performance.")
            else:
                print("  Kein signifikanter Vorteil durch Pretraining (Paper-Erkenntnis bestätigt).")
            print()


def generate_offline_demonstrations(env, num_episodes: int = 100, policy: str = "random",
                                    expert_noise: float = 0.2) -> OfflineDataset:
    all_states, all_actions, all_rewards, all_next_states, all_dones = [], [], [], [], []
    for ep in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = np.array(state, dtype=np.float32)
        else:
            state = np.array([state], dtype=np.float32)
        done = False
        for _ in range(200):
            if policy == "random":
                action = random.randrange(4)
            elif policy == "expert" and hasattr(env, 'goal'):
                goal = np.array(env.goal, dtype=np.float32)
                if random.random() < expert_noise:
                    action = random.randrange(4)
                else:
                    diff = goal - state
                    action = 2 if diff[0] > 0 else 0 if abs(diff[0]) > abs(diff[1]) else (1 if diff[1] > 0 else 3)
            else:
                action = random.randrange(4)
            next_state, reward, done = env.step(action)
            if isinstance(next_state, tuple):
                next_state = np.array(next_state, dtype=np.float32)
            else:
                next_state = np.array([next_state], dtype=np.float32)
            all_states.append(state)
            all_actions.append(action)
            all_rewards.append(reward)
            all_next_states.append(next_state)
            all_dones.append(float(done))
            state = next_state
            if done:
                break
    return OfflineDataset(
        states=np.array(all_states, dtype=np.float32),
        actions=np.array(all_actions),
        rewards=np.array(all_rewards, dtype=np.float32),
        next_states=np.array(all_next_states, dtype=np.float32),
        dones=np.array(all_dones, dtype=np.float32),
    )


def save_results(pretrained: ExperimentResult, random_init: ExperimentResult, output_dir: str = "results"):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    results = {
        "experiment": "q_function_pretraining_study",
        "paper": "arXiv:2607.27203",
        "paper_title": "Do You Really Need to Pretrain Q-Functions for Online RL Fine-Tuning?",
        "pretrained": {"summary": pretrained.summary(), "rewards": pretrained.rewards,
                       "losses": pretrained.losses, "metadata": pretrained.metadata},
        "random": {"summary": random_init.summary(), "rewards": random_init.rewards,
                   "losses": random_init.losses, "metadata": random_init.metadata},
    }
    filepath = output_path / "q_pretraining_results.json"
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Ergebnisse gespeichert -> {filepath}")
    return filepath


def run_demo():
    if not TORCH_AVAILABLE:
        print("PyTorch nicht verfügbar — Demo übersprungen.")
        return
    print("Q-Function Pretraining Study — Demo (arXiv:2607.27203)")

    class DummyEnv:
        def __init__(self):
            self.state_dim = 4
            self.action_dim = 2
            self.goal = np.array([1.0, 1.0, 1.0, 1.0])
        def reset(self):
            return np.random.randn(self.state_dim).astype(np.float32) * 0.1
        def step(self, action):
            ns = np.random.randn(self.state_dim).astype(np.float32) * 0.1
            r = 1.0 if action == 0 else -0.1
            d = np.random.random() < 0.02
            return ns, r, d

    env = DummyEnv()
    print("Generiere Offline-Daten...")
    offline_data = generate_offline_demonstrations(env, num_episodes=50, policy="random")
    print(f"  {len(offline_data)} Transitionen gesammelt")

    study = QFunctionPretraining(state_dim=env.state_dim, action_dim=env.action_dim,
                                 hidden_dim=64, lr=0.001, gamma=0.99)
    pretrained_result, random_result = study.run_comparison_experiment(
        env=env, offline_data=offline_data, online_episodes=200, pretrain_epochs=20,
        batch_size=32, epsilon_decay=0.99, target_update=5, memory_size=2000, verbose=True)
    save_results(pretrained_result, random_result)
    print("Demo abgeschlossen!")


if __name__ == "__main__":
    run_demo()
