# 🎮 Agent Reinforcement Learning

**Q-Learning, Policy Gradient & DQN — mit W&B Experiment Tracking & OpenPipe Fine-Tuning.**

## 📦 Installation

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## 🚀 Quick Start

```bash
# Demo (alle Algorithmen)
python rl_agent.py

# Volles Training
python train_rl.py --env standard --algo all --episodes 500

# Cliff Walking
python train_rl.py --env cliff --episodes 1000

# GridWorld mit Hindernissen
python train_rl.py --env obstacles --size 6

# Nur Policy Gradient
python train_rl.py --algo pg --episodes 300
```

## 🔍 Hyperparameter Sweep

```bash
# Lokaler Grid-Search (kein W&B API-Key nötig)
python sweep_runner.py --count 2 --offline --episodes 200

# W&B Cloud Sweep (braucht API-Key)
python sweep_runner.py --count 20 --env cliff
```

## 📊 Experiment Tracking

- **W&B**: `wandb.init(mode="offline")` falls kein API-Key — alle Metriken werden lokal geloggt
- **OpenPipe**: Episoden-Daten werden als JSONL exportiert für Fine-Tuning

## 🧠 Algorithmen

| Algorithmus | Umgebung | Typ |
|---|---|---|
| Q-Learning (Tabular) | GridWorld 4×4/6×6/8×8 | Value-based |
| Q-Learning (Tabular) | Cliff Walking | Value-based |
| Q-Learning (Tabular) | GridWorld + Obstacles | Value-based |
| Policy Gradient (REINFORCE) | CartPole-Simulation | Policy-based |

## 📁 Struktur

```
agent-reinforcement-learning/
├── rl_agent.py          # Core: GridWorld, QLearning, PolicyGradient, W&B, OpenPipe
├── train_rl.py          # Training-Pipeline mit CLI
├── sweep_runner.py      # Hyperparameter-Sweep (lokal + W&B Cloud)
├── checkpoints/         # Gespeicherte Modelle
├── wandb/               # W&B Offline-Runs
└── rl_training_data.jsonl  # OpenPipe Fine-Tuning-Daten
```

## 🏆 Beste Hyperparameter (GridSearch)

| Parameter | Wert |
|---|---|
| Learning Rate | 0.5 |
| Gamma | 0.99 |
| Epsilon Start | 0.1 |
| Avg Reward (100 Ep.) | 0.936 |
