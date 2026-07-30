# 🤖 Agenten-Verstärkungslernen

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![W&B](https://img.shields.io/badge/W%26B-Tracking-orange.svg)](https://wandb.ai/)

**Grundlagen des Reinforcement Learning für KI-Agenten** — Q-Learning, Policy Gradient und Deep Q-Networks mit W&B Experiment Tracking und OpenPipe Fine-Tuning Logging.

## 📋 Beschreibung

Dieses Repository implementiert die drei fundamentalen Reinforcement-Learning-Algorithmen von Grund auf — ohne externe RL-Bibliotheken. Jeder Algorithmus ist in reinem Python/NumPy geschrieben und enthält eine interaktive Streamlit-App zur Visualisierung.

- **Q-Learning (tabular)** auf GridWorld mit Hindernissen und Cliff-Walking
- **Policy Gradient (REINFORCE)** mit manuellem 2-Layer-Netzwerk
- **Deep Q-Network (DQN)** mit PyTorch, Double DQN, Experience Replay und Target Network
- **W&B Integration** für Experiment-Tracking, Hyperparameter-Sweeps und Model-Artifacts
- **OpenPipe Integration** für Fine-Tuning-Datenlogging

## ✨ Features

- 🧠 **Drei RL-Algorithmen** — Q-Learning, REINFORCE, DQN in einem Repo
- 🎮 **GridWorld-Umgebungen** — Standard, Hindernisse, Cliff Walking
- 🎯 **CartPole-Simulation** — Physik-basierte Umgebung für DQN (kein gymnasium nötig)
- 📊 **W&B Tracking** — Automatisches Logging von Rewards, Loss, Epsilon, Policy-Visualisierung
- 🔧 **OpenPipe Logging** — Export von Trainingsdaten als JSONL für Fine-Tuning
- 💾 **Checkpointing** — Speichern/Laden von Q-Tables und Netzwerk-Gewichten
- 🖥️ **Streamlit-App** — Interaktive UI für Training, Policy-Visualisierung und Tracking-Status
- 🧪 **Test-Suite** — pytest-basierte Tests für alle Komponenten

## 🚀 Installation

```bash
# Repository klonen
git clone https://github.com/mark-baumann/agenten-verstaerkungslernen.git
cd agenten-verstaerkungslernen

# Virtuelle Umgebung erstellen
python3 -m venv .venv
source .venv/bin/activate

# Abhängigkeiten installieren
pip install -r requirements.txt

# Optional: W&B und OpenPipe
pip install wandb openpipe
```

## 🎮 Nutzung

### Streamlit-App starten

```bash
streamlit run app.py
```

Die App bietet vier Modi:
1. **Q-Learning GridWorld** — Grid-Größe, Episoden, Lernrate einstellen und Policy live sehen
2. **Policy Gradient Demo** — REINFORCE mit konfigurierbaren Parametern
3. **DQN CartPole** — Deep Q-Network auf CartPole-Simulation
4. **Tracking-Status** — W&B/OpenPipe Verbindungsstatus prüfen

### Training per CLI

```bash
# Alle Algorithmen trainieren
python train_rl.py

# Nur Q-Learning auf Cliff Walking
python train_rl.py --algo ql --env cliff --episodes 1000

# Nur DQN mit Hindernissen
python train_rl.py --algo dqn --env obstacles --size 6

# Ohne W&B/OpenPipe
python train_rl.py --no-wandb --no-openpipe
```

### Tests ausführen

```bash
pytest tests/ -v
```

## 🏗️ Tech-Stack

| Komponente | Technologie |
|---|---|
| **Sprache** | Python 3.10+ |
| **RL-Algorithmen** | NumPy, PyTorch (DQN) |
| **UI** | Streamlit |
| **Tracking** | Weights & Biases, OpenPipe |
| **Testing** | pytest |

## 📁 Projektstruktur

```
agenten-verstaerkungslernen/
├── app.py                  # Streamlit-App
├── rl_agent.py             # Q-Learning, Policy Gradient, DQN (Kernmodul)
├── train_rl.py             # CLI-Training mit W&B/OpenPipe
├── sweep_runner.py         # W&B Hyperparameter-Sweeps
├── wandb_utils.py          # W&B/OpenPipe Hilfsfunktionen
├── tests/
│   ├── conftest.py
│   ├── test_rl_agent.py
│   └── test_wandb_utils.py
├── checkpoints/            # Gespeicherte Modelle
└── wandb_runs/             # W&B Offline-Runs
```

## 👤 Autor

**Mark Baumann** — [GitHub](https://github.com/mark-baumann)

---

*Teil der Agenten-Toolkit-Sammlung. Für Fragen oder Beiträge: Issue erstellen oder Pull Request öffnen.*
