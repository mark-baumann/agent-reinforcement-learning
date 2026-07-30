"""
Streamlit-App: Agenten-Verstärkungslernen
=========================================
GridWorld Q-Learning live, Policy Gradient Demo, DQN CartPole, W&B/OpenPipe Tracking-Status.
"""

import streamlit as st
import numpy as np
import time
import os
from collections import defaultdict
import random

# ── RL-Agent-Module (keine Code-Duplizierung) ────────────────
from rl_agent import (
    GridWorld, QLearning, PolicyGradient,
    WANDB_AVAILABLE, OPENPIPE_AVAILABLE, TORCH_AVAILABLE,
)

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Agenten-Verstärkungslernen",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Agenten-Verstärkungslernen")
st.markdown("GridWorld Q-Learning · Policy Gradient · DQN CartPole · W&B/OpenPipe Tracking")

# ── Sidebar: Modus-Auswahl ───────────────────────────────────
mode = st.sidebar.selectbox(
    "Modus wählen",
    ["Q-Learning GridWorld", "Policy Gradient Demo", "DQN CartPole", "Tracking-Status"],
)

# ═══════════════════════════════════════════════════════════════
# 1. Q-Learning GridWorld
# ═══════════════════════════════════════════════════════════════

if mode == "Q-Learning GridWorld":
    st.header("🧠 Q-Learning — GridWorld")

    col1, col2, col3 = st.columns(3)
    with col1:
        size = st.slider("Grid-Größe", 3, 8, 4)
    with col2:
        episodes = st.slider("Episoden", 100, 2000, 500, 100)
    with col3:
        lr = st.selectbox("Lernrate", [0.01, 0.05, 0.1, 0.2, 0.5], index=2)

    col4, col5 = st.columns(2)
    with col4:
        gamma = st.selectbox("Gamma (Discount)", [0.9, 0.95, 0.99], index=2)
    with col5:
        epsilon_start = st.slider("Epsilon (Exploration)", 0.01, 1.0, 0.3, 0.05)

    env_type = st.radio("Umgebung", ["Standard", "Hindernisse", "Cliff"], horizontal=True)

    if st.button("🚀 Training starten", type="primary"):
        # ── Environment ──────────────────────────────────────
        obstacles = None
        cliff = False
        if env_type == "Hindernisse":
            obstacles = [(1, 1), (2, 2), (1, 3)] if size >= 4 else [(1, 1)]
        elif env_type == "Cliff":
            cliff = True

        env = GridWorld(size, obstacles=obstacles, cliff=cliff)
        agent = QLearning(env, lr=lr, gamma=gamma, epsilon=epsilon_start)
        rewards_history = agent.train(episodes=episodes)

        progress_bar = st.progress(1.0)
        status_text = st.empty()
        status_text.text(f"✅ Training abgeschlossen! {episodes} Episoden.")
        chart_placeholder = st.empty()

        # ── Reward-Verlauf ──────────────────────────────────
        chart_placeholder.line_chart(rewards_history, height=300)

        # ── Policy-Grid ─────────────────────────────────────
        st.subheader("📋 Gelernte Policy")
        action_names = {0: "↑", 1: "→", 2: "↓", 3: "←"}
        policy = agent.get_policy()
        grid = np.zeros((size, size), dtype=object)
        for r in range(size):
            for c in range(size):
                if (r, c) == env.goal:
                    grid[r, c] = "🎯"
                elif (r, c) in (obstacles or set()):
                    grid[r, c] = "🧱"
                else:
                    grid[r, c] = action_names.get(int(policy[r, c]), "?")

        # Als Tabelle anzeigen
        cols = st.columns(size)
        for r in range(size):
            for c in range(size):
                bg = "#e8f5e9" if (r, c) == env.goal else "#ffebee" if (r, c) in (obstacles or set()) else "#f5f5f5"
                cols[c].markdown(
                    f"<div style='text-align:center;padding:10px;background:{bg};border-radius:5px;font-size:20px;'>{grid[r,c]}</div>",
                    unsafe_allow_html=True,
                )

        # ── W&B Tracking Status ──────────────────────────────
        st.subheader("📊 W&B Tracking-Status")
        if WANDB_AVAILABLE:
            api_key = os.environ.get("WANDB_API_KEY", "")
            if api_key:
                st.success("✅ W&B verfügbar (online) — API-Key gesetzt")
            else:
                st.warning("⚠️ W&B verfügbar (offline) — kein API-Key gesetzt. Läuft im Offline-Modus.")
        else:
            st.info("ℹ️ W&B nicht installiert. `pip install wandb` für Experiment-Tracking.")

        if OPENPIPE_AVAILABLE:
            op_key = os.environ.get("OPENPIPE_API_KEY", "")
            if op_key:
                st.success("✅ OpenPipe verfügbar — API-Key gesetzt")
            else:
                st.warning("⚠️ OpenPipe installiert, aber kein API-Key gesetzt")
        else:
            st.info("ℹ️ OpenPipe nicht installiert. `pip install openpipe` für Fine-Tuning-Logging.")

# ═══════════════════════════════════════════════════════════════
# 2. Policy Gradient Demo
# ═══════════════════════════════════════════════════════════════

elif mode == "Policy Gradient Demo":
    st.header("📈 Policy Gradient (REINFORCE) — Demo")

    st.markdown("""
    Der **REINFORCE**-Algorithmus lernt eine Policy direkt durch Gradientenabstieg.
    Hier: Ein 2-Layer-Netzwerk lernt, einen 4-dimensionalen State auf 2 Aktionen abzubilden.
    """)

    col1, col2 = st.columns(2)
    with col1:
        episodes_pg = st.slider("Episoden", 50, 1000, 200, 50, key="pg_episodes")
        lr_pg = st.selectbox("Lernrate", [0.001, 0.005, 0.01, 0.05, 0.1], index=2, key="pg_lr")
    with col2:
        gamma_pg = st.selectbox("Gamma", [0.9, 0.95, 0.99], index=2, key="pg_gamma")
        state_dim = st.slider("State-Dimension", 2, 8, 4, key="pg_state")

    if st.button("🚀 Policy Gradient starten", type="primary", key="pg_btn"):
        action_dim = 2
        pg = PolicyGradient(state_dim=state_dim, action_dim=action_dim,
                           lr=lr_pg, gamma=gamma_pg)

        rewards_pg = []
        progress = st.progress(0)
        status = st.empty()
        chart_ph = st.empty()

        for ep in range(episodes_pg):
            episode_data = []
            state = np.random.randn(state_dim) * 0.5
            for _ in range(20):
                action, probs = pg.sample_action(state)
                reward = 1.0 if action == 0 else -0.5
                reward += np.dot(state, np.ones(state_dim)) * 0.1
                episode_data.append((state.copy(), action, reward))
                state = np.random.randn(state_dim) * 0.5

            pg.update(episode_data)

            total_r = sum(r for _, _, r in episode_data)
            rewards_pg.append(total_r)

            if ep % 10 == 0:
                progress.progress((ep + 1) / episodes_pg)
                status.text(f"Episode {ep+1}/{episodes_pg} — Reward: {total_r:.3f}")

        progress.progress(1.0)
        status.text(f"✅ Policy Gradient abgeschlossen!")
        chart_ph.line_chart(rewards_pg, height=300)

        st.subheader("📊 Finale Policy-Verteilung")
        test_states = np.random.randn(5, state_dim) * 0.5
        for i, s in enumerate(test_states):
            probs = pg.forward(s)
            st.write(f"State {i+1}: Aktion 0 = {probs[0]:.3f}, Aktion 1 = {probs[1]:.3f}")

# ═══════════════════════════════════════════════════════════════
# 3. DQN CartPole
# ═══════════════════════════════════════════════════════════════

elif mode == "DQN CartPole":
    st.header("🎮 DQN — CartPole Simulation")

    st.markdown("""
    **Deep Q-Network** für CartPole: Ein neuronales Netz approximiert Q-Werte.
    Die Umgebung wird simuliert (kein gymnasium/gym nötig).
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        dqn_episodes = st.slider("Episoden", 50, 500, 200, 25, key="dqn_ep")
    with col2:
        dqn_lr = st.selectbox("Lernrate", [0.0005, 0.001, 0.005, 0.01], index=1, key="dqn_lr")
    with col3:
        dqn_hidden = st.selectbox("Hidden-Dim", [32, 64, 128], index=1, key="dqn_hid")

    if st.button("🚀 DQN Training starten", type="primary", key="dqn_btn"):
        # Einfache CartPole-Simulation
        class CartPoleSim:
            def __init__(self):
                self.reset()

            def reset(self):
                self.x = np.random.uniform(-0.05, 0.05)
                self.x_dot = np.random.uniform(-0.05, 0.05)
                self.theta = np.random.uniform(-0.05, 0.05)
                self.theta_dot = np.random.uniform(-0.05, 0.05)
                return np.array([self.x, self.x_dot, self.theta, self.theta_dot])

            def step(self, action):
                force = 10.0 if action == 1 else -10.0
                gravity = 9.8
                masscart = 1.0
                masspole = 0.1
                total_mass = masscart + masspole
                length = 0.5
                polemass_length = masspole * length
                tau = 0.02

                temp = (force + polemass_length * self.theta_dot**2 * np.sin(self.theta)) / total_mass
                theta_acc = (gravity * np.sin(self.theta) - np.cos(self.theta) * temp) / (
                    length * (4.0/3.0 - masspole * np.cos(self.theta)**2 / total_mass)
                )
                x_acc = temp - polemass_length * theta_acc * np.cos(self.theta) / total_mass

                self.x += tau * self.x_dot
                self.x_dot += tau * x_acc
                self.theta += tau * self.theta_dot
                self.theta_dot += tau * theta_acc

                done = abs(self.x) > 2.4 or abs(self.theta) > 0.2095
                reward = 1.0 if not done else 0.0
                return np.array([self.x, self.x_dot, self.theta, self.theta_dot]), reward, done

        # Einfaches neuronales Netz (numpy-basiert, kein PyTorch nötig)
        class SimpleDQN:
            def __init__(self, state_dim, action_dim, hidden_dim, lr):
                self.W1 = np.random.randn(state_dim, hidden_dim) * 0.1
                self.b1 = np.zeros(hidden_dim)
                self.W2 = np.random.randn(hidden_dim, hidden_dim) * 0.1
                self.b2 = np.zeros(hidden_dim)
                self.W3 = np.random.randn(hidden_dim, action_dim) * 0.1
                self.b3 = np.zeros(action_dim)
                self.lr = lr

            def forward(self, x):
                self.z1 = x @ self.W1 + self.b1
                self.a1 = np.maximum(0, self.z1)
                self.z2 = self.a1 @ self.W2 + self.b2
                self.a2 = np.maximum(0, self.z2)
                self.z3 = self.a2 @ self.W3 + self.b3
                return self.z3

            def update(self, x, y):
                pred = self.forward(x)
                error = pred - y
                dW3 = np.outer(self.a2, error)
                db3 = error
                da2 = error @ self.W3.T
                dz2 = da2 * (self.z2 > 0)
                dW2 = np.outer(self.a1, dz2)
                db2 = dz2
                da1 = dz2 @ self.W2.T
                dz1 = da1 * (self.z1 > 0)
                dW1 = np.outer(x, dz1)
                db1 = dz1
                self.W3 -= self.lr * dW3
                self.b3 -= self.lr * db3
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1

        env = CartPoleSim()
        agent = SimpleDQN(4, 2, dqn_hidden, dqn_lr)
        gamma = 0.99
        epsilon = 1.0
        rewards_dqn = []
        memory = []
        batch_size = 32

        progress = st.progress(0)
        status = st.empty()
        chart_ph = st.empty()

        for ep in range(dqn_episodes):
            state = env.reset()
            total_reward = 0
            done = False
            while not done:
                if random.random() < epsilon:
                    action = random.randint(0, 1)
                else:
                    q_vals = agent.forward(state)
                    action = int(np.argmax(q_vals))

                next_state, reward, done = env.step(action)
                memory.append((state, action, reward, next_state, done))
                if len(memory) > 2000:
                    memory.pop(0)

                if len(memory) >= batch_size:
                    batch = random.sample(memory, batch_size)
                    for s, a, r, ns, d in batch:
                        target = r
                        if not d:
                            target += gamma * np.max(agent.forward(ns))
                        q_vals = agent.forward(s)
                        q_vals[a] = target
                        agent.update(s, q_vals)

                state = next_state
                total_reward += reward

            rewards_dqn.append(total_reward)
            epsilon = max(0.01, epsilon * 0.995)

            if ep % 10 == 0:
                progress.progress((ep + 1) / dqn_episodes)
                status.text(f"Episode {ep+1}/{dqn_episodes} — Reward: {total_reward:.1f} — ε: {epsilon:.3f}")

        progress.progress(1.0)
        status.text(f"✅ DQN Training abgeschlossen! Beste Episode: {max(rewards_dqn):.0f}")
        chart_ph.line_chart(rewards_dqn, height=300)

        st.metric("Durchschnitt letzte 50 Episoden", f"{np.mean(rewards_dqn[-50:]):.1f}")
        st.metric("Maximale Episode", f"{max(rewards_dqn):.0f}")

# ═══════════════════════════════════════════════════════════════
# 4. Tracking-Status
# ═══════════════════════════════════════════════════════════════

elif mode == "Tracking-Status":
    st.header("📊 W&B & OpenPipe Tracking-Status")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔵 Weights & Biases")
        wandb_ok = False
        try:
            import wandb
            wandb_ok = True
            st.success("✅ W&B installiert")
        except ImportError:
            st.error("❌ W&B nicht installiert")

        if wandb_ok:
            api_key = os.environ.get("WANDB_API_KEY", "")
            if api_key:
                st.success(f"✅ API-Key gesetzt: {api_key[:8]}...")
                st.info("Modus: **online** — Runs werden in die Cloud synchronisiert")
            else:
                st.warning("⚠️ Kein API-Key — W&B läuft im **Offline-Modus**")
                st.code("export WANDB_API_KEY=your_key_here", language="bash")

            # Zeige letzte Runs
            wandb_dir = os.path.join(os.path.dirname(__file__), "wandb_runs")
            if os.path.exists(wandb_dir):
                run_dirs = [d for d in os.listdir(wandb_dir) if os.path.isdir(os.path.join(wandb_dir, d))]
                if run_dirs:
                    st.write(f"**{len(run_dirs)} lokale Run-Ordner** gefunden:")
                    for rd in sorted(run_dirs)[-5:]:
                        st.text(f"📁 {rd}")

    with col2:
        st.subheader("🟢 OpenPipe")
        op_ok = False
        try:
            from openpipe import OpenAI
            op_ok = True
            st.success("✅ OpenPipe installiert")
        except ImportError:
            st.error("❌ OpenPipe nicht installiert")

        if op_ok:
            api_key = os.environ.get("OPENPIPE_API_KEY", "")
            if api_key:
                st.success(f"✅ API-Key gesetzt: {api_key[:8]}...")
            else:
                st.warning("⚠️ Kein API-Key — OpenPipe nicht aktiv")
                st.code("export OPENPIPE_API_KEY=your_key_here", language="bash")

    st.divider()
    st.subheader("📋 Empfohlene Setup-Befehle")
    st.code("""# W&B
pip install wandb
wandb login

# OpenPipe
pip install openpipe
export OPENPIPE_API_KEY="opk_..."

# Beide zusammen
export WANDB_API_KEY="..."
export OPENPIPE_API_KEY="opk_..."
""", language="bash")

st.sidebar.markdown("---")
st.sidebar.caption("Agenten-Verstärkungslernen · Streamlit App")
