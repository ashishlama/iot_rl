import gymnasium as gym
import numpy as np
import random
from collections import defaultdict
import time
import matplotlib.pyplot as plt

# Environment setup (use is_slippery=True to keep FrozenLake stochastic)
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
n_states = env.observation_space.n
n_actions = env.action_space.n

# Monte Carlo Agent
class MonteCarloAgent:
    def __init__(self, n_states, n_actions, epsilon=0.1, gamma=0.99):
        self.Q = defaultdict(lambda: np.zeros(n_actions))
        self.returns = defaultdict(list)
        self.epsilon = epsilon
        self.gamma = gamma
        self.n_actions = n_actions

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(range(self.n_actions))
        return np.argmax(self.Q[state])

    def generate_episode(self, env):
        episode = []
        state, _ = env.reset()
        done = False
        while not done:
            action = self.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            done = terminated or truncated
        return episode

    def update(self, episode):
        G = 0
        visited = set()
        for i in reversed(range(len(episode))):
            state, action, reward = episode[i]
            G = self.gamma * G + reward
            if (state, action) not in visited:
                self.returns[(state, action)].append(G)
                self.Q[state][action] = np.mean(self.returns[(state, action)])
                visited.add((state, action))

# Training setup
agent = MonteCarloAgent(n_states, n_actions)
n_episodes = 20000
rolling_returns = []
all_returns = []
success_count = 0
start_time = time.time()

# Training loop
for ep in range(n_episodes):
    ep_start = time.time()
    episode = agent.generate_episode(env)
    agent.update(episode)

    total_reward = sum([x[2] for x in episode])
    all_returns.append(total_reward)
    rolling_returns.append(np.mean(all_returns[-100:]))

    if total_reward > 0:
        success_count += 1

    if ep % 500 == 0:
        print(f"Ep {ep} | Avg Return (last 100): {rolling_returns[-1]:.3f} | Time/ep: {time.time() - ep_start:.3f}s")

# Plotting average return
plt.figure(figsize=(10, 5))
plt.plot(rolling_returns)
plt.title("Rolling Average Return (last 100 episodes)")
plt.xlabel("Episode")
plt.ylabel("Avg Return")
plt.grid(True)
plt.show()

# Statistics
returns_variance = np.var(all_returns)
print(f"\n🔢 Variance of Episodic Returns: {returns_variance:.4f}")

best_avg = max(rolling_returns)
threshold = 0.9 * best_avg
for idx, val in enumerate(rolling_returns):
    if val >= threshold:
        print(f"✅ Reached 90% of best avg return at episode: {idx}")
        break

avg_time_per_episode = (time.time() - start_time) / n_episodes
print(f"🕒 Avg time per episode: {avg_time_per_episode:.4f} seconds")

# Detect convergence (low std over window)
window = 100
converged_at = None
for i in range(len(rolling_returns) - window):
    if np.std(rolling_returns[i:i+window]) < 0.01:
        converged_at = i + window
        break
print(f"📉 Converged (std < 0.01) at episode: {converged_at}")

# Final success rate
success_rate = success_count / n_episodes
print(f"🎯 Success rate (goal reached): {success_rate:.4f}")
