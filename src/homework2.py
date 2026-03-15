import csv
import glob
import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import environment


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
MEMORY_SIZE = 10000
NUM_EPISODES = 2500
BATCH_SIZE = 128
EPS_DECAY = 10000
EPS_END = 0.05
EPS_START = 0.9
GAMMA = 0.99
LEARNING_RATE = 0.0001
TAU = 0.005
N_ACTIONS = 8
STATE_DIM = 6

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class Hw2Env(environment.BaseEnv):
    def __init__(self, n_actions=8, n_splits=15, **kwargs) -> None:
        super().__init__(**kwargs)
        self._n_actions = n_actions
        self._n_splits = n_splits
        self._delta = 0.05

        theta = np.linspace(0, 2*np.pi, n_actions)
        actions = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        self._actions = {i: action for i, action in enumerate(actions)}

        self._goal_thresh = 0.01
        self._max_timesteps = 50

    def _create_scene(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        scene = environment.create_tabletop_scene()
        obj_pos = [np.random.uniform(0.25, 0.75),
                   np.random.uniform(-0.3, 0.3),
                   1.5]
        goal_pos = [np.random.uniform(0.25, 0.75),
                    np.random.uniform(-0.3, 0.3),
                    1.025]
        environment.create_object(scene, "box", pos=obj_pos, quat=[0, 0, 0, 1],
                                  size=[0.03, 0.03, 0.03], rgba=[0.8, 0.2, 0.2, 1],
                                  name="obj1")
        environment.create_visual(scene, "cylinder", pos=goal_pos, quat=[0, 0, 0, 1],
                                  size=[0.05, 0.005], rgba=[0.2, 1.0, 0.2, 1],
                                  name="goal")
        return scene

    def state(self):
        if self._render_mode == "offscreen":
            self.viewer.update_scene(self.data, camera="topdown")
            pixels = torch.tensor(self.viewer.render().copy(), dtype=torch.uint8).permute(2, 0, 1)
        else:
            pixels = self.viewer.read_pixels(camid=1).copy()
            pixels = torch.tensor(pixels, dtype=torch.uint8).permute(2, 0, 1)
            pixels = transforms.functional.center_crop(pixels, min(pixels.shape[1:]))
            pixels = transforms.functional.resize(pixels, (128, 128))
        return pixels / 255.0

    def high_level_state(self):
        ee_pos = self.data.site(self._ee_site).xpos[:2]
        obj_pos = self.data.body("obj1").xpos[:2]
        goal_pos = self.data.site("goal").xpos[:2]
        return np.concatenate([ee_pos, obj_pos, goal_pos])

    def reward(self):
        state = self.high_level_state()
        ee_pos = state[:2]
        obj_pos = state[2:4]
        goal_pos = state[4:6]
        ee_to_obj = max(100*np.linalg.norm(ee_pos - obj_pos), 1)
        obj_to_goal = max(100*np.linalg.norm(obj_pos - goal_pos), 1)
        return 1/(ee_to_obj) + 1/(obj_to_goal)

    def is_terminal(self):
        obj_pos = self.data.body("obj1").xpos[:2]
        goal_pos = self.data.site("goal").xpos[:2]
        return np.linalg.norm(obj_pos - goal_pos) < self._goal_thresh

    def is_truncated(self):
        return self._t >= self._max_timesteps

    def step(self, action_id):
        action = self._actions[action_id] * self._delta
        ee_pos = self.data.site(self._ee_site).xpos[:2]
        target_pos = np.concatenate([ee_pos, [1.06]])
        target_pos[:2] = np.clip(target_pos[:2] + action, [0.25, -0.3], [0.75, 0.3])
        self._set_ee_in_cartesian(target_pos, rotation=[-90, 0, 180],
                                  n_splits=self._n_splits, threshold=0.04)
        self._t += 1

        state = self.state()
        reward = self.reward()
        terminal = self.is_terminal()
        truncated = self.is_truncated()
        return state, reward, terminal, truncated


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32).to(DEVICE),
            torch.tensor(actions, dtype=torch.long).to(DEVICE),
            torch.tensor(rewards, dtype=torch.float32).to(DEVICE),
            torch.tensor(np.array(next_states), dtype=torch.float32).to(DEVICE),
            torch.tensor(dones, dtype=torch.float32).to(DEVICE),
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
class DQNNetwork(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class DQNAgent:
    def __init__(self):
        self.policy_net = DQNNetwork(STATE_DIM, N_ACTIONS).to(DEVICE)
        self.target_net = DQNNetwork(STATE_DIM, N_ACTIONS).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.steps_done = 0

    def select_action(self, state):
        eps = EPS_END + (EPS_START - EPS_END) * max(0, (EPS_DECAY - self.steps_done) / EPS_DECAY)
        self.steps_done += 1
        if random.random() < eps:
            return random.randrange(N_ACTIONS)
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            return self.policy_net(state_t).argmax(dim=1).item()

    def update(self):
        if len(self.memory) < BATCH_SIZE:
            return None
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + GAMMA * next_q * (1 - dones)

        loss = nn.functional.mse_loss(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for param, target_param in zip(self.policy_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        return loss.item()

    def push(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(num_episodes=NUM_EPISODES, render_mode="offscreen", run_name="run1",
          n_splits=15, checkpoint_every=500):
    """Train DQN agent.

    Saves checkpoints to hw2_checkpoints/{run_name}_ep{N}.pt every
    `checkpoint_every` episodes and resumes automatically if a checkpoint
    for this run_name already exists.  Per-episode metrics are appended to
    hw2_{run_name}_log.csv.
    """
    print(f"Using device: {DEVICE}")

    os.makedirs("hw2_checkpoints", exist_ok=True)
    log_path = f"hw2_{run_name}_log.csv"

    env = Hw2Env(n_actions=N_ACTIONS, n_splits=n_splits, render_mode=render_mode)
    agent = DQNAgent()

    episode_rewards = []
    episode_rps = []
    start_episode = 0

    # Resume from latest checkpoint if available
    existing = sorted(glob.glob(f"hw2_checkpoints/{run_name}_ep*.pt"))
    if existing:
        ckpt_path = existing[-1]
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        agent.policy_net.load_state_dict(ckpt["policy_net"])
        agent.target_net.load_state_dict(ckpt["target_net"])
        agent.optimizer.load_state_dict(ckpt["optimizer"])
        agent.steps_done = ckpt["steps_done"]
        episode_rewards = ckpt["episode_rewards"]
        episode_rps = ckpt["episode_rps"]
        start_episode = ckpt["episode"] + 1
        print(f"Resumed from {ckpt_path} (episode {start_episode})")

    # Write CSV header only for new runs
    if start_episode == 0:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["episode", "total_reward", "rps", "avg_reward_100", "avg_rps_100", "epsilon", "loss"]
            )

    for episode in range(start_episode, num_episodes):
        env.reset()
        state = env.high_level_state()
        done = False
        total_reward = 0.0
        steps = 0
        episode_loss = []

        while not done:
            action = agent.select_action(state)
            _, reward, terminal, truncated = env.step(action)
            next_state = env.high_level_state()
            done = terminal or truncated

            agent.push(state, action, reward, next_state, float(done))
            loss = agent.update()
            if loss is not None:
                episode_loss.append(loss)

            state = next_state
            total_reward += reward
            steps += 1

        rps = total_reward / steps
        episode_rewards.append(total_reward)
        episode_rps.append(rps)

        avg_r = float(np.mean(episode_rewards[-100:]))
        avg_rps = float(np.mean(episode_rps[-100:]))
        eps = EPS_END + (EPS_START - EPS_END) * max(0, (EPS_DECAY - agent.steps_done) / EPS_DECAY)
        avg_loss = float(np.mean(episode_loss)) if episode_loss else 0.0

        # CSV log every episode
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [episode + 1, round(total_reward, 4), round(rps, 4),
                 round(avg_r, 4), round(avg_rps, 4), round(eps, 4), round(avg_loss, 6)]
            )

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"Avg Reward: {avg_r:.3f} | Avg RPS: {avg_rps:.3f} | "
                  f"Eps: {eps:.3f} | Loss: {avg_loss:.5f}")

        # Checkpoint
        if (episode + 1) % checkpoint_every == 0:
            ckpt_path = f"hw2_checkpoints/{run_name}_ep{episode+1}.pt"
            torch.save({
                "episode": episode,
                "policy_net": agent.policy_net.state_dict(),
                "target_net": agent.target_net.state_dict(),
                "optimizer": agent.optimizer.state_dict(),
                "steps_done": agent.steps_done,
                "episode_rewards": episode_rewards,
                "episode_rps": episode_rps,
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    # Final model save
    model_path = f"hw2_policy_{run_name}.pt"
    torch.save(agent.policy_net.state_dict(), model_path)
    print(f"Training complete. Model saved to {model_path}")

    return agent, episode_rewards, episode_rps


# ---------------------------------------------------------------------------
# Entry point (demo / quick sanity check)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    agent, rewards, rps = train()
