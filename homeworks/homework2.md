# Homework 2 (Deep Q Network)

In this homework, you will train a deep Q network (DQN) that learns to push the object to the desired position. There has been a couple of updates in the environment file, so make sure to pull the latest version of the repository by running `git pull`. You can run the following code to interact with the environment (also see `homework2.py` after pulling the latest version):

```python
import numpy as np

from homework2 import Hw2Env

N_ACTIONS = 8
env = Hw2Env(n_actions=N_ACTIONS, render_mode="gui")
for episode in range(10):
    env.reset()
    done = False
    cumulative_reward = 0.0
    episode_steps = 0
    while not done:
        action = np.random.randint(N_ACTIONS)
        state, reward, is_terminal, is_truncated = env.step(action)
        done = is_terminal or is_truncated
        cumulative_reward += reward
        episode_steps += 1
    print(f"Episode={episode}, reward={cumulative_reward}, RPS={cumulative_reward/episode_steps}")
```

If you want to work on a remote machine with no screen, make sure you set the following environment variables:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

The reward is set to the following *1/distance(ee, obj)+1/distance(obj, goal)* where *ee* is the end-effector position, *obj* is the object position, and *goal* is the goal position. Tuning with hyperparameters can be tricky, so you can use the following hyperparameters:

```
Network(
    Conv2d(3, 32, 4, 2, 1), ReLU(),  # (-1, 3, 128, 128) -> (-1, 32, 64, 64)
    Conv2d(32, 64, 4, 2, 1), ReLU(),  # (-1, 32, 64, 64) -> (-1, 64, 32, 32)
    Conv2d(64, 128, 4, 2, 1), ReLU(),  # (-1, 64, 32, 32) -> (-1, 128, 16, 16)
    Conv2d(128, 256, 4, 2, 1), ReLU(),  # (-1, 128, 16, 16) -> (-1, 256, 8, 8)
    Conv2d(256, 512, 4, 2, 1), ReLU(),  # (-1, 256, 8, 8) -> (-1, 512, 4, 4)
    Avg([2, 3])  # average pooling over the spatial dimensions  (-1, 512, 4, 4) -> (-1, 512),
    Linear(512, N_ACTIONS)
)
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.999 # decay epsilon by 0.999 every EPSILON_DECAY_ITER
EPSILON_DECAY_ITER = 10 # decay epsilon every 100 updates
MIN_EPSILON = 0.1 # minimum epsilon
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
UPDATE_FREQ = 4 # update the network every 4 steps
TARGET_NETWORK_UPDATE_FREQ = 100 # update the target network every 1000 steps
BUFFER_LENGTH = 10000
```

This set is not definitive, but it seems to converge to a good policy. Feel free to share your good set of hyperparameters with the class. Plot (1) the reward, (2) the RPS (reward per step) over episodes and add it to your submission. You can use `high_level_state` to get a higher-level state, instead of raw pixels. This might speed up your experimentation as you would not need to train a convolutional network.
