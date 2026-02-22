# Homework 1 (Training a DNN using PyTorch)

In this homework, you will train deep neural networks that take the **initial image** (top-down view of the environment) and the **executed action**, and predict either the resulting object position or the post-action image. Below are some example states.

*(See `docs/images/hw1_states.png` for example states.)*

There are two object types (cube and sphere) with random sizes between 2cm and 3cm, and the robot randomly pushes the object in four main directions. Based on the object's type and size, the resulting object position changes. Assuming that you have already cloned the repository, you can run the following code for sampling the data:

```python
import numpy as np
from homework1 import Hw1Env

env = Hw1Env(render_mode="gui")
for _ in range(100):
    env.reset()
    action_id = np.random.randint(4)
    _, img_before = env.state()
    env.step(action_id)
    pos_after, img_after = env.state()
    env.reset()
```

You might also want to check the main part of `homework1.py` to see how to collect data with multiple processes. Sample at least 1000 data points for training.

---

## Deliverables

1. **Object position prediction from initial image and action using MLP**  
   Train a multi-layer perceptron (MLP) that, given the initial image and the action, predicts the object position after the action.

2. **Object position prediction from initial image and action using CNN**  
   Train a convolutional neural network (CNN) that, given the initial image and the action, predicts the object position after the action.

3. **Post-action image reconstruction from initial image and action (any method)**  
   Train a model (MLP, CNN, or any other method) that, given the initial image and the action, reconstructs the image of the environment after the action.

All models should be implemented and trained using PyTorch.
