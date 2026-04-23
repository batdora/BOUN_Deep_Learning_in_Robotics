# BOUN Deep Learning in Robotics — Lecture Homeworks

This repository contains the **homework assignments** for the **Deep Learning in Robotics** course (CMPE591) at **Boğaziçi University (BOUN)**. It provides a MuJoCo-based simulation environment (UR5e robot with Robotiq gripper), shared code, and instructions for four homeworks covering deep neural networks, deep reinforcement learning, and learning from demonstration.

## What’s in this repo

- **`src/`** — Main code:
  - **`environment.py`** — Shared simulation environment (tabletop, pushing, etc.).
  - **`mujoco_menagerie/`** — Robot assets (Universal Robots UR5e, Robotiq 2F-85).
  - **`homework1.py`**, **`homework2.py`**, **`homework3.py`**, **`homework4.py`** — Homework entry points and task definitions.
  - **`demo.py`** — Demo script to run the environment.
- **`homeworks/`** — Written instructions for each homework (Markdown):
  - [Homework 1](homeworks/homework1.md) — Train a DNN (MLP and CNN) with PyTorch. **Implementation (Assignment 1):** [README-assignment1.md](README-assignment1.md) and notebook [src/hw1.ipynb](src/hw1.ipynb).
  - [Homework 2](homeworks/homework2.md) — Deep Q-Network (DQN). **Implementation (Assignment 2):** [README-assignment2.md](README-assignment2.md).
  - [Homework 3](homeworks/homework3.md) — Policy gradient (REINFORCE, SAC). *(Not submitted — skipped in the assignment sequence.)*
  - [Homework 4](homeworks/homework4.md) — Learning from demonstration with CNMPs. **Implementation (Assignment 3):** [README-assignment3.md](README-assignment3.md).

> **Assignment ↔ Homework mapping.** The course homeworks (HW1–HW4) are numbered by topic. The submitted assignments are numbered sequentially in submission order, and HW3 was skipped — so Assignment 1 = HW1, Assignment 2 = HW2, Assignment 3 = HW4. Future assignments may continue to skip; the `README-assignmentN.md` naming keeps submission order stable even when the HW numbering does not.
- **`docs/`** — Course docs and full “Preparing the Environment” guide ([homeworks.html](docs/homeworks.html)).

## Requirements

- **Python 3.9**
- **MuJoCo** and **dm_control** for simulation
- **PyTorch** (and torchvision) for training
- **mujoco-python-viewer** for GUI

## Installation

### 1. Create a virtual environment (recommended)

Use **Conda** or **Mamba** (Python 3.9):

```bash
# Conda
conda create -n boun_robotics python=3.9
conda activate boun_robotics

# Or Mamba (faster)
mamba create -n boun_robotics python=3.9
mamba activate boun_robotics
```

You can use another name instead of `boun_robotics`. Activate this environment whenever you work on the homeworks.

### 2. Install simulation stack (order matters)

Install **MuJoCo first**, then **dm_control** (to avoid build issues):

```bash
pip install mujoco==2.3.2
pip install dm_control==1.0.10
```

Then install the viewer and other dependencies:

```bash
pip install git+https://github.com/alper111/mujoco-python-viewer.git
pip install PyYAML
conda install numpy   # or: pip install numpy
```

### 3. Install PyTorch

Install PyTorch and torchvision for your OS and hardware from the [official instructions](https://pytorch.org/get-started/locally/). For example, CPU-only on macOS:

```bash
pip install torch torchvision
```

### 4. Install from this repo (optional)

From the repo root you can install the rest of the Python dependencies (mujoco and dm_control should already be installed as above):

```bash
pip install -r requirements.txt
```

**Note:** `requirements.txt` lists `mujoco==2.3.2` and `dm_control==1.0.10` at the top so that, if you install with `pip install -r requirements.txt` in a clean environment, pip installs them in that order. If you already followed step 2, you can still run `pip install -r requirements.txt` to get numpy, scipy, PyYAML, matplotlib, torch, etc.

### 5. Run the demo

From the repository root:

```bash
cd src
python demo.py
```

You should see the simulation window and episodes running. If you are on a **headless/remote machine** (no display), set:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

before running.

## Quick reference

| Item | Location |
|------|----------|
| Full course install guide | [docs/homeworks.html](docs/homeworks.html) (section “Preparing the Environment”) |
| Homework 1–4 instructions | [homeworks/](homeworks/) (Markdown files) |
| **Assignment 1 (HW1) implementation** | [README-assignment1.md](README-assignment1.md), notebook [src/hw1.ipynb](src/hw1.ipynb) |
| **Assignment 2 (HW2) implementation** | [README-assignment2.md](README-assignment2.md) |
| **Assignment 3 (HW4) implementation** | [README-assignment3.md](README-assignment3.md) |
| Shared environment + robot assets | `src/environment.py`, `src/mujoco_menagerie/` |
| Demo | `src/demo.py` |

## License and attribution

Course materials and homework structure are from CMPE591 (Deep Learning in Robotics) at Boğaziçi University. Robot models are from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie). See `src/mujoco_menagerie/LICENSE` and course docs for details.
