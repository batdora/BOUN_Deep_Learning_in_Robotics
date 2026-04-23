# BOUN Deep Learning in Robotics — Lecture Homeworks

This repository contains the **homework assignments** for the **Deep Learning in Robotics** course (CMPE591) at **Boğaziçi University (BOUN)**. It provides a MuJoCo-based simulation environment (UR5e robot with Robotiq 2F-85 gripper), shared code, and per-homework implementations covering deep neural networks, deep reinforcement learning, and learning from demonstration.

## What's in this repo

- **`src/`** — Main code:
  - **`environment.py`** — Shared simulation environment (tabletop, pushing, IK helpers).
  - **`demo.py`** — Minimal script that runs the environment with random actions.
  - **`hw1/`, `hw2/`, `hw3/`, `hw4/`** — Per-homework implementations. Each directory contains the homework's environment subclass, training/evaluation scripts, and any generated artifacts (models, logs, plots).
  - **`mujoco_menagerie/`** — Robot assets (Universal Robots UR5e, Robotiq 2F-85).
- **`homeworks/`** — Official task descriptions (Markdown) provided by the course.
- **`docs/`** — Course docs and the full "Preparing the Environment" guide ([homeworks.html](docs/homeworks.html)).
- **`README-assignmentN.md`** — Submission-facing writeups, one per submitted assignment.

## Assignments

| Homework (topic) | Task spec | Submitted as | Implementation report |
|---|---|---|---|
| HW1 — DNN (MLP + CNN) | [homeworks/homework1.md](homeworks/homework1.md) | Assignment 1 | [README-assignment1.md](README-assignment1.md) + notebook [`src/hw1/hw1.ipynb`](src/hw1/hw1.ipynb) |
| HW2 — Deep Q-Network | [homeworks/homework2.md](homeworks/homework2.md) | Assignment 2 | [README-assignment2.md](README-assignment2.md) |
| HW3 — Policy gradient | [homeworks/homework3.md](homeworks/homework3.md) | *(skipped)* | — |
| HW4 — CNMP (learning from demonstration) | [homeworks/homework4.md](homeworks/homework4.md) | Assignment 3 | [README-assignment3.md](README-assignment3.md) |

The course homeworks (HW1–HW4) are numbered by topic, while the submitted assignments are numbered sequentially in submission order. HW3 was skipped, so Assignment 3 is HW4. Future assignments may continue to skip; the `README-assignmentN.md` naming keeps submission order stable even when the HW numbering does not.

## Requirements

- **Python 3.9** (the `mujoco==2.3.2` / `dm_control==1.0.10` pin requires 3.9 on the tested stack).
- **MuJoCo**, **dm_control**, **mujoco-python-viewer** for simulation.
- **PyTorch** (+ torchvision) for training.

### Apple Silicon note

On an M-series Mac you must use an **arm64 build of Python**. An Intel (x86_64) miniconda installed under Rosetta will fail with `ImportError: You are running an x86_64 build of Python on an Apple Silicon machine` when importing MuJoCo. Install a native arm64 Python — e.g. via [Miniforge](https://github.com/conda-forge/miniforge) (`brew install miniforge`) — and create the env from there.

## Installation

### 1. Create a Python 3.9 environment

Using Conda or Mamba (any name works; we use `boun_robotics` below as a placeholder):

```bash
conda create -n boun_robotics python=3.9
conda activate boun_robotics
```

### 2. Install dependencies

Dependencies are pinned in [`requirements.txt`](requirements.txt) in the correct install order (MuJoCo first, then `dm_control`, then everything else). One command is enough:

```bash
pip install -r requirements.txt
```

This installs `mujoco==2.3.2`, `dm_control==1.0.10`, `mujoco-python-viewer` (from the `alper111` fork), `numpy`, `scipy`, `matplotlib`, `PyYAML`, `torch`, and `torchvision`.

> If you need a CUDA-specific PyTorch build, install it first from the [official instructions](https://pytorch.org/get-started/locally/) and then run `pip install -r requirements.txt`, which will leave PyTorch untouched.

### 3. Run the demo

```bash
cd src
python demo.py
```

You should see the simulation window with the UR5e arm executing random actions.

### Headless machines

If you're on a machine with no display (e.g. a remote server), set these before any script that opens a simulation:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

## Quick reference

| Item | Location |
|---|---|
| Full course install guide | [docs/homeworks.html](docs/homeworks.html) (section "Preparing the Environment") |
| Homework task descriptions | [homeworks/](homeworks/) |
| Shared environment + robot assets | `src/environment.py`, `src/mujoco_menagerie/` |
| Demo | `src/demo.py` |
| **Assignment 1 (HW1)** | [README-assignment1.md](README-assignment1.md) |
| **Assignment 2 (HW2)** | [README-assignment2.md](README-assignment2.md) |
| **Assignment 3 (HW4)** | [README-assignment3.md](README-assignment3.md) |

## License and attribution

Course materials and homework structure are from CMPE591 (Deep Learning in Robotics) at Boğaziçi University. Robot models are from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie). See `src/mujoco_menagerie/LICENSE` and course docs for details.
