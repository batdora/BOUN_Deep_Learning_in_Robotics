# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

**Python 3.9 required.** Install in this exact order to avoid build issues:

```bash
conda create -n boun_robotics python=3.9 && conda activate boun_robotics
pip install mujoco==2.3.2
pip install dm_control==1.0.10
pip install git+https://github.com/alper111/mujoco-python-viewer.git
pip install -r requirements.txt  # numpy, scipy, PyYAML, matplotlib, torch, torchvision
```

For headless/remote machines (no display), set before running any script:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

## Running Scripts

**All scripts must be run from the `src/` directory** because `environment.py` loads robot assets via relative paths (`mujoco_menagerie/universal_robots_ur5e/ur5e.xml`).

```bash
cd src
python demo.py               # sanity-check: runs random-action episodes in GUI
python homework1.py          # collects training data (4 parallel processes)
python homework2.py          # runs random DQN-style episodes
python hw3/homework3.py      # runs REINFORCE training loop
python homework4.py          # collects CNMP demonstrations
```

## Architecture

### Core simulation layer (`src/environment.py`)

`BaseEnv` wraps a MuJoCo simulation of a UR5e robot arm with a Robotiq 2F-85 gripper on a tabletop:
- `reset()` rebuilds the scene from scratch each episode via `_create_scene()` (subclass hook)
- `render_mode="gui"` uses `mujoco_viewer`; `render_mode="offscreen"` uses `mujoco.Renderer` (128×128)
- Movement is done via IK: `_set_ee_pose()` / `_set_ee_in_cartesian()` (Cartesian-space trajectory with SLERP for orientation) → `_set_joint_position()` for gripper
- The custom IK solver (`qpos_from_site_pose` / `nullspace_method`) is adapted from dm_control

### Homework environments (each extends `BaseEnv`)

| File | Class | Task | State | Action |
|------|-------|------|-------|--------|
| `src/homework1.py` | `Hw1Env` | Push cube/sphere in 4 cardinal directions | `(obj_pos, 128×128 image)` | discrete action_id 0–3 |
| `src/homework2.py` | `Hw2Env` | Push object to goal (pixel obs) | 128×128 RGB image / 6-dim `high_level_state` | discrete (8 directions) |
| `src/hw3/homework3.py` | `Hw3Env` | Push object to goal (continuous actions) | 6-dim `high_level_state` | continuous 2D delta |
| `src/homework4.py` | `Hw5Env` | Collect arm trajectory demonstrations | `(e_y, e_z, o_y, o_z, height)` | Bezier curve waypoints |

### HW3 is a self-contained subdirectory (`src/hw3/`)

It has its own copy of `environment.py`, plus `agent.py` (REINFORCE `Agent` class) and `model.py` (`VPG` policy network). The agent skeleton's `decide_action` and `update_model` methods are intentionally incomplete for students to implement.

### CNMP model (`src/homework4.py`)

`CNP` is a Conditional Neural Process: encoder aggregates context points into representation `r`, which is concatenated with query points and decoded to `(mean, std)`. Training uses NLL loss. Used for HW4 (learning from demonstration).

### Multi-process data collection pattern

`homework1.py` and `_homework3.py` show two patterns:
- `multiprocessing.Process` (HW1): spawn N processes, each saves `.pt` files independently
- `torch.multiprocessing` with shared queue (HW3): collector workers push to a shared `mp.Queue`, main process drains the queue and updates the model

## Key Conventions

- `high_level_state()` returns a low-dimensional numpy array `[ee_x, ee_y, obj_x, obj_y, goal_x, goal_y]`; used as an alternative to raw pixel observations for faster experimentation
- `state()` returns a normalised `float` tensor `[0,1]` (pixels ÷ 255); HW1 returns raw `uint8`
- `step()` increments `self._t`; `is_truncated()` checks `_t >= _max_timesteps`
- Camera `camid=0` = "frontface" (side view), `camid=1` = "topdown"
