"""Collect robot push demonstrations for HW4 (CNMP).

Each trajectory is a sequence of high_level_state() readings
(e_y, e_z, o_y, o_z, h) where h is the object height (constant per episode).
We store one (N, T, 4) array of positional states and one (N,) array of heights.
"""

import argparse
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # src/hw4/

import numpy as np
import torch

import environment  # noqa: F401  (ensures src/environment.py is importable)
from homework4 import Hw5Env, bezier


HERE = os.path.dirname(os.path.abspath(__file__))


def collect(n_trajectories: int, render_mode: str, out_path: str, seed: Optional[int]):
    if seed is not None:
        np.random.seed(seed)

    env = Hw5Env(render_mode=render_mode)
    traj_buffer = []
    height_buffer = []

    for i in range(n_trajectories):
        env.reset()
        p_1 = np.array([0.5, 0.3, 1.04])
        p_2 = np.array([0.5, 0.15, np.random.uniform(1.04, 1.4)])
        p_3 = np.array([0.5, -0.15, np.random.uniform(1.04, 1.4)])
        p_4 = np.array([0.5, -0.3, 1.04])
        points = np.stack([p_1, p_2, p_3, p_4], axis=0)
        curve = bezier(points)

        env._set_ee_in_cartesian(curve[0], rotation=[-90, 0, 180],
                                 n_splits=100, max_iters=100, threshold=0.05)
        states = []
        for p in curve:
            env._set_ee_pose(p, rotation=[-90, 0, 180], max_iters=10)
            states.append(env.high_level_state())
        states = np.stack(states)  # (T, 5): [e_y, e_z, o_y, o_z, h]

        traj_buffer.append(states[:, :4].astype(np.float32))
        height_buffer.append(float(states[0, 4]))
        print(f"Collected {i + 1}/{n_trajectories}", end="\r", flush=True)
    print()

    trajectories = np.stack(traj_buffer, axis=0)  # (N, T, 4)
    heights = np.array(height_buffer, dtype=np.float32)  # (N,)
    torch.save({"trajectories": trajectories, "heights": heights}, out_path)
    print(f"Saved {trajectories.shape[0]} trajectories "
          f"(T={trajectories.shape[1]}, d=4) to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n-trajectories", type=int, default=150)
    parser.add_argument("--render-mode", choices=["gui", "offscreen"], default="offscreen")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=os.path.join(HERE, "trajectories.pt"))
    args = parser.parse_args()

    collect(args.n_trajectories, args.render_mode, args.out, args.seed)


if __name__ == "__main__":
    main()
