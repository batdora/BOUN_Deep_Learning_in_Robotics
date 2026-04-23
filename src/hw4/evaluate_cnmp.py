"""Evaluate the trained CNMP on random observation/query subsets.

For each of N_TESTS tests we
  1. Pick a random trajectory.
  2. Sample random, disjoint context and target subsets with random sizes
     drawn uniformly from [1, N_CONTEXT_MAX] and [1, N_TARGET_MAX].
  3. Run the model and compute per-test MSE separately for the end-effector
     (e_y, e_z) and the object (o_y, o_z).

Produces a bar plot with two bars (mean +/- std) and a CSV of per-test errors.
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # src/hw4/

import matplotlib.pyplot as plt
import numpy as np
import torch

from homework4 import CNP


HERE = os.path.dirname(os.path.abspath(__file__))

D_X = 2
D_Y = 4


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_dataset(path: str):
    data = torch.load(path, weights_only=False)
    trajectories = torch.from_numpy(np.asarray(data["trajectories"])).float()
    heights = torch.from_numpy(np.asarray(data["heights"])).float()
    N, T, _ = trajectories.shape
    t = torch.linspace(0.0, 1.0, T).view(1, T, 1).expand(N, -1, -1)
    h = heights.view(N, 1, 1).expand(-1, T, 1)
    x = torch.cat([t, h], dim=-1)
    return x, trajectories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(HERE, "trajectories.pt"))
    parser.add_argument("--model", default=os.path.join(HERE, "cnmp_model.pt"))
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-hidden-layers", type=int, default=3)
    parser.add_argument("--min-std", type=float, default=0.1)
    parser.add_argument("--n-tests", type=int, default=100)
    parser.add_argument("--n-context-max", type=int, default=10)
    parser.add_argument("--n-target-max", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv-out", default=os.path.join(HERE, "mse_results.csv"))
    parser.add_argument("--plot-out", default=os.path.join(HERE, "mse_barplot.png"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    print(f"Device: {device}")

    x, y = build_dataset(args.data)
    N, T, _ = x.shape

    model = CNP(in_shape=(D_X, D_Y),
                hidden_size=args.hidden_size,
                num_hidden_layers=args.num_hidden_layers,
                min_std=args.min_std).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()

    ee_mses = np.zeros(args.n_tests, dtype=np.float64)
    obj_mses = np.zeros(args.n_tests, dtype=np.float64)

    with torch.no_grad():
        for i in range(args.n_tests):
            traj_idx = int(np.random.randint(0, N))
            n_context = int(np.random.randint(1, args.n_context_max + 1))
            n_target = int(np.random.randint(1, args.n_target_max + 1))

            perm = torch.randperm(T)
            ctx_idx = perm[:n_context]
            tgt_idx = perm[n_context:n_context + n_target]

            ctx = torch.cat([x[traj_idx, ctx_idx], y[traj_idx, ctx_idx]], dim=-1)
            obs = ctx.unsqueeze(0).to(device)
            target_x = x[traj_idx, tgt_idx].unsqueeze(0).to(device)
            target_truth = y[traj_idx, tgt_idx].to(device)

            mean, _ = model(obs, target_x)
            pred = mean.squeeze(0)

            se = (pred - target_truth) ** 2
            ee_mses[i] = se[:, :2].mean().item()
            obj_mses[i] = se[:, 2:].mean().item()

    ee_mean, ee_std = ee_mses.mean(), ee_mses.std()
    obj_mean, obj_std = obj_mses.mean(), obj_mses.std()
    print(f"End-effector  MSE: mean={ee_mean:.6f}  std={ee_std:.6f}")
    print(f"Object        MSE: mean={obj_mean:.6f}  std={obj_std:.6f}")

    with open(args.csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["test_index", "ee_mse", "obj_mse"])
        for i, (e, o) in enumerate(zip(ee_mses, obj_mses)):
            w.writerow([i, e, o])

    fig, ax = plt.subplots(figsize=(6, 5))
    means = [ee_mean, obj_mean]
    stds = [ee_std, obj_std]
    x_pos = np.arange(2)
    colors = ["#1f77b4", "#d62728"]
    ax.bar(x_pos, means, yerr=stds, capsize=10,
           color=colors, alpha=0.85, edgecolor="black")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["end-effector", "object"])
    ax.set_ylabel("MSE")
    ax.set_title(f"CNMP prediction MSE over {args.n_tests} random tests")
    ax.grid(axis="y", alpha=0.3)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + max(means) * 0.03,
                f"{m:.4f}\n±{s:.4f}", ha="center", fontsize=9)
    fig.savefig(args.plot_out, dpi=120, bbox_inches="tight")

    print(f"CSV  -> {args.csv_out}")
    print(f"PNG  -> {args.plot_out}")


if __name__ == "__main__":
    main()
