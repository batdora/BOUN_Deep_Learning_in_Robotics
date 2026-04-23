"""Train a CNMP on the collected push demonstrations.

Query dim d_x = 2 ([t, h]), target dim d_y = 4 ([e_y, e_z, o_y, o_z]).
Time t is normalised to [0, 1]. The height h is replicated along T so that
it enters both the context observation and each target query; the TA spec asks
for h to be a condition to the decoder, and because h appears in every target
query (concatenated with r), this is exactly what happens.
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
from anp import AttentiveCNP


HERE = os.path.dirname(os.path.abspath(__file__))

D_X = 2  # [t, h]
D_Y = 4  # [e_y, e_z, o_y, o_z]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_dataset(path: str):
    data = torch.load(path, weights_only=False)
    trajectories = torch.from_numpy(np.asarray(data["trajectories"])).float()  # (N, T, 4)
    heights = torch.from_numpy(np.asarray(data["heights"])).float()  # (N,)

    N, T, _ = trajectories.shape
    t = torch.linspace(0.0, 1.0, T).view(1, T, 1).expand(N, -1, -1)  # (N, T, 1)
    h = heights.view(N, 1, 1).expand(-1, T, 1)  # (N, T, 1)
    x = torch.cat([t, h], dim=-1)  # (N, T, 2)
    y = trajectories  # (N, T, 4)
    return x, y


def sample_batch(x, y, batch_size, n_context, n_target, device):
    """Build a training batch where every entry shares (n_context, n_target).

    Context and target indices are sampled disjointly per trajectory so the
    model never receives the target point as part of its context.
    """
    N, T, _ = x.shape
    traj_idx = torch.randint(0, N, (batch_size,))
    observations = torch.empty(batch_size, n_context, D_X + D_Y)
    targets = torch.empty(batch_size, n_target, D_X)
    target_truths = torch.empty(batch_size, n_target, D_Y)

    for i, ti in enumerate(traj_idx):
        perm = torch.randperm(T)
        ctx_idx = perm[:n_context]
        tgt_idx = perm[n_context:n_context + n_target]

        observations[i] = torch.cat([x[ti, ctx_idx], y[ti, ctx_idx]], dim=-1)
        targets[i] = x[ti, tgt_idx]
        target_truths[i] = y[ti, tgt_idx]

    return (observations.to(device),
            targets.to(device),
            target_truths.to(device))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(HERE, "trajectories.pt"))
    parser.add_argument("--model-type", choices=["cnp", "anp"], default="cnp",
                        help="cnp: mean-aggregator CNMP. anp: cross-attention AttentiveCNP.")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-hidden-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4,
                        help="Attention heads (anp only).")
    parser.add_argument("--min-std", type=float, default=0.1)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-context-max", type=int, default=10)
    parser.add_argument("--n-target-max", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-out", default=os.path.join(HERE, "cnmp_model.pt"))
    parser.add_argument("--loss-csv-out", default=os.path.join(HERE, "training_loss.csv"))
    parser.add_argument("--loss-plot-out", default=os.path.join(HERE, "training_loss.png"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    print(f"Device: {device}")

    x, y = build_dataset(args.data)
    N, T, _ = x.shape
    print(f"Dataset: N={N} trajectories, T={T} steps, d_x={D_X}, d_y={D_Y}")

    if args.model_type == "cnp":
        model = CNP(in_shape=(D_X, D_Y),
                    hidden_size=args.hidden_size,
                    num_hidden_layers=args.num_hidden_layers,
                    min_std=args.min_std).to(device)
    else:
        model = AttentiveCNP(in_shape=(D_X, D_Y),
                             hidden_size=args.hidden_size,
                             num_hidden_layers=args.num_hidden_layers,
                             num_heads=args.num_heads,
                             min_std=args.min_std).to(device)
    print(f"Model: {args.model_type}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    losses = []
    for step in range(args.iterations):
        n_context = np.random.randint(1, args.n_context_max + 1)
        n_target = np.random.randint(1, args.n_target_max + 1)
        obs, tgt, tgt_truth = sample_batch(x, y, args.batch_size,
                                           n_context, n_target, device)
        loss = model.nll_loss(obs, tgt, tgt_truth)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if (step + 1) % args.log_every == 0:
            recent = np.mean(losses[-args.log_every:])
            print(f"iter {step + 1:6d}/{args.iterations}  "
                  f"loss(mean@{args.log_every})={recent:.4f}")

    torch.save(model.state_dict(), args.model_out)

    with open(args.loss_csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "nll_loss"])
        for i, l in enumerate(losses):
            w.writerow([i, l])

    losses_arr = np.asarray(losses)
    window = max(1, min(200, len(losses_arr) // 50))
    smoothed = np.convolve(losses_arr, np.ones(window) / window, mode="valid")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(losses_arr, color="#1f77b4", alpha=0.25, label="per-iteration")
    ax.plot(np.arange(window - 1, len(losses_arr)), smoothed,
            color="#1f77b4", label=f"moving avg (w={window})")
    ax.set_xlabel("iteration")
    ax.set_ylabel("NLL loss")
    ax.set_title("CNMP training loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(args.loss_plot_out, dpi=120, bbox_inches="tight")

    print(f"Model     -> {args.model_out}")
    print(f"Loss CSV  -> {args.loss_csv_out}")
    print(f"Loss PNG  -> {args.loss_plot_out}")


if __name__ == "__main__":
    main()
