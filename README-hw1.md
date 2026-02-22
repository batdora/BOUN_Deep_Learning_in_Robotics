# Homework 1 — Training a DNN using PyTorch

This document describes the **Homework 1** implementation in this repository. For the official task description, see [homeworks/homework1.md](homeworks/homework1.md).

## Summary

Homework 1 trains deep networks in **PyTorch** to predict the outcome of a robot push from the **initial top-down image** and the **executed action**, using data from a MuJoCo simulation (UR5e arm pushing a cube or sphere in four directions).

**Deliverables:**

1. **MLP** — Predicts final object (x, y) from flattened image + one-hot action.
2. **CNN** — Same task with convolutional spatial awareness.
3. **Image reconstruction** — Predicts the full post-action image from initial image + action (plain encoder–decoder and U-Net with skip connections).

## Where to find the solution

| Item | Path |
|------|------|
| **Notebook** (data loading, training, plots) | [src/hw1.ipynb](src/hw1.ipynb) |
| **Environment & data collection** | [src/homework1.py](src/homework1.py) |

All models are implemented and trained inside the notebook.

## How to run

1. **Environment** — Follow the main [README.md](README.md) (conda, MuJoCo, dm_control, PyTorch). Run everything from `src/`:

   ```bash
   cd src
   ```

2. **Collect data** (if not already in `src/data/`):

   ```bash
   python homework1.py
   ```

   This runs 4 processes × 250 episodes (1000 samples) and saves sharded `.pt` files under `src/data/`.

3. **Train and evaluate** — Open and run [src/hw1.ipynb](src/hw1.ipynb). It loads the data, trains the MLP, CNN, and encoder–decoder / U-Net, and shows validation plots and reconstruction samples.

## Notes and limitations

- **Arm vs object:** Arm prediction in the image models is strong; the **object** is a small region and is underemphasised by pixel-wise MSE, so object reconstruction is weaker.
- **Next steps:** A pix2pix-style setup with a U-Net generator (and e.g. patch discriminator) could improve object fidelity; this was not implemented due to time.

For full context and plots, see the final cells of [src/hw1.ipynb](src/hw1.ipynb).
