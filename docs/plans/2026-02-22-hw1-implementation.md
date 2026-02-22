# HW1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a Jupyter notebook (`src/hw1.ipynb`) that collects simulation data and trains three PyTorch models — an MLP and CNN for object position prediction, and a CNN encoder-decoder for post-push image reconstruction.

**Architecture:** Fix `homework1.py`'s `collect()` to capture both before/after states, then save 1000 samples to `src/data/`. The notebook loads those files, defines all three models, trains them, and visualises results.

**Tech Stack:** Python 3.9, PyTorch (MPS → CPU fallback), MuJoCo 2.3.2 + dm_control, Jupyter notebook

---

## Prerequisites

All commands run from `src/`. Conda env `boun_robotics` must be active.

```bash
cd /Users/batdora/Desktop/BOUN/Robotics/homeworks/src
conda activate boun_robotics
```

---

### Task 1: Fix data collection in `homework1.py`

**Files:**
- Modify: `src/homework1.py`

The current `collect()` only captures the post-action state. We need `img_before` too, and need to bump samples per process from 100 to 250 (4 × 250 = 1000 total). Data is saved to `src/data/` (already gitignored).

**Step 1: Create the data directory**

```bash
mkdir -p /Users/batdora/Desktop/BOUN/Robotics/homeworks/src/data
```

**Step 2: Replace `collect()` and `__main__` block in `homework1.py`**

Replace from line 70 to end of file with:

```python
def collect(idx, N):
    env = Hw1Env(render_mode="offscreen")
    imgs_before = torch.zeros(N, 3, 128, 128, dtype=torch.uint8)
    imgs_after  = torch.zeros(N, 3, 128, 128, dtype=torch.uint8)
    positions   = torch.zeros(N, 2, dtype=torch.float)
    actions     = torch.zeros(N, dtype=torch.uint8)

    for i in range(N):
        env.reset()
        action_id = np.random.randint(4)
        _, img_before = env.state()
        env.step(action_id)
        pos_after, img_after = env.state()

        imgs_before[i] = img_before
        imgs_after[i]  = img_after
        positions[i]   = torch.tensor(pos_after)
        actions[i]     = action_id

    torch.save(imgs_before, f"data/imgs_before_{idx}.pt")
    torch.save(imgs_after,  f"data/imgs_after_{idx}.pt")
    torch.save(positions,   f"data/positions_{idx}.pt")
    torch.save(actions,     f"data/actions_{idx}.pt")
    print(f"Process {idx} done.")


if __name__ == "__main__":
    processes = []
    for i in range(4):
        p = Process(target=collect, args=(i, 250))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print("All done. 1000 samples saved to src/data/")
```

**Step 3: Verify the fix looks correct by dry-running one sample**

```bash
python -c "
import numpy as np, torch
from homework1 import Hw1Env
env = Hw1Env(render_mode='offscreen')
env.reset()
_, img_b = env.state()
env.step(0)
pos, img_a = env.state()
print('img_before shape:', img_b.shape)   # expect torch.Size([3, 128, 128])
print('img_after shape:', img_a.shape)    # expect torch.Size([3, 128, 128])
print('pos shape:', pos.shape)            # expect (2,)
print('OK')
"
```

Expected output:
```
img_before shape: torch.Size([3, 128, 128])
img_after shape: torch.Size([3, 128, 128])
pos shape: (2,)
OK
```

**Step 4: Run full data collection (~10–20 min on MPS/CPU)**

```bash
python homework1.py
```

Expected output (order may vary):
```
Process 0 done.
Process 1 done.
Process 2 done.
Process 3 done.
All done. 1000 samples saved to src/data/
```

Verify files exist:
```bash
ls data/
# imgs_before_0.pt  imgs_after_0.pt  positions_0.pt  actions_0.pt
# imgs_before_1.pt  imgs_after_1.pt  positions_1.pt  actions_1.pt
# imgs_before_2.pt  imgs_after_2.pt  positions_2.pt  actions_2.pt
# imgs_before_3.pt  imgs_after_3.pt  positions_3.pt  actions_3.pt
```

**Step 5: Commit**

```bash
git add src/homework1.py
git commit -m "fix(hw1): collect img_before and increase to 1000 samples"
```

---

### Task 2: Notebook — data loading and preprocessing

**Files:**
- Create: `src/hw1.ipynb`

**Step 1: Create notebook with Cell 1 — imports**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)
```

Expected output: `Using device: mps`  (or `cpu`)

**Step 2: Cell 2 — load and merge shards**

```python
imgs_before = torch.cat([torch.load(f"data/imgs_before_{i}.pt") for i in range(4)])  # (1000, 3, 128, 128)
imgs_after  = torch.cat([torch.load(f"data/imgs_after_{i}.pt")  for i in range(4)])  # (1000, 3, 128, 128)
positions   = torch.cat([torch.load(f"data/positions_{i}.pt")   for i in range(4)])  # (1000, 2)
actions_raw = torch.cat([torch.load(f"data/actions_{i}.pt")     for i in range(4)])  # (1000,)

# Normalise images and one-hot encode actions
imgs_before_f = imgs_before.float() / 255.0
imgs_after_f  = imgs_after.float()  / 255.0
actions_oh    = F.one_hot(actions_raw.long(), num_classes=4).float()  # (1000, 4)

print("imgs_before_f:", imgs_before_f.shape, imgs_before_f.min().item(), imgs_before_f.max().item())
print("positions:", positions.shape, positions.min().item(), positions.max().item())
print("actions_oh:", actions_oh.shape)
```

Expected output:
```
imgs_before_f: torch.Size([1000, 3, 128, 128]) 0.0 1.0
positions: torch.Size([1000, 2]) ~0.25 ~0.75
actions_oh: torch.Size([1000, 4])
```

**Step 3: Cell 3 — Dataset class and DataLoaders**

```python
class PushDataset(Dataset):
    def __init__(self, imgs_before, imgs_after, positions, actions_oh):
        self.imgs_before = imgs_before
        self.imgs_after  = imgs_after
        self.positions   = positions
        self.actions_oh  = actions_oh

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return (self.imgs_before[idx], self.imgs_after[idx],
                self.positions[idx], self.actions_oh[idx])

dataset = PushDataset(imgs_before_f, imgs_after_f, positions, actions_oh)
n_train = int(0.8 * len(dataset))
n_val   = len(dataset) - n_train
train_set, val_set = random_split(dataset, [n_train, n_val],
                                  generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=64, shuffle=False)
print(f"Train: {n_train}  Val: {n_val}")
```

Expected output: `Train: 800  Val: 200`

**Step 4: Cell 4 — visualise a few samples (sanity check)**

```python
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i in range(5):
    axes[0, i].imshow(imgs_before[i].permute(1, 2, 0))
    axes[0, i].set_title(f"before | action={actions_raw[i].item()}")
    axes[1, i].imshow(imgs_after[i].permute(1, 2, 0))
    axes[1, i].set_title(f"after | pos=({positions[i,0]:.2f},{positions[i,1]:.2f})")
    for ax in axes[:, i]: ax.axis("off")
plt.tight_layout()
plt.show()
```

Should display 5 before/after image pairs with action labels and final positions.

**Step 5: Commit**

```bash
git add src/hw1.ipynb
git commit -m "feat(hw1): notebook skeleton with data loading and visualisation"
```

---

### Task 3: MLP for position prediction

**Files:**
- Modify: `src/hw1.ipynb` (add cells)

**Step 1: Cell — MLP definition**

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * 128 * 128 + 4, 1024), nn.ReLU(),
            nn.Linear(1024, 512),               nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, img, action):
        x = torch.cat([img.flatten(start_dim=1), action], dim=1)
        return self.net(x)

# Verify forward pass shape
mlp = MLP().to(device)
dummy_img = torch.zeros(4, 3, 128, 128).to(device)
dummy_act = torch.zeros(4, 4).to(device)
out = mlp(dummy_img, dummy_act)
print("MLP output shape:", out.shape)  # expect torch.Size([4, 2])
```

**Step 2: Cell — MLP training loop**

```python
def train_model(model, train_loader, val_loader, criterion, epochs, label):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        batch_loss = []
        for img_b, img_a, pos, act in train_loader:
            img_b, img_a, pos, act = img_b.to(device), img_a.to(device), pos.to(device), act.to(device)
            optimizer.zero_grad()
            pred = model(img_b, act) if not isinstance(model, EncDec) else model(img_b, act)
            loss = criterion(pred, pos if not isinstance(model, EncDec) else img_a)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        train_losses.append(np.mean(batch_loss))

        model.eval()
        with torch.no_grad():
            v_losses = []
            for img_b, img_a, pos, act in val_loader:
                img_b, img_a, pos, act = img_b.to(device), img_a.to(device), pos.to(device), act.to(device)
                pred = model(img_b, act)
                loss = criterion(pred, pos if not isinstance(model, EncDec) else img_a)
                v_losses.append(loss.item())
        val_losses.append(np.mean(v_losses))

        if (epoch + 1) % 10 == 0:
            print(f"[{label}] Epoch {epoch+1}/{epochs}  train={train_losses[-1]:.4f}  val={val_losses[-1]:.4f}")

    return train_losses, val_losses

mlp_criterion = nn.MSELoss()
mlp_train_losses, mlp_val_losses = train_model(mlp, train_loader, val_loader, mlp_criterion, epochs=50, label="MLP")
```

Expected: loss should decrease each 10 epochs; final val MSE typically 0.001–0.01 range.

**Step 3: Cell — MLP evaluation plot**

```python
# Loss curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(mlp_train_losses, label="train")
axes[0].plot(mlp_val_losses,   label="val")
axes[0].set_title("MLP Loss"); axes[0].legend()

# Predicted vs actual scatter
mlp.eval()
preds, trues = [], []
with torch.no_grad():
    for img_b, _, pos, act in val_loader:
        p = mlp(img_b.to(device), act.to(device)).cpu()
        preds.append(p); trues.append(pos)
preds = torch.cat(preds); trues = torch.cat(trues)

axes[1].scatter(trues[:,0], preds[:,0], alpha=0.4, label="x")
axes[1].scatter(trues[:,1], preds[:,1], alpha=0.4, label="y")
axes[1].plot([0.2, 0.8], [0.2, 0.8], 'k--')
axes[1].set_title("MLP: Predicted vs Actual"); axes[1].legend()
plt.tight_layout(); plt.show()

print(f"MLP val MSE: {F.mse_loss(preds, trues).item():.5f}")
```

**Step 4: Commit**

```bash
git add src/hw1.ipynb
git commit -m "feat(hw1): MLP position prediction model and training"
```

---

### Task 4: CNN for position prediction

**Files:**
- Modify: `src/hw1.ipynb` (add cells)

**Step 1: Cell — CNN definition**

```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3,  32,  4, stride=2, padding=1), nn.ReLU(),  # 128→64
            nn.Conv2d(32, 64,  4, stride=2, padding=1), nn.ReLU(),  # 64→32
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),  # 32→16
            nn.Conv2d(128,256, 4, stride=2, padding=1), nn.ReLU(),  # 16→8
            nn.AdaptiveAvgPool2d(1)                                  # →(256,1,1)
        )
        self.head = nn.Sequential(
            nn.Linear(256 + 4, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, img, action):
        feat = self.backbone(img).flatten(start_dim=1)  # (B, 256)
        x    = torch.cat([feat, action], dim=1)          # (B, 260)
        return self.head(x)

cnn = CNN().to(device)
out = cnn(dummy_img, dummy_act)
print("CNN output shape:", out.shape)  # expect torch.Size([4, 2])
```

**Step 2: Cell — CNN training**

```python
# Wrap CNN forward for train_model compatibility
class CNNWrapper(nn.Module):
    def __init__(self, cnn): super().__init__(); self.cnn = cnn
    def forward(self, img, act): return self.cnn(img, act)

cnn_criterion = nn.MSELoss()
cnn_train_losses, cnn_val_losses = train_model(cnn, train_loader, val_loader, cnn_criterion, epochs=50, label="CNN")
```

**Step 3: Cell — CNN evaluation + comparison with MLP**

```python
cnn.eval()
preds_cnn, trues_cnn = [], []
with torch.no_grad():
    for img_b, _, pos, act in val_loader:
        p = cnn(img_b.to(device), act.to(device)).cpu()
        preds_cnn.append(p); trues_cnn.append(pos)
preds_cnn = torch.cat(preds_cnn); trues_cnn = torch.cat(trues_cnn)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(mlp_train_losses, label="MLP train"); axes[0].plot(mlp_val_losses, label="MLP val")
axes[0].plot(cnn_train_losses, label="CNN train"); axes[0].plot(cnn_val_losses, label="CNN val")
axes[0].set_title("Loss comparison"); axes[0].legend()

axes[1].scatter(trues_cnn[:,0], preds_cnn[:,0], alpha=0.4, label="x")
axes[1].scatter(trues_cnn[:,1], preds_cnn[:,1], alpha=0.4, label="y")
axes[1].plot([0.2, 0.8], [0.2, 0.8], 'k--')
axes[1].set_title("CNN: Predicted vs Actual"); axes[1].legend()
plt.tight_layout(); plt.show()

mlp_mse = F.mse_loss(preds, trues).item()
cnn_mse = F.mse_loss(preds_cnn, trues_cnn).item()
print(f"MLP val MSE: {mlp_mse:.5f}")
print(f"CNN val MSE: {cnn_mse:.5f}")
print(f"CNN improvement: {(mlp_mse - cnn_mse)/mlp_mse * 100:.1f}%")
```

**Step 4: Commit**

```bash
git add src/hw1.ipynb
git commit -m "feat(hw1): CNN position prediction model and MLP/CNN comparison"
```

---

### Task 5: CNN Encoder-Decoder for image reconstruction

**Files:**
- Modify: `src/hw1.ipynb` (add cells)

**Step 1: Cell — Encoder-Decoder definition**

```python
class EncDec(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,   32,  4, stride=2, padding=1), nn.ReLU(),  # 128→64
            nn.Conv2d(32,  64,  4, stride=2, padding=1), nn.ReLU(),  # 64→32
            nn.Conv2d(64,  128, 4, stride=2, padding=1), nn.ReLU(),  # 32→16
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU(),  # 16→8
            nn.Conv2d(256, 256, 4, stride=2, padding=1), nn.ReLU(),  # 8→4
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(260, 256, 4, stride=2, padding=1), nn.ReLU(),  # 4→8
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(),  # 8→16
            nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1), nn.ReLU(),  # 16→32
            nn.ConvTranspose2d(64,  32,  4, stride=2, padding=1), nn.ReLU(),  # 32→64
            nn.ConvTranspose2d(32,  3,   4, stride=2, padding=1), nn.Sigmoid() # 64→128
        )

    def forward(self, img, action):
        z = self.encoder(img)                                         # (B, 256, 4, 4)
        act_spatial = action.view(-1, 4, 1, 1).expand(-1, 4, 4, 4)   # (B, 4, 4, 4)
        z = torch.cat([z, act_spatial], dim=1)                        # (B, 260, 4, 4)
        return self.decoder(z)                                         # (B, 3, 128, 128)

enc_dec = EncDec().to(device)
dummy_out = enc_dec(dummy_img, dummy_act)
print("EncDec output shape:", dummy_out.shape)  # expect torch.Size([4, 3, 128, 128])
```

**Step 2: Cell — Encoder-Decoder training loop**

```python
def train_encdec(model, train_loader, val_loader, epochs):
    optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion  = nn.MSELoss()
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        batch_loss = []
        for img_b, img_a, _, act in train_loader:
            img_b, img_a, act = img_b.to(device), img_a.to(device), act.to(device)
            optimizer.zero_grad()
            pred = model(img_b, act)
            loss = criterion(pred, img_a)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        train_losses.append(np.mean(batch_loss))

        model.eval()
        with torch.no_grad():
            v_losses = [criterion(model(ib.to(device), a.to(device)), ia.to(device)).item()
                        for ib, ia, _, a in val_loader]
        val_losses.append(np.mean(v_losses))

        if (epoch + 1) % 10 == 0:
            print(f"[EncDec] Epoch {epoch+1}/{epochs}  train={train_losses[-1]:.4f}  val={val_losses[-1]:.4f}")

    return train_losses, val_losses

ed_train_losses, ed_val_losses = train_encdec(enc_dec, train_loader, val_loader, epochs=100)
```

**Step 3: Cell — Image reconstruction visualisation**

```python
enc_dec.eval()
img_b_sample, img_a_sample, _, act_sample = next(iter(val_loader))
with torch.no_grad():
    pred_imgs = enc_dec(img_b_sample[:5].to(device), act_sample[:5].to(device)).cpu()

fig, axes = plt.subplots(3, 5, figsize=(15, 9))
titles = ["Before", "Predicted After", "True After"]
rows   = [img_b_sample[:5], pred_imgs, img_a_sample[:5]]
for row_idx, (row_data, title) in enumerate(zip(rows, titles)):
    for col_idx in range(5):
        axes[row_idx, col_idx].imshow(row_data[col_idx].permute(1, 2, 0).clamp(0, 1))
        axes[row_idx, col_idx].axis("off")
    axes[row_idx, 0].set_ylabel(title, fontsize=12)
plt.suptitle("Image Reconstruction: Before | Predicted | True After")
plt.tight_layout(); plt.show()
```

Should show rows: original scene / decoder's prediction / ground-truth post-push scene.

**Step 4: Commit**

```bash
git add src/hw1.ipynb
git commit -m "feat(hw1): CNN encoder-decoder image reconstruction and visualisation"
```

---

### Task 6: Final summary cell and push

**Files:**
- Modify: `src/hw1.ipynb` (add summary cell)

**Step 1: Cell — Results summary**

```python
print("=" * 50)
print("HW1 Results Summary")
print("=" * 50)
print(f"Dataset:        1000 samples (800 train / 200 val)")
print(f"MLP val MSE:    {mlp_mse:.5f}")
print(f"CNN val MSE:    {cnn_mse:.5f}")
print(f"EncDec val MSE: {np.mean(ed_val_losses[-5:]):.5f}  (pixel-wise, last 5 epochs avg)")
print(f"CNN vs MLP improvement: {(mlp_mse - cnn_mse)/mlp_mse * 100:.1f}%")
```

**Step 2: Restart kernel and run all cells top-to-bottom**

Kernel → Restart & Run All. Verify no errors and all plots render.

**Step 3: Final commit and push**

```bash
git add src/hw1.ipynb src/homework1.py
git commit -m "feat(hw1): complete notebook — MLP, CNN, encoder-decoder trained and evaluated"
git push origin main
```
