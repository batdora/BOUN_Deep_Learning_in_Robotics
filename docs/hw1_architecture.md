# HW1 Architecture

## What is this homework?

A MuJoCo simulation runs a UR5e robot arm that pushes an object on a tabletop. The physics engine computes exactly where the object ends up. We collect 1000 such push episodes and use them as supervised training data. The goal is to train neural networks that can predict the outcome of a push **without running the simulation** — just from a top-down image of the scene and the chosen push direction.

This is called **forward modelling**: the model learns the mapping `(scene, action) → outcome`.

## The Simulation as a Data Oracle

```
reset()
  └─ spawns object at (0.6, 0.0) with random type (cube/sphere) and size (2–3 cm)

state()  →  img_before (128×128 RGB),  pos_before (x, y)
                                  │
                            step(action_id)
                                  │
                    MuJoCo physics: contact, friction, inertia
                                  │
state()  →  img_after  (128×128 RGB),  pos_after  (x, y)   ← labels
```

There are 4 discrete actions (0–3), each a fixed sweep trajectory:

| Action | Direction | Trajectory |
|--------|-----------|------------|
| 0 | +x (forward) | x: 0.4 → 0.8, y fixed at 0 |
| 1 | -x (backward) | x: 0.8 → 0.4, y fixed at 0 |
| 2 | +y (right) | y: -0.2 → +0.2, x fixed at 0.6 |
| 3 | -y (left) | y: +0.2 → -0.2, x fixed at 0.6 |

All trajectories pass through the object's spawn point — contact is guaranteed every episode. The outcome varies based on object shape and size, both of which are visible in the image.

## Three Models

### Model 1: MLP — Position Prediction

The simplest possible approach. The image is flattened into a 49,152-dimensional vector and concatenated with a one-hot action vector (length 4). A fully connected network regresses directly to the (x, y) position.

```
[img_before flattened (49152)] ++ [one_hot(action) (4)]
                    │
           Linear 49156 → 1024 → ReLU
           Linear  1024 → 512  → ReLU
           Linear   512 → 2
                    │
              predicted (x, y)
```

**Weakness**: No concept of spatial structure. Every pixel is treated independently — the network cannot directly reason about object shape or size as geometric features.

---

### Model 2: CNN — Position Prediction

A convolutional backbone extracts spatial features from the image. The action is injected after the backbone, just before the regression head.

```
img_before (3×128×128)
        │
   Conv(3→32)  →  64×64
   Conv(32→64) →  32×32
   Conv(64→128)→  16×16
   Conv(128→256)→  8×8
   AvgPool → flatten → (256,)
        │
   concat one_hot(action) → (260,)
        │
   Linear(260→128) → Linear(128→2)
        │
   predicted (x, y)
```

**Why better than MLP**: Convolutional layers detect edges, shapes, and textures — features that directly encode whether the object is a cube (sharp edges) or sphere (round). The global average pool compresses spatial information into a fixed-size descriptor before the action conditions the final output.

---

### Model 3: CNN Encoder-Decoder — Image Reconstruction

Predicts what the scene will *look like* after the push, rather than just where the object lands. An encoder compresses the input image to a bottleneck; the action is injected there; a symmetric decoder reconstructs the output image.

```
img_before (3×128×128)
        │
   ┌─── Encoder ───────────────────────┐
   │  Conv(3→32)   128→64              │
   │  Conv(32→64)   64→32              │
   │  Conv(64→128)  32→16              │
   │  Conv(128→256) 16→8               │
   │  Conv(256→256)  8→4  ← bottleneck │
   └───────────────────────────────────┘
        │ (256×4×4)
   concat broadcast(one_hot(action)) → (260×4×4)
        │
   ┌─── Decoder ───────────────────────┐
   │  ConvT(260→256) 4→8               │
   │  ConvT(256→128) 8→16              │
   │  ConvT(128→64) 16→32              │
   │  ConvT(64→32)  32→64              │
   │  ConvT(32→3)  64→128 → Sigmoid    │
   └───────────────────────────────────┘
        │
   img_after predicted (3×128×128) ∈ [0,1]
```

**Why action goes at the bottleneck**: The bottleneck is the most compressed, abstract representation of the scene — 256 numbers encoding "what object is here and how big." Injecting the action here lets the decoder ask "given this object and *this* push direction, how should I reconstruct the resulting scene?" Injecting earlier (at the input) would force the conv layers to jointly process image content and action simultaneously, which is less clean.

## Loss Functions

| Model | Loss | Why |
|-------|------|-----|
| MLP (pos) | MSE | Regression to continuous (x, y) |
| CNN (pos) | MSE | Same |
| Encoder-Decoder | MSE on pixels | Pixel-wise reconstruction; images are normalised [0,1] |

## What to Expect

- **MLP vs CNN on position**: CNN should achieve lower MSE. The gap demonstrates the value of spatial inductive biases for visual inputs.
- **Image reconstruction**: The decoder will learn to place a blurred/approximate object at roughly the right location. Exact pixel reconstruction is hard — the model cannot know the exact final position with certainty from the image alone.
- **Failure cases**: If the object is at the edge of the table and the push sends it further, the model may struggle since those are rarer samples.
