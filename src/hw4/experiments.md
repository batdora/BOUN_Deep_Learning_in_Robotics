# HW4 experiments journal

A chronological record of what we tried, why we tried it, what the numbers looked like, and what they pushed us toward next. Each iteration is a self-contained section so the document grows naturally as we loop.

---

## Iteration 1 — Baseline

### Setup

The task is to train a Conditional Neural Movement Primitive (CNMP) on push demonstrations where a UR5e arm sweeps along a Bezier curve in the y–z plane and occasionally hits a box of random height `h`. For each demonstration we record the high-level state `(e_y, e_z, o_y, o_z, h)` at 100 evenly-spaced time steps; the height `h` is constant within a demonstration.

For the first run we took the most literal reading of the TA spec and changed nothing about the stock `CNP` class. The query dimension was set to `d_x = 2` (time `t`, replicated per-step height `h`) and the target dimension to `d_y = 4` (`e_y, e_z, o_y, o_z`). Because `h` is replicated to every target query, it reaches the decoder on every forward pass — this is the spec's "condition given to the decoder" interpreted in the cheapest way that does not require modifying the provided network.

Data collection used the starter Bezier sampler unchanged: 150 trajectories, heights `h ~ U[0.03, 0.10]`, middle control points `p_2.z, p_3.z ~ U[1.04, 1.40]`. No bias toward collisions; we deliberately took the natural distribution so we could see what the default task looks like before tuning anything.

The CNMP itself was left at the provided architecture (three hidden layers of width 128, ReLU, a `softplus(...) + 0.1` uncertainty head). We trained for 20 000 iterations of batch size 16 with Adam at `lr = 1e-4`, NLL loss. For each iteration we sampled `n_context, n_target ~ U{1, ..., 10}` independently, and within the chosen trajectory we picked disjoint random index subsets of those sizes, so the model never saw a target point in its own context. Evaluation ran 100 independent tests with the same sampling distribution, reporting separate MSEs for the end-effector `(e_y, e_z)` and the object `(o_y, o_z)`.

### Results

Training converged cleanly. The moving-average NLL began at about `+1.20`, dropped to roughly `-0.90` within a few hundred iterations, plateaued briefly, then descended again around iteration 2 000–3 000 to `-1.30` and drifted slowly down to **`-1.364`** over the last 500 iterations. Gaussian densities above 1 pull NLL negative, so this indicates the model is emitting reasonably tight predictive distributions rather than collapsing std to the `0.1` floor. See [`training_loss.png`](training_loss.png) and [`training_loss.csv`](training_loss.csv).

Evaluation over 100 random tests gave:

| group        | mean MSE  | std MSE   |
|--------------|-----------|-----------|
| end-effector | 0.000520  | 0.000473  |
| object       | 0.000433  | 0.000914  |

(see [`mse_barplot.png`](mse_barplot.png), [`mse_results.csv`](mse_results.csv)).

### What the numbers tell us

The first thing worth noticing is that the two means are in the same ballpark; the object mean is even marginally lower. Taken alone, this would suggest the model does just as well on the object as on the arm. The second column disagrees: the object's standard deviation is about twice the end-effector's, and its lower error-bar whisker crosses zero.

Reading those two columns together points at a bimodal distribution of object errors. Most tests contribute nearly zero to the object MSE — these are the runs where the arm clears the box and the object never moves, so any smooth predicted trajectory that stays near the initial `(o_y, o_z)` scores well. A minority of tests contribute much larger errors — these are the collisions, where the final object position depends sensitively on exactly where and how the arm clipped the box. The mixture of "almost zero" and "occasionally large" produces a low mean with a heavy upper tail, which is exactly the shape the std captures.

The end-effector, by contrast, is effectively deterministic given `(t, h)` plus a handful of context points. The Bezier controller is independent of `h`, so the model just has to fit a smooth mapping from time to arm position, and it does.

That leaves a structural limitation visible in the numbers: the two Bezier middle-control points `p_2.z` and `p_3.z` are the actual drivers of whether a collision occurs, and they are *not* in the model's input. `(t, h)` alone cannot tell the model "this run is a collision run"; that information is only implicit in the y–z values of the observed context points, which the CNP currently aggregates with a plain mean. A context point near the collision moment carries far more predictive weight for the object than a context point sampled at `t = 0.05`, but mean pooling gives them equal say.

So the failure mode is specific: the model is not *refusing* to learn the object, it is behaving rationally given what it can see. It outputs a broad Gaussian around the no-motion prior and absorbs the collision cases as variance, because honest uncertainty scores better on NLL than a confident wrong guess.

### What this pushes us toward next

Three candidate directions follow from the diagnosis above, in roughly increasing order of effort:

1. **Give the current architecture more to work with.** Raise `n_context_max` (so each training update sees more of the arm's shape), widen and deepen the CNP, train longer. If this noticeably shrinks the object std, the model was information- or capacity-starved.
2. **Shift the data distribution toward collisions.** 150 trajectories with uniformly random middle control points probably produced very few clean collision examples. Biasing `p_2.z` and `p_3.z` downward, or simply collecting more trajectories, gives the model more collision samples to fit. Do this if step 1 stalls.
3. **Change the aggregator.** Replace the mean with cross-attention (Attentive NP, Kim et al. 2019) so that each target query can selectively weight the context points that matter — particularly the ones near the collision moment. This is the textbook fix for "some context points are dramatically more informative than others" and should directly reduce object-side variance if the bottleneck is really the aggregator.

We will run step 1 next and see whether the object std responds.

---

## Iteration 2 — Bigger model, more context, longer training

### Setup

We held the data, the loss, the architecture family, and the evaluation protocol fixed, and changed four knobs at once on the training side:

| knob                 | v1 (baseline) | v2     |
|----------------------|---------------|--------|
| hidden size          | 128           | 256    |
| hidden layers        | 3             | 5      |
| `n_context_max`      | 10            | 30     |
| `n_target_max`       | 10            | 30     |
| training iterations  | 20 000        | 60 000 |

We bundled all four together because they target the same hypothesis — *the baseline is starved of information and capacity* — and we wanted a cheap test of that hypothesis before considering the more invasive architectural change (attention). The evaluation seed was kept at `42`, so the 100 random tests are the *same* trajectories, context indices, and target indices as in iteration 1; only the model changed.

The larger context range is especially worth highlighting. Sampling up to 30 context points from a 100-step trajectory means a typical training batch sees roughly a third of the arm's path at once, which is enough to resolve the Bezier shape. Under `n_context_max = 10` the model often had to guess the arm arc from barely-observable evidence; at 30 the arc is essentially visible.

### Results

Training loss proceeded through two plateaus. The first descent mirrored v1 — a fast drop to about `-0.9`, then a brief plateau, then the familiar second step that took the loss to `-1.35` by iteration 3 000. A *third* step then appeared around iteration 11 000–12 000 that pulled the moving average down to about `-1.38` and held it there through iteration 60 000. The final 500-iteration average was **`-1.380`** versus v1's `-1.364`. In NLL terms this is a small gain, but it is consistent with the extra capacity being used rather than sitting idle. See [`training_loss_v2.png`](training_loss_v2.png), [`training_loss_v2.csv`](training_loss_v2.csv).

Test MSE (same 100 seeded tests as v1):

| group        | v1 mean   | v2 mean   | v1 std    | v2 std    |
|--------------|-----------|-----------|-----------|-----------|
| end-effector | 0.000520  | **0.000062** | 0.000473  | **0.000105** |
| object       | 0.000433  | **0.000103** | 0.000914  | **0.000626** |

See [`mse_barplot_v2.png`](mse_barplot_v2.png), [`mse_results_v2.csv`](mse_results_v2.csv).

### What the numbers tell us

Every cell improved, but the improvements are unevenly distributed.

The end-effector benefited most. Mean MSE dropped by a factor of about 8, and std dropped by about 4.5. This is consistent with the iteration-1 diagnosis that the arm trajectory is a smooth, fully-determined function of `(t, h)` plus enough context to reveal the Bezier arc. Once we gave the model more context per update and more capacity to fit that function, it did. The improvement here is not *interesting* in a scientific sense — it is the baseline-getting-out-of-its-own-way kind of improvement.

The object improved too, but the shape of the improvement is telling. Mean MSE dropped by about 4x — good — but std only fell by about 30%. The ratio `obj_std / obj_mean` actually *grew*, from 2.1 in v1 to 6.1 in v2. In plain language: the easy cases (arm clears the box, object stays put) got much easier, while the hard cases (collisions) stayed essentially as hard as before, so the heavy-tailed mixture became *more* heavy-tailed in relative terms.

This is the clearest piece of evidence we have so far that the remaining error is structural, not a capacity problem. Adding width, depth, iterations, and context points all reduce average error, but none of them change how information flows between context and target inside the network. Mean aggregation cannot emphasise the handful of context points near the collision moment over the rest, so when the decoder is asked to predict `o_y`, `o_z` at some post-contact time it still sees a smeared-out summary that washes out the very features (arm low and near the object during the collision window) it would need in order to commit to a specific post-contact location.

There is a secondary observation in the training curve worth noting. The new third descent around iteration 11 000–12 000 is unlikely to be the model finally "cracking collisions" — if it were, we would have seen the object std fall harder than it did. More likely it corresponds to the extra layers learning a slightly better smooth fit for the deterministic parts of the task. The collision mode continues to be absorbed as predictive variance.

### What this pushes us toward next

The remaining heavy tail is exactly the scenario Attentive NP was designed for, so that becomes the natural iteration 3: replace the mean aggregator with cross-attention so each target can weight context points by relevance rather than uniformly. We will keep the v2 hyperparameters otherwise, so any further improvement in object std is attributable to the aggregator change and not to architectural bulk.

An alternative, cheaper move is to first try expanding or biasing the dataset — more trajectories, or trajectories with `p_2.z, p_3.z` sampled from a lower-height distribution so collisions are over-represented. The reason we are *not* prioritising this is that iteration 2 already shows we can reduce easy-case error on demand; the bottleneck is no longer "we haven't seen enough easy cases", it is "we cannot turn observed evidence into a precise collision prediction". That is an aggregator problem first, a data problem second.
