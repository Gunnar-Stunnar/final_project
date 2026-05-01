# Motor Control Final Project — Design Log

**Course:** Motor Control, Spring 2026 — CU Denver  
**Model:** MoBL-ARMS upper limb OpenSim model (shoulder + elbow, 2 active DOFs)  
**Language:** Python (OpenSim scripting environment, `opensim_scripting` conda env)

---

## Overview

This project builds a real-time simulation of a human arm reaching to randomly placed targets, with a biologically-inspired **cerebellar forward model** (Echo State Network) that learns to predict the arm's motion one step ahead. The visualizer shows both the real arm (white) and the cerebellum's internal prediction (blue ghost). A real-time plot tracks what the cerebellum expected versus what actually happened.

---

## Phase 1 — Arm Control & Reaching

### Decision: Inverse Kinematics for target sampling
Rather than placing targets at random 3D coordinates (many of which are unreachable), we sample reachable targets by:
1. Drawing random joint angles within physiological limits
2. Running **forward kinematics** (FK) to get the resulting 3D hand position
3. Using those FK-generated joint angles directly as `q_goal`

This guarantees every target is reachable with zero IK residual, eliminating a whole class of "arm can't reach the ball" failures.

### Decision: PID torque control with anti-windup
A proportional-integral-derivative (PID) controller drives each joint actuator. Key tuning decisions:
- **Kp** (proportional): high enough to overcome inertia
- **Kd** (derivative): provides viscous damping, prevents overshoot
- **Ki** (integral): eliminates steady-state gravity error that pure PD couldn't correct
- **Anti-windup**: integral accumulator clamped to `KI_MAX` to prevent wind-up during long transients
- **Integral reset** on every new target to avoid the old error fighting the new goal

### Decision: Lock uncontrolled DOFs
The OpenSim arm model has additional degrees of freedom (elevation angle, shoulder rotation, pro/supination) that are not actively controlled. Without locking them, these DOFs drift under gravity and corrupt the arm's trajectory. They are locked in both the simulation state and the IK model.

### Decision: CoordinateActuator with `optimal_force = 1.0`
Setting `optimal_force = 1.0` makes the control signal equal the torque in Nm directly, with no hidden scaling. This was essential for tuning — early versions had the effective torque 80× too large, causing explosive instability.

---

## Phase 2 — Visualization

### Decision: Live OpenSim/SimTK visualizer
The built-in SimTK visualizer renders the arm in real time. Key challenges:
- **`simbody-visualizer` path**: manually prepended the build output directory to `PATH`
- **Ground plane removal**: set background to `SolidColor` (dark navy) to eliminate the ground reference
- **Moving the target ball**: `DecorativeGeometry.setTransform()` returns a copy, not a reference — a fresh decoration lookup is required each frame

### Decision: Red target ball via `DecorativeSphere`
The active target is shown as a red sphere attached to the model's Ground body. Its transform is updated every frame by removing and re-adding the decoration with `setTransform`.

### Decision: Ghost arm as stick figure with `DecorativeCylinder`
A "holographic" blue stick figure overlays the real arm to show the cerebellum's predicted position. `DecorativeLine` couldn't be updated dynamically (`.setPoint1/.setPoint2` not accessible post-creation). Replaced with thin `DecorativeCylinder` objects whose transforms are recomputed from the predicted joint positions each frame.

---

## Phase 3 — Echo State Network (Cerebellum Forward Model)

### Biological motivation
The cerebellum acts as a **forward model** of the body: it receives the current state (sensory afferents) and efference copy (motor command) and predicts the resulting next state. This prediction runs faster than sensory feedback, enabling smooth, anticipatory motor control.

### Decision: Echo State Network (reservoir computer)
An ESN was chosen over a standard feedforward network because:
- **Reservoir dynamics** naturally capture the temporal context of ongoing arm movement
- **Fixed recurrent weights** (only `W_out` is trained) makes online learning tractable — no backpropagation through time
- **Spectral radius and leak rate** control the effective memory horizon of the reservoir

**Architecture:**
- `N_reservoir = 500` neurons (sparse random recurrent connections)
- `spectral_radius = 0.95` — near the edge of chaos for rich dynamics
- `leak_rate = 0.3` — leaky integrator, smooths reservoir response
- **Quadratic readout**: augmented state `φ = [r; r²; 1]` captures nonlinear arm dynamics

**Input vector (10-dim):** `[q, qd, qdd, τ, q_goal]` — current joint angles, velocities, accelerations, PID torques, and target joint angles

**Output vector (4-dim):** `[q_next, qd_next]` — predicted next joint state

### Decision: Velocity scaling in the target vector
Raw joint velocity (rad/s) has much larger magnitude than joint position (rad). Ridge regression weighted velocity residuals ~100× more than position residuals. Fix: multiply `qd` by `dt` (the timestep, 0.005 s) to convert to rad/step, placing both on the same numerical scale.

### Decision: Velocity clipping at target transitions
When a new target fires, the PID commands a burst torque that can launch the arm at 10–20 rad/s in one timestep. Storing this spike in the training data corrupts the covariance matrices. Fix: clip `qd` in the target vector to `±5·dt` rad/step — preserves fast-but-real motion while rejecting physically implausible discontinuities.

---

## Phase 4 — Online Learning: Recursive Least Squares (RLS)

### Why RLS over batch ridge regression
| | Batch replay buffer | Online RLS |
|---|---|---|
| W_out update | Every N steps (batch solve) | Every single step |
| Storage | Ring buffer of (φ, y) pairs | Just P (N×N) + W_out |
| Forgetting | Hard eviction | Smooth exponential (λ) |
| Adapts to new dynamics | After buffer turns over | Immediately |

### Decision: Sherman-Morrison rank-1 update
At each step, the covariance matrix `P = (ΦᵀΦ)⁻¹` is updated via the rank-1 formula:
```
gain   = P·φ / (λ + φᵀ·P·φ)
W_out += error ⊗ gain
P      = (P − gain·Ppᵀ) / λ
```
This is O(N²) per step — the same cost as accumulating outer products, but always gives the exact optimal readout weights.

### Decision: Forgetting factor λ = 0.999
A forgetting factor < 1 gives exponential decay to past observations, so old arm movements naturally fade as new ones replace them. At λ = 0.999, a sample from 1000 steps ago has weight `0.999^1000 ≈ 0.37`.

### Decision: High initial uncertainty (δ = 1e4)
Initializing `P = δ·I` with large δ means the first few samples have large gain and immediately shape `W_out`. This accelerates early learning from a cold start.

---

## Phase 5 — Threshold-Gated Learning (Climbing Fiber Model)

### Biological motivation
In the cerebellum, **climbing fibers** (from the inferior olive) only fire when there is a significant mismatch between the predicted and actual state. They are sparse, event-driven error signals — not continuous. Between climbing fiber events, Purkinje cells run pure inference.

### Decision: Error-threshold gate on RLS update
The RLS update only fires when `||Δq|| > ESN_ERROR_LEARN_THRESH` (default: 0.05 rad ≈ 3°). Below the threshold the cerebellum runs pure inference with no weight change. This:
- Reduces unnecessary computation
- Prevents noise from corrupting `W_out` when the model is already accurate
- Maps directly to the climbing fiber biology

### Decision: Per-target washout on reservoir reset
When a new target fires, `q_goal` jumps discontinuously in the ESN input. The reservoir has memory of the old goal baked into its recurrent state. Without resetting, the first post-transition prediction is corrupted by stale context.

**Fix:** On each new target:
1. `_esn_r[:] = 0` — clean slate, no old-goal memory
2. `_esn_target_washout_rem = 30` — suppress RLS updates and ghost display for ~0.15 s
3. During washout: reservoir fills with 30 steps of real arm + new `q_goal` context
4. After washout: predictions and learning resume from clean context

---

## Phase 6 — Learning Termination & Generalization Test

### Decision: Hard stop after N reaches
After `ESN_LEARN_STOP_REACHES` (default: 1) successful reaches, `W_out` and `P` are permanently frozen. The cerebellum then runs **pure inference** — the readout weights never change again.

This allows direct observation of generalization: how well does a model trained on N reaches predict the arm's behavior on novel targets in unseen parts of the workspace?

**Design rationale:** The biological cerebellum reaches "motor consolidation" — a point where further error signals are not required for accurate prediction. This hard stop simulates that consolidation.

---

## Phase 7 — Real-Time Perturbation System

### Decision: Keyboard-driven wrist force (SPACE key)
A `pynput` background listener detects SPACE key press/release. On press, a random 3D unit vector is drawn, scaled to 150 N, and stored as `_perturb_force`. On release, the force is zeroed.

**Force → joint torques via Jacobian transpose:**
```
J = numerical ∂(hand_pos) / ∂(q)   [3 × n_coords, finite differences]
τ_perturbation = Jᵀ · F_wrist
```
The perturbation torque is applied **after** the PID clamp, with its own limit (`PERTURBATION_TAU_LIM = 80 Nm`), so it cannot be absorbed by the PID budget.

### Decision: Runtime-adjustable damping (`[` / `]` keys)
The PID derivative gain `Kd` acts as viscous joint damping. Halving it makes the arm oscillatory and unpredictable — a test of whether the frozen cerebellum model can still track an arm with different dynamics.

- **`[`** — drops `Kd` to 15% of baseline → springy, oscillatory arm
- **`]`** — restores baseline `Kd` → smooth, damped motion returns

Both events are marked on the prediction plot with colored vertical lines (magenta / violet).

**Biological analogy:** Changing `Kd` at runtime is equivalent to changing the arm's mechanical impedance (e.g., muscle fatigue, external load, tone change). The cerebellum's frozen model will exhibit prediction error if it was trained under different dynamics — analogous to cerebellar adaptation failure after sudden limb loading.

---

## Phase 8 — Real-Time Prediction Monitor

### Decision: Single-window multi-subplot monitor
All prediction data is displayed in one matplotlib window with shared x-axis:

```
┌─────────────────────────────────────────────────┐
│  shoulder_elv:  actual (solid) vs predicted (──) │
├─────────────────────────────────────────────────┤
│  elbow_flexion: actual (solid) vs predicted (──) │
├─────────────────────────────────────────────────┤
│  ||ghost_hand − real_hand||  (m, green)         │
└────────────────────── time (s) ─────────────────┘
```

**Event markers (shared across all rows):**
| Marker | Meaning |
|---|---|
| Cyan ticks | RLS update fired (error crossed threshold) |
| Orange vertical | Successful reach completed |
| Red dashed | Learning permanently stopped |
| Magenta dash-dot | Damping reduced (`[` key) |
| Violet dash-dot | Damping restored (`]` key) |
| Lime dashed | Convergence milestone (EMA error below threshold) |

### Decision: Thread-safe plot queue
`pynput` callbacks run on a background thread. Matplotlib's Tk backend only allows GUI calls from the main thread. Solution: keyboard thread pushes event dicts to a `queue.SimpleQueue`; the main simulation loop drains the queue and draws annotations safely on each plot refresh cycle.

---

## Key Parameter Reference

| Parameter | Value | Role |
|---|---|---|
| `ESN_N_RESERVOIR` | 500 | Reservoir size |
| `ESN_RLS_LAMBDA` | 0.999 | Forgetting factor |
| `ESN_RLS_DELTA` | 1e4 | Initial P uncertainty |
| `ESN_RLS_WARMUP` | 200 | Steps before ghost shown |
| `ESN_WASHOUT_STEPS` | 100 | Initial reservoir settling |
| `ESN_TARGET_WASHOUT_STEPS` | 30 | Per-target reservoir reset window |
| `ESN_ERROR_LEARN_THRESH` | 0.05 rad | Climbing-fiber threshold |
| `ESN_LEARN_STOP_REACHES` | 1 | Reaches before W_out frozen |
| `PERTURBATION_FORCE_N` | 150 N | Wrist perturbation magnitude |
| `PERTURBATION_TAU_LIM` | 80 Nm | Perturbation torque cap |
| `stepsize` | 0.005 s | Simulation timestep |
| `Kp` | {shoulder: 100, elbow: 60} Nm/rad | PID proportional gain |
| `Kd` | {shoulder: 28, elbow: 14} Nm·s/rad | PID derivative / damping |
| `Ki` | {shoulder: 15, elbow: 8} Nm/(rad·s) | PID integral gain |

---

## Files

| File | Purpose |
|---|---|
| `main_script_incomplete.py` | Main simulation: arm control, ESN, visualization, plotting |
| `echoState.py` | `EchoStateNetwork` class with RLS, quadratic readout, Lorenz test |
| `PC.py` | Predictive Coding network implementations (generative + discriminative) |
| `PC_test.py` | PC regression test (nonlinear function fitting) |
| `activation.py` | Activation functions: `tanh`, `relu`, `relu_prime`, `linear` |
| `DESIGN_LOG.md` | This file |
