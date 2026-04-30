"""
Echo State Network (Reservoir Computer)
========================================
A Reservoir Computer projects the input into a high-dimensional nonlinear
dynamical system (the reservoir), then fits a *linear* readout via ridge
regression.  Only W_out is trained — the reservoir is fixed at construction.

Architecture
------------
    u(t) ──W_in──► r(t+1) = (1-α)·r(t) + α·tanh(W_res r(t) + W_in u(t) + b)
                      │
                    W_out (trained, linear, on augmented [r;1])
                      │
                    ŷ(t+1)

Why it works for chaos: the fixed nonlinear reservoir expands the input into
a rich feature space.  The linear readout then finds the combination of
features that best predicts the next state — requiring far fewer training
samples than a fully trainable RNN.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 — registers 3D projection


# ── Echo State Network ────────────────────────────────────────────────────────

class EchoStateNetwork:
    """
    Parameters
    ----------
    n_inputs        : input dimension
    n_reservoir     : number of reservoir nodes
    n_outputs       : output dimension
    spectral_radius : largest |eigenvalue| of W_res (< 1 for echo-state property).
                      0.9–0.95 is a safe default for chaotic systems.
    connectivity    : fraction of non-zero weights in W_res (e.g. 0.05 = 5 %)
    leak_rate       : leaky-integrator coefficient α ∈ (0, 1].
                      r(t+1) = (1-α)·r(t)  +  α·tanh(W_res r(t) + W_in u(t) + b)
                      α = 1 → standard ESN; α < 1 → each node has memory ~1/α steps.
                      For Lorenz (dt=0.02, ~50-step period) α ≈ 0.3 works well.
    input_scaling   : W_in entries drawn from Uniform(±input_scaling)
    bias_scaling    : reservoir bias entries drawn from Uniform(±bias_scaling)
    regularization  : L2 (ridge) coefficient for the output weight solve
    seed            : RNG seed
    """

    def __init__(
        self,
        n_inputs: int,
        n_reservoir: int,
        n_outputs: int,
        spectral_radius: float = 0.9,
        connectivity: float = 0.05,
        leak_rate: float = 1.0,
        input_scaling: float = 1.0,
        bias_scaling: float = 0.1,
        regularization: float = 1e-6,
        seed: int = 42,
    ) -> None:
        self.n_inputs        = n_inputs
        self.n_reservoir     = n_reservoir
        self.n_outputs       = n_outputs
        self.spectral_radius = spectral_radius
        self.leak_rate       = leak_rate
        self.regularization  = regularization

        rng = np.random.default_rng(seed)

        # Fixed input weights: (n_reservoir, n_inputs)
        self.W_in = rng.uniform(-input_scaling, input_scaling,
                                size=(n_reservoir, n_inputs))

        # Sparse reservoir matrix stored as a CSR matrix for fast matrix-vector products.
        # We use scipy.sparse to avoid building the full dense matrix and to compute
        # only the single largest eigenvalue efficiently (Arnoldi/ARPACK).
        W_dense = rng.uniform(-1.0, 1.0, size=(n_reservoir, n_reservoir))
        mask    = rng.random(size=(n_reservoir, n_reservoir)) < connectivity
        W_dense[~mask] = 0.0
        W_sp    = sp.csr_matrix(W_dense)
        # eigs uses ARPACK — fast even for N=2000
        sr      = float(np.abs(spla.eigs(W_sp, k=1, which="LM", return_eigenvectors=False)[0]))
        self.W_res = (W_sp * (spectral_radius / sr) if sr > 1e-10 else W_sp)

        # Reservoir bias
        self.b = rng.uniform(-bias_scaling, bias_scaling, size=n_reservoir)

        # Output weights — set by train().
        # We augment the state with 1 to give the readout a free bias term.
        self.W_out: np.ndarray | None = None

    # ── core dynamics ─────────────────────────────────────────────────────────

    def _step(self, r: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Leaky-integrator update:
        r(t+1) = (1-α)·r(t)  +  α·tanh(W_res r(t) + W_in u(t) + b)
        """
        α = self.leak_rate
        return (1.0 - α) * r + α * np.tanh(self.W_res @ r + self.W_in @ u + self.b)

    def step(self, r: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Public single-step reservoir update.

        Advances the reservoir state by one timestep given input u.
        Use this when you need to drive the reservoir manually (e.g. online
        learning loops) and track the state r yourself between calls.

        Parameters
        ----------
        r : (n_reservoir,)  current reservoir state
        u : (n_inputs,)     input at this timestep

        Returns
        -------
        r_next : (n_reservoir,)  next reservoir state
        """
        return self._step(r, u)

    def online_update(
        self,
        r:             np.ndarray,
        error:         np.ndarray,
        learning_rate: float = 1e-3,
    ) -> None:
        """Delta rule (Widrow-Hoff / LMS) update of the output weights.

        Adjusts W_out by one gradient-descent step on the squared prediction
        error, using the current (augmented) reservoir state as the feature
        vector:

            W_out += lr · error ⊗ φ(r)

        where φ(r) = [r; r²; 1]  (same quadratic augmentation used in train).

        This is the online analog of ridge regression — equivalent to the
        biological LTD/LTP rule the user referenced:

            W_pc += learning_rate * error * reservoir_state

        Parameters
        ----------
        r             : (n_reservoir,)  current reservoir state (from step())
        error         : (n_outputs,)    error signal = target − prediction.
                        Positive error → weights increase to reduce future error.
        learning_rate : step size (default 1e-3).  Too large → unstable;
                        too small → slow adaptation.  Try 1e-4 to 1e-2.

        Raises
        ------
        RuntimeError  if train() has not been called yet (W_out is None).
        """
        if self.W_out is None:
            raise RuntimeError("Call train() before online_update() — W_out is not initialised.")
        phi = self._augment(r)                       # (2N+1,)
        # outer product: (n_outputs, 1) * (1, 2N+1)  →  (n_outputs, 2N+1)
        self.W_out += learning_rate * np.outer(error, phi)

    def _augment(self, r: np.ndarray) -> np.ndarray:
        """Quadratic readout: [r; r²; 1].

        The Lorenz equations have quadratic nonlinearities (xy, xz terms).
        Giving W_out access to r² lets the linear readout model those
        interactions without needing a much larger reservoir.
        """
        return np.concatenate([r, r * r, [1.0]])

    def _readout(self, r: np.ndarray) -> np.ndarray:
        if self.W_out is None:
            raise RuntimeError("Call train() before predicting.")
        return self.W_out @ self._augment(r)

    # ── training ──────────────────────────────────────────────────────────────

    def train(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        washout: int = 200,
    ) -> tuple[float, np.ndarray]:
        """
        Drive the reservoir with `inputs`, collect states after the washout
        period, then solve for W_out via ridge regression.

        Parameters
        ----------
        inputs  : (T, n_inputs)  — driving signal
        targets : (T, n_outputs) — desired output at each timestep
        washout : first N steps discarded so the reservoir forgets its initial state

        Returns
        -------
        train_mse : MSE on the training window
        r_final   : final reservoir state (seed for downstream prediction)
        """
        T     = inputs.shape[0]
        N     = self.n_reservoir
        N_aug = 2 * N + 1                     # [r; r²; 1]
        α     = self.leak_rate
        r     = np.zeros(N)

        W_in_u      = inputs @ self.W_in.T    # (T, N) — pre-batched input projection
        W_res_dense = (self.W_res.toarray()
                       if sp.issparse(self.W_res) else self.W_res)

        states = np.zeros((T - washout, N_aug))
        for t in range(T):
            r = (1.0 - α) * r + α * np.tanh(W_res_dense @ r + W_in_u[t] + self.b)
            if t >= washout:
                i = t - washout
                states[i, :N]      = r
                states[i, N:2*N]   = r * r    # quadratic features
                states[i, 2*N]     = 1.0      # bias node

        Y = targets[washout:]

        λ = self.regularization
        self.W_out = np.linalg.solve(
            states.T @ states + λ * np.eye(N_aug),
            states.T @ Y,
        ).T                                   # (n_outputs, N_aug)

        train_mse = float(np.mean((states @ self.W_out.T - Y) ** 2))
        return train_mse, r

    # ── prediction ────────────────────────────────────────────────────────────

    def predict_open_loop(
        self,
        inputs: np.ndarray,
        r0: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Open-loop prediction: reservoir is driven by the *true* input each step.
        Isolates readout accuracy from error accumulation.
        Returns (T, n_outputs).
        """
        α   = self.leak_rate
        W_res_dense = (self.W_res.toarray()
                       if sp.issparse(self.W_res) else self.W_res)
        r   = np.zeros(self.n_reservoir) if r0 is None else r0.copy()
        W_in_u = inputs @ self.W_in.T         # (T, N) — pre-batched
        out = np.zeros((len(inputs), self.n_outputs))
        for t in range(len(inputs)):
            r      = (1.0 - α) * r + α * np.tanh(W_res_dense @ r + W_in_u[t] + self.b)
            out[t] = self._readout(r)
        return out

    def predict_closed_loop(
        self,
        seed_inputs: np.ndarray,
        n_steps: int,
        r0: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Closed-loop (autonomous) generation.

        1. Warm-up: drive the reservoir with seed_inputs (teacher-forced).
        2. Generate: feed the network's own output back as the next input.

        This lets the ESN hallucinate a trajectory on the attractor
        without ever seeing the true signal again.

        Returns generated trajectory of shape (n_steps, n_outputs).
        """
        α           = self.leak_rate
        W_res_dense = (self.W_res.toarray()
                       if sp.issparse(self.W_res) else self.W_res)
        r = np.zeros(self.n_reservoir) if r0 is None else r0.copy()

        # Teacher-forced warm-up
        W_in_seed = seed_inputs @ self.W_in.T
        for t in range(len(seed_inputs)):
            r = (1.0 - α) * r + α * np.tanh(W_res_dense @ r + W_in_seed[t] + self.b)

        # Autonomous generation — feed own output back as next input
        generated = np.zeros((n_steps, self.n_outputs))
        u = self._readout(r)
        for t in range(n_steps):
            r            = (1.0 - α) * r + α * np.tanh(W_res_dense @ r
                                                         + self.W_in @ u + self.b)
            u            = self._readout(r)
            generated[t] = u

        return generated


# ── Lorenz attractor ──────────────────────────────────────────────────────────

def _lorenz_deriv(
    xyz: np.ndarray,
    sigma: float = 10.0,
    rho: float   = 28.0,
    beta: float  = 8.0 / 3.0,
) -> np.ndarray:
    x, y, z = xyz
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])


def generate_lorenz(
    n_steps: int = 10_000,
    dt: float = 0.02,
    init: np.ndarray | None = None,
    sigma: float = 10.0,
    rho: float   = 28.0,
    beta: float  = 8.0 / 3.0,
) -> np.ndarray:
    """
    Integrate the Lorenz system with RK4.
    Returns trajectory of shape (n_steps, 3).
    """
    xyz  = np.array([1.0, 0.0, 0.0] if init is None else init, dtype=float)
    traj = np.zeros((n_steps, 3))
    traj[0] = xyz
    for t in range(1, n_steps):
        k1 = _lorenz_deriv(xyz,           sigma, rho, beta) * dt
        k2 = _lorenz_deriv(xyz + k1 / 2,  sigma, rho, beta) * dt
        k3 = _lorenz_deriv(xyz + k2 / 2,  sigma, rho, beta) * dt
        k4 = _lorenz_deriv(xyz + k3,       sigma, rho, beta) * dt
        xyz     = xyz + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        traj[t] = xyz
    return traj


# ── test: predict the Lorenz attractor ───────────────────────────────────────

if __name__ == "__main__":

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("Generating Lorenz trajectory …")
    N_TOTAL  = 15_000    # more data for a richer attractor sample
    WASHOUT  = 500
    N_TRAIN  = 10_000    # longer training window
    SEED_LEN = 500       # steps used to sync reservoir state before closed-loop
    N_GEN    = 3_000     # closed-loop generation (covers well past 1000-step target)
    DT       = 0.02

    traj = generate_lorenz(n_steps=N_TOTAL + 1, dt=DT)

    # Normalise to zero mean / unit std per coordinate
    mean, std = traj.mean(0), traj.std(0)
    data = (traj - mean) / std                        # (N_TOTAL+1, 3)

    # One-step-ahead task: input=x(t), target=x(t+1)
    u_all = data[:-1]                                 # (N_TOTAL, 3)
    y_all = data[1:]                                  # (N_TOTAL, 3)

    u_train = u_all[:N_TRAIN];  y_train = y_all[:N_TRAIN]
    # Test window starts right after training
    u_test  = u_all[N_TRAIN:]
    y_test  = y_all[N_TRAIN:]

    # ── 2. Build & train ──────────────────────────────────────────────────────
    print("Building reservoir and computing spectral radius …")
    esn = EchoStateNetwork(
        n_inputs        = 3,
        n_reservoir     = 1_500,    # more neurons → richer nonlinear feature space
        n_outputs       = 3,
        spectral_radius = 0.95,
        connectivity    = 0.1,
        leak_rate       = 0.3,      # leaky integrator — key for Lorenz
        input_scaling   = 1.0,
        bias_scaling    = 0.1,
        regularization  = 1e-6,     # slightly looser ridge for the 3001-dim quadratic readout
        seed            = 0,
    )

    print("Training …")
    train_mse, r_final = esn.train(u_train, y_train, washout=WASHOUT)
    print(f"  Train MSE : {train_mse:.2e}")

    # ── 3. Open-loop test ─────────────────────────────────────────────────────
    # Run open-loop from the reservoir state at the end of training
    N_TEST   = min(2_000, len(u_test))
    preds_open = esn.predict_open_loop(u_test[:N_TEST], r0=r_final)
    test_mse   = float(np.mean((preds_open - y_test[:N_TEST]) ** 2))
    nrmse      = float(np.sqrt(test_mse) / y_test[:N_TEST].std())
    print(f"  Test MSE  : {test_mse:.2e}  |  NRMSE : {nrmse:.4f}")
    print(f"  (NRMSE < 0.05 is excellent one-step-ahead fit)")

    # ── 4. Aligned closed-loop generation ────────────────────────────────────
    # Seed the reservoir with the first SEED_LEN steps of test data (teacher-
    # forced) so the reservoir state matches the true trajectory.  Then generate
    # autonomously and compare with the *continuation* of that true trajectory.
    print("Running aligned closed-loop autonomous generation …")
    seed_seq  = u_test[:SEED_LEN]                     # true signal warm-up
    true_cont = y_test[SEED_LEN:SEED_LEN + N_GEN]    # ground-truth we compare against

    generated = esn.predict_closed_loop(seed_seq, n_steps=N_GEN, r0=r_final)

    # Denormalise for physical units
    true_phys = true_cont * std + mean
    gen_phys  = generated * std + mean

    # Valid-time metric: steps until cumulative RMSE exceeds threshold
    cumrmse = np.sqrt(np.cumsum(np.mean((generated - true_cont)**2, axis=1))
                      / np.arange(1, N_GEN + 1))
    threshold   = 0.4 * true_cont.std()
    valid_steps = int(np.argmax(cumrmse > threshold)) if np.any(cumrmse > threshold) else N_GEN
    print(f"  Valid prediction time ≈ {valid_steps * DT:.2f} s  "
          f"({valid_steps} steps, threshold = {threshold:.3f})")

    # ── 5. Plot ───────────────────────────────────────────────────────────────
    T_ts  = min(1_200, N_GEN)                         # show 1200 steps (24 s) in time-series
    t_ax  = np.arange(T_ts) * DT

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("Reservoir Computer — Lorenz Attractor", fontsize=14, fontweight="bold")

    # Row 1: 3-D attractors
    for col, (data3d, title, color) in enumerate([
        (true_phys[:N_GEN], "True attractor (test)", "steelblue"),
        (gen_phys,           "ESN closed-loop",       "tomato"),
    ]):
        ax = fig.add_subplot(2, 3, col + 1, projection="3d")
        ax.plot(*data3d.T, lw=0.35, color=color, alpha=0.85)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.tick_params(labelsize=7)

    ax_ov = fig.add_subplot(2, 3, 3, projection="3d")
    ax_ov.plot(*true_phys[:N_GEN].T, lw=0.3, color="steelblue", alpha=0.5, label="true")
    ax_ov.plot(*gen_phys.T,          lw=0.3, color="tomato",    alpha=0.5, label="ESN")
    ax_ov.set_title("Overlay", fontsize=10)
    ax_ov.legend(fontsize=7)
    ax_ov.tick_params(labelsize=7)

    # Row 2: per-coordinate time series (aligned — both start from same point)
    for col, coord in enumerate(["x", "y", "z"]):
        ax = fig.add_subplot(2, 3, 4 + col)
        ax.plot(t_ax, true_phys[:T_ts, col], color="steelblue", lw=1.2, label="true")
        ax.plot(t_ax, gen_phys[:T_ts,  col], color="tomato",    lw=1.2, ls="--", label="ESN")
        # Shade the region where prediction is still valid
        if valid_steps > 0 and valid_steps * DT <= T_ts * DT:
            ax.axvline(valid_steps * DT, color="gray", ls=":", lw=1.0,
                       label=f"diverge ≈ {valid_steps * DT:.1f}s")
        ax.set_title(f"{coord}(t) — aligned from same IC", fontsize=10)
        ax.set_xlabel("time (s)", fontsize=8)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()
