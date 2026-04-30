from turtle import forward
import numpy as np
from typing import Callable, List
from math import inf

class PredictiveCodingLayer:

    def __init__(self, dim, dim_above, h, h_prime, rng, is_top_layer,
                 state_lr=0.1, weight_lr=0.05):
        self.dim          = dim
        self.dim_above    = dim_above
        self.h            = h
        self.h_prime      = h_prime
        self.rng          = rng
        self.is_top_layer = is_top_layer
        self.state_lr     = state_lr
        self.weight_lr    = weight_lr

        self.mask = np.ones(dim)
        self.z    = np.zeros(dim)
        self.mu   = np.zeros(dim)
        self.e    = np.zeros(dim)

        # He initialisation: std = sqrt(2 / fan_in) keeps relu pre-activations
        # in a healthy range so neurons don't start dead.  std=0.01 kills every
        # neuron immediately — relu output is 0, h_prime is 0, no gradient flows.
        _std = np.sqrt(2.0 / dim_above) if (not is_top_layer and dim_above > 0) else 0.01
        self.W = rng.normal(0, _std, size=(dim, dim_above)) \
                 if not is_top_layer else None

    def reset(self):
        self.z = self.rng.normal(0, 0.01, size=self.dim)

    def prediction(self, z_above):
        self.mu = np.zeros(self.dim) if self.is_top_layer \
                  else self.W @ self.h(z_above)

    def forward(self, z_above):
        if self.W is None:
            return z_above
        return self.h(self.W.T @ z_above)

    def error_calc(self):
        self.e = self.z - self.mu

    def state_update(self, W_below, e_below):
        bottom_up = (W_below.T @ e_below) * self.h_prime(self.z) \
                    if W_below is not None else np.zeros(self.dim)
        self.z += self.state_lr * (-self.e + bottom_up) * self.mask

    def weight_update(self, z_above):
        if not self.is_top_layer:
            self.W += self.weight_lr * np.outer(self.e, self.h(z_above))

    def clamp(self, values):
        self.mask = np.zeros(self.dim)
        self.z    = values.copy()

    def unclamp(self):
        self.mask = np.ones(self.dim)

    def free_energy(self):
        return 0.5 * np.dot(self.e, self.e)


class PredictiveCodingNetwork:

    def __init__(self, layer_dims, h, h_prime, seed=42,
                 state_lr=0.1, weight_lr=0.05):
        self.layer_dims = layer_dims
        self.rng        = np.random.default_rng(seed)
        self.L          = len(layer_dims)
        self.layers: List[PredictiveCodingLayer] = []

        for i in range(self.L):
            is_top    = (i == self.L - 1)
            dim_above = layer_dims[i + 1] if not is_top else 0
            self.layers.append(
                PredictiveCodingLayer(
                    dim=layer_dims[i], dim_above=dim_above,
                    h=h, h_prime=h_prime, rng=self.rng,
                    is_top_layer=is_top,
                    state_lr=state_lr,
                    weight_lr=weight_lr,
                )
            )

    def _reset_free_layers(self):
        for l in self.layers:
            if np.any(l.mask == 1):
                l.reset()

    def _unclamp_all(self):
        for l in self.layers:
            l.unclamp()

    def _inference(self, free_energy_thresh=0.001, max_iter=500):
        self._reset_free_layers()
        F_prev = inf
        for _ in range(max_iter):

            # Step 1 — top-down predictions
            for i in range(self.L - 1):
                self.layers[i].prediction(self.layers[i + 1].z)
            self.layers[-1].prediction(None)

            # Step 2 — errors
            for layer in self.layers:
                layer.error_calc()

            # Step 3 — state updates.
            # mask=0 on clamped layers makes updates no-ops for those layers.
            for i in range(self.L):
                W_below = self.layers[i - 1].W if i > 0 else None
                e_below = self.layers[i - 1].e if i > 0 else None
                self.layers[i].state_update(W_below, e_below)

            F = sum(l.free_energy() for l in self.layers)
            if abs(F_prev - F) < free_energy_thresh:
                break
            F_prev = F
        return F

    # ------------------------------------------------------------------ #
    #  Training — clamp z^0, let everything above settle, update weights
    # ------------------------------------------------------------------ #
    def learn_one(self, x, free_energy_thresh=0.001):
        self._unclamp_all()
        self.layers[0].clamp(x)
        F = self._inference(free_energy_thresh)
        for i in range(self.L - 1):
            self.layers[i].weight_update(self.layers[i + 1].z)
        # return converged top state — this is the internal code for this input
        return F, self.layers[-1].z.copy()

    def learn_stream(self, X, free_energy_thresh=0.001, verbose=True):
        F_history  = []
        top_states = []
        for n, x in enumerate(X):
            F, z_top = self.learn_one(x, free_energy_thresh)
            F_history.append(F)
            top_states.append(z_top)
            if verbose and n % 20 == 0:
                print(f"  sample {n:4d}  F={F:.4f}  "
                      f"z^L range=[{z_top.min():.3f}, {z_top.max():.3f}]")
        mean_top = np.mean(top_states, axis=0)
        return np.array(F_history), mean_top, np.array(top_states)

    # ------------------------------------------------------------------ #
    #  Supervised — clamp z^0 to x AND z^L to y simultaneously.
    #  Hidden layers settle, then weights are updated.
    #  This teaches the generative model the association x ↔ y.
    # ------------------------------------------------------------------ #
    def supervised(self, x, y, free_energy_thresh=0.001):
        """
        Clamp the bottom layer to x (observation) and the top layer to y (cause).
        Run inference to settle the hidden layers, then update weights.

        After training the generative model knows: given cause y, generate x.
        Use predict(x) to invert this at test time via inference.
        """
        self._unclamp_all()
        self.layers[0].clamp(x)
        self.layers[-1].clamp(y)
        F = self._inference(free_energy_thresh)
        for i in range(self.L - 1):
            self.layers[i].weight_update(self.layers[i + 1].z)
        self._unclamp_all()
        return F

    # ------------------------------------------------------------------ #
    #  Predict — clamp z^0 to x, let z^L settle freely via inference.
    #  The converged top state is the network's inferred cause ≈ y.
    #  No weight update — pure inference (the "few-shot" property of PC).
    # ------------------------------------------------------------------ #
    def predict(self, x, free_energy_thresh=0.0001, max_iter=1000):
        """
        Inference-based prediction: clamp the observation (x) at the bottom,
        let ALL other layers — including the top — relax to equilibrium.
        The settled top-layer state is the network's best guess for the
        latent cause y that would generate this observation.

        This is the key few-shot property of generative PC:
        no weight update is needed — a handful of inference iterations adapts
        the representation to the new input.

        NOTE: requires an activation with h'(0) > 0 (e.g. tanh, sigmoid).
        With ReLU the top layer is initialised at z=0 and relu'(0)=0 so
        bottom-up corrections are immediately zeroed — inference is stuck.
        """
        self._unclamp_all()
        self.layers[0].clamp(x)
        # Top layer is FREE — inference will drive it toward the inferred y.
        self._inference(free_energy_thresh, max_iter=max_iter)
        return self.layers[-1].z.copy()

    # ------------------------------------------------------------------ #
    #  Generation — clamp z^L to learned mean, free everything below
    #  At convergence z^0 IS the network's reconstruction
    # ------------------------------------------------------------------ #
    def generate(self, top_state, free_energy_thresh=0.001):
        """
        Clamp the top layer to the learned representation.
        Free all layers below — they self-organize via top-down predictions.
        Read z^0 at convergence — this is the hallucinated input.
        """
        self._unclamp_all()
        self.layers[-1].clamp(top_state)
        self._inference(free_energy_thresh)
        return self.layers[0].z.copy()   # z^0 has settled — this is the image



class predictiveCodingForward:
    """
    Discriminative (forward) predictive coding network for supervised regression.

    Key design decisions vs the generative PredictiveCodingNetwork
    --------------------------------------------------------------
    • Predictions flow BOTTOM-UP:  μᵢ = Wᵢ @ z_{i-1}
      (h is applied to hidden-layer outputs, never to the raw input so that
      negative inputs are not zeroed by ReLU before any linear expansion.)
    • Free-layer state updates receive TOP-DOWN corrections:
          Δzᵢ = lr * (−eᵢ  +  W_{i+1}ᵀ e_{i+1} · h′(zᵢ)) · mask
    • Weight update:  Wᵢ += lr * eᵢ ⊗ z_{i-1}^activated
      so Wᵢ directly encodes the i-1 → i mapping.
    • forward(x) is therefore a clean bottom-up pass — no Wᵀ inversion needed,
      unlike the old generative approach which learned y→x then tried to invert.

    Previous implementation learned the GENERATIVE direction (y at top → x at
    bottom) and used Wᵀ in forward() to approximately invert it.  For nonlinear
    targets that inversion is poor, producing nearly linear predictions.
    """

    def __init__(self, layer_dims: list, h, h_prime, seed: int = 42,
                 state_lr: float = 0.1, weight_lr: float = 0.05):
        self.layer_dims = layer_dims
        self.rng        = np.random.default_rng(seed)
        self.L          = len(layer_dims)
        self.h          = h
        self.h_prime    = h_prime
        self.state_lr   = state_lr
        self.weight_lr  = weight_lr

        # W[i] shape (layer_dims[i], layer_dims[i-1]): encodes layer i-1 → i.
        # W[0] = None (input layer has no bottom prediction).
        self.W: list = [None]
        for i in range(1, self.L):
            fan_in = layer_dims[i - 1]
            std    = np.sqrt(2.0 / fan_in)   # He initialisation for ReLU
            self.W.append(self.rng.normal(0, std, size=(layer_dims[i], fan_in)))

        # Per-layer state buffers
        self.z    = [np.zeros(d) for d in layer_dims]
        self.mu   = [np.zeros(d) for d in layer_dims]
        self.e    = [np.zeros(d) for d in layer_dims]
        self.mask = [np.ones(d)  for d in layer_dims]

    # ── helpers ──────────────────────────────────────────────────────────────

    def _clamp(self, i: int, values) -> None:
        self.mask[i] = np.zeros(self.layer_dims[i])
        self.z[i]    = np.asarray(values).ravel().copy()

    def _unclamp(self, i: int) -> None:
        self.mask[i] = np.ones(self.layer_dims[i])

    def _unclamp_all(self) -> None:
        for i in range(self.L):
            self._unclamp(i)

    def _reset_free_layers(self) -> None:
        """Re-initialise each free layer to break symmetry before inference."""
        for i in range(self.L):
            if np.any(self.mask[i] == 1):
                self.z[i] = self.rng.normal(0, 0.01, size=self.layer_dims[i])

    def _activated(self, i: int) -> np.ndarray:
        """Return h(zᵢ) for hidden layers; return zᵢ raw for the input layer (i=0)."""
        return self.z[i] if i == 0 else self.h(self.z[i])

    # ── inference ────────────────────────────────────────────────────────────

    def _compute_predictions(self) -> None:
        self.mu[0] = np.zeros(self.layer_dims[0])          # input layer: no prediction
        for i in range(1, self.L):
            self.mu[i] = self.W[i] @ self._activated(i - 1)

    def _compute_errors(self) -> None:
        for i in range(self.L):
            self.e[i] = self.z[i] - self.mu[i]

    def _state_update(self) -> None:
        for i in range(self.L):
            if np.all(self.mask[i] == 0):               # skip clamped layers
                continue
            # Top-down correction from the layer above (if not top layer)
            if i < self.L - 1:
                correction = self.W[i + 1].T @ self.e[i + 1] * self.h_prime(self.z[i])
            else:
                correction = np.zeros(self.layer_dims[i])
            self.z[i] += self.state_lr * (-self.e[i] + correction) * self.mask[i]

    def _inference(self, free_energy_thresh: float = 0.001, max_iter: int = 500) -> float:
        self._reset_free_layers()
        F_prev = float("inf")
        for _ in range(max_iter):
            self._compute_predictions()
            self._compute_errors()
            self._state_update()
            F = 0.5 * sum(float(np.dot(e, e)) for e in self.e)
            if abs(F_prev - F) < free_energy_thresh:
                break
            F_prev = F
        return F

    # ── public API ────────────────────────────────────────────────────────────

    def supervised(self, x, y, free_energy_thresh: float = 0.0001) -> float:
        """
        One supervised step: clamp input at bottom, label at top.
        Inference settles the hidden layers; weights updated to reduce prediction errors.
        Returns the converged free energy for this sample.
        """
        self._unclamp_all()
        self._clamp(0,          x)   # input  → bottom layer
        self._clamp(self.L - 1, y)   # label  → top layer

        F = self._inference(free_energy_thresh)

        # Wᵢ += lr * eᵢ ⊗ z_{i-1}^activated
        for i in range(1, self.L):
            self.W[i] += self.weight_lr * np.outer(self.e[i], self._activated(i - 1))

        self._unclamp_all()
        return F

    def forward(self, x) -> np.ndarray:
        """
        Discriminative forward pass: x → y_pred.
        Computes W_L h(… h(W₂ h(W₁ x)) …) directly — no Wᵀ inversion.
        h is not applied to the raw input (layer 0 passes through linearly).
        """
        z = np.asarray(x).ravel()
        for i in range(1, self.L):
            z_in = z if i == 1 else self.h(z)   # no activation on raw input
            z = self.W[i] @ z_in
        return z
