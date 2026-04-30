"""
Activation functions and their derivatives for the predictive coding network.

Each function h must have a paired h_prime (its derivative) since both
are needed in the network dynamics (Eq. 53):

    phi_dot_i = -e_p_i + h'(phi_i) * theta_i^T * e_u_{i-1}

Usage
-----
    from predictive_coding.activations import tanh, tanh_prime
    net = PredictiveCodingNetwork(layer_dims=[64, 32, 10], h=tanh, h_prime=tanh_prime)
"""

import numpy as np


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def tanh_prime(x: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(x) ** 2


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def sigmoid_prime(x: np.ndarray) -> np.ndarray:
    s = sigmoid(x)
    return s * (1.0 - s)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_prime(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


def linear(x: np.ndarray) -> np.ndarray:
    return x.copy()


def linear_prime(x: np.ndarray) -> np.ndarray:
    return np.ones_like(x)