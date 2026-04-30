from PC import PredictiveCodingNetwork
from activation import tanh, tanh_prime
import numpy as np
import matplotlib.pyplot as plt

# Dataset: x is the observation (bottom layer), y is the latent cause (top layer).
x = np.linspace(0, 1, num=500)
y = x**3 + 2*x**2
y /= y.max()

data_set = np.vstack((x, y)).T

# Generative PC: learns P(x | y) — "given cause y, generate observation x".
# Prediction at test time uses INFERENCE (not W.T inversion):
#   clamp x at bottom, let top settle freely → inferred y.
#
# Activation MUST be tanh (not relu).
# Reason: predict() initialises z_top ≈ 0; relu'(0) = 0 permanently blocks
# bottom-up corrections from reaching the top — inference is frozen.
# tanh'(0) = 1 so corrections flow from the very first iteration.
layers = [1, 64, 64, 1]
pc = PredictiveCodingNetwork(layers, tanh, tanh_prime,
                             state_lr=0.05, weight_lr=0.01)

N_EPOCHS = 50

if __name__ == "__main__":

    for epoch in range(N_EPOCHS):
        F_ = 0.0
        for i in range(data_set.shape[0]):
            F_ += pc.supervised(data_set[i][:1], data_set[i][1:])

        if (epoch + 1) % 10 == 0 or epoch == 0:
            # predict() runs inference only (no weight update) — the few-shot path.
            preds = np.array([pc.predict(np.array([xi]))[0] for xi in x])
            mse   = float(np.mean((preds - y) ** 2))
            print(f"Epoch {epoch+1:3d}  Free Energy: {F_:.2f}  Pred MSE: {mse:.5f}")

    preds = np.array([pc.predict(np.array([xi]))[0] for xi in x])
    plt.plot(x, preds, label="PC inference prediction")
    plt.plot(x, y,     label="target")
    plt.legend()
    plt.title("Generative PC — inference-based prediction")
    plt.show()
