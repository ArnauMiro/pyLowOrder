#!/usr/bin/env python
#
# Example of POD.
#
# Last revision: 07/02/2025
from __future__ import print_function, division

from pyLOM.GPR import GPR, MF_GPR

import os, numpy as np
import matplotlib.pyplot as plt

# =========================
# Single Fidelity GPR
# =========================

# NOISELESS DATA
low_x = np.linspace(0, 1, 20)
low_y = np.sin(low_x * 2 * np.pi)
x_pred = np.linspace(0, 1, 100)

# GPR
gpr = GPR(input_dim=1)
print(gpr.kernel.available_kernels)  # Dict of available kernels
print(
    gpr.kernel.get_kernel_parameters("RBF")
)  # Dict of available parameters for the selecter kernel
kernel = gpr.kernel.RBF(
    variance=1.0,
    lengthscale=1.0,
    limits={"variance": (0.1, 2.0), "lengthscale": (1, 2)},
)
gpr.fit(low_x, low_y, kernel, noise_var=1e-10)
gpr.display_model()
result = gpr.predict(x_pred)

# PLOT
plt.plot(x_pred, result["mean"], label="Prediction")
plt.scatter(low_x, low_y, color="red", label="Data")
plt.legend()
plt.show()

############################

# NOISY DATA
low_x = np.linspace(0, 1, 40)
true_y = np.sin(low_x * 2 * np.pi)
noise = np.random.normal(scale=0.1, size=low_x.shape)
low_y = true_y + noise
x_pred = np.linspace(0, 1, 100)

# FPR
gpr = GPR(input_dim=1)
kernel = gpr.kernel.RBF(variance=1.0, lengthscale=1.0)
gpr.fit(low_x, low_y, kernel, noise_var=1e-2)
result = gpr.predict(x_pred)

# PLOT
plt.plot(x_pred, np.sin(2 * np.pi * x_pred), "--", label="True function", color="green")
plt.plot(x_pred, result["mean"], label="Prediction", color="blue")
plt.scatter(low_x, low_y, color="red", label="Noisy Data")
plt.legend()
plt.show()

# =========================
# Multi Fidelity GPR
# =========================


# Forrester Function
def forrester_high(x):
    """High-fidelity function"""
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4)


def forrester_low(x):
    """Low-fidelity function"""
    return 0.5 * forrester_high(x) + 10 * (x - 0.5) - 5


x_train_low = np.linspace(0, 1, 11)[:, None]
x_train_high = x_train_low[[0, 4, 6, 10]]
y_train_low = forrester_low(x_train_low)
y_train_high = forrester_high(x_train_high)

x_pred_low = np.linspace(0, 1, 100)[:, None]
x_pred_high = np.linspace(0, 1, 100)[:, None]

# MF_GPR
mf_gpr = MF_GPR(input_dim=1)
print(mf_gpr.kernel.available_kernels)  # Dict of available kernels
print(
    mf_gpr.kernel.get_kernel_parameters("Matern32")
)  # Dict of available parameters for the selecter kernel
kernel_low = mf_gpr.kernel.Matern32(
    variance=1.0,
    lengthscale=1.0,
)
kernel_high = mf_gpr.kernel.Matern32(
    variance=1.0,
    lengthscale=1.0,
)

mf_gpr.fit(
    train_features_list=[x_train_low, x_train_high],
    train_labels_list=[y_train_low, y_train_high],
    kernels=[kernel_low, kernel_high],
    noise_vars=[1e-6, 1e-4],
    num_restarts=7,
    verbose=True,
)
mf_gpr.display_model()
results = mf_gpr.predict([x_pred_low, x_pred_high])
mean_MF_LF = results["fidelity_1"]["mean"]
std_MF_LF = results["fidelity_1"]["std"]
mean_MF_HF = results["fidelity_2"]["mean"]
std_MF_HF = results["fidelity_2"]["std"]

plot_x = np.linspace(0, 1, 501).reshape(-1, 1)
plt.figure(figsize=(6, 5), dpi=600)
plt.plot(plot_x, forrester_high(plot_x), linewidth=2, color="red", label="$True HF$")
plt.plot(
    plot_x,
    forrester_low(plot_x),
    linewidth=2,
    color="blue",
    linestyle="--",
    label="$True LF$",
)
plt.scatter(
    x_train_high,
    y_train_high,
    marker="o",
    facecolors="none",
    color="red",
    label="$Data HF$",
)
plt.scatter(
    x_train_low,
    y_train_low,
    marker="s",
    facecolors="none",
    color="blue",
    label="$Data LF$",
)
plt.plot(x_pred_low[:, 0], mean_MF_LF, linewidth=2, color="green", label="$Pred LF$")
plt.fill_between(
    x_pred_low[:, 0],
    (mean_MF_LF - 1.96 * std_MF_LF).flatten(),
    (mean_MF_LF + 1.96 * std_MF_LF).flatten(),
    facecolor="green",
    alpha=0.3,
    edgecolor="green",
    linewidth=1.5,
    label="LF 95% CI",
    zorder=5,
)
plt.plot(x_pred_high[:, 0], mean_MF_HF, linewidth=2, color="orange", label="$Pred HF$")
plt.fill_between(
    x_pred_high[:, 0],
    (mean_MF_HF - 1.96 * std_MF_HF).flatten(),
    (mean_MF_HF + 1.96 * std_MF_HF).flatten(),
    facecolor="orange",
    alpha=0.3,
    edgecolor="orange",
    linewidth=1.5,
    label="HF 95% CI",
    zorder=5,
)
plt.xlim([0, 1])
plt.ylim([-10, 20])
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend(loc=2)
plt.tight_layout()
plt.show()
