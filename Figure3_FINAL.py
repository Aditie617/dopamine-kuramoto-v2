"""
Figure3_FINAL.py

Multi-seed robustness analysis, canonical main-model configuration.
N = 100 and N = 200, 10 independent seeds, T = 3000, dt = 0.002 s (6 s).
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "axes.titlesize": 14})

T = 3000
DT = 0.002

TARGET_HZ = 35.0
SIGMA_HZ = 1.5

K_MIN = 0.2
K_MAX = 25.0

MEAN_DELAY_STEPS = 2
NOISE_D = 1.5

DA0 = 0.5
ALPHA = 10.0
GAMMA = 0.4

N_SEEDS = 10


def run_one_seed(N, seed):

    rng = np.random.default_rng(seed)

    f_i = rng.normal(TARGET_HZ, SIGMA_HZ, N)
    omega = 2 * np.pi * f_i

    G = nx.watts_strogatz_graph(N, 10, 0.1, seed=seed)
    A = nx.to_numpy_array(G)
    deg = A.sum(axis=1)
    deg[deg == 0] = 1

    delays = rng.poisson(lam=MEAN_DELAY_STEPS, size=(N, N))
    max_delay = int(delays.max()) + 1
    buffer = rng.uniform(0, 2 * np.pi, size=(max_delay, N))
    buffer_index = max_delay - 1
    columns = np.arange(N)
    theta = buffer[-1].copy()

    da = 1.0
    R_hist = np.zeros(T)

    for t in range(T):

        da *= np.exp(-GAMMA * DT)
        exponent = np.clip(-ALPHA * (DA0 - da), -700.0, 700.0)
        K = K_MIN + (K_MAX - K_MIN) / (1 + np.exp(exponent))

        R_hist[t] = np.abs(np.mean(np.exp(1j * theta)))

        delayed_indices = (buffer_index - delays) % max_delay
        phase_delayed = buffer[delayed_indices, columns[None, :]]
        phase_difference = phase_delayed - theta[:, None]
        coupling = (K / deg) * np.sum(A * np.sin(phase_difference), axis=1)

        noise_increment = np.sqrt(2 * NOISE_D * DT) * rng.standard_normal(N)
        theta = (theta + (omega + coupling) * DT + noise_increment) % (2 * np.pi)

        buffer_index = (buffer_index + 1) % max_delay
        buffer[buffer_index, :] = theta

    return R_hist


def main():

    time = np.arange(T) * DT
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    summary = {}

    for ax, N in zip(axes, [100, 200]):

        all_R = np.array([run_one_seed(N, seed) for seed in range(N_SEEDS)])
        mean_R = all_R.mean(axis=0)
        std_R = all_R.std(axis=0)

        ax.plot(time, mean_R, lw=2, label="Mean R")
        ax.fill_between(time, mean_R - std_R, mean_R + std_R, alpha=0.25, label="+/- SD")
        ax.axhline(0.951, color="gray", ls=":", lw=1.2, label="R=0.951 (preliminary all-to-all result, Fig. 1)")

        ax.set_title(f"N = {N} ({N_SEEDS} seeds)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("R(t)")
        ax.set_ylim(0, 1)
        ax.legend()

        summary[N] = (mean_R[-1], std_R[-1])
        print(f"N={N}: final mean R = {mean_R[-1]:.3f} +/- {std_R[-1]:.3f}")

    plt.tight_layout()
    plt.savefig("Figure3_FINAL.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSimulation duration = {T * DT:.3f} s")
    print("Saved Figure3_FINAL.png")
    return summary


if __name__ == "__main__":
    main()
