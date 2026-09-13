"""
Figure4_FINAL.py
Representative single-seed LFP and spectrogram, canonical main-model config.
N = 100, seed = 42, T = 3000, dt = 0.002 s (6 s).
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import signal

plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "axes.titlesize": 14})

N = 100
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

BETA_LO = 13
BETA_HI = 30

SEED = 42


def simulate():

    rng = np.random.default_rng(SEED)

    f_i = rng.normal(TARGET_HZ, SIGMA_HZ, N)
    omega = 2 * np.pi * f_i

    G = nx.watts_strogatz_graph(N, 10, 0.1, seed=SEED)
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
    lfp = np.zeros(T)

    for t in range(T):

        da *= np.exp(-GAMMA * DT)
        exponent = np.clip(-ALPHA * (DA0 - da), -700.0, 700.0)
        K = K_MIN + (K_MAX - K_MIN) / (1 + np.exp(exponent))

        lfp[t] = np.mean(np.sin(theta))

        delayed_indices = (buffer_index - delays) % max_delay
        phase_delayed = buffer[delayed_indices, columns[None, :]]
        phase_difference = phase_delayed - theta[:, None]
        coupling = (K / deg) * np.sum(A * np.sin(phase_difference), axis=1)

        noise_increment = np.sqrt(2 * NOISE_D * DT) * rng.standard_normal(N)
        theta = (theta + (omega + coupling) * DT + noise_increment) % (2 * np.pi)

        buffer_index = (buffer_index + 1) % max_delay
        buffer[buffer_index, :] = theta

    return lfp


def plot(lfp):

    time = np.arange(T) * DT
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(time, lfp)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("LFP (a.u.)")
    axes[0].set_title("(A) Single-seed LFP (N=100)")

    fs = 1 / DT
    f, t_spec, Sxx = signal.spectrogram(
        lfp, fs=fs, nperseg=512, noverlap=256, detrend="constant", scaling="density"
    )
    Sxx_dB = 10 * np.log10(Sxx + 1e-12)

    pcm = axes[1].pcolormesh(t_spec, f, Sxx_dB, shading="gouraud")
    axes[1].axhspan(BETA_LO, BETA_HI, alpha=0.15)
    axes[1].axhline(BETA_LO, ls="--")
    axes[1].axhline(BETA_HI, ls="--")
    axes[1].set_ylim(0, 50)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_title("(B) Spectrogram (shaded band = clinical beta, reference only)")

    fig.colorbar(pcm, ax=axes[1], label="PSD (dB)")

    plt.tight_layout()
    plt.savefig("Figure4_FINAL.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":

    lfp = simulate()
    plot(lfp)

    print("\nFigure 4 (FINAL, canonical config) Results")
    print("-------------------------------------------")
    print(f"Simulation duration = {T * DT:.3f} s")
    print("Saved Figure4_FINAL.png")
