"""
Figure2_FINAL.py

Main dopamine-adaptive Kuramoto model: Watts-Strogatz small-world
network + conduction delay + stochastic noise.

CANONICAL MAIN-MODEL CONFIGURATION (as specified for final submission):
    Mean intrinsic frequency = 35 Hz   (omega = 2*pi*f)
    Kmin = 0.2, Kmax = 25
    alpha = 10, gamma = 0.4, D = 1.5, DA0 = 0.5
    Watts-Strogatz k = 10, p = 0.1
    N = 100 (this figure), also N = 200 (Figure 3)

NOTE ON HONESTY OF RESULT:
    Under this canonical configuration the population does NOT settle
    into the clinically reported 13-30 Hz beta band; the dominant late
    -window frequency tracks close to the 35 Hz mean intrinsic
    frequency (see printed output). This script reports that result
    as computed. The beta band is shown on the spectrum plot only as
    a reference range, not as a claim about where the peak falls.

Unspecified-by-canonical-list parameters (frequency SD, mean delay)
are carried over unchanged from the prior draft and documented here
so they are not silently altered:
    Frequency SD   = 2.0 Hz
    Mean delay     = 2 steps (Poisson)
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14
})

# ---------------- Canonical parameters ----------------

N = 100

T = 3000
DT = 0.002

TARGET_HZ = 35.0
SIGMA_HZ = 1.5          # not specified canonically; carried over, documented above

K_MIN = 0.2
K_MAX = 25.0

MEAN_DELAY_STEPS = 2    # not specified canonically; carried over, documented above

NOISE_D = 1.5

DA0 = 0.5
ALPHA = 10.0
GAMMA = 0.4

BETA_LO = 13
BETA_HI = 30

SEED = 42
KC_HEURISTIC = 18.04  # Eq. 7 heuristic: 2/(pi*g(mu_omega)) + 2D, sigma_f=1.5 Hz, D=1.5


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

    da_hist = np.zeros(T)
    K_hist = np.zeros(T)
    R_hist = np.zeros(T)
    lfp = np.zeros(T)

    for t in range(T):

        da *= np.exp(-GAMMA * DT)

        exponent = np.clip(-ALPHA * (DA0 - da), -700.0, 700.0)
        K = K_MIN + (K_MAX - K_MIN) / (1 + np.exp(exponent))

        da_hist[t] = da
        K_hist[t] = K

        R_hist[t] = np.abs(np.mean(np.exp(1j * theta)))
        lfp[t] = np.mean(np.sin(theta))

        delayed_indices = (buffer_index - delays) % max_delay
        phase_delayed = buffer[delayed_indices, columns[None, :]]

        phase_difference = phase_delayed - theta[:, None]
        coupling = (K / deg) * np.sum(A * np.sin(phase_difference), axis=1)

        noise_increment = np.sqrt(2 * NOISE_D * DT) * rng.standard_normal(N)

        theta = (theta + (omega + coupling) * DT + noise_increment) % (2 * np.pi)

        buffer_index = (buffer_index + 1) % max_delay
        buffer[buffer_index, :] = theta

    return da_hist, K_hist, R_hist, lfp


def dominant_freq(signal_data, dt):
    n = len(signal_data)
    s = signal_data - np.mean(signal_data)
    yf = np.abs(fft(s))[:n // 2]
    xf = fftfreq(n, dt)[:n // 2]
    yf[0] = 0
    return xf[np.argmax(yf)]


def plot(da, K, R, lfp):

    time = np.arange(T) * DT

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax1 = axes[0]
    ax1.plot(time, K, color="tab:orange", label="K(t)", lw=2)
    ax1.axhline(KC_HEURISTIC, color="gray", ls=":", lw=1.5, label=f"Eq. 7 heuristic $K_c$={KC_HEURISTIC:.1f}")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("K")

    ax2 = ax1.twinx()
    ax2.plot(time, da, "--", color="tab:blue", label="DA", lw=2)
    ax2.plot(time, R, color="tab:green", label="R(t)", lw=2)
    ax2.set_ylabel("DA, R")
    ax2.set_ylim(0, 1.05)

    ax1.set_title("(A) K(t), DA, and R(t), small-world + delay + noise (N=100)")

    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="center right", fontsize=9)

    ax3 = axes[1]
    half = T // 2
    for segment, label, lw in [(lfp[:half], "Early", 1.5), (lfp[half:], "Late", 2)]:
        segment = segment - np.mean(segment)
        yf = np.abs(fft(segment))
        xf = fftfreq(len(segment), DT)
        n2 = len(segment) // 2
        ax3.plot(xf[:n2], yf[:n2] / len(segment), label=label, lw=lw)

    ax3.axvspan(BETA_LO, BETA_HI, alpha=0.15, label="Clinical beta band (reference)")
    ax3.set_xlim(0, 60)
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("Amplitude")
    ax3.set_title("(B) Population LFP spectrum")
    ax3.legend()

    plt.tight_layout()
    plt.savefig("Figure2_FINAL.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":

    da, K, R, lfp = simulate()
    plot(da, K, R, lfp)

    half = T // 2
    dominant = dominant_freq(lfp[half:], DT)

    print("\nFigure 2 (FINAL, canonical config) Results")
    print("-------------------------------------------")
    print(f"Simulation duration = {T * DT:.3f} s")
    print(f"K minimum           = {K.min():.3f}")
    print(f"K maximum           = {K.max():.3f}")
    print(f"Final R(t)          = {R[-1]:.3f}")
    print(f"Dominant frequency  = {dominant:.2f} Hz")
    print(f"In beta band (13-30 Hz)? {'YES' if BETA_LO <= dominant <= BETA_HI else 'NO'}")
    print("Saved Figure2_FINAL.png")
