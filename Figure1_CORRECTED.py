import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "axes.titlesize": 14})

N = 200
T = 1000
DT = 0.01

TARGET_HZ = 20.0
SIGMA_HZ = 2.0

K_MIN = 0.5
K_MAX = 40.0

DA0 = 0.5
ALPHA = 8.0
GAMMA = 0.4

BETA_LO = 13
BETA_HI = 30

SEED = 42
KC_HEURISTIC = 20.05  # Eq. 7 heuristic, D=0 (no noise term in this configuration)


def simulate():
    rng = np.random.default_rng(SEED)
    f_i = rng.normal(TARGET_HZ, SIGMA_HZ, N)
    omega = 2 * np.pi * f_i
    theta = rng.uniform(0, 2 * np.pi, N)
    da = 1.0
    da_hist = np.zeros(T)
    K_hist = np.zeros(T)
    R_hist = np.zeros(T)
    lfp = np.zeros(T)

    for t in range(T):
        da *= np.exp(-GAMMA * DT)
        K = K_MIN + ((K_MAX - K_MIN) / (1 + np.exp(-ALPHA * (DA0 - da))))
        K = np.clip(K, K_MIN, K_MAX)
        da_hist[t] = da
        K_hist[t] = K
        R_hist[t] = np.abs(np.mean(np.exp(1j * theta)))
        lfp[t] = np.mean(np.sin(theta))
        phase_difference = theta[None, :] - theta[:, None]
        coupling_sum = np.sum(np.sin(phase_difference), axis=1)
        dtheta = omega + (K / N) * coupling_sum
        theta = (theta + dtheta * DT) % (2 * np.pi)

    return da_hist, K_hist, R_hist, lfp


def dominant_freq(signal_data, dt):
    n = len(signal_data)
    signal_centered = signal_data - np.mean(signal_data)
    yf = np.abs(fft(signal_centered))[:n // 2]
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

    ax1.set_title("(A) K(t), DA, and R(t), all-to-all (N=200)")
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
    ax3.axvspan(BETA_LO, BETA_HI, alpha=0.15, label="Clinical beta band")
    ax3.set_xlim(0, 40)
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("Amplitude")
    ax3.set_title("(B) Population LFP spectrum")
    ax3.legend()

    plt.tight_layout()
    plt.savefig("Figure1_FINAL.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    da, K, R, lfp = simulate()
    plot(da, K, R, lfp)
    half = T // 2
    dominant = dominant_freq(lfp[half:], DT)
    print("\nFigure 1 Results (preliminary all-to-all, unchanged config)")
    print("------------------------------------------------------------")
    print(f"Simulation duration = {T * DT:.3f} s")
    print(f"K minimum          = {K.min():.3f}")
    print(f"K maximum           = {K.max():.3f}")
    print(f"Final R(t)          = {R[-1]:.3f}")
    print(f"Dominant frequency  = {dominant:.2f} Hz")
    print(f"In beta band (13-30 Hz)? {'YES' if BETA_LO <= dominant <= BETA_HI else 'NO'}")
    print("Saved Figure1_FINAL.png")
