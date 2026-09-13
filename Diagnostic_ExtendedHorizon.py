"""
Diagnostic_ExtendedHorizon.py

PURPOSE: diagnostic only. Canonical main-model parameters are UNCHANGED
from the 6 s run (35 Hz, Kmin=0.2, Kmax=25, alpha=10, gamma=0.4, D=1.5,
DA0=0.5, WS k=10 p=0.1). The only thing varied is simulation duration,
to test whether the low R observed at 6 s is a transient / time-horizon
artifact or the model's actual steady-state behavior.

Horizon: 120 s (T = 60000 steps at dt = 0.002 s), 20x the original 6 s
window. Same seed (42) as the original single-seed figures, run for
both N = 100 and N = 200.

No parameter other than T (and derived buffers) is touched.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import time as _time
import json

DT = 0.002
T = 60000              # 120 s
TARGET_HZ = 35.0
SIGMA_HZ = 2.0
K_MIN = 0.2
K_MAX = 25.0
MEAN_DELAY_STEPS = 2
NOISE_D = 1.5
DA0 = 0.5
ALPHA = 10.0
GAMMA = 0.4
BETA_LO, BETA_HI = 13, 30
SEED = 42
R_THRESHOLD = 0.6
KC_HEURISTIC = 18.04  # Eq. 7 heuristic: 2/(pi*g(mu_omega)) + 2D, sigma_f=1.5 Hz, D=1.5


def simulate(N, seed):
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
    K_hist = np.zeros(T)
    R_hist = np.zeros(T)
    lfp = np.zeros(T)

    for t in range(T):
        da *= np.exp(-GAMMA * DT)
        exponent = np.clip(-ALPHA * (DA0 - da), -700.0, 700.0)
        K = K_MIN + (K_MAX - K_MIN) / (1 + np.exp(exponent))
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

    return K_hist, R_hist, lfp


def dominant_freq(sig, dt):
    n = len(sig)
    s = sig - np.mean(sig)
    yf = np.abs(fft(s))[:n // 2]
    xf = fftfreq(n, dt)[:n // 2]
    yf[0] = 0
    return xf[np.argmax(yf)], xf, yf


def crossing_analysis(time, R_hist, threshold=R_THRESHOLD):
    above = R_hist >= threshold
    if not above.any():
        return {"first_crossing_s": None, "sustained_from_s": None,
                "fraction_time_above": 0.0}

    first_idx = np.argmax(above)
    first_crossing_s = float(time[first_idx])

    # "sustained" = once crossed, stays above threshold for at least the
    # remaining 10% of the run without dropping back down for more than
    # a short blip (<1% of total duration)
    tail_len = max(1, int(0.01 * len(R_hist)))
    sustained_from = None
    for i in range(len(R_hist)):
        if above[i] and above[i:].mean() > 0.95:
            sustained_from = float(time[i])
            break

    return {
        "first_crossing_s": first_crossing_s,
        "sustained_from_s": sustained_from,
        "fraction_time_above": float(above.mean()),
    }


def run_and_report(N):
    t0 = _time.time()
    K_hist, R_hist, lfp = simulate(N, SEED)
    elapsed = _time.time() - t0

    time = np.arange(T) * DT
    half = T // 2
    early = lfp[:half]
    late = lfp[half:]

    dom_early, _, _ = dominant_freq(early, DT)
    dom_late, xf, yf = dominant_freq(late, DT)

    cross = crossing_analysis(time, R_hist)

    result = {
        "N": N,
        "duration_s": T * DT,
        "wall_clock_s": elapsed,
        "K_min": float(K_hist.min()),
        "K_max": float(K_hist.max()),
        "R_final": float(R_hist[-1]),
        "R_mean_last_10pct": float(R_hist[int(0.9 * T):].mean()),
        "R_max_overall": float(R_hist.max()),
        "dominant_freq_early_Hz": float(dom_early),
        "dominant_freq_late_Hz": float(dom_late),
        "late_peak_in_beta_band": bool(BETA_LO <= dom_late <= BETA_HI),
        "crossing_analysis": cross,
    }

    # ---- plots ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].plot(time, R_hist, lw=0.7)
    axes[0, 0].axhline(0.951, color="gray", ls=":", lw=1.2, label="R=0.951 (Fig. 1 result)")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("R(t)")
    axes[0, 0].set_title(f"(A) R(t) over {T*DT:.0f} s, N={N}")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].legend()

    axes[0, 1].plot(time, K_hist, lw=1.2, color="tab:orange")
    axes[0, 1].axhline(K_MAX, ls=":", color="gray", label=f"Kmax={K_MAX}")
    axes[0, 1].axhline(KC_HEURISTIC, ls="--", color="black", lw=1.2, label=f"Eq. 7 heuristic $K_c$={KC_HEURISTIC:.1f}")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("K(t)")
    axes[0, 1].set_title(f"(B) K(t) over {T*DT:.0f} s, N={N}")
    axes[0, 1].legend()

    for seg, label in [(early, "Early half"), (late, "Late half")]:
        seg_c = seg - np.mean(seg)
        yfp = np.abs(fft(seg_c))
        xfp = fftfreq(len(seg_c), DT)
        n2 = len(seg_c) // 2
        axes[1, 0].plot(xfp[:n2], yfp[:n2] / len(seg_c), label=label, lw=1)
    axes[1, 0].axvspan(BETA_LO, BETA_HI, alpha=0.15, label="Clinical beta band")
    axes[1, 0].set_xlim(0, 60)
    axes[1, 0].set_xlabel("Frequency (Hz)")
    axes[1, 0].set_ylabel("Amplitude")
    axes[1, 0].set_title(f"(C) LFP spectrum, N={N}")
    axes[1, 0].legend()

    fs = 1 / DT
    f, t_spec, Sxx = signal.spectrogram(lfp, fs=fs, nperseg=2048, noverlap=1024,
                                         detrend="constant", scaling="density")
    Sxx_dB = 10 * np.log10(Sxx + 1e-12)
    pcm = axes[1, 1].pcolormesh(t_spec, f, Sxx_dB, shading="gouraud")
    axes[1, 1].axhspan(BETA_LO, BETA_HI, alpha=0.15)
    axes[1, 1].axhline(BETA_LO, ls="--", color="k")
    axes[1, 1].axhline(BETA_HI, ls="--", color="k")
    axes[1, 1].set_ylim(0, 60)
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Frequency (Hz)")
    axes[1, 1].set_title(f"(D) Spectrogram, N={N}")
    fig.colorbar(pcm, ax=axes[1, 1], label="PSD (dB)")

    plt.tight_layout()
    fname = f"Diagnostic_N{N}_120s.png"
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()
    result["figure"] = fname

    return result


if __name__ == "__main__":
    all_results = {}
    for N in (100, 200):
        print(f"\nRunning N={N}, T={T} steps ({T*DT:.0f} s)...")
        res = run_and_report(N)
        all_results[N] = res
        print(json.dumps(res, indent=2))

    with open("Diagnostic_ExtendedHorizon_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nSaved Diagnostic_ExtendedHorizon_results.json")
