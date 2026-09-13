# Dopamine-Modulated Adaptive Kuramoto Network for Parkinsonian Neural Dynamics

This repository contains the simulation scripts and diagnostic analyses associated with the manuscript:

**“A Dopamine-Modulated Adaptive Kuramoto Network for Parkinsonian Neural Dynamics: Effects of Delay and Topology on Beta-Band Synchronization”**

The project investigates whether a dopamine-dependent adaptive coupling mechanism can produce a robust synchronization transition when additional network features—small-world topology, conduction delays, and stochastic forcing—are incorporated into a Kuramoto-type neural oscillator model.

---

## Repository Overview

The repository contains the Python scripts used to generate the main figures and perform extended-horizon diagnostic analyses reported in the manuscript.

The simulations are organized into:

1. **Preliminary all-to-all simulations**
2. **Main dopamine-adaptive Kuramoto model**
3. **Multi-seed robustness analysis**
4. **LFP and spectrogram analysis**
5. **Extended-horizon diagnostic simulations**

The code is provided to support transparency and reproducibility of the reported computational results.

---

## File Descriptions

### `Figure1_CORRECTED.py`

This script implements the **preliminary all-to-all Kuramoto simulation** used to demonstrate synchronization under a simplified network configuration.

The simulation uses global/all-to-all coupling without the full combination of small-world topology, conduction delays, and stochastic forcing used in the main model.

This simulation serves as a preliminary reference case and should not be interpreted as the full Parkinsonian network model.

**Purpose:**

* Demonstrate the basic dopamine-dependent adaptive coupling mechanism.
* Provide a simplified synchronization reference.
* Generate the preliminary results associated with Figure 1.

---

### `Figure1b_AllToAll35Hz.py`

This script provides an additional **all-to-all reference simulation using a 35 Hz mean intrinsic frequency**.

It is included to distinguish the behavior of the simplified all-to-all configuration at the frequency scale used in the main model.

**Purpose:**

* Examine the simplified model at 35 Hz.
* Provide a direct frequency-reference comparison with the main configuration.
* Support interpretation of the difference between the preliminary and full network models.

---

### `Figure2_FINAL.py`

This is the primary simulation script for the **main dopamine-modulated adaptive Kuramoto model**.

The model incorporates:

* Dopamine-dependent adaptive coupling
* Watts–Strogatz small-world network topology
* Conduction delays
* Stochastic forcing
* Heterogeneous intrinsic oscillator frequencies
* Euler–Maruyama numerical integration
* Kuramoto order parameter analysis

The script generates the principal results used for **Figure 2**.

The main simulation is intended to test whether dopamine-driven coupling alone remains sufficient to produce strong synchronization after structural and stochastic constraints are introduced.

---

### `Figure3_FINAL.py`

This script performs the **multi-seed robustness analysis** of the main model.

Multiple independent random realizations are simulated for the specified network sizes, allowing the synchronization behavior to be evaluated across different stochastic/network realizations rather than relying on a single simulation.

**Purpose:**

* Quantify variability across random seeds.
* Assess robustness of the observed synchronization behavior.
* Report mean and standard deviation of the final synchronization level.
* Generate the results associated with Figure 3.

---

### `Figure4_FINAL.py`

This script generates the representative **local-field-potential (LFP) and spectrogram analysis** associated with Figure 4.

The analysis examines the temporal and frequency-domain behavior of the simulated network, including the dominant spectral component.

**Purpose:**

* Visualize representative LFP activity.
* Compute and display the frequency spectrum/spectrogram.
* Examine whether the simulated population activity develops a beta-band-dominant oscillation.
* Provide the frequency-domain diagnostic associated with Figure 4.

---

### `Diagnostic_ExtendedHorizon.py`

This script performs the **extended-horizon diagnostic analysis**.

The purpose of this analysis is to determine whether the behavior observed during the shorter simulations is simply a transient phenomenon or persists over a substantially longer simulation interval.

The extended simulations use the main-model configuration and evaluate:

* Long-term synchronization behavior
* Dopamine/coupling evolution
* Population coherence
* Dominant frequency
* Persistence or absence of a beta-band transition

These simulations provide an additional robustness check for the interpretation of the main results.

---

## Model Structure

The main model extends the classical Kuramoto framework by introducing a dopamine-dependent adaptive coupling strength:

**Dopamine → adaptive coupling → network synchronization**

The model additionally incorporates:

**Small-world topology + conduction delays + stochastic forcing**

This combination allows the study to move beyond an idealized all-to-all deterministic network and examine synchronization under more constrained network conditions.

---

## Main Computational Components

### Network topology

The main model uses a **Watts–Strogatz small-world network** to represent non-all-to-all interactions.

### Dopamine-dependent coupling

The effective coupling strength varies continuously with the dopamine variable rather than switching between manually defined fixed coupling regimes.

### Conduction delays

Interaction terms incorporate finite propagation delays between oscillators.

### Stochastic forcing

Noise is incorporated using an Euler–Maruyama discretization with the stochastic increment scaled according to the diffusion coefficient and simulation timestep.

### Synchronization measure

Synchronization is quantified using the Kuramoto order parameter:

**R(t)**

where values closer to 1 indicate stronger phase coherence across the oscillator population.

### Spectral analysis

Frequency-domain analyses are used to evaluate the dominant oscillatory component and characterize changes in spectral organization.

---

## Important Interpretation of the Results

A key result of the computational study is that the **main delayed, noisy small-world configuration does not produce a robust beta-band synchronization transition under the tested parameterization**.

This is an important distinction from the simplified preliminary all-to-all simulation.

The repository therefore preserves the negative/limiting result rather than presenting beta-band synchronization as an outcome that was obtained by the full model.

The simulations instead support the conclusion that **dopamine-dependent global adaptive coupling, by itself, may be insufficient to produce robust beta-band synchronization when network topology, conduction delays, and stochastic forcing are simultaneously included.**

This result should be interpreted within the tested parameter range and model assumptions.

---

## Reproducibility

The scripts contain the simulation parameters, numerical settings, and random-seed information required to reproduce the corresponding computational analyses.

Important reproducibility variables include:

* Network size (`N`)
* Simulation duration (`T`)
* Integration timestep (`dt`)
* Mean intrinsic frequency
* Frequency heterogeneity
* Minimum and maximum coupling
* Dopamine parameters
* Noise intensity
* Network topology parameters
* Conduction delay
* Random seed

The same parameterization should be used when reproducing a specific manuscript figure.

---

## Relationship Between Scripts and Manuscript Figures

| Script                          | Analysis                          | Manuscript Output                      |
| ------------------------------- | --------------------------------- | -------------------------------------- |
| `Figure1_CORRECTED.py`          | Preliminary all-to-all model      | Figure 1                               |
| `Figure1b_AllToAll35Hz.py`      | All-to-all 35 Hz reference        | Figure 1b                              |
| `Figure2_FINAL.py`              | Main adaptive Kuramoto model      | Figure 2                               |
| `Figure3_FINAL.py`              | Multi-seed robustness analysis    | Figure 3                               |
| `Figure4_FINAL.py`              | LFP and spectrogram analysis      | Figure 4                               |
| `Diagnostic_ExtendedHorizon.py` | Long-duration diagnostic analysis | Supplementary/extended-horizon results |

---

## Software Requirements

The simulations require Python 3 and the following packages:

```text
numpy
networkx
pandas
matplotlib
scipy
```

Install the required packages using:

```bash
pip install numpy networkx pandas matplotlib scipy
```

---

## Running the Simulations

Each script can be executed independently from the repository directory.

For example:

```bash
python Figure2_FINAL.py
```

The corresponding scripts can be run similarly:

```bash
python Figure1_CORRECTED.py
python Figure1b_AllToAll35Hz.py
python Figure3_FINAL.py
python Figure4_FINAL.py
python Diagnostic_ExtendedHorizon.py
```

Exact outputs depend on the parameters and random seeds specified within each script.

---

## Scientific Scope

This repository contains a **computational and phenomenological network model**.

The model is not intended to:

* Fit patient-specific clinical data
* Reproduce individual patient trajectories
* Model limb biomechanics
* Model muscle activity
* Simulate tremor directly
* Establish a clinical diagnostic or therapeutic prediction

The focus is specifically on the relationship between dopamine-dependent adaptive coupling and population-level neural synchronization in a delayed, noisy network.

---

## Reproducibility and Transparency Note

The repository intentionally retains both the simplified reference simulations and the more constrained main-model simulations.

This separation is important because the preliminary all-to-all configuration and the full delayed/noisy small-world configuration exhibit different synchronization behavior.

All reported conclusions should therefore be interpreted in relation to the specific parameterization and network assumptions used in each simulation.

---

## Citation

If you use the code or model in academic work, please cite the associated manuscript:

> *A Dopamine-Modulated Adaptive Kuramoto Network for Parkinsonian Neural Dynamics: Effects of Delay and Topology on Beta-Band Synchronization.*

---

## License

Please refer to the repository license for terms governing reuse and redistribution of the code.
