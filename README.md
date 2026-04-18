# CPT Atomic Clock Simulation Model

A physics-first simulation of a CW-CPT (Coherent Population Trapping) ⁸⁷Rb
atomic clock in a compact vapour-cell. The model covers optical physics,
decoherence, systematic frequency shifts, noise budgets, and Allan deviation
for both **buffer-gas** and **no-buffer-gas** cell configurations.

> **Physics accuracy:** All constants and formulas are benchmarked against
> primary literature (Steck2021, Vanier2005, Boudot2011, Rotondaro1997,
> Bouchiat1966). See [`references.md`](references.md) for the full citation map.

---

## What's new (April 2026)

A comprehensive physics verification and correction pass was applied.
Key changes relative to earlier versions:

### Critical bug fix — Saturation intensity

`I_sat_D1` was computed via `h·ν·Γ/(3λ²)` = 0.476 mW/cm², which is
**9.4× below** the correct Steck2021 Table 7 value of **1.669 mW/cm²**
(lin‖lin polarisation, D1 line). This caused the Rabi frequency to be
overestimated by ~3× and power-broadening to be overestimated by ~9×.
Fixed to use `I_sat_D1_CPT = 16.70 W/m²` from `constants.py`.

### New — No-buffer-gas cell model

`total_ground_decoherence()` now auto-dispatches to the correct physics
regime based on buffer-gas pressure:

| Pressure | Regime | Dominant channel |
|---|---|---|
| `P_N2 > 0` or `P_Ar > 0` | **Diffusion** | Brownian diffusion to walls |
| `P_N2 = P_Ar = 0` | **Ballistic** | Knudsen wall collisions |

Anti-relaxation wall coatings (paraffin, PDMS/OTS) reduce the wall
depolarisation rate by 10³–10⁴×, enabling narrow CPT linewidths without
buffer gas.

### New — N₂ quenching + optical pressure broadening

`optical_homogeneous_rate(P_N2, P_Ar, T_K)` computes the total
excited-state decay rate Γ_hom = Γ_nat + Γ_press + Γ_quench, which feeds
directly into linewidth, contrast, and optical depth calculations.

### Other fixes

- Diffusion scaling T^1.5 → **T^1.75** (hard-sphere Chapman-Enskog)
- Corrected `Kozlova2011` (Cs paper) → **Boudot2011** (correct Rb reference)
- Added **flicker noise floor** (`σ_y_flicker`) to noise budget
- **Cell-type-aware** contrast ceiling: CSAC 10%, lab 20%, macro 25%
- Test suite expanded 87 → **122 tests**, all passing

---

## Features

- **CPT signal model** — Rabi frequency, power-broadened linewidth, resonance
  contrast, Lorentzian lineshape, discriminator slope
- **Two cell regimes** — buffer-gas diffusion model and no-buffer-gas
  ballistic model with optional anti-relaxation coatings
- **Excited-state decay** — natural linewidth + pressure broadening (N₂/Ar)
  + N₂ collisional quenching, all auto-computed from gas pressures
- **VCSEL modulation** — Bessel sideband power spectrum, optimal modulation
  index
- **Systematic frequency shifts** — buffer-gas (quadratic T), light shift,
  Zeeman (Breit–Rabi), spin-exchange, BBR, barometric
- **Noise budget (9 sources)** — shot noise, RIN, FM, PM-AM, flicker (1/f),
  temperature, magnetic, electronics, long-term drift
- **Allan deviation** — τ-grid, slope characterisation, noise source ranking
- **Environment sweeps** — temperature and pressure-ratio optimisation
- **Tkinter GUI** — interactive parameter panel and live plots

---

## Physics Scope

### Supported operating modes

**Buffer-gas cell** (`P_N2 > 0` or `P_Ar > 0`):
- Atoms diffuse to walls; wall loss suppressed by buffer gas
- Decoherence: γ₂ = γ_diff + γ_se + γ_bg
- Inversion temperature ΔT optimisation for zero-crossing shift

**No-buffer-gas cell** (`P_N2 = P_Ar = 0`):
- Atoms travel ballistically at thermal speed v̄ = √(8kT/πm)
- Bare glass: γ_wall = v̄·A/(4V) ~ 5 kHz (dominates)
- Paraffin coating (P_depol = 10⁻³): γ_wall reduced by 1000×
- PDMS/OTS coating (P_depol = 10⁻⁴): γ_wall reduced by 10000×

### Key equations

CPT linewidth (two-photon resonance):
```
γ_CPT = 2·γ₂ + Ω² / (2·Γ_hom)
```

Rabi frequency (corrected I_sat):
```
Ω = Γ_D1 · √(I / (2·I_sat))    I_sat = 1.669 mW/cm² [Steck2021]
```

Total excited-state decay rate:
```
Γ_hom = Γ_nat + 2π·(γ_N2·P_N2 + γ_Ar·P_Ar) + k_Q·n_N2
         γ_N2 = 17 MHz/Torr,  γ_Ar = 8 MHz/Torr   [Vanier2005]
         k_Q  = 1.4×10⁻¹⁰ cm³/s                     [Rotondaro1997]
```

Wall collision depolarisation rate (no-buffer-gas cell):
```
γ_wall = v̄ · (A_cell / 4V_cell) · P_depol    [Bouchiat1966]
```

Fractional frequency instability (quadrature noise budget):
```
σ_y,total(τ) = √(Σᵢ σᵢ(τ)²)
```

Short-term white-noise scaling:
```
σ_y(τ) ∝ τ⁻¹/²
```

Flicker noise floor (1/f, τ-independent):
```
σ_y,flicker(τ) = σ_y,floor    (default 5×10⁻¹³)
```

Long-term drift (linear):
```
σ_drift(τ) = (d/86400) · τ / √2    [Riley2008]
```

---

## Repository Structure

### Entry points

| File | Purpose |
|---|---|
| `main.py` | GUI entry point |
| `gui.py` | Tkinter parameter panel, plotting, orchestration |
| `launcher.py` | Windows helper launcher |

### Core physics modules

| Module | Responsibility |
|---|---|
| `constants.py` | Physical constants + Rb/buffer-gas coefficients |
| `rb_vapor.py` | Vapour pressure, number density, mean thermal speed |
| `cell_model.py` | Decoherence rates: diffusion, ballistic, spin-exchange, wall coating |
| `vcsel_model.py` | Bessel sideband amplitudes, modulation-index optimisation |
| `cpt_signal.py` | Rabi freq, Γ_hom, CPT linewidth/contrast, lineshape, OD |
| `frequency_shifts.py` | All systematic shifts + combined budget |
| `noise_budget.py` | All noise σ_y(τ) models + total aggregation |
| `allan_deviation.py` | τ-grid, ADEV assembly, sensitivity ranking |
| `environment_sweep.py` | Temperature and pressure-ratio sweeps |

### Key constants (from `constants.py`)

| Constant | Value | Source |
|---|---|---|
| `I_sat_D1_CPT` | 16.70 W/m² (1.669 mW/cm²) | Steck2021 Table 7 lin-pol |
| `I_sat_D1_isotropic` | 44.84 W/m² (4.484 mW/cm²) | Steck2021 Table 7 |
| `gamma_opt_N2_Hz_Torr` | 17 MHz/Torr | Vanier2005 sec. 6.2 |
| `gamma_opt_Ar_Hz_Torr` | 8 MHz/Torr | Vanier2005 sec. 6.2 |
| `k_Q_N2` | 1.4×10⁻¹⁶ m³/s | Rotondaro1997 |
| `P_depol_bare` | 1.0 | Bouchiat1966 |
| `P_depol_paraffin` | 1×10⁻³ | Straessle2014 |
| `P_depol_PDMS` | 1×10⁻⁴ | Straessle2014 |

### Verification and documentation

| File | Purpose |
|---|---|
| `test_physics.py` | 122-test suite with literature-based expected ranges |
| `verify.py` | Quick sanity-check script (~30 checks) |
| `references.md` | Full literature map by subsystem |
| `report.tex` | Technical write-up and equation derivations |
| `requirements.txt` | Python dependencies |

---

## Requirements

- Python ≥ 3.10 (3.12.x recommended; repo includes `.python-version = 3.12.10`)
- numpy ≥ 1.24
- scipy ≥ 1.10
- matplotlib ≥ 3.7
- Tkinter (bundled with standard Python; required for GUI only)

---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

---

## How to Run

### GUI (primary workflow)

```bash
python main.py
```

Click **Run Simulation** to compute and plot all outputs interactively.

### Comprehensive physics validation

```bash
python test_physics.py
```

Expected output: `122 passed, 0 failed, 122 total` with a green `ALL TESTS PASSED`.
Exits non-zero on any failure — suitable for CI regression checks.

### Quick sanity check

```bash
python verify.py
```

Fast smoke-test over constants, shifts, decoherence, and Allan behaviour.

---

## Programmatic Usage

### Buffer-gas cell (standard CSAC)

```python
import numpy as np
from cell_model import total_ground_decoherence
from cpt_signal import compute_cpt_signal
from frequency_shifts import total_shift_budget
from allan_deviation import tau_array, compute_allan_deviation

params = {
    "T_K": 348.15,          # 75 °C
    "P_N2_Torr": 20.0,
    "P_Ar_Torr": 34.0,
    "cell_R_m": 1.5e-3,
    "cell_L_m": 3.0e-3,
    "P_total_uW": 15.0,
    "mod_index": 1.85,
    "beam_diam_mm": 2.0,
    "B_Gauss": 50e-3,
    "eta_QE": 0.7,
    "RIN_dBHz": -135.0,
    "LO_PN_dBcHz": -130.0,
    "servo_bw_Hz": 100.0,
    "NEP_W_rtHz": 1e-12,
    "sigma_T_mK": 1.0,
    "tau_thermal_s": 10.0,
    "sigma_B_uG": 5.0,
    "glass_thickness_mm": 0.5,
}

# Step 1: decoherence (auto-selects diffusion regime since P_bg > 0)
cell = total_ground_decoherence(
    params["cell_R_m"], params["cell_L_m"],
    params["P_N2_Torr"], params["P_Ar_Torr"], params["T_K"]
)
params["gamma2_Hz"] = cell["gamma2_total_Hz"]
print(f"Regime: {cell['regime']}")          # 'diffusion'

# Step 2: CPT signal (auto-computes Γ_hom from P_N2/P_Ar)
cpt = compute_cpt_signal(params)
print(f"FWHM: {cpt['gamma_CPT_Hz']:.0f} Hz")
print(f"Contrast: {100*cpt['contrast']:.2f}%")
print(f"Γ_hom: {cpt['Gamma_hom_rad_s']/6.283e6:.0f} MHz")

# Step 3: Allan deviation
noise_params = {**params,
    "gamma_CPT_Hz": cpt["gamma_CPT_Hz"],
    "contrast": cpt["contrast"],
    "P_det_W": params["P_total_uW"] * 1e-6 * (1.0 - cpt["contrast"]),
    "ddnu_dT_Hz_K": 100.0,
    "Delta_P_atm_mbar": 0.0,
    "laser_detuning_MHz": 0.0,
}
tau = tau_array(1.0, 1e5, 20)
adev = compute_allan_deviation(tau, noise_params)
print(f"σ_y(1s): {adev['sigma_1s']:.2e}")
```

### No-buffer-gas cell with anti-relaxation coating

```python
from cell_model import total_ground_decoherence
from constants import P_depol_paraffin

# Paraffin-coated 10mm × 20mm cell, no buffer gas
cell = total_ground_decoherence(
    R_m=5e-3, L_m=20e-3,
    P_N2_Torr=0.0, P_Ar_Torr=0.0,
    T_K=333.15,
    beam_diam_m=3e-3,
    P_depol_per_bounce=P_depol_paraffin,   # 1e-3
)
print(f"Regime: {cell['regime']}")          # 'ballistic'
print(f"γ_wall: {cell['gamma_wall_Hz']:.1f} Hz")
print(f"γ_se:   {cell['gamma_se_Hz']:.1f} Hz")
print(f"γ₂:     {cell['gamma2_total_Hz']:.1f} Hz")
```

### Inspecting optical physics

```python
from cpt_signal import optical_homogeneous_rate, rabi_frequency
import math

# No buffer gas — natural linewidth only
G_nat = optical_homogeneous_rate(0.0, 0.0, 333.15)
print(f"Γ_nat:          {G_nat / (2*math.pi) / 1e6:.3f} MHz")   # 5.746 MHz

# CSAC buffer gas mixture
G_csac = optical_homogeneous_rate(20.0, 34.0, 348.15)
print(f"Γ_hom (20+34T): {G_csac / (2*math.pi) / 1e6:.0f} MHz")  # ~1.3 GHz

# Corrected Rabi frequency at 10 µW in 3 mm beam
import math
Omega = rabi_frequency(10e-6, math.pi * (1.5e-3)**2)
print(f"Ω/(2π):         {Omega / (2*math.pi) / 1e6:.3f} MHz")  # ~1.2 MHz
```

---

## Noise Budget Parameters

The noise budget accepts the following keys in the `params` dict:

| Key | Default | Description |
|---|---|---|
| `P_det_W` | — | Detected optical power [W] |
| `gamma_CPT_Hz` | — | CPT linewidth FWHM [Hz] |
| `contrast` | — | CPT resonance contrast (0–1) |
| `eta_QE` | — | Detector quantum efficiency |
| `RIN_dBHz` | — | Laser relative intensity noise [dBc/Hz] |
| `LO_PN_dBcHz` | -110 | LO phase noise [dBc/Hz at f_mod] |
| `servo_bw_Hz` | 10 | Servo bandwidth [Hz] |
| `sigma_T_mK` | — | Temperature stability 1σ [mK] |
| `tau_thermal_s` | — | Thermal time constant [s] |
| `ddnu_dT_Hz_K` | — | Buffer-gas shift sensitivity [Hz/K] |
| `B_Gauss` | — | Bias field magnitude [G] |
| `sigma_B_uG` | — | B-field stability 1σ [µG] |
| `NEP_W_rtHz` | — | Detector noise equivalent power [W/√Hz] |
| `sigma_y_flicker_floor` | 5×10⁻¹³ | Flicker (1/f) ADEV floor |
| `drift_rate_frac_per_day` | 3×10⁻¹¹ | Linear drift [frac/day] |
| `glass_thickness_mm` | — | (If set) enables He permeation drift model |

---

## Model Assumptions and Limits

- **CW operation only** — Dick effect is zero (no dead time). Pulsed CPT
  requires adding the Dick-effect term from `noise_budget.dick_effect_sigma()`.
- **Analytical / reduced-order** — Not a full multilevel density-matrix solver.
  Suitable for design-space exploration and sensitivity analysis.
- **Homogeneous Doppler integration** — Beer-Lambert OD uses the
  Doppler-averaged cross-section in the Voigt profile limit.
- **Symmetric bichromatic field** — Both CPT sidebands assumed equal power.
- **GUI defaults** target a CSAC-like operating point (75 °C, 54 Torr total,
  2 mm beam, 15 µW).

---

## Validation

Two complementary scripts are provided:

| Script | Tests | Purpose |
|---|---|---|
| `test_physics.py` | 122 | Full regression suite with literature expected values |
| `verify.py` | ~30 | Quick smoke-test for fast iteration |

Run before committing any model changes:

```bash
python test_physics.py && python verify.py
```

Test sections in `test_physics.py`:

1. Physical constants (Steck2021)
2. Rb vapour model (Nesmeyanov)
3. Cell geometry and decoherence (Vanier2005, Bouchiat1966)
4. VCSEL modulation (Bessel identities, Shah2006)
5. CPT signal (Wynands1999, Knappe2004)
6. Frequency shifts (Kozlova→Boudot2011, Breit–Rabi, Bize1999)
7. Noise budget (Wynands1999, Audoin1998, Riley2008)
8. Allan deviation slopes and quadrature properties
9. Environment sweeps (inversion temperature, pressure ratio)
10. CSAC cross-checks (Knappe2004)
11. **New physics corrections** (I_sat, Γ_hom, ballistic model, flicker noise)

---

## Roadmap

- [ ] pytest-style test harness wrapping `test_physics.py`
- [ ] CLI batch runner with CSV/JSON export (no GUI dependency)
- [ ] Parameter preset files (YAML) for reproducible experiments
- [ ] Uncertainty propagation on key coefficients
- [ ] CI workflow (GitHub Actions) to run physics regression on every push
- [ ] Full multilevel density-matrix solver option for high-accuracy regime
- [ ] Pulsed-CPT / Ramsey mode with Dick effect

---

## References

| Tag | Citation |
|---|---|
| [Steck2021] | D. A. Steck, *Rubidium 87 D Line Data*, rev. 2.2.2 (2021) |
| [Vanier2005] | J. Vanier & C. Audoin, *The Quantum Physics of Atomic Frequency Standards*, IOP (2005) |
| [Boudot2011] | R. Boudot et al., *Opt. Express* **19**, 3106 (2011) |
| [Wynands1999] | R. Wynands & A. Nagel, *Appl. Phys. B* **68**, 1 (1999) |
| [Rotondaro1997] | A. D. Rotondaro & G. P. Perram, *JQSRT* **57**, 497 (1997) |
| [Bouchiat1966] | M. A. Bouchiat & J. Brossel, *Phys. Rev.* **147**, 41 (1966) |
| [Beverini1971] | N. Beverini et al., *Phys. Rev. A* **4**, 550 (1971) |
| [Straessle2014] | R. Straessle et al., *J. Phys. B* **47**, 075502 (2014) |
| [Knappe2004] | S. Knappe et al., *Opt. Lett.* **29**, 388 (2004) |
| [Riley2008] | W. J. Riley, *NIST SP-1065* (2008) |
| [Santarelli1998] | G. Santarelli et al., *IEEE Trans. UFFC* **45**, 887 (1998) |
| [Camparo1999] | J. C. Camparo & J. G. Coffer, *Phys. Rev. A* **59**, 728 (1999) |

See [`references.md`](references.md) for the complete citation map with per-equation attribution.
