# CPT Atomic Clock Simulation (87Rb CW-CPT)

Comprehensive simulation and analysis toolkit for a compact coherent population trapping (CPT) atomic clock based on 87Rb vapor with Ar/N2 buffer gas.

The project combines:
- First-principles and literature-based physics models
- An interactive Tkinter + Matplotlib GUI for design exploration
- Parametric sweeps for optimization
- Noise and Allan deviation prediction
- Automated validation scripts against published ranges

## Table of Contents

1. [What This Repository Does](#what-this-repository-does)
2. [Key Features](#key-features)
3. [Physics Scope](#physics-scope)
4. [Repository Structure](#repository-structure)
5. [Requirements](#requirements)
6. [Installation](#installation)
7. [How to Run](#how-to-run)
8. [Programmatic Usage Example](#programmatic-usage-example)
9. [Validation and Testing](#validation-and-testing)
10. [Model Assumptions and Limits](#model-assumptions-and-limits)
11. [Troubleshooting](#troubleshooting)
12. [Roadmap Ideas](#roadmap-ideas)

## What This Repository Does

This codebase models the performance of a continuous-wave CPT clock by connecting the full chain:

1. Cell physics
   - Rb vapor density
   - Diffusion, spin exchange, and buffer-gas broadening
2. Optical interaction
   - VCSEL sideband spectrum from modulation index
   - CPT linewidth, contrast, lineshape, and discriminator slope
3. Systematic shifts
   - Buffer-gas, light shift, second-order Zeeman, spin exchange, BBR, barometric
4. Stability
   - Noise-source decomposition
   - Allan deviation across averaging time
   - Holdover/time-error prediction
5. Optimization
   - Temperature sweep
   - Ar/N2 ratio sweep for inversion tuning

## Key Features

- Interactive GUI with 7 plot tabs:
  - CPT signal lineshape
  - Shift budget
  - Allan deviation
  - Noise budget at 1 s
  - Temperature sweep
  - Pressure-ratio optimization
  - Holdover time error
- Full systematic-shift budget with component-level introspection
- Noise budget including:
  - Shot noise
  - Laser RIN
  - LO FM noise
  - Laser PM-to-AM conversion
  - Temperature and magnetic noise
  - Detector/electronics noise
  - Long-term drift terms (helium permeation, VCSEL aging, Rb reactivity)
- Physics validation scripts with literature-based expected ranges
- LaTeX report source for documentation and derivations

## Physics Scope

The implementation models a CW-CPT clock in a compact-vapor-cell regime.

Representative equations used by the code include:

- CPT linewidth (symmetric two-photon driving):

  gamma_CPT = 2*gamma_2 + Omega^2 / (2*Gamma_hom)

- Fractional instability decomposition:

  sigma_y_total(tau) = sqrt(sum_i sigma_i(tau)^2)

- Typical white-noise scaling:

  sigma_y(tau) ~ tau^(-1/2)

- Drift contribution (linear fractional drift d):

  sigma_drift(tau) = (d/86400) * tau / sqrt(2)

For detailed derivations, see `report.tex` and module-level docstrings.

## Repository Structure

### Entry points

- `main.py`
  - Main GUI entry point.
- `gui.py`
  - Tkinter app, parameter panel, plotting, orchestration of all model blocks.
- `launcher.py`
  - Windows helper launcher (tries pythonw / MSYS2 pythonw).
- `launcher.c`
  - Native Win32 launcher source.

### Core physics modules

- `constants.py`
  - Fundamental constants and Rb/buffer-gas coefficients.
- `rb_vapor.py`
  - Vapor pressure, number density, mean thermal speed.
- `cell_model.py`
  - Diffusion and decoherence terms (gamma_diff, gamma_se, gamma_bg).
- `vcsel_model.py`
  - Bessel sideband amplitudes/power and modulation-index optimization.
- `cpt_signal.py`
  - Rabi frequency, CPT linewidth/contrast, lineshape, discriminator slope.
- `frequency_shifts.py`
  - All systematic shift components and combined shift budget.
- `noise_budget.py`
  - Noise-source sigma_y models and total noise aggregation.
- `allan_deviation.py`
  - Tau grid generation, ADEV assembly, sensitivity ranking.
- `environment_sweep.py`
  - Temperature and pressure-ratio optimization sweeps.

### Verification and documentation

- `test_physics.py`
  - Comprehensive test/validation script against expected physical behavior.
- `verify.py`
  - Quick sanity check script.
- `references.md`
  - Literature map by subsystem.
- `report.tex`
  - Technical write-up and equations.
- `requirements.txt`
  - Python dependencies.

### Artifacts

- `report_updated.pdf`
  - Generated report output.
- `CPTClock_Local.exe`
  - Packaged executable artifact.

## Requirements

- Python: 3.12.x recommended (repo contains `.python-version` = 3.12.10)
- OS:
  - macOS, Linux, Windows for Python scripts
  - launcher helpers target Windows
- Python packages:
  - numpy>=1.24
  - scipy>=1.10
  - matplotlib>=3.7
- Tkinter:
  - Required by the GUI (`main.py`, `gui.py`)
  - Usually bundled with standard Python installers

## Installation

From repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## How to Run

### 1) Launch the GUI (primary workflow)

```bash
python main.py
```

Then click Run Simulation in the GUI.

### 2) Run comprehensive physics validation

```bash
python test_physics.py
```

Expected behavior:
- Prints PASS/FAIL checks by section
- Exits non-zero if any check fails

### 3) Run quick sanity checks

```bash
python verify.py
```

Expected behavior:
- Fast range checks over constants, shifts, decoherence, and Allan behavior
- Exits non-zero on failure

## Programmatic Usage Example

Use the model directly in scripts/notebooks:

```python
import numpy as np

from cell_model import total_ground_decoherence
from cpt_signal import compute_cpt_signal
from frequency_shifts import total_shift_budget
from allan_deviation import tau_array, compute_allan_deviation

params = {
    "T_C": 75.0,
    "T_K": 75.0 + 273.15,
    "P_N2_Torr": 20.0,
    "P_Ar_Torr": 34.0,
    "cell_R_m": 1.5e-3,
    "cell_L_m": 3.0e-3,
    "B_Gauss": 50e-3,
    "P_total_uW": 15.0,
    "mod_index": 1.85,
    "beam_diam_mm": 2.0,
    "laser_detuning_MHz": 0.0,
    "eta_QE": 0.7,
    "RIN_dBHz": -135.0,
    "LO_PN_dBcHz": -130.0,
    "servo_bw_Hz": 100.0,
    "NEP_W_rtHz": 1e-12,
    "sigma_T_mK": 1.0,
    "tau_thermal_s": 10.0,
    "sigma_B_uG": 5.0,
    "Delta_P_atm_mbar": 0.0,
    "glass_thickness_mm": 0.5,
    "vcsel_aging_rate": 1.0,
    "rb_reaction_rate": 0.5,
}

cell = total_ground_decoherence(
    params["cell_R_m"], params["cell_L_m"],
    params["P_N2_Torr"], params["P_Ar_Torr"], params["T_K"]
)
params["gamma2_Hz"] = cell["gamma2_total_Hz"]

cpt = compute_cpt_signal(params)
shifts = total_shift_budget(
    params["T_K"], params["P_N2_Torr"], params["P_Ar_Torr"],
    params["B_Gauss"], params["P_total_uW"],
    params["beam_diam_mm"], params["mod_index"],
    params["laser_detuning_MHz"], params["Delta_P_atm_mbar"]
)

noise_params = dict(params)
noise_params["gamma_CPT_Hz"] = cpt["gamma_CPT_Hz"]
noise_params["contrast"] = cpt["contrast"]
noise_params["P_det_W"] = params["P_total_uW"] * 1e-6 * (1.0 - cpt["contrast"])
noise_params["ddnu_dT_Hz_K"] = shifts["ddnu_bg_dT_Hz_K"]

tau = tau_array(1.0, 1e6, 25)
adev = compute_allan_deviation(tau, noise_params)

print("gamma_CPT [Hz]:", cpt["gamma_CPT_Hz"])
print("contrast [%]:", 100.0 * cpt["contrast"])
print("total shift [Hz]:", shifts["dnu_total_Hz"])
print("sigma_y(1s):", adev["sigma_1s"])
```

## Validation and Testing

The repository provides two complementary scripts:

- `test_physics.py` (comprehensive)
  - Multi-section checks across constants, vapor model, linewidth/contrast,
    shifts, noise, Allan slopes, and sweep behavior.
  - Better choice for regression testing after model edits.

- `verify.py` (quick)
  - Lightweight sanity checks for fast smoke-testing.

Recommended workflow before committing model changes:

```bash
python test_physics.py
python verify.py
```

## Model Assumptions and Limits

- Assumes continuous-wave CPT operation.
  - Dick effect is set to zero in the combined noise budget for CW use.
- Uses reduced-order analytical forms for several subsystems.
  - Suitable for design-space exploration and sensitivity studies.
  - Not a full density-matrix time-domain solver with complete multilevel pumping.
- Some coefficients are representative/empirical.
  - Tune coefficients for a specific hardware realization and measured data.
- GUI defaults target a CSAC-like operating point.

## Troubleshooting

### GUI does not open

- Ensure Tkinter is available in your Python build.
- Verify your virtual environment is active and dependencies are installed.

### Import errors

- Confirm you are running from repository root.
- Reinstall dependencies:

```bash
pip install -r requirements.txt
```

### Plot window appears but values look unrealistic

- Check units in GUI entries (mG, Torr, mK, etc.).
- Start from defaults and vary one parameter at a time.
- Run `test_physics.py` to confirm model integrity.

### Validation script failures

- Review first failing section and compare to recent coefficient/equation edits.
- Ensure you did not accidentally change unit conversions.

## Roadmap Ideas

- Add automated pytest-style test harness around current validation scripts.
- Add non-GUI CLI runner for batch sweeps and CSV export.
- Add uncertainty propagation on coefficients and environmental inputs.
- Add parameter-set files (YAML/JSON) for reproducible experiments.
- Add CI workflow to run physics regression checks on every push.

---

If you want, this README can also be expanded with:
- a section matching each GUI tab to exact formulas,
- benchmark parameter presets (short-term stability optimized vs low-drift optimized),
- and a publication-ready methods section extracted from `report.tex`.
