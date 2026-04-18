# verify.py - quick sanity-check script for all physics modules.
# Run: C:/msys64/ucrt64/bin/python.exe verify.py
import sys, math
sys.path.insert(0, 'd:/Google_antigravity/CPTClockModel')

from constants import nu_hfs, K_Zeeman, Gamma_D1
from rb_vapor import number_density
from cell_model import total_ground_decoherence
from vcsel_model import sideband_powers
from cpt_signal import cpt_linewidth_Hz, cpt_contrast, rabi_frequency
from frequency_shifts import (total_shift_budget,
                               buffer_gas_inversion_temperature, zeeman_shift)
from noise_budget import total_noise_budget
from allan_deviation import tau_array, compute_allan_deviation
import numpy as np

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

def check(name, val, low, high):
    ok = low <= val <= high
    tag = PASS if ok else FAIL
    print(f"  [{tag}]  {name}: {val:.4g}  (expect {low:.4g} – {high:.4g})")
    return ok

errors = 0
print("\n=== CPT CLOCK MODEL — SELF-VERIFICATION ===\n")

# ── 1. Fundamental constants ──────────────────────────────────────────────
print("1. CONSTANTS")
ok = check("nu_hfs [Hz]", nu_hfs, 6.8346826e9, 6.8346827e9)
ok &= check("K_Zeeman [Hz/G^2]", K_Zeeman, 574.0, 576.5)
if not ok: errors += 1

# ── 2. Vapour pressure / density ─────────────────────────────────────────
print("\n2. Rb VAPOUR DENSITY")
n_25  = number_density(298.15)
n_60  = number_density(333.15)
n_80  = number_density(353.15)
ok  = check("n_Rb(25°C) [m-3]",  n_25, 1e15, 1e17)
ok &= check("n_Rb(60°C) [m-3]",  n_60, 1e16, 1e18)
ok &= check("n_Rb(80°C) [m-3]",  n_80, 5e16, 5e18)
if not ok: errors += 1

# ── 3. Buffer gas inversion temperature ──────────────────────────────────
print("\n3. BUFFER GAS INVERSION TEMPERATURE")
# P_Ar/P_N2 = 8/5 = 1.6 → expect ~55–70 °C
T_inv = buffer_gas_inversion_temperature(5.0, 8.0) - 273.15
ok = check("T_inv(N2=5T, Ar=8T) [°C]", T_inv, 40.0, 80.0)
if not ok: errors += 1

# ── 4. Zeeman shift ───────────────────────────────────────────────────────
print("\n4. ZEEMAN SHIFT")
zm50  = zeeman_shift(0.050)   # 50 mG
zm70  = zeeman_shift(0.070)   # 70 mG
ok  = check("Zeeman(50 mG) [Hz]", zm50["dnu_Z2_Hz"], 1.40, 1.50)   # 575.15×0.05^2=1.438
ok &= check("Zeeman(70 mG) [Hz]", zm70["dnu_Z2_Hz"], 2.80, 2.85)   # 575.15×0.07^2=2.818
if not ok: errors += 1

# ── 5. Cell decoherence ───────────────────────────────────────────────────
print("\n5. CELL DECOHERENCE RATES")
cell = total_ground_decoherence(5e-3, 20e-3, 5.0, 8.0, 333.15)
ok  = check("gamma_diff  [Hz]", cell["gamma_diff_Hz"], 10.0, 5000.0)
ok &= check("gamma_se    [Hz]", cell["gamma_se_Hz"],   0.1,  1000.0)
ok &= check("gamma_bg    [Hz]", cell["gamma_bg_Hz"],   500.0, 5000.0)
ok &= check("gamma2_tot  [Hz]", cell["gamma2_total_Hz"], 500.0, 6000.0)
if not ok: errors += 1

# ── 6. CPT linewidth & contrast ────────────────────────────────────────────
print("\n6. CPT SIGNAL")
pw      = sideband_powers(1.85, 100e-6)
beam_a  = math.pi * (3e-3 / 2) ** 2
Omega   = rabi_frequency(pw[1], beam_a)
Gamma_buf = 2.0 * math.pi * 500e6
lw      = cpt_linewidth_Hz(cell["gamma2_total_Hz"], Omega, Gamma_hom_rad_s=(Gamma_D1 + Gamma_buf))
C       = cpt_contrast(Omega, cell["gamma2_total_Hz"])
ok  = check("gamma_CPT [Hz]", lw, 500.0, 5e6)   # power-broadened at 100uW/3mm
ok &= check("contrast  [%]",  C * 100, 0.01, 10.0)
if not ok: errors += 1

# ── 7. Full shift budget ──────────────────────────────────────────────────
print("\n7. SHIFT BUDGET")
sb = total_shift_budget(333.15, 5.0, 8.0, 0.07, 100.0, 3.0, 1.85)
ok  = check("Buffer gas shift [Hz]",  sb["dnu_buffer_gas_Hz"],  -2000.0, 5000.0)
ok &= check("Total shift      [Hz]",  sb["dnu_total_Hz"],       -3000.0, 6000.0)
ok &= check("BBR shift        [Hz]",  sb["dnu_BBR_Hz"],         -0.2, 0.0)
if not ok: errors += 1
print(f"         T_inversion = {sb['T_inversion_C']:.1f} °C")
print(f"         dnu/dT      = {sb['ddnu_bg_dT_Hz_K']:.4f} Hz/K")

# ── 8. Allan deviation ────────────────────────────────────────────────────
print("\n8. ALLAN DEVIATION")
noise_p = {
    "T_K": 333.15, "P_N2_Torr": 5.0, "P_Ar_Torr": 8.0,
    "B_Gauss": 0.07, "P_total_uW": 100.0, "mod_index": 1.85,
    "beam_diam_mm": 3.0, "cell_R_m": 5e-3, "cell_L_m": 20e-3,
    "gamma_CPT_Hz": lw, "contrast": C,
    "P_det_W": 100e-6 * (1 - C),
    "eta_QE": 0.7, "RIN_dBHz": -135.0,
    "LO_PN_dBcHz": -110.0, "servo_bw_Hz": 10.0,
    "NEP_W_rtHz": 2e-12,
    "sigma_T_mK": 1.0, "tau_thermal_s": 100.0,
    "sigma_B_uG": 10.0, "Delta_P_atm_mbar": 0.0,
    "drift_rate_frac_per_day": 5e-15,
    "ddnu_dT_Hz_K": sb["ddnu_bg_dT_Hz_K"],
}
tau  = tau_array(1.0, 1e5, 20)
adev = compute_allan_deviation(tau, noise_p)
ok  = check("sigma_y(1s)", adev["sigma_1s"], 1e-14, 1e-8)
ok &= check("slope (short tau)", adev["slope_short"], -0.65, -0.35)
if not ok: errors += 1

# ── Result ────────────────────────────────────────────────────────────────
print()
if errors == 0:
    print("\033[92m=== ALL CHECKS PASSED ===\033[0m")
else:
    print(f"\033[91m=== {errors} CHECK(S) FAILED ===\033[0m")

sys.exit(errors)
