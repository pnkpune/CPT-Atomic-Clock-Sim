"""
environment_sweep.py
====================
Temperature and pressure-ratio sweeps for CPT clock optimisation.

References
----------
[Kozlova2011] O. Kozlova et al., Proc. EFTF (2011).
[Vanier2005]  J. Vanier & C. Audoin (2005), sec. 6.3, 6.4.
"""

import numpy as np
from frequency_shifts import buffer_gas_shift, buffer_gas_inversion_temperature, total_shift_budget
from cell_model import total_ground_decoherence
from cpt_signal import cpt_linewidth_Hz, cpt_contrast, rabi_frequency
from vcsel_model import sideband_powers
from noise_budget import total_noise_budget

# ─────────────────────────────────────────────────────────────────────────────

def temperature_sweep(params: dict,
                      T_min_C: float = 30.0,
                      T_max_C: float = 90.0,
                      n_points: int = 200) -> dict:
    """
    Sweep cell temperature and compute clock performance metrics.

    Parameters
    ----------
    params : dict
        Base parameter dict (see gui.py). T_K is overridden.
    T_min_C, T_max_C : float
        Sweep range [°C].
    n_points : int
        Number of temperature points.

    Returns
    -------
    dict
        Arrays indexed by temperature.
    """
    T_arr_C  = np.linspace(T_min_C, T_max_C, n_points)
    T_arr_K  = T_arr_C + 273.15

    dnu_bg   = np.zeros(n_points)
    ddnu_dT  = np.zeros(n_points)
    dnu_total= np.zeros(n_points)
    gamma2   = np.zeros(n_points)
    gamma_CPT= np.zeros(n_points)
    contrast = np.zeros(n_points)
    sigma_1s = np.zeros(n_points)

    pw = sideband_powers(params["mod_index"], params["P_total_uW"] * 1e-6)
    P_sb = pw.get(1, 0.0)
    beam_area = np.pi * (params["beam_diam_mm"] * 1e-3 / 2) ** 2

    for i, T_K in enumerate(T_arr_K):
        p = dict(params)
        p["T_K"] = T_K

        # Cell model
        cell = total_ground_decoherence(
            p["cell_R_m"], p["cell_L_m"],
            p["P_N2_Torr"], p["P_Ar_Torr"], T_K)
        g2 = cell["gamma2_total_Hz"]
        gamma2[i] = g2

        # CPT signal
        Omega = rabi_frequency(P_sb, beam_area)
        lw    = cpt_linewidth_Hz(g2, Omega)
        C     = cpt_contrast(Omega, g2)    # pass actual gamma2 for correct saturation
        gamma_CPT[i] = lw
        contrast[i]  = C

        # Frequency shifts
        bg = buffer_gas_shift(T_K, p["P_N2_Torr"], p["P_Ar_Torr"])
        dnu_bg[i]  = bg["dnu_total_Hz"]
        ddnu_dT[i] = bg["ddnu_dT_Hz_K"]

        sb = total_shift_budget(
            T_K, p["P_N2_Torr"], p["P_Ar_Torr"],
            p["B_Gauss"], p["P_total_uW"],
            p["beam_diam_mm"], p["mod_index"],
            p.get("laser_detuning_MHz", 0.0),
            p.get("Delta_P_atm_mbar", 0.0))
        dnu_total[i] = sb["dnu_total_Hz"]

        # Stability at τ = 1 s
        noise_p = dict(p)
        noise_p["gamma_CPT_Hz"]  = lw
        noise_p["contrast"]      = C
        noise_p["ddnu_dT_Hz_K"]  = bg["ddnu_dT_Hz_K"]
        noise_p["P_det_W"]       = p["P_total_uW"] * 1e-6 * (1 - C)
        nb = total_noise_budget(np.array([1.0]), noise_p)
        sigma_1s[i] = float(nb["sigma_total"][0])

    return {
        "T_C":       T_arr_C,
        "T_K":       T_arr_K,
        "dnu_bg_Hz": dnu_bg,
        "ddnu_dT_Hz_K": ddnu_dT,
        "dnu_total_Hz": dnu_total,
        "gamma2_Hz": gamma2,
        "gamma_CPT_Hz": gamma_CPT,
        "contrast":  contrast,
        "sigma_1s":  sigma_1s,
    }


def pressure_ratio_sweep(params: dict,
                          ratio_min: float = 0.5,
                          ratio_max: float = 4.0,
                          n_points: int = 200) -> dict:
    """
    Sweep P_Ar / P_N2 at fixed total pressure and compute:
        - Buffer gas total shift at operating temperature
        - Inversion temperature
        - dν/dT at operating temperature

    Parameters
    ----------
    params : dict
        Base parameters; P_N2_Torr and P_Ar_Torr are recomputed.
    ratio_min, ratio_max : float
        Range of P_Ar / P_N2 to sweep.
    n_points : int
        Number of ratio points.

    Returns
    -------
    dict
        Arrays indexed by ratio.
    """
    P_total  = params["P_N2_Torr"] + params["P_Ar_Torr"]
    T_K      = params["T_K"]
    ratios   = np.linspace(ratio_min, ratio_max, n_points)

    dnu_bg   = np.zeros(n_points)
    ddnu_dT  = np.zeros(n_points)
    T_inv    = np.zeros(n_points)

    for i, r in enumerate(ratios):
        # r = P_Ar / P_N2,  P_total = P_N2 + P_Ar = P_N2*(1+r)
        P_N2 = P_total / (1.0 + r)
        P_Ar = P_total - P_N2

        bg = buffer_gas_shift(T_K, P_N2, P_Ar)
        dnu_bg[i]  = bg["dnu_total_Hz"]
        ddnu_dT[i] = bg["ddnu_dT_Hz_K"]
        T_inv[i]   = buffer_gas_inversion_temperature(P_N2, P_Ar) - 273.15  # to °C

    return {
        "ratio":     ratios,
        "dnu_bg_Hz": dnu_bg,
        "ddnu_dT_Hz_K": ddnu_dT,
        "T_inv_C":   T_inv,
        "P_total_Torr": P_total,
    }
