"""
frequency_shifts.py
===================
All systematic frequency shifts for a CW-CPT ⁸⁷Rb/buffer-gas clock.

Each function returns the shift in Hz (positive = blue, negative = red).

References
----------
[Kozlova2011] O. Kozlova et al., Phys. Rev. A 83, 062714 (2011). [buffer gas shift]
[Vanier2005]  J. Vanier & C. Audoin (2005).          [Zeeman, barometric]
[Bize1999]    S. Bize et al., Europhys. Lett. 45, 558 (1999). [BBR]
[Levi2000]    F. Levi et al., Eur. Phys. J. D 12, 53 (2000). [light shift]
[Allard2004]  F. Allard et al., Phys. Rev. A 70, 012513 (2004). [spin-exchange]
[Steck2021]   D. A. Steck, Rb-87 D Line Data (2021).
"""

import math
import numpy as np
from scipy.special import jv

from constants import (
    T0_bg,
    N2_beta0, N2_beta1, N2_beta2,
    Ar_beta0, Ar_beta1, Ar_beta2,
    K_Zeeman,
    kappa_se, T_ref_BBR, beta_BBR, eps_BBR, nu_hfs,
    kappa_baro,
    Gamma_D1,
    alpha_LS_typ,
)
from rb_vapor import number_density
from cpt_signal import doppler_linewidth_Hz

# ─────────────────────────────────────────────────────────────────────────────
# 1. BUFFER GAS COLLISIONAL SHIFT  [Kozlova2011]
# ─────────────────────────────────────────────────────────────────────────────

def buffer_gas_shift(T_K: float, P_N2_Torr: float, P_Ar_Torr: float) -> dict:
    """
    Frequency shift due to Rb + buffer-gas elastic collisions.

    For each gas:
        Δν_gas = P_gas · [β₀ + β₁·(T−T₀) + β₂·(T−T₀)²]

    where T₀ = 273.15 K (0 °C).  [Kozlova2011, Table 1]

    The temperature sensitivity (useful for error budget):
        dΔν/dT|_gas = P_gas · [β₁ + 2·β₂·(T−T₀)]

    Parameters
    ----------
    T_K : float
        Cell temperature [K].
    P_N2_Torr : float
        N₂ partial pressure [Torr].
    P_Ar_Torr : float
        Ar partial pressure [Torr].

    Returns
    -------
    dict
        Individual shifts, total shift, and total dν/dT [Hz/K].
    """
    dT = T_K - T0_bg

    dnu_N2 = P_N2_Torr * (N2_beta0 + N2_beta1 * dT + N2_beta2 * dT ** 2)
    dnu_Ar = P_Ar_Torr * (Ar_beta0 + Ar_beta1 * dT + Ar_beta2 * dT ** 2)

    ddnu_N2_dT = P_N2_Torr * (N2_beta1 + 2 * N2_beta2 * dT)
    ddnu_Ar_dT = P_Ar_Torr * (Ar_beta1 + 2 * Ar_beta2 * dT)

    return {
        "dnu_N2_Hz":      dnu_N2,
        "dnu_Ar_Hz":      dnu_Ar,
        "dnu_total_Hz":   dnu_N2 + dnu_Ar,
        "ddnu_dT_Hz_K":   ddnu_N2_dT + ddnu_Ar_dT,
    }


def buffer_gas_inversion_temperature(P_N2_Torr: float, P_Ar_Torr: float) -> float:
    """
    Temperature at which dΔν_bg/dT = 0 (zero T-coefficient).

    Analytical formulation for the inversion temperature T_i:
        T_i = T_0 - 2(gamma_1 + a*gamma_2) / (delta_1 + a*delta_2)

    Parameters
    ----------
    P_N2_Torr, P_Ar_Torr : float
        Partial pressures [Torr].

    Returns
    -------
    float
        Inversion temperature [K] (NaN if β₂ terms are zero).
    """
    numerator   = -(P_N2_Torr * N2_beta1 + P_Ar_Torr * Ar_beta1)
    denominator =  (P_N2_Torr * 2 * N2_beta2 + P_Ar_Torr * 2 * Ar_beta2)
    if abs(denominator) < 1e-30:
        return float("nan")
    return T0_bg + numerator / denominator


# ─────────────────────────────────────────────────────────────────────────────
# 2. LIGHT SHIFT (AC STARK SHIFT)  [Levi2000, Shah2006]
# ─────────────────────────────────────────────────────────────────────────────

def light_shift(P_total_uW: float, beam_diam_mm: float,
                mod_index: float, laser_detuning_MHz: float = 0.0, T_K: float = 333.15) -> dict:
    """
    AC Stark (light) shift of the CPT clock frequency.

    Model structure
    ---------------
    Two contributions are computed separately:

    (A) INTENSITY SHIFT — semi-empirical, based on alpha_LS_typ [constants.py]

        dnu_int = alpha_eff(m) * I   [Hz]

        where I is in muW/cm^2 and alpha_LS_typ is an empirically measured
        coefficient (Representative value: -1e-3 Hz/(muW/cm^2); sign and
        magnitude depend on cell density, polarisation, and detuning history).
        alpha_eff(m) accounts for partial cancellation between k=+/-1
        sidebands (which shift one way) and the carrier + k=+/-2 sidebands
        (which shift the other way)  [Levi2000 eq. 11; Shah2006].

    (B) DETUNING SHIFT — Voigt-profile dispersive contribution

        When the laser optical frequency is detuned from line centre by Delta,
        the dispersive slope of the Doppler-broadened absorption profile
        generates a frequency-dependent modulation of the light shift.
        The normalised slope is computed from Im[wofz(z)] (Faddeeva function)
        evaluated at z = (Delta + i*Gamma_nat/2) / Delta_nu_D  [Camparo1999]:

            dnu_det = alpha_LS_typ * (2*Delta/Delta_nu_D)
                      * (sqrt(pi)/2) * Im[wofz(z)] * I   [Hz]

        All intermediate calculations are performed in Hz to keep units
        explicit and traceable.

    Parameters
    ----------
    P_total_uW : float
        Total laser power [muW].
    beam_diam_mm : float
        Beam diameter [mm] (for intensity calculation).
    mod_index : float
        VCSEL modulation index m.
    laser_detuning_MHz : float
        Laser optical detuning from D1 line centre [MHz].
    T_K : float, optional
        Cell temperature [K] for Doppler width. Defaults to 333.15 K (60 C).

    Returns
    -------
    dict
        alpha_eff [Hz/(muW/cm^2)], I [muW/cm^2],
        dnu_intensity_Hz, dnu_detuning_Hz, dnu_total_Hz, zero_crossing_m.
    """
    if T_K is None:
        T_K = 333.15
    beam_area_cm2 = math.pi * (beam_diam_mm * 0.1 / 2) ** 2   # cm^2
    I_cm2         = P_total_uW / beam_area_cm2                  # muW/cm^2

    # --- (A) Intensity shift with sideband weighting ---
    J0 = float(jv(0, mod_index))
    J1 = float(jv(1, mod_index))
    J2 = float(jv(2, mod_index))
    J1sq = J1 ** 2 if abs(J1) > 1e-10 else 1e-10
    # Effective sideband weighting [Levi2000 eq. 11]
    alpha_eff = alpha_LS_typ * (J0 ** 2 - J2 ** 2) / (J0 ** 2 + J1sq + J2 ** 2)
    dnu_intensity = alpha_eff * I_cm2   # Hz

    # --- (B) Detuning shift via Voigt dispersion, all in Hz ---
    from scipy.special import wofz
    laser_detuning_Hz = laser_detuning_MHz * 1e6      # Hz
    Doppler_width_Hz  = doppler_linewidth_Hz(T_K)     # Hz (FWHM)
    Gamma_nat_Hz      = Gamma_D1 / (2.0 * math.pi)   # Hz (natural linewidth)

    # Faddeeva function argument: z = (detuning + i*Gamma_nat/2) / Delta_nu_D
    z_det      = complex(laser_detuning_Hz, Gamma_nat_Hz / 2.0) / Doppler_width_Hz
    dispersion = np.imag(wofz(z_det))   # dimensionless dispersive profile

    # Normalised dispersive slope (in Hz): (2*Delta/Delta_nu_D)*(sqrt(pi)/2)*Im[W(z)]
    # Multiplied by alpha_LS_typ [Hz/(muW/cm^2)] gives Hz output.
    dnu_detuning = (alpha_LS_typ
                    * (2.0 * laser_detuning_Hz / Doppler_width_Hz)
                    * (math.sqrt(math.pi) / 2.0)
                    * dispersion
                    * I_cm2)   # Hz

    dnu_total = dnu_intensity + dnu_detuning

    return {
        "alpha_eff_Hz_per_uW_cm2": alpha_eff,
        "I_uW_cm2":                I_cm2,
        "dnu_intensity_Hz":        dnu_intensity,
        "dnu_detuning_Hz":         dnu_detuning,
        "dnu_total_Hz":            dnu_total,
        "zero_crossing_m":         2.4,   # approx m where LS ~ 0 [Shah2006]
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. SECOND-ORDER ZEEMAN SHIFT  [Vanier2005, sec. 3.5]
# ─────────────────────────────────────────────────────────────────────────────

def zeeman_shift(B_Gauss: float) -> dict:
    """
    Second-order Zeeman shift of the (F=1,mF=0)→(F=2,mF=0) clock transition.

        Δν_Z2 = K_Z · B²         [Vanier2005, eq. 3.48]

    K_Z = 575.15 Hz/G² for ⁸⁷Rb (derived from Breit-Rabi formula).

    Parameters
    ----------
    B_Gauss : float
        DC magnetic (C-field) magnitude [Gauss].

    Returns
    -------
    dict
        Zeeman shift [Hz] and dν/dB [Hz/G].
    """
    dnu = K_Zeeman * B_Gauss ** 2
    sensitivity = 2 * K_Zeeman * B_Gauss   # Hz/G
    return {
        "dnu_Z2_Hz":         dnu,
        "sensitivity_Hz_G":  sensitivity,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. SPIN-EXCHANGE SHIFT  [Allard2004]
# ─────────────────────────────────────────────────────────────────────────────

def spin_exchange_shift(T_K: float) -> dict:
    """
    Spin-exchange frequency shift due to Rb–Rb collisions.

        Δν_se = κ_se · n_Rb(T)     [Allard2004]

    κ_se is negative for Rb HFS; the shift is typically < 1 Hz for
    CPT cells operating below 80 °C.

    Parameters
    ----------
    T_K : float
        Temperature [K].

    Returns
    -------
    dict
        Shift [Hz] and density [m⁻³].
    """
    n   = number_density(T_K)
    dnu = kappa_se * n
    return {
        "n_Rb_m3":       n,
        "dnu_se_Hz":     dnu,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. BLACKBODY RADIATION SHIFT  [Bize1999]
# ─────────────────────────────────────────────────────────────────────────────

def bbr_shift(T_K: float) -> dict:
    """
    Fractional frequency shift due to ambient blackbody radiation.

        Δν_BBR / ν₀ = −β_BBR · (T/T_ref)⁴ · [1 + ε·(T/T_ref)²]   [Bize1999]

    For Rb: β_BBR = 1.26 × 10⁻¹⁴, ε = 0.013 at T_ref = 300 K.

    Parameters
    ----------
    T_K : float
        Cell environment temperature [K].

    Returns
    -------
    dict
        BBR shift [Hz] and fractional shift.
    """
    ratio = T_K / T_ref_BBR
    frac  = -beta_BBR * ratio ** 4 * (1.0 + eps_BBR * ratio ** 2)
    dnu   = frac * nu_hfs
    return {
        "fractional_BBR":  frac,
        "dnu_BBR_Hz":      dnu,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. BAROMETRIC SHIFT  [Vanier2005, sec. 6.7]
# ─────────────────────────────────────────────────────────────────────────────

def barometric_shift(Delta_P_atm_mbar: float) -> dict:
    """
    Frequency shift due to changes in atmospheric pressure.

    For a hermetically sealed cell the buffer gas pressure inside is
    fixed; however, external pressure can flex the cell walls and alter
    the internal pressure slightly.

        Δν_baro / ν₀ ≈ κ_baro · ΔP_atm          [Vanier2005]

    κ_baro ≈ 10⁻¹² / mbar for glass cells (can vary with cell design).

    Parameters
    ----------
    Delta_P_atm_mbar : float
        Deviation from nominal atmospheric pressure [mbar].

    Returns
    -------
    dict
        Barometric shift [Hz].
    """
    frac  = kappa_baro * Delta_P_atm_mbar
    dnu   = frac * nu_hfs
    return {
        "fractional_baro":  frac,
        "dnu_baro_Hz":      dnu,
    }


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED SHIFT BUDGET
# ─────────────────────────────────────────────────────────────────────────────

def total_shift_budget(T_K: float, P_N2_Torr: float, P_Ar_Torr: float,
                       B_Gauss: float, P_total_uW: float,
                       beam_diam_mm: float, mod_index: float,
                       laser_detuning_MHz: float = 0.0,
                       Delta_P_atm_mbar: float = 0.0) -> dict:
    """
    Compute all systematic frequency shifts and return a comprehensive budget.

    Returns
    -------
    dict
        All individual shifts and their sum [Hz].
    """
    bg   = buffer_gas_shift(T_K, P_N2_Torr, P_Ar_Torr)
    ls   = light_shift(P_total_uW, beam_diam_mm, mod_index, laser_detuning_MHz, T_K=T_K)
    zm   = zeeman_shift(B_Gauss)
    se   = spin_exchange_shift(T_K)
    bbr  = bbr_shift(T_K)
    baro = barometric_shift(Delta_P_atm_mbar)

    total = (bg["dnu_total_Hz"]  + ls["dnu_total_Hz"] +
             zm["dnu_Z2_Hz"]     + se["dnu_se_Hz"]    +
             bbr["dnu_BBR_Hz"]   + baro["dnu_baro_Hz"])

    T_inv = buffer_gas_inversion_temperature(P_N2_Torr, P_Ar_Torr)

    return {
        # Component shifts [Hz]
        "dnu_buffer_gas_Hz":    bg["dnu_total_Hz"],
        "dnu_light_shift_Hz":   ls["dnu_total_Hz"],
        "dnu_zeeman_Hz":        zm["dnu_Z2_Hz"],
        "dnu_spin_exchange_Hz": se["dnu_se_Hz"],
        "dnu_BBR_Hz":           bbr["dnu_BBR_Hz"],
        "dnu_barometric_Hz":    baro["dnu_baro_Hz"],
        "dnu_total_Hz":         total,
        # Sensitivity info
        "ddnu_bg_dT_Hz_K":      bg["ddnu_dT_Hz_K"],
        "ddnu_zeeman_dB_Hz_G":  zm["sensitivity_Hz_G"],
        "T_inversion_K":        T_inv,
        "T_inversion_C":        T_inv - 273.15 if not math.isnan(T_inv) else float("nan"),
        # Sub-dicts for detail
        "_bg_detail":  bg,
        "_ls_detail":  ls,
        "_zm_detail":  zm,
        "_se_detail":  se,
        "_bbr_detail": bbr,
        "_baro_detail": baro,
    }
