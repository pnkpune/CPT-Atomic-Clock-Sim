"""
cpt_signal.py
=============
Analytical model of the CW-CPT dark-resonance signal.

Model
-----
For a 3-level Λ system driven by two resonant optical fields (first-order
VCSEL sidebands k = ±1), the transmitted power vs two-photon detuning δ is
modelled as a Lorentzian CPT dark-resonance *dip* on a Beer-Lambert
off-resonance background [Wynands1999, Vanier2005 sec 5.2]:

    T(δ, B) = T_bg · [ 1 − C · Σ_j w_j · (γ_CPT/2)²
                              / ( (δ − j·ξ·B)² + (γ_CPT/2)² ) ]

where:
    T_bg   = Beer-Lambert background transmission (Doppler + buffer-gas)
    C      = peak-to-T_bg contrast (≤ 1)
    γ_CPT  = FWHM of the CPT dark resonance [Hz]
    w_j    = relative weights for Zeeman sub-resonances (normalised so Σw_j=1)
    ξ      = first-order Zeeman coefficient [Hz/G]

The three contributing Zeeman sub-resonances (j = −1, 0, +1) are summed
incoherently, which is valid when the Zeeman splitting ξ·B ≫ γ_CPT.  In the
zero-field limit (or when B → 0) the formula reduces to the standard single
Lorentzian dip.

Power-broadened FWHM for symmetric two-photon driving (Ω₁ = Ω₂ ≡ Ω):
    γ_CPT = 2γ₂ + Ω² / (2·Γ_hom)          [Wynands1999 eq. 7; Vanier2005 5.15]
where Γ_hom is the *homogeneous* excited-state decay rate (natural + collisional
quenching), NOT the Doppler width.

References
----------
[Wynands1999] R. Wynands & A. Nagel, Appl. Phys. B 68, 1 (1999).
              Density-matrix derivation of contrast & linewidth.
[Vanier2005]  J. Vanier & C. Audoin (2005), sec. 5.2, 5.3.
[Levi2000]    F. Levi et al., Eur. Phys. J. D 12, 53 (2000).
[Knappe2004]  S. Knappe et al., Opt. Lett. 29, 388 (2004). [CSAC signal]
[Steck2021]   D. A. Steck, Rb-87 D Line Data, rev. 2.2.2 (2021).
"""

import numpy as np
import math
from typing import Optional
from constants import (
    Gamma_D1, nu_hfs, kB, m_Rb, c, lambda_D1, h, eps0,
    I_sat_D1_CPT, I_sat_D1_lin, I_sat_D1_isotropic,
    gamma_opt_N2_Hz_Torr, gamma_opt_Ar_Hz_Torr,
    k_Q_N2,
)
from vcsel_model import sideband_amplitudes
from rb_vapor import number_density

# ─────────────────────────────────────────────────────────────────────────────
# Saturation intensity for Rb D1 line — Steck2021 Table 7.
# Two values are provided for different polarisation geometries:
#   I_sat_D1_lin       = 1.669 mW/cm² = 16.70 W/m²  (lin‖lin, used for CPT)
#   I_sat_D1_isotropic = 4.484 mW/cm² = 44.84 W/m²  (unpolarised / isotropic)
# The earlier formula h·ν·Γ/(3λ²) gave 0.476 mW/cm², which was ~9.4× too small,
# leading to a 3.07× overestimate of Ω and 9.4× overestimate of power broadening.
# We import the correct value from constants.py.
_I_sat_D1 = I_sat_D1_CPT  # [W/m²]  — lin‖lin polarisation, Steck2021 Table 7

# First-order Zeeman splitting coefficient for Rb-87 ground hyperfine states.
# ξ = (g_J·μB − g_I·μN) / h  ≈  700.2 kHz/G  [Steck2021, Table 10]
# In the low-field limit both |F=1,mF=0⟩→|F=2,mF=0⟩ Zeeman sidebands shift
# by ±ξ·B.   We use sign-consistent, Steck-derived value.
_XI_Hz_G = 700.2e3   # [Hz/G]

# ─────────────────────────────────────────────────────────────────────────────


def optical_homogeneous_rate(P_N2_Torr: float = 0.0, P_Ar_Torr: float = 0.0,
                             T_K: float = 333.15) -> float:
    """
    Total homogeneous (Lorentzian) decay rate of the excited 5²P₁/₂ state.

    Three contributions are summed in quadrature:
      (1) Natural radiative decay: Γ_nat = Γ_D1
      (2) Pressure (dephasing) broadening by buffer gas:
              Γ_press = 2π × [γ_N2 × P_N2 + γ_Ar × P_Ar]  [Vanier2005 sec. 6.2]
      (3) N₂ collisional quenching of the excited state:
              Γ_quench = k_Q_N2 × n_N2(T, P_N2)             [Rotondaro1997]

    Argon is a noble gas and does not quench the excited state appreciably.

    Parameters
    ----------
    P_N2_Torr, P_Ar_Torr : float
        Buffer gas partial pressures [Torr].
    T_K : float
        Cell temperature [K].

    Returns
    -------
    float
        Total excited-state homogeneous decay rate [rad/s].
    """
    # (1) natural linewidth
    Gamma_nat = Gamma_D1

    # (2) pressure broadening: HWHM in Hz/Torr → angular frequency
    Gamma_press = 2.0 * math.pi * (
        gamma_opt_N2_Hz_Torr * P_N2_Torr
        + gamma_opt_Ar_Hz_Torr * P_Ar_Torr
    )  # [rad/s]

    # (3) N₂ quenching: n_N2 = P_N2 / (kB * T)  [ideal gas]
    n_N2 = (P_N2_Torr * 133.322) / (kB * T_K)
    Gamma_quench = k_Q_N2 * n_N2   # [rad/s]

    return Gamma_nat + Gamma_press + Gamma_quench



def rabi_frequency(P_sideband_W: float, beam_area_m2: float) -> float:
    """
    Single-photon Rabi frequency for one CPT sideband field.

        Ω = Γ_D1 · sqrt(I / (2·I_sat))        [Steck2021; Wynands1999 eq. 3]

    I_sat is derived from first principles (see module header).  The
    two-photon effective Rabi frequency for symmetric driving is Ω itself
    (enters linewidth as Ω²/(2·Γ_hom)).

    Parameters
    ----------
    P_sideband_W : float
        Optical power in the sideband [W].
    beam_area_m2 : float
        Beam cross-section area [m²].

    Returns
    -------
    float
        Single-sideband Rabi frequency [rad/s].
    """
    I = P_sideband_W / beam_area_m2
    return Gamma_D1 * math.sqrt(I / (2.0 * _I_sat_D1))


def cpt_linewidth_Hz(gamma2_Hz: float, Omega_rad_s: float,
                     Gamma_hom_rad_s: Optional[float] = None) -> float:
    """
    FWHM linewidth of the CW-CPT dark resonance, including power broadening.

    For symmetric two-photon driving (Ω₁ = Ω₂ ≡ Ω), the density-matrix
    steady-state solution gives [Wynands1999 eq. 7; Vanier2005 eq. 5.15]:

        γ_CPT = 2γ₂ + Ω² / (2·Γ_hom)

    where:
      - 2γ₂    : zero-power linewidth (twice the ground-state HWHM)
      - Ω²/(2Γ): power-broadening term (note factor of 2 in denominator vs
                  the one-photon case)
      - Γ_hom  : *homogeneous* optical decay rate = Γ_nat + Γ_quench (NOT
                  the Doppler width, which is an inhomogeneous contribution)

    Parameters
    ----------
    gamma2_Hz : float
        Total ground-state coherence relaxation rate [Hz] (single-sided HWHM,
        so zero-power FWHM = 2·γ₂).
    Omega_rad_s : float
        Single-sideband Rabi frequency [rad/s].
    Gamma_hom_rad_s : float, optional
        Homogeneous excited-state decay rate [rad/s].
        Defaults to Γ_D1 (natural linewidth only — conservative; add
        collisional quenching rate if N₂ quenching is significant).

    Returns
    -------
    float
        CPT resonance FWHM [Hz].
    """
    if Gamma_hom_rad_s is None:
        Gamma_hom_rad_s = Gamma_D1   # natural linewidth; add quenching externally

    gamma2_rad = 2.0 * math.pi * gamma2_Hz
    # Correct factor-of-2 denominator for symmetric two-photon driving
    power_broadening = Omega_rad_s ** 2 / (2.0 * Gamma_hom_rad_s)
    gamma_CPT_rad = 2.0 * gamma2_rad + power_broadening
    return gamma_CPT_rad / (2.0 * math.pi)


def cpt_contrast(Omega_rad_s: float, gamma2_Hz: float,
                 Gamma_hom_rad_s: Optional[float] = None,
                 C_max: Optional[float] = None) -> float:
    """
    CPT resonance contrast C (dimensionless, 0 < C < 1).

    From the 3-level density-matrix solution in the weak-field regime
    [Wynands1999 eq. 8; Vanier2005 sec. 5.2]:

        C ≈ Ω² / (2·Γ_hom·γ₂)           (unsaturated limit)

    Saturated form (valid at all powers):

        C = C_max · Ω² / (Ω² + 2·Γ_hom·γ₂)

    C_max ceiling depends on cell type and polarisation scheme:
      - CSAC small cell:        C_max ≈ 0.10  [Knappe2004]
      - Lab macroscopic cell:   C_max ≈ 0.20  [Boudot2011]
      - Anti-relaxation coated: C_max ≈ 0.25  [Levi2000]

    Parameters
    ----------
    Omega_rad_s : float
        Single-sideband Rabi frequency [rad/s].
    gamma2_Hz : float
        Ground-state coherence relaxation rate [Hz].  Used to compute the
        saturation denominator; coupling the contrast to the actual cell and
        temperature via the full cell model.
    Gamma_hom_rad_s : float, optional
        Homogeneous excited-state decay rate [rad/s].  Defaults to Γ_D1.
    C_max : float, optional
        Empirical contrast ceiling (0–1).  Defaults to 0.10 (CSAC regime).

    Returns
    -------
    float
        Signal contrast (dimensionless).
    """
    if Gamma_hom_rad_s is None:
        Gamma_hom_rad_s = Gamma_D1

    C_max_val = C_max if C_max is not None else 0.10
    gamma2_rad = 2.0 * math.pi * gamma2_Hz
    sat_term = 2.0 * Gamma_hom_rad_s * gamma2_rad   # [rad²/s²]
    denom = Omega_rad_s ** 2 + sat_term
    return C_max_val * Omega_rad_s ** 2 / denom if denom > 0 else 0.0


def doppler_linewidth_Hz(T_K: float) -> float:
    """
    Doppler FWHM of the Rb D1 optical transition [Hz].

        Δν_D = (1/λ) · sqrt(8·ln(2)·k_B·T / m)

    Parameters
    ----------
    T_K : float
        Temperature [K].

    Returns
    -------
    float
        Doppler FWHM [Hz].
    """
    return (1.0 / lambda_D1) * math.sqrt(8.0 * math.log(2.0) * kB * T_K / m_Rb)


def absorption_od(T_K: float, L_m: float, Gamma_buffer_rad_s: float = 0.0,
                 P_N2_Torr: float = 0.0, P_Ar_Torr: float = 0.0) -> float:
    """
    Beer-Lambert optical depth α·L for the Rb D1 transition.

    The peak absorption coefficient is [Steck2021 eq. 5; Dem tröder]:

        α₀ = n · σ₀

    where the Doppler-averaged peak cross-section for a Gaussian velocity
    distribution is:

        σ₀ = (λ²/2π) · (Γ_nat/2) · sqrt(π/ln2) / Δν_D

    This is the correct Gaussian lineshape peak, consistent with the normalised
    Voigt profile in the Doppler-broadened (Γ_nat ≪ Δν_D) limit.
    In a buffer-gas cell the homogeneous linewidth Γ_hom = Γ_nat + Γ_press + Γ_quench
    replaces Γ_nat in the Voigt kernel width.

    If P_N2_Torr / P_Ar_Torr are provided, Γ_hom is computed automatically via
    optical_homogeneous_rate().  Otherwise Gamma_buffer_rad_s is used directly
    (backward-compatible).

    Parameters
    ----------
    T_K : float
        Cell temperature [K].
    L_m : float
        Cell length [m].
    Gamma_buffer_rad_s : float
        Extra homogeneous (Lorentzian) broadening from buffer-gas collisions
        [rad/s].  Used only if P_N2_Torr = P_Ar_Torr = 0 (backward-compat).
    P_N2_Torr, P_Ar_Torr : float
        Buffer-gas partial pressures [Torr].  When non-zero, the total
        homogeneous rate is computed from optical_homogeneous_rate().

    Returns
    -------
    float
        Optical depth (dimensionless, >0).
    """
    n = number_density(T_K)
    nu_D = doppler_linewidth_Hz(T_K)         # Gaussian FWHM [Hz]

    # Determine homogeneous decay rate
    if P_N2_Torr > 0 or P_Ar_Torr > 0:
        Gamma_hom = optical_homogeneous_rate(P_N2_Torr, P_Ar_Torr, T_K)
    else:
        Gamma_hom = Gamma_D1 + Gamma_buffer_rad_s   # backward-compatible

    # The area under the absorption cross section is strictly proportional to the
    # NATURAL linewidth (Gamma_D1), because buffer gas collisions only dephase
    # atoms without creating new oscillator strength.
    # We use an effective Voigt FWHM to properly capture pressure broadening.
    nu_eff = math.sqrt(nu_D**2 + (Gamma_hom / (2.0 * math.pi))**2)

    sigma0 = (lambda_D1 ** 2 / (2.0 * math.pi)) * \
             (Gamma_D1 / (2.0 * math.pi)) * math.sqrt(math.pi / math.log(2.0)) / nu_eff
    return n * sigma0 * L_m

# Backward-compatible alias (old name used in test_physics.py)
absorption_cross_section = absorption_od


def mean_intensity_factor(params: dict) -> tuple:
    """
    Calculate the off-resonance (background) Beer-Lambert transmission T_bg
    and the spatially-averaged intensity factor inside the cell.

    Returns
    -------
    (T_bg, intensity_factor)
        T_bg             : exp(-αL), off-resonance transmission (0–1)
        intensity_factor : (1 − T_bg)/(αL), mean intensity fraction (0–1)
    """
    T_K  = params["T_K"]
    L_m  = params.get("cell_L_mm", 20.0) * 1e-3
    P_N2 = params.get("P_N2_Torr", 0.0)
    P_Ar = params.get("P_Ar_Torr", 0.0)

    # Prefer pressure-based computation; fall back to explicit Gamma_buffer_rad_s
    if P_N2 > 0 or P_Ar > 0:
        alpha_L = absorption_od(T_K, L_m, P_N2_Torr=P_N2, P_Ar_Torr=P_Ar)
    else:
        # Legacy path: explicit Gamma_buffer_rad_s parameter or zero
        Gamma_buf = params.get("Gamma_buffer_rad_s", 0.0)
        alpha_L = absorption_od(T_K, L_m, Gamma_buffer_rad_s=Gamma_buf)

    if alpha_L < 1e-4:
        return (1.0, 1.0)

    T_bg = math.exp(-alpha_L)
    # Spatial average: (1/L) ∫₀ᴸ exp(−α·z) dz = (1 − T_bg)/(α·L)
    intensity_factor = (1.0 - T_bg) / alpha_L
    return (T_bg, intensity_factor)


def lineshape(delta_Hz: np.ndarray, gamma_CPT_Hz: float,
              contrast: float, B_G: float = 0.0, T_bg: float = 1.0) -> np.ndarray:
    """
    Proper Lorentzian CPT dark-resonance lineshape on Beer-Lambert background.

    Derived from the density-matrix steady-state solution for a 3-level Λ
    system [Wynands1999; Vanier2005 eq. 5.15]:

        T(δ, B) = T_bg · [ 1 − C · Σ_j w_j · (γ/2)² / ((δ−j·ξ·B)² + (γ/2)²) ]

    The three magnetic sub-resonances (j = −1, 0, +1) are summed incoherently,
    valid for Zeeman splitting ξ·B ≫ γ_CPT.  Weights w_j are normalised so
    that Σ w_j = 1 (the central m_F=0 component carries half the oscillator
    strength; the two flanking ΔmF=±1 carry a quarter each for lin‖lin
    polarisation).

    Parameters
    ----------
    delta_Hz : np.ndarray
        Two-photon detuning from the unperturbed clock resonance [Hz].
        Positive = red of resonance.
    gamma_CPT_Hz : float
        CPT resonance FWHM [Hz].
    contrast : float
        Peak-to-background contrast C (0–1).
    B_G : float
        Applied static magnetic field [Gauss].
    T_bg : float
        Off-resonance (background) transmission (0–1).

    Returns
    -------
    np.ndarray
        Normalised transmitted power at each detuning.
    """
    # Zeeman coefficient (Steck2021-derived, see module header)
    xi_Hz_G = _XI_Hz_G

    # Incoherent Zeeman weights, normalised to unity (lin‖lin polarisation)
    # j=0 (clock transition): weight 0.5
    # j=±1 (ΔmF=±1 sidebands): weight 0.25 each
    w = np.array([0.25, 0.50, 0.25])
    j_vals = np.array([-1, 0, 1])

    half = gamma_CPT_Hz / 2.0

    is_scalar = np.isscalar(delta_Hz)
    delta_arr = np.atleast_1d(np.asarray(delta_Hz, dtype=float))

    # Weighted sum of Lorentzian dips
    L_sum = np.zeros_like(delta_arr)
    for j, wj in zip(j_vals, w):
        center = j * xi_Hz_G * B_G
        L_sum += wj * half ** 2 / ((delta_arr - center) ** 2 + half ** 2)

    # Transmitted power: T_bg · (1 − C · L_sum)
    # Clamp to ensure physical positivity
    result = T_bg * np.maximum(1.0 - contrast * L_sum, 0.0)
    if is_scalar:
        return float(result[0])
    return result


def discriminator_slope(gamma_CPT_Hz: float, contrast: float,
                        T_bg: float = 1.0) -> float:
    """
    Maximum slope |dT/dδ| of the CPT lineshape, used as the FM discriminator.

    For a Lorentzian dip T(δ) = T_bg·[1 − C·(γ/2)²/(δ²+(γ/2)²)],
    the slope is maximum at δ = ±γ/2 and equals [Vanier2005 sec. 5.3]:

        |dT/dδ|_max = C · T_bg / (2·γ_CPT)   [1/Hz]

    Parameters
    ----------
    gamma_CPT_Hz : float
        CPT FWHM [Hz].
    contrast : float
        Signal contrast.
    T_bg : float
        Background transmission.

    Returns
    -------
    float
        Discriminator slope [1/Hz].
    """
    return contrast * T_bg / (2.0 * gamma_CPT_Hz)


def compute_cpt_signal(params: dict) -> dict:
    """
    High-level convenience function: compute all CPT signal quantities.

    Parameters
    ----------
    params : dict
        Must contain:
            T_K          : cell temperature [K]
            P_total_uW   : total laser power [μW]
            mod_index    : VCSEL modulation index m
            beam_diam_mm : beam diameter [mm]
            gamma2_Hz    : total ground-state decoherence rate [Hz]
        Optional:
            P_N2_Torr       : N₂ partial pressure [Torr]  (auto-computes Γ_hom)
            P_Ar_Torr       : Ar partial pressure [Torr]  (auto-computes Γ_hom)
            Gamma_hom_rad_s  : explicit homogeneous excited-state decay rate [rad/s]
                               (used only if P_N2/P_Ar not given, falls back to Γ_D1)
            cell_L_mm       : cell length [mm] (for Beer-Lambert OD), default 20
            cell_type       : 'csac' | 'lab' | 'macro' — sets C_max empirically
                               'csac'  : small CSAC cell, C_max ≈ 0.10  [Knappe2004]
                               'lab'   : lab macroscopic cell, C_max ≈ 0.20  [Boudot2011]
                               'macro' : large cell / coated, C_max ≈ 0.25  [Levi2000]
                               default : 'csac'

    Returns
    -------
    dict
        Omega_rad_s, gamma_CPT_Hz, contrast, disc_slope_1_Hz, T_bg,
        Gamma_hom_rad_s (for diagnostics)
    """
    from vcsel_model import sideband_powers

    T_K      = params["T_K"]
    P_uW     = params["P_total_uW"]
    m        = params["mod_index"]
    d_mm     = params["beam_diam_mm"]
    gamma2   = params["gamma2_Hz"]

    # ---- Determine total homogeneous excited-state decay rate ----
    P_N2 = params.get("P_N2_Torr", 0.0)
    P_Ar = params.get("P_Ar_Torr", 0.0)
    if P_N2 > 0 or P_Ar > 0:
        # Auto-compute: natural + pressure broadening + N2 quenching
        Gamma_hom_total = optical_homogeneous_rate(P_N2, P_Ar, T_K)
    else:
        # Legacy / explicit path
        Gamma_hom_base = params.get("Gamma_hom_rad_s", Gamma_D1)
        Gamma_buf      = params.get("Gamma_buffer_rad_s", 0.0)
        Gamma_hom_total = Gamma_hom_base + Gamma_buf

    # ---- C_max depends on cell regime ----
    cell_type = params.get("cell_type", "csac")
    _C_max_map = {"csac": 0.10, "lab": 0.20, "macro": 0.25}
    C_max = _C_max_map.get(cell_type, 0.10)

    beam_area = math.pi * (d_mm * 1e-3 / 2.0) ** 2   # [m²]

    # Background transmission and mean intensity inside cell
    T_bg, i_factor = mean_intensity_factor(params)

    # Power in k=+1 sideband [W]
    pw   = sideband_powers(m, P_uW * 1e-6)
    P_sb = pw.get(1, 0.0)

    # Effective Rabi frequency weighted by mean intensity factor
    Omega = rabi_frequency(P_sb, beam_area) * math.sqrt(i_factor)

    lw    = cpt_linewidth_Hz(gamma2, Omega, Gamma_hom_rad_s=Gamma_hom_total)
    C     = cpt_contrast(Omega, gamma2, Gamma_hom_rad_s=Gamma_hom_total,
                         C_max=C_max)
    slope = discriminator_slope(lw, C, T_bg=T_bg)

    return {
        "Omega_rad_s":      Omega,
        "gamma_CPT_Hz":     lw,
        "contrast":         C,
        "disc_slope_1_Hz":  slope,
        "T_bg":             T_bg,
        "Gamma_hom_rad_s":  Gamma_hom_total,
    }
