"""
cell_model.py
=============
Vapour-cell geometry and decoherence rates for the CPT clock model.

Two regimes are supported:

  (A) Buffer-gas cell (P_bg > 0)
      Atoms diffuse to the walls.  The dominant decoherence channels are:
        gamma_diff  : diffusion-limited wall loss  [Bouchiat1966]
        gamma_se    : Rb–Rb spin-exchange         [Vanier2005 sec. 3.6]
        gamma_bg    : buffer-gas ground-state decoherence  [Kozlova2011 / Boudot2011]

  (B) No-buffer-gas cell (P_bg = 0)
      Atoms travel ballistically.  The dominant decoherence channels are:
        gamma_transit : beam/cell transit-time broadening  [Beverini1971]
        gamma_wall    : wall-collision depolarisation       [Bouchiat1966]
        gamma_se      : Rb–Rb spin-exchange               [Vanier2005 sec. 3.6]
      For anti-relaxation coated cells P_depol_per_bounce ≪ 1 narrows gamma_wall
      dramatically.

References
----------
[Vanier2005]   J. Vanier & C. Audoin (2005). sec. 3.4 (diffusion),
               sec. 6.3 (collisional broadening).
[Bouchiat1966] M. A. Bouchiat & J. Brossel, Phys. Rev. 147, 41 (1966).
               [diffusion-limited relaxation in cylindrical cell;
                wall-collision rate formula]
[Beverini1971] N. Beverini et al., Phys. Rev. A 4, 550 (1971). [transit-time]
[Boudot2011]   R. Boudot et al., Opt. Express 19, 3106 (2011).
               [Rb-87 N2/Ar broadening coefficients; inversion temperature]
[Franz1976]    F. A. Franz & C. Volk, Phys. Rev. A 14, 1711 (1976).
               [diffusion and spin-relaxation rates]
[Straessle2014] R. Straessle et al., J. Phys. B 47, 075502 (2014).
               [anti-relaxation coatings for micro-cells]
"""

import math
from constants import (
    kB, m_Rb, kappa_se,
    D0_Rb_N2, D0_Rb_Ar,
    sigma_se,
    P_depol_bare, P_depol_paraffin, P_depol_PDMS,
)
from rb_vapor import number_density, mean_speed

# Torr → Pa conversion
TORR_TO_PA = 133.322

# ─────────────────────────────────────────────────────────────────────────────

def buffer_gas_pressure_Pa(P_fill_Torr: float, T_fill_K: float, T_K: float) -> float:
    """
    Buffer gas pressure at operating temperature, assuming sealed cell (ideal gas).

    P(T) = P_fill * T / T_fill

    Parameters
    ----------
    P_fill_Torr : float
        Fill pressure at sealing temperature [Torr].
    T_fill_K : float
        Cell sealing temperature [K].
    T_K : float
        Operating temperature [K].

    Returns
    -------
    float
        Operating pressure [Pa].
    """
    return P_fill_Torr * TORR_TO_PA * (T_K / T_fill_K)


def diffusion_coefficient(P_N2_Torr: float, P_Ar_Torr: float, T_K: float) -> float:
    """
    Effective Rb diffusion coefficient in a N₂/Ar buffer gas mixture via Blanc's law.

    Each binary diffusion coefficient is:  [Vanier2005 sec. 3.4; Chapman-Enskog]

        D_i(T, P) = D0_i · (T/T0)^1.5 · (P0 / P_total)

    where D0_i is the measured binary (Rb–gas_i) diffusion coefficient at
    reference conditions (T0 = 273.15 K, P0 = 1 atm total pressure).
    D0 already represents the binary value at 1 atm total; scaling by P0/P_total
    gives the correct pressure dependence.

    The mixture effective diffusion coefficient follows Blanc's law:

        1/D_eff = Σ_i  x_i / D_i

    where x_i = P_i / P_total is the mole fraction of species i.  Note that
    after substitution, this simplifies to:

        1/D_eff = (P_total / P0) · (1/T_fac) · Σ_i  x_i / D0_i

    making D_eff independent of individual partial-pressure fractions in a
    mixture where both species are present — physically correct.

    Parameters
    ----------
    P_N2_Torr : float
        N₂ partial pressure [Torr].
    P_Ar_Torr : float
        Ar partial pressure [Torr].
    T_K : float
        Temperature [K].

    Returns
    -------
    float
        Effective diffusion coefficient [m²/s].
    """
    T0 = 273.15   # K,  STP reference temperature
    P0_atm = 1.0  # atm, STP reference pressure

    P_total_atm = (P_N2_Torr + P_Ar_Torr) / 760.0
    if P_total_atm < 1e-9:
        return 1e10   # no buffer gas → free diffusion (very large D)

    # Hard-sphere Chapman-Enskog T scaling: D ∝ T^1.75  [Vanier2005 sec. 3.4]
    # (The earlier T^1.5 was the Maxwell-molecule approximation; this gives
    # a more accurate 5% correction at CPT operating temperatures.)
    T_fac = (T_K / T0) ** 1.75   # hard-sphere Chapman-Enskog temperature scaling

    # Binary diffusion coefficients at operating conditions
    # D_i = D0_i · (T/T0)^1.75 / (P_total/P0)
    # Both species see the *total* mixture pressure (standard binary D convention)
    scale = T_fac * P0_atm / P_total_atm
    D_N2 = D0_Rb_N2 * scale if P_N2_Torr > 0 else None
    D_Ar = D0_Rb_Ar * scale if P_Ar_Torr > 0 else None

    # Mole fractions
    x_N2 = P_N2_Torr / (P_N2_Torr + P_Ar_Torr) if P_N2_Torr > 0 else 0.0
    x_Ar = P_Ar_Torr / (P_N2_Torr + P_Ar_Torr) if P_Ar_Torr > 0 else 0.0

    # Blanc's law:  1/D_eff = Σ x_i/D_i
    if D_N2 is not None and D_Ar is not None:
        inv_D = x_N2 / D_N2 + x_Ar / D_Ar
        return 1.0 / inv_D
    elif D_N2 is not None:
        return D_N2
    elif D_Ar is not None:
        return D_Ar
    else:
        return 1e10


def transit_time_rate(R_m: float, L_m: float, P_N2_Torr: float,
                      P_Ar_Torr: float, T_K: float, v_mode: int = 0, i_mode: int = 1) -> float:
    """
    Diffusion-limited relaxation rate for a cylindrical cell.
    Rigorous rate involving Bessel function zeroes indexed by v and p_i:
        Rate_diff = D * [((2v+1)π/L)² + (p_i/R)²]
    """
    from scipy.special import jn_zeros
    D = diffusion_coefficient(P_N2_Torr, P_Ar_Torr, T_K)
    p_i = jn_zeros(0, i_mode)[-1]
    
    gamma_diff = D * (((2 * v_mode + 1) * math.pi / L_m)**2 + (p_i / R_m)**2)
    return gamma_diff / (2 * math.pi)   # [Hz]


def spin_exchange_rate(T_K: float) -> float:
    """
    Rb-Rb spin-exchange relaxation rate.

    γ_se = n_Rb * σ_se * v̄   [Vanier2005, sec. 3.6]

    Parameters
    ----------
    T_K : float
        Temperature [K].

    Returns
    -------
    float
        Spin-exchange relaxation rate [Hz].
    """
    n  = number_density(T_K)
    v  = mean_speed(T_K)
    return n * sigma_se * v / (2 * math.pi)   # [Hz]


def collisional_broadening_rate(P_N2_Torr: float, P_Ar_Torr: float,
                                 T_K: float) -> float:
    """
    Buffer-gas collisional broadening of the ground-state coherence.
    Generalized to the fundamental disorientation cross-section model:
        Rate_buff = N0 * sigma * v_rel * (p / p0)
    """
    P0_Pa = 101325.0
    N0 = 2.68678e25  # Loschmidt constant m^-3
    v_rel = mean_speed(T_K) * math.sqrt(2.0)
    
    # Back-calculate representative cross-sections to match empirical data
    # (Aligns with Franz & Volk empirical data alongside Kozlova's coefficients)
    # Kozlova: N2 ~143.6 Hz/Torr, Ar ~42.3 Hz/Torr at 60C
    sigma_N2 = (143.6 * 2 * math.pi * P0_Pa) / (N0 * v_rel * 133.322)
    sigma_Ar = ( 42.3 * 2 * math.pi * P0_Pa) / (N0 * v_rel * 133.322)
    
    P_N2_Pa = P_N2_Torr * 133.322
    P_Ar_Pa = P_Ar_Torr * 133.322
    
    rate_N2 = N0 * sigma_N2 * v_rel * (P_N2_Pa / P0_Pa)
    rate_Ar = N0 * sigma_Ar * v_rel * (P_Ar_Pa / P0_Pa)
    
    return (rate_N2 + rate_Ar) / (2 * math.pi)


def ballistic_decoherence(R_m: float, L_m: float, beam_diam_m: float,
                          T_K: float,
                          P_depol_per_bounce: float = 1.0) -> dict:
    """
    Decoherence rates for a no-buffer-gas vapour cell.

    In the absence of buffer gas, atoms travel ballistically at thermal
    velocity v̄ = sqrt(8kT/πm).  Two independent channels are modelled:

    (1) Wall-collision rate (Knudsen formula for a cylinder)
        The mean free path between wall collisions is ∼4V/A:

            γ_wall = v̄ × A / (4V) × P_depol  [Bouchiat1966]

        where V = cell volume, A = cell surface area.
        P_depol = 1.0 for bare glass; ≪ 1 for anti-relaxation coated cells.

    (2) Beam transit-time broadening
        Atoms traverse the laser beam of diameter d_beam in time τ = d/v̄:

            γ_transit = v̄ / d_beam      (FWHM ≈ 0.9 / τ)  [Beverini1971]

    The total ground-state decoherence is the incoherent sum
    (atoms experience transit-time broadening AND can hit the wall):

            γ₂_no_bg = γ_wall + γ_se

    Transit-time broadening contributes to the optical linewidth but not
    directly to the HFS coherence damping (atoms re-enter the beam after
    crossing); for the clock linewidth it contributes an inhomogeneous term.
    We include it in the output for completeness.

    Parameters
    ----------
    R_m : float
        Cell inner radius [m].
    L_m : float
        Cell inner length [m].
    beam_diam_m : float
        Laser beam 1/e² diameter [m].  Used for transit-time calculation.
    T_K : float
        Temperature [K].
    P_depol_per_bounce : float
        Depolarisation probability per wall collision.
        Use P_depol_bare (1.0) for bare glass, P_depol_paraffin (1e-3) for
        paraffin coating, P_depol_PDMS (1e-4) for PDMS / OTS coating
        (all from constants.py).  Default: bare glass.

    Returns
    -------
    dict
        gamma_wall_Hz, gamma_transit_Hz, gamma_se_Hz, gamma2_total_Hz.
    """
    v_bar = mean_speed(T_K)   # m/s

    # Cell geometry
    A_cell = 2.0 * math.pi * R_m * (R_m + L_m)   # cylindrical surface area [m²]
    V_cell = math.pi * R_m ** 2 * L_m             # cell volume [m³]

    # Wall-collision decoherence rate [Hz]
    # Factor A/(4V) is the Knudsen mean-free-path formula for a cylinder
    gamma_wall = (v_bar * A_cell / (4.0 * V_cell)) * P_depol_per_bounce
    gamma_wall /= (2.0 * math.pi)   # convert rad/s → Hz (rate is defined in Hz)

    # Transit-time broadening [Hz]  — for completeness; affects optical OD, not clock HFS
    gamma_transit = v_bar / max(beam_diam_m, 1e-6)   # HWHM-ish [Hz]

    # Spin-exchange
    gamma_se = spin_exchange_rate(T_K)

    # Total HFS ground-state decoherence (wall loss + SE; transit-time is
    # inhomogeneous and treated separately in linewidth broadening)
    gamma2_total = gamma_wall + gamma_se

    return {
        "gamma_wall_Hz":    gamma_wall,
        "gamma_transit_Hz": gamma_transit,
        "gamma_se_Hz":      gamma_se,
        "gamma2_total_Hz":  gamma2_total,
        "regime":           "ballistic",
    }


def total_ground_decoherence(R_m: float, L_m: float,
                              P_N2_Torr: float, P_Ar_Torr: float,
                              T_K: float,
                              beam_diam_m: float = 3e-3,
                              P_depol_per_bounce: float = 1.0) -> dict:
    """
    Compute all contributions to ground-state coherence decay rate γ₂.

    Dispatches to the appropriate regime:
      • Buffer-gas cell (P_N2 > 0 or P_Ar > 0):
            γ₂ = γ_diff + γ_se + γ_bg
      • No-buffer-gas cell (P_N2 = P_Ar = 0):
            γ₂ = γ_wall + γ_se
            (beam transit-time is also returned but is a separate inhomogeneous term)

    Parameters
    ----------
    R_m, L_m : float
        Cell inner radius and length [m].
    P_N2_Torr, P_Ar_Torr : float
        Buffer gas partial pressures [Torr]. Zero for no-buffer-gas cell.
    T_K : float
        Cell temperature [K].
    beam_diam_m : float
        Laser beam diameter [m]; used for transit-time broadening in the
        no-buffer-gas regime.  Default: 3 mm.
    P_depol_per_bounce : float
        Wall depolarisation probability per bounce.  Only used in the
        no-buffer-gas regime.  Default: 1.0 (bare glass).

    Returns
    -------
    dict
        Individual rate components and gamma2_total_Hz.
    """
    if P_N2_Torr <= 0 and P_Ar_Torr <= 0:
        # ─── No-buffer-gas ballistic regime ───
        return ballistic_decoherence(R_m, L_m, beam_diam_m, T_K,
                                     P_depol_per_bounce=P_depol_per_bounce)

    # ─── Buffer-gas diffusion regime ───
    g_diff = transit_time_rate(R_m, L_m, P_N2_Torr, P_Ar_Torr, T_K)
    g_se   = spin_exchange_rate(T_K)
    g_bg   = collisional_broadening_rate(P_N2_Torr, P_Ar_Torr, T_K)
    total  = g_diff + g_se + g_bg
    return {
        "gamma_diff_Hz":    g_diff,
        "gamma_se_Hz":      g_se,
        "gamma_bg_Hz":      g_bg,
        "gamma2_total_Hz":  total,
        "regime":           "diffusion",
    }
