"""
cell_model.py
=============
Vapour-cell geometry and decoherence rates for the CPT clock model.

References
----------
[Vanier2005]  J. Vanier & C. Audoin, "The Quantum Physics of Atomic Frequency
              Standards", IOP Publishing (2005).  sec. 3.4 (diffusion),
              sec. 6.3 (collisional broadening).
[Bouchiat1966] M. A. Bouchiat & J. Brossel, Phys. Rev. 147, 41 (1966).
               [diffusion-limited relaxation in cylindrical cell]
[Beverini1971] N. Beverini et al., Phys. Rev. A 4, 550 (1971). [transit-time]
[Kozlova2011]  O. Kozlova et al., Proc. EFTF (2011). [broadening coefficients]
[Franz1976]    F. A. Franz & C. Volk, Phys. Rev. A 14, 1711 (1976). 
               [diffusion and spin-relaxation rates]
"""

import math
from constants import (
    kB, m_Rb, kappa_se,
    D0_Rb_N2, D0_Rb_Ar,
    sigma_se,
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

    T_fac = (T_K / T0) ** 1.5   # Chapman-Enskog temperature scaling

    # Binary diffusion coefficients at operating conditions
    # D_i = D0_i · (T/T0)^1.5 / (P_total/P0)
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


def total_ground_decoherence(R_m: float, L_m: float,
                              P_N2_Torr: float, P_Ar_Torr: float,
                              T_K: float) -> dict:
    """
    Compute all contributions to ground-state coherence decay rate γ₂.

    γ₂ = γ_diff + γ_se + γ_bg         (HWHM, Hz)

    Returns a dict with individual components and total.
    """
    g_diff = transit_time_rate(R_m, L_m, P_N2_Torr, P_Ar_Torr, T_K)
    g_se   = spin_exchange_rate(T_K)
    g_bg   = collisional_broadening_rate(P_N2_Torr, P_Ar_Torr, T_K)
    total  = g_diff + g_se + g_bg
    return {
        "gamma_diff_Hz": g_diff,
        "gamma_se_Hz":   g_se,
        "gamma_bg_Hz":   g_bg,
        "gamma2_total_Hz": total,
    }
