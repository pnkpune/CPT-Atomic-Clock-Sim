"""
rb_vapor.py
===========
Rb-87 vapour pressure and atom number density as a function of temperature.

References
----------
[Steck2021]  D. A. Steck, "Rubidium 87 D Line Data", revision 2.2.2 (2021).
             Table 2, Nesmeyanov vapour pressure relation.
[Alcock1984] C. B. Alcock, V. P. Itkin, M. K. Horrigan, Can. Metall. Q. 23,
             309 (1984). — cross-check for alkali vapour pressures.
"""

import math
from constants import (
    VP_A_solid, VP_B_solid, VP_A_liquid, VP_B_liquid,
    kB, m_Rb
)

# Rb melting point
T_melt_Rb = 312.46   # [K]   [Steck2021]

# ─────────────────────────────────────────────────────────────────────────────

def vapour_pressure_Pa(T_K: float) -> float:
    """
    Return Rb vapour pressure in Pascal.

    Uses Nesmeyanov relation from [Steck2021], Table 2:
        log10(P / Pa) = A  -  B / T

    Valid for T > 200 K.

    Parameters
    ----------
    T_K : float
        Temperature in Kelvin.

    Returns
    -------
    float
        Vapour pressure [Pa].
    """
    if T_K < T_melt_Rb:
        log10_P = VP_A_solid  - VP_B_solid  / T_K
    else:
        log10_P = VP_A_liquid - VP_B_liquid / T_K
    return 10.0 ** log10_P


def number_density(T_K: float) -> float:
    """
    Rb atom number density assuming vapour in thermal equilibrium with
    liquid/solid.  Uses ideal gas: n = P / (kB * T).

    Parameters
    ----------
    T_K : float
        Temperature in Kelvin.

    Returns
    -------
    float
        Number density [m⁻³].
    """
    P = vapour_pressure_Pa(T_K)
    return P / (kB * T_K)


def mean_speed(T_K: float) -> float:
    """
    Mean thermal speed of Rb atoms: v̄ = sqrt(8 kB T / (π m))  [Reif].

    Parameters
    ----------
    T_K : float
        Temperature in Kelvin.

    Returns
    -------
    float
        Mean speed [m/s].
    """
    return math.sqrt(8.0 * kB * T_K / (math.pi * m_Rb))


# ─── quick sanity checks ─────────────────────────────────────────────────────
if __name__ == "__main__":
    for T_C in [25, 40, 60, 70, 80]:
        T = T_C + 273.15
        n = number_density(T)
        print(f"T = {T_C:4d} °C  →  n = {n:.3e} m⁻³  ({n*1e-6:.3e} cm⁻³)")
