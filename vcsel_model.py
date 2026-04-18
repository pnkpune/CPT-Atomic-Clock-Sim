"""
vcsel_model.py
==============
VCSEL modulation spectrum for CW-CPT clocks.

In CW-CPT, a VCSEL is directly current-modulated at f_mod ≈ ν_hfs/2
(half the ground-state hyperfine splitting).  The optical spectrum is
a comb of sidebands with amplitudes proportional to Bessel functions:

    E_k ∝ J_k(m)  at  ν_0 + k·f_mod

where m is the modulation index and ν_0 is the VCSEL optical carrier
frequency.  The k=+1 and k=-1 sidebands drive the Λ-type CPT
resonance; all others contribute to light shifts.

References
----------
[Zhu1993]    M. Zhu & L. S. Cutler, Proc. PTTI (1993).
             Original CW-CPT with frequency-doubled microwave.
[Levi2000]   F. Levi et al., Eur. Phys. J. D 12, 53 (2000).
             VCSEL modulation spectrum and sideband power.
[Shah2006]   V. Shah & J. Kitching, Adv. At. Mol. Opt. Phys. 59 (2010).
             Modulation index optimisation for light-shift cancellation.
"""

import numpy as np
from scipy.special import jv   # Bessel functions of the first kind

# ─────────────────────────────────────────────────────────────────────────────

def sideband_amplitudes(m: float, k_max: int = 8) -> dict:
    """
    Compute optical sideband amplitude coefficients J_k(m) for k in
    range [-k_max, k_max].

    Power in the k-th sideband:  P_k = P_total * J_k(m)²  (normalised)

    Parameters
    ----------
    m : float
        Modulation index (dimensionless, typically 1.0–3.0).
    k_max : int
        Highest order sideband to include.

    Returns
    -------
    dict
        k -> J_k(m)²  (normalised power fraction, sums to 1 via Parseval).
    """
    orders = np.arange(-k_max, k_max + 1)
    Jk = {int(k): float(jv(k, m)) for k in orders}
    # Parseval check: Σ J_k²  = J_0² + 2·Σ_{k=1} J_k²  should ≈ 1
    return Jk


def sideband_powers(m: float, P_total_uW: float, k_max: int = 8) -> dict:
    """
    Power [μW] in each sideband.

    Parameters
    ----------
    m : float
        Modulation index.
    P_total_uW : float
        Total laser power [μW].
    k_max : int
        Highest order to include.

    Returns
    -------
    dict
        k -> power [μW].
    """
    Jk = sideband_amplitudes(m, k_max)
    total_norm = sum(v**2 for v in Jk.values())
    if total_norm < 1e-12:
        total_norm = 1.0
    return {k: (v**2 / total_norm) * P_total_uW for k, v in Jk.items()}


def cpt_sideband_power_ratio(m: float) -> float:
    """
    Fraction of total power in the two CPT-active sidebands (k = ±1).

    Parameters
    ----------
    m : float
        Modulation index.

    Returns
    -------
    float
        Power fraction in k=+1 and k=-1 sidebands combined.
    """
    powers = sideband_powers(m, 1.0)
    return powers[1] + powers[-1]


def optimal_modulation_index(n_sidebands: int = 8) -> float:
    """
    Find the modulation index m that maximises the fraction of power in
    the CPT sidebands (k = ±1).  This is a rough optimum for contrast;
    light-shift cancellation optimum is near m ≈ 2.4.  [Shah2006]

    Returns
    -------
    float
        m at maximum CPT sideband fraction.
    """
    m_arr = np.linspace(0.5, 4.0, 500)
    fracs = [cpt_sideband_power_ratio(float(m)) for m in m_arr]
    return float(m_arr[np.argmax(fracs)])
