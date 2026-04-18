"""
allan_deviation.py
==================
Allan deviation computation and stability analysis.

References
----------
[Allan1966]  D. W. Allan, Proc. IEEE 54, 221 (1966). [definition]
[Vanier2005] J. Vanier & C. Audoin (2005), sec. 5.2 [clock stability metrics].
[Riley2008]  W. J. Riley, "Handbook of Frequency Stability Analysis",
             NIST Special Publication 1065 (2008). [MDEV, TDEV formulas]
"""

import numpy as np
from noise_budget import total_noise_budget


# ─────────────────────────────────────────────────────────────────────────────

def tau_array(tau_min: float = 1.0, tau_max: float = 1e6,
              points_per_decade: int = 20) -> np.ndarray:
    """
    Logarithmically-spaced averaging time array.

    Parameters
    ----------
    tau_min, tau_max : float
        Start and end averaging times [s].
    points_per_decade : int
        Number of τ points per decade.

    Returns
    -------
    np.ndarray
        Averaging times [s].
    """
    n_dec = np.log10(tau_max / tau_min)
    n_pts = max(int(n_dec * points_per_decade) + 1, 2)
    return np.logspace(np.log10(tau_min), np.log10(tau_max), n_pts)


def compute_allan_deviation(tau: np.ndarray, params: dict) -> dict:
    """
    Compute the Allan deviation (ADEV) for all noise sources and their
    quadrature sum.

    The Allan deviation σ_y(τ) for white frequency noise is:
        σ_y(τ) = sqrt(h₀ / (2·τ))           [Allan1966]

    For flicker frequency noise:
        σ_y(τ) = sqrt(2·ln(2)·h₋₁)          (τ-independent floor)

    For random-walk frequency noise (linear drift dominance):
        σ_y(τ) ∝ τ^(+1/2)

    This function computes each contribution and returns them along with
    the total.

    Parameters
    ----------
    tau : np.ndarray
        Averaging time array [s].
    params : dict
        All clock parameters dict (see gui.py).

    Returns
    -------
    dict
        Per-source and total σ_y arrays, plus slope characterisation.
    """
    budget = total_noise_budget(tau, params)

    # Identify short-term dominant source
    sigma_total = budget["sigma_total"]
    at_1s_idx   = np.argmin(np.abs(tau - 1.0))
    sigma_1s    = sigma_total[at_1s_idx]

    # Identify the Allan deviation slope at short τ (should be −1/2)
    if len(tau) >= 4:
        log_tau   = np.log10(tau[:len(tau)//2])
        log_sigma = np.log10(sigma_total[:len(tau)//2] + 1e-30)
        slope_short = float(np.polyfit(log_tau, log_sigma, 1)[0])
    else:
        slope_short = -0.5

    return {
        **budget,
        "tau":          tau,
        "sigma_1s":     sigma_1s,
        "slope_short":  slope_short,
    }


def sensitivity_table(params: dict) -> list:
    """
    Tabulate the sensitivity coefficients / limiting factors at τ = 1 s.

    Returns
    -------
    list of (name, sigma_y_at_1s) tuples, sorted by magnitude.
    """
    tau_1s = np.array([1.0])
    budget = total_noise_budget(tau_1s, params)
    results = []
    for key, val in budget.items():
        if key.startswith("sigma_") and hasattr(val, "__len__"):
            results.append((key, float(val[0])))
    results.sort(key=lambda x: -abs(x[1]))
    return results
