pyenv rehash"""
noise_budget.py
===============
Noise power spectral densities (PSDs) and frequency instability contributions
for a CW-CPT ⁸⁷Rb atomic clock.

All noise sources produce a fractional frequency instability σ_y(τ).  For white
frequency noise: σ_y(τ) = h₀^(1/2) · τ^(-1/2) where h₀ is the one-sided PSD.

References
----------
[Vanier2005] J. Vanier & C. Audoin (2005), sec. 5.3, 6.4. [general noise]
[Wynands1999] R. Wynands & A. Nagel, Appl. Phys. B 68 (1999). [shot noise]
[Santarelli1998] G. Santarelli et al., IEEE Trans. UFFC 45, 887 (1998).
                 [Dick effect formula]
[Audoin1998]  C. Audoin et al., IEEE Trans. UFFC 45, 877 (1998).
              [FM-AM noise conversion]
[Knappe2004]  S. Knappe et al., Opt. Lett. 29, 388 (2004).  [CSAC noise floor]
[Marlow2021]  J. Marlow & J. Scherer, MITRE Tech. Rep. (2021). [drift prediction]
[Camparo1999] J. C. Camparo & J. G. Coffer, Phys. Rev. A 59, 728 (1999). [PM-AM]
"""

import math
import numpy as np
from constants import e, nu_hfs, Gamma_D1


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: convert noise source to Allan deviation floor
# For white phase/frequency noise → σ_y(τ) = A·τ^(-1/2)
# ─────────────────────────────────────────────────────────────────────────────

def _white_noise_sigma(h0: float, tau: np.ndarray) -> np.ndarray:
    """σ_y(τ) = sqrt(h0 / (2·τ))  for white frequency noise PSD h0."""
    return np.sqrt(h0 / (2.0 * tau))


# ─────────────────────────────────────────────────────────────────────────────
# 1. PHOTON SHOT NOISE  [Wynands1999, eq. 12]
# ─────────────────────────────────────────────────────────────────────────────

def shot_noise_sigma(
    tau: np.ndarray,
    P_det_W: float,
    gamma_CPT_Hz: float,
    contrast: float,
    eta_QE: float = 0.7,
    lambda_m: float = 795e-9,
) -> np.ndarray:
    """
    Fractional frequency instability from photon shot noise.

    The shot-noise-limited signal-to-noise ratio per unit bandwidth:
        SNR_shot = C · P_det / sqrt(2 · e_rate · Δf)
                 = C · P_det / sqrt(2 · P_det / (η·hν) · Δf)

    Combined with the discriminator slope D = C / (2·γ_CPT) [Hz⁻¹]:

        σ_y,shot(τ) = (1/ν₀) · (γ_CPT/2) · sqrt(hν / (η·P_det)) · τ^(-1/2)

    [Wynands1999, eq. 12; Vanier2005, sec. 5.3.2]

    Parameters
    ----------
    P_det_W : float
        Detected optical power (background, off-resonance) [W].
    gamma_CPT_Hz : float
        CPT resonance FWHM [Hz].
    contrast : float
        Signal contrast (dimensionless).
    eta_QE : float
        Photodetector quantum efficiency.
    lambda_m : float
        Optical wavelength [m].

    Returns
    -------
    np.ndarray
        σ_y,shot(τ).
    """
    h_nu   = 6.626e-34 * 3e8 / lambda_m   # photon energy [J]
    # Noise coefficient [s^(1/2)]
    if contrast < 1e-9 or P_det_W < 1e-20:
        return np.zeros_like(tau, dtype=float)
    A_shot = (1.0 / nu_hfs) * (gamma_CPT_Hz / 2.0) * \
             math.sqrt(h_nu / (eta_QE * P_det_W)) / contrast
    return A_shot * tau ** (-0.5)


# ─────────────────────────────────────────────────────────────────────────────
# 2. LASER RELATIVE INTENSITY NOISE (RIN)  [Vanier2005, sec. 5.3.3]
# ─────────────────────────────────────────────────────────────────────────────

def rin_noise_sigma(
    tau: np.ndarray,
    RIN_dBHz: float,
    gamma_CPT_Hz: float,
    contrast: float,
) -> np.ndarray:
    """
    Fractional frequency instability from laser RIN.

    RIN contributes an intensity noise floor that limits the ability to
    resolve the CPT dip.  For a discriminator-based servo:

        σ_y,RIN(τ) = (1/ν₀) · (γ_CPT/2) · sqrt(RIN) · τ^(-1/2)   [Vanier2005]

    Parameters
    ----------
    RIN_dBHz : float
        Laser one-sided RIN [dBc/Hz] (typically −140 to −120 dBc/Hz).
    gamma_CPT_Hz : float
        CPT FWHM [Hz].
    contrast : float
        Signal contrast.

    Returns
    -------
    np.ndarray
        σ_y,RIN(τ).
    """
    RIN_lin = 10.0 ** (RIN_dBHz / 10.0)   # linear [1/Hz]
    if contrast < 1e-9:
        return np.zeros_like(tau, dtype=float)
    A_rin = (1.0 / nu_hfs) * (gamma_CPT_Hz / 2.0) * \
            math.sqrt(RIN_lin) / contrast
    return A_rin * tau ** (-0.5)


# ─────────────────────────────────────────────────────────────────────────────
# 3. LO FM NOISE  (microwave LO frequency noise via CPT discriminator)
#    [Audoin1998, Vanier2005 §5.3]
# ─────────────────────────────────────────────────────────────────────────────

def fm_noise_sigma(
    tau: np.ndarray,
    LO_phase_noise_dBcHz: float,
    gamma_CPT_Hz: float,
    contrast: float,
    servo_bw_Hz: float = 10.0,
) -> np.ndarray:
    """
    Fractional frequency instability from microwave LO frequency noise
    leaking through the servo loop and converting to clock error via the
    CPT discriminator slope [Audoin1998, Vanier2005].

    Physical derivation
    -------------------
    The microwave local oscillator (LO) that drives the VCSEL modulation
    at ν_hfs/2 has a single-sideband phase noise L(f) [dBc/Hz].  At the
    servo modulation offset f_m, the one-sided phase noise PSD is:

        S_φ(f_m) = 10^(L(f_m) / 10)   [rad²/Hz]

    This phase noise converts to fractional frequency noise through the
    CPT discriminator slope D = C / (2·γ_CPT):

        σ_y,FM(τ) = (γ_CPT / (2·C·ν₀)) · √S_φ(f_m) · τ^(-1/2)

    IMPORTANT: This term is driven by the MICROWAVE LO, not the VCSEL
    optical linewidth.  The VCSEL optical linewidth (typically 50–100 MHz)
    drives the PM-to-AM conversion noise (Camparo effect), handled
    separately by laser_pm_am_noise_sigma().

    Parameters
    ----------
    LO_phase_noise_dBcHz : float
        Single-sideband phase noise of the LO at the servo offset
        frequency [dBc/Hz].  Typical values: -90 to -130 dBc/Hz.
    gamma_CPT_Hz : float
        CPT resonance FWHM [Hz].
    contrast : float
        Signal contrast C (0–1).
    servo_bw_Hz : float
        Servo modulation frequency offset [Hz] (default 10 Hz).
        Used only for documentation; the phase noise value should
        already be specified at this offset.

    Returns
    -------
    np.ndarray
        σ_y,FM(τ).
    """
    tau = np.asarray(tau, dtype=float)
    if contrast < 1e-9 or gamma_CPT_Hz < 1e-9:
        return np.zeros_like(tau, dtype=float)
    S_phi = 10.0 ** (LO_phase_noise_dBcHz / 10.0)   # rad²/Hz
    # Noise coefficient:  A_fm = (γ_CPT / (2·C·ν₀)) · √S_φ
    A_fm = (gamma_CPT_Hz / (2.0 * contrast * nu_hfs)) * math.sqrt(S_phi)
    return A_fm * tau ** (-0.5)


# ─────────────────────────────────────────────────────────────────────────────
# 3b. LASER PM-AM NOISE CONVERSION  [Camparo1999]
# ─────────────────────────────────────────────────────────────────────────────

def laser_pm_am_noise_sigma(
    tau: np.ndarray,
    linewidth_Hz: float,
    optical_detuning_error_Hz: float,
    doppler_width_Hz: float,
    gamma_CPT_Hz: float,
    contrast: float,
) -> np.ndarray:
    """
    Computes the fractional frequency instability due to the conversion of
    laser phase noise (linewidth) to amplitude noise (PM-AM conversion).

    References
    ----------
    [Camparo1999] J. C. Camparo & J. G. Coffer, Phys. Rev. A 59, 728 (1999).
    """
    from constants import nu_hfs
    tau = np.asarray(tau, dtype=float)

    if contrast < 1e-9 or doppler_width_Hz < 1e3:
        return np.zeros_like(tau)

    # Calculate the normalized slope of the Doppler-broadened Gaussian absorption
    # evaluated at the slight optical detuning error.
    # d(T)/d(nu) / T  ≈  -(4 * ln(2) * detuning) / (doppler_width^2)
    slope_factor = abs(4.0 * math.log(2) * optical_detuning_error_Hz / (doppler_width_Hz ** 2))

    # PM-AM effective RIN density (one-sided)
    S_RIN_pm_am = (slope_factor ** 2) * (linewidth_Hz / math.pi)

    # Convert to Allan deviation
    h0_pm_am = (gamma_CPT_Hz / (2.0 * contrast * nu_hfs))**2 * S_RIN_pm_am

    return np.sqrt(h0_pm_am) * tau**(-0.5)


# ─────────────────────────────────────────────────────────────────────────────
# 4. DICK EFFECT (LOCAL OSCILLATOR PHASE NOISE)  [Santarelli1998]
# ─────────────────────────────────────────────────────────────────────────────

def dick_effect_sigma(
    tau: np.ndarray,
    LO_phase_noise_dBcHz: float,
    f_servo_Hz: float = 10.0,
) -> np.ndarray:
    """
    Dick effect: LO phase noise aliased at servo modulation frequency.

    For CW-CPT with a sinusoidal FM servo at frequency f_servo:

        σ_y,Dick(τ) ≈ (1/ν₀) · sqrt(S_φ(f_servo)) · τ^(-1/2)

    where S_φ is the single-sided phase noise PSD [rad²/Hz] of the LO.
    [Santarelli1998, eq. 4; simplified for CW interrogation]

    Parameters
    ----------
    LO_phase_noise_dBcHz : float
        Single-sideband phase noise of LO at offset f_servo [dBc/Hz].
    f_servo_Hz : float
        FM servo modulation frequency [Hz] (sets the relevant noise offset).

    Returns
    -------
    np.ndarray
        σ_y,Dick(τ).
    """
    S_phi_lin = 10.0 ** (LO_phase_noise_dBcHz / 10.0)   # [rad²/Hz]
    A_dick    = math.sqrt(S_phi_lin) / nu_hfs
    return A_dick * tau ** (-0.5)


# ─────────────────────────────────────────────────────────────────────────────
# 5. TEMPERATURE FLUCTUATION NOISE  [Vanier2005, sec. 6.4]
# ─────────────────────────────────────────────────────────────────────────────

def temperature_noise_sigma(
    tau: np.ndarray,
    sigma_T_mK: float,
    ddnu_dT_Hz_K: float,
    tau_thermal_s: float = 100.0,
) -> np.ndarray:
    """
    Fractional frequency instability from cell temperature fluctuations.

        σ_y,T(τ) = (1/ν₀) · |dν/dT| · σ_T(τ)

    σ_T(τ) is modelled as a random walk (τ^(1/2)) for short τ (dominated
    by white temperature noise in the oven servo) crossing over to a flat
    floor controlled by the oven stability:

        σ_T(τ) = σ_T,0 · sqrt(τ / τ_thermal)   for τ < τ_thermal
        σ_T(τ) = σ_T,0                          for τ > τ_thermal

    Parameters
    ----------
    sigma_T_mK : float
        Short-term temperature stability of oven [mK] (typical: 0.1–2 mK).
    ddnu_dT_Hz_K : float
        Frequency-temperature sensitivity [Hz/K] from buffer_gas_shift.
    tau_thermal_s : float
        Thermal time constant of oven [s].

    Returns
    -------
    np.ndarray
        σ_y,T(τ).
    """
    sigma_T_K  = sigma_T_mK * 1e-3   # mK → K
    tau        = np.asarray(tau, dtype=float)
    # Effective temperature noise at each τ
    sigma_T_tau = sigma_T_K * np.minimum(np.sqrt(tau / tau_thermal_s), 1.0)
    return np.abs(ddnu_dT_Hz_K) * sigma_T_tau / nu_hfs


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAGNETIC FIELD FLUCTUATION NOISE  [Vanier2005, sec. 6.5]
# ─────────────────────────────────────────────────────────────────────────────

def magnetic_noise_sigma(
    tau: np.ndarray,
    B_Gauss: float,
    sigma_B_uG: float,
) -> np.ndarray:
    """
    Fractional frequency instability from residual B-field fluctuations.

        σ_y,B(τ) = (1/ν₀) · 2·K_Z·B · σ_B          [Vanier2005, sec. 6.5]

    Modelled as white noise (constant σ_B over all τ, since shielded
    environments have fairly flat B-noise PSDs in the relevant range).

    Parameters
    ----------
    B_Gauss : float
        Applied C-field magnitude [Gauss].
    sigma_B_uG : float
        B-field fluctuation RMS [μGauss].

    Returns
    -------
    np.ndarray
        σ_y,B(τ).
    """
    from constants import K_Zeeman
    sigma_B_G = sigma_B_uG * 1e-6
    sensitivity = 2.0 * K_Zeeman * B_Gauss   # Hz/G
    A_B = sensitivity * sigma_B_G / nu_hfs
    return A_B * tau ** (-0.5)


# ─────────────────────────────────────────────────────────────────────────────
# 7. ELECTRONICS / DETECTION NOISE  [Knappe2004]
# ─────────────────────────────────────────────────────────────────────────────

def electronics_noise_sigma(
    tau: np.ndarray,
    NEP_W_rtHz: float,
    P_det_W: float,
    gamma_CPT_Hz: float,
    contrast: float,
) -> np.ndarray:
    """
    Fractional frequency instability from detector/amplifier electronics noise.

    Models photodetector noise equivalent power (NEP) and amplifier noise.
    The resulting amplitude noise is converted to frequency noise via the
    discriminator slope.

        σ_y,elec(τ) = (1/ν₀) · (NEP / (C · P_det)) · (γ_CPT/2) · τ^(-1/2)

    [Knappe2004; Vanier2005 sec. 5.3.4]

    Parameters
    ----------
    NEP_W_rtHz : float
        Noise equivalent power of photodetector [W/√Hz].
    P_det_W : float
        Detected optical power [W].
    gamma_CPT_Hz : float
        CPT FWHM [Hz].
    contrast : float
        Signal contrast.

    Returns
    -------
    np.ndarray
        σ_y,elec(τ).
    """
    if contrast < 1e-9 or P_det_W < 1e-20:
        return np.zeros_like(tau, dtype=float)
    A_elec = (1.0 / nu_hfs) * (gamma_CPT_Hz / 2.0) * \
             NEP_W_rtHz / (contrast * P_det_W)
    return A_elec * tau ** (-0.5)


# ─────────────────────────────────────────────────────────────────────────────
# LONG-TERM DRIFT MODEL
# ─────────────────────────────────────────────────────────────────────────────

def long_term_drift_sigma(
    tau: np.ndarray,
    drift_rate_frac_per_day: float,
) -> np.ndarray:
    """
    Allan deviation contribution from a linear fractional frequency drift
    (dominated by buffer-gas aging and Helium permeation).

    For a linear drift d [frac/s]:
        σ_y,drift(τ) = d·τ / sqrt(2)   [Riley2008, Table 2.1]

    See [Marlow2021] for physical mechanisms of drift (permeation, aging).

    Parameters
    ----------
    drift_rate_frac_per_day : float
        Fractional frequency drift rate [1/day].

    Returns
    -------
    np.ndarray
        σ_y,drift(τ).
    """
    d_per_s = drift_rate_frac_per_day / 86400.0
    return d_per_s * np.asarray(tau, dtype=float) / math.sqrt(2.0)

def helium_permeation_drift(
    R_m: float,
    L_m: float,
    thickness_m: float,
    T_K: float,
) -> float:
    """
    Compute fractional frequency drift rate [1/day] from atmospheric Helium
    permeating through the glass cell walls via Fick's law.
    
    dP_He/dt = (K * A / (V * d)) * P_He,atmos
    K(T) = K_0 * exp(-E_a / (k_B * T))
    
    Parameters
    ----------
    R_m : float
        Cell radius [m]
    L_m : float
        Cell length [m]
    thickness_m : float
        Glass wall thickness [m]
    T_K : float
        Cell temperature [K]
        
    Returns
    -------
    float
        Fractional drift rate per day.
    """
    from constants import kB, nu_hfs
    
    P_He_atmos_Torr = 5.24e-6 * 760.0   # [Torr]
    He_beta0 = 718.0                    # [Hz/Torr] shift for Rb87
    He_K0 = 1.7e-9                      # [m^2/s] typical borosilicate permeation prefactor
    He_Ea = 4.3e-20                     # [J] activation energy
    
    area_m2 = 2.0 * math.pi * R_m * L_m + 2.0 * math.pi * (R_m**2)
    vol_m3  = math.pi * (R_m**2) * L_m
    
    if thickness_m <= 0:
        return 0.0
        
    K_m2_s = He_K0 * math.exp(-He_Ea / (kB * T_K))
    dP_dt_Torr_s = K_m2_s * (area_m2 / (vol_m3 * thickness_m)) * P_He_atmos_Torr
    
    dnu_dt_Hz_s = dP_dt_Torr_s * He_beta0
    return (dnu_dt_Hz_s / nu_hfs) * 86400.0

def vcsel_aging_drift(
    aging_rate_current_ppm_day: float = 1.0,
    light_shift_scaling: float = 1e-5,
) -> float:
    """
    Compute drift from VCSEL aging.
    aging_rate_current_ppm_day: fractional change in injection current.
    light_shift_scaling: fractional freq sensitivity to current change.
    """
    # result: (1e-6) * (1e-5) = 1e-11 / day typical
    return aging_rate_current_ppm_day * 1e-6 * light_shift_scaling

def rb_reactivity_drift(
    reaction_rate_ppm_day: float = 0.5,
) -> float:
    """
    Compute drift from chemical reactivity/depletion of Rb.
    """
    # 0.5 ppm/day -> 0.5e-6 ppb-like scale -> 5e-13 / day
    return reaction_rate_ppm_day * 1e-12 


def time_error_prediction(
    tau_s: np.ndarray,
    T0_offset_s: float = 0.0,
    fractional_freq_offset: float = 0.0,
    drift_rate_A: float = 0.0,
    sigma_x_tau: np.ndarray = None,
    integral_env_freq_offset: np.ndarray = None,
) -> np.ndarray:
    """
    Time Error Prediction (Holdover) long-term stability model.

        Delta T(tau) = T_0 + (Delta f / f)*tau + 1/2 A tau^2 + sigma_x(tau) + Integral_0^tau [Delta nu_env(t) / nu_0] dt

    Computes the absolute phase/time error accumulated over holdover time tau.

    Parameters
    ----------
    tau_s : np.ndarray
        Holdover duration array [s].
    T0_offset_s : float
        Initial time offset [s].
    fractional_freq_offset : float
        Initial frequency offset Delta f / f.
    drift_rate_A : float
        Linear fractional frequency drift rate A [s^-1].
    sigma_x_tau : np.ndarray, optional
        Stochastic time deviation (TDEV) [s].
    integral_env_freq_offset : np.ndarray, optional
        Accumulated environmental shift integral [s].
    """
    tau = np.asarray(tau_s, dtype=float)
    if sigma_x_tau is None:
        sigma_x_tau = np.zeros_like(tau)
    if integral_env_freq_offset is None:
        integral_env_freq_offset = np.zeros_like(tau)
        
    return T0_offset_s + fractional_freq_offset * tau + 0.5 * drift_rate_A * tau**2 + sigma_x_tau + integral_env_freq_offset


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED NOISE BUDGET
# ─────────────────────────────────────────────────────────────────────────────

def total_noise_budget(tau: np.ndarray, params: dict) -> dict:
    """
    Compute all noise contributions and the combined Allan deviation.
    
    The total instability is evaluated explicitly as the quadrature sum:
        sigma_y,total(tau) = sqrt(sigma_shot^2 + sigma_rin^2 + sigma_fm^2 + sigma_pm_am^2 +
                                  sigma_T^2 + sigma_B^2 + sigma_elec^2 + sigma_drift^2)

    [Riley2008] / [Marlow2021] stability classification.

    Note: The Dick effect (sigma_dick) is omitted for CW systems as it scales 
    with dead-time, which is zero in continuous interrogation. [Audoin1998]
    """
    pc = params
    tau = np.asarray(tau, dtype=float)

    sigma_shot  = shot_noise_sigma(
        tau, pc["P_det_W"], pc["gamma_CPT_Hz"], pc["contrast"], pc["eta_QE"])
    sigma_rin   = rin_noise_sigma(
        tau, pc["RIN_dBHz"], pc["gamma_CPT_Hz"], pc["contrast"])
    sigma_fm    = fm_noise_sigma(
        tau, pc.get("LO_PN_dBcHz", -110.0),
        pc["gamma_CPT_Hz"], pc["contrast"],
        pc.get("servo_bw_Hz", 10.0))
    sigma_pm_am = laser_pm_am_noise_sigma(
        tau,
        pc.get("laser_linewidth_Hz", 50e6),              # Default 50 MHz for standard VCSEL
        pc.get("optical_detuning_error_Hz", 10e6),       # Default 10 MHz lock error
        pc.get("doppler_linewidth_Hz", 500e6),
        pc["gamma_CPT_Hz"],
        pc["contrast"]
    )
    
    # Dick effect is nominally zero for CW clocks
    sigma_dick_cw = np.zeros_like(tau)

    sigma_T     = temperature_noise_sigma(
        tau, pc["sigma_T_mK"], pc["ddnu_dT_Hz_K"], pc["tau_thermal_s"])
    sigma_B     = magnetic_noise_sigma(
        tau, pc["B_Gauss"], pc["sigma_B_uG"])
    sigma_elec  = electronics_noise_sigma(
        tau, pc["NEP_W_rtHz"], pc["P_det_W"], pc["gamma_CPT_Hz"], pc["contrast"])
    
    if "glass_thickness_mm" in pc:
        drift_he = helium_permeation_drift(
            pc["cell_R_m"], pc["cell_L_m"], pc["glass_thickness_mm"]*1e-3, pc["T_K"]
        )
        drift_vcsel = vcsel_aging_drift(pc.get("vcsel_aging_rate", 1.0))
        drift_rb = rb_reactivity_drift(pc.get("rb_reaction_rate", 0.5))
        drift_rate = drift_he + drift_vcsel + drift_rb
    else:
        drift_rate = pc.get("drift_rate_frac_per_day", 3e-11)

    sigma_drift = long_term_drift_sigma(tau, drift_rate)

    sigma_total = np.sqrt(sigma_shot**2  + sigma_rin**2   + sigma_fm**2   +
                          sigma_pm_am**2 + sigma_dick_cw**2  + sigma_T**2      +
                          sigma_B**2    + sigma_elec**2  + sigma_drift**2)

    return {
        "sigma_shot":   sigma_shot,
        "sigma_rin":    sigma_rin,
        "sigma_fm":     sigma_fm,
        "sigma_pm_am":  sigma_pm_am,
        "sigma_dick":   sigma_dick_cw,
        "sigma_T":      sigma_T,
        "sigma_B":      sigma_B,
        "sigma_elec":   sigma_elec,
        "sigma_drift":  sigma_drift,
        "sigma_total":  sigma_total,
    }
