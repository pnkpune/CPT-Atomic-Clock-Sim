"""
constants.py
============
Physical constants and 87-Rb specific data for the CPT clock model.

References
----------
[Steck2021]    D. A. Steck, "Rubidium 87 D Line Data", revision 2.2.2 (2021).
               https://steck.us/alkalidata/rubidium87numbers.pdf
[Vanier2005]   J. Vanier and C. Audoin, "The Quantum Physics of Atomic
               Frequency Standards", IOP Publishing (2005).
[Boudot2011]   R. Boudot et al., Opt. Express 19, 3106 (2011).
               [Rb-87 buffer-gas shift coefficients for N2/Ar mixtures]
[Bize1999]     S. Bize et al., Europhys. Lett. 45, 558 (1999).  [BBR shift]
[Allard2004]   F. Allard et al., Phys. Rev. A 70, 012513 (2004). [Se shift]
[Danos1958]    R. B. Danos & A. M. Ruderman, Phys. Rev. 109, 1036 (1958).
               [diffusion constants]
[Rotondaro1997] A. D. Rotondaro & G. P. Perram, J. Quant. Spectrosc. Radiat.
               Transfer 57, 497 (1997). [N2 quenching of Rb excited state]
[Straessle2014] R. Straessle et al., J. Phys. B 47, 075502 (2014).
               [anti-relaxation coatings for micro-cells]
"""

import math

# ─────────────────────────────────────────────────────────────────────────────
# FUNDAMENTAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
h   = 6.62607015e-34      # Planck constant  [J·s]
hbar = h / (2 * math.pi)  # Reduced Planck   [J·s]
kB  = 1.380649e-23        # Boltzmann        [J/K]
c   = 2.99792458e8        # Speed of light   [m/s]
e   = 1.602176634e-19     # Elementary charge[C]
eps0 = 8.8541878128e-12   # Permittivity     [F/m]
amu  = 1.66053906660e-27  # Atomic mass unit [kg]
sigma_SB = 5.670374419e-8 # Stefan-Boltzmann [W/m²/K⁴]

# ─────────────────────────────────────────────────────────────────────────────
# 87-Rb ATOMIC DATA  [Steck2021]
# ─────────────────────────────────────────────────────────────────────────────
# Ground-state hyperfine splitting (clock frequency)
nu_hfs    = 6_834_682_610.904      # [Hz]  [Steck2021, Table 1]

# D1 line  (5²S₁/₂ → 5²P₁/₂)
lambda_D1 = 794.978969929e-9       # [m]   [Steck2021]
nu_D1     = c / lambda_D1          # [Hz]
Gamma_D1  = 2 * math.pi * 5.746e6 # [rad/s] natural linewidth [Steck2021]

# D2 line  (5²S₁/₂ → 5²P₃/₂)
lambda_D2 = 780.241209686e-9       # [m]   [Steck2021]
Gamma_D2  = 2 * math.pi * 6.067e6 # [rad/s] [Steck2021]

# Rb-87 atom mass
m_Rb      = 86.909180527 * amu     # [kg]  [Steck2021]

# Rb-87 vapour pressure constants (solid phase, T < 312.46 K; liquid, T >= 312.46 K)
# log10(P/Pa) = A1 - B1/T  (solid),   A2 - B2/T  (liquid)  [Steck2021, Table 2]
VP_A_solid  =  9.863
VP_B_solid  = 4215.0                # [K]
VP_A_liquid =  9.318
VP_B_liquid = 4040.0                # [K]

# ─────────────────────────────────────────────────────────────────────────────
# ZEEMAN SHIFT  [Vanier2005, sec. 3.5]
# Δν_Z2 = K_Z · B²    (second-order, field-insensitive 0-0 transition)
# ─────────────────────────────────────────────────────────────────────────────
K_Zeeman = 575.152 # [Hz / G²] for ⁸⁷Rb clock transition [Vanier2005]

# ─────────────────────────────────────────────────────────────────────────────
# BUFFER GAS COLLISIONAL SHIFT COEFFICIENTS  [Kozlova2011, Table 1]
# Δν_gas(T) = P_gas * [beta0 + beta1*(T-T0) + beta2*(T-T0)^2]
# T0 = 273.15 K (0 °C),  P in Torr,  Δν in Hz
# ─────────────────────────────────────────────────────────────────────────────
T0_bg = 273.15   # [K]  reference temperature for shift coefficients

# Nitrogen (N₂)
N2_beta0 =  543.0    # [Hz/Torr]
N2_beta1 =   0.36    # [Hz/Torr/K]
N2_beta2 =  -5.3e-4  # [Hz/Torr/K²]

# Argon (Ar)
Ar_beta0 = -54.5    # [Hz/Torr]
Ar_beta1 =  -0.29   # [Hz/Torr/K]
Ar_beta2 =   8.0e-4 # [Hz/Torr/K²]

# ─────────────────────────────────────────────────────────────────────────────
# SPIN-EXCHANGE SHIFT  [Allard2004]
# Δν_se = kappa_se * n_Rb(T)
# ─────────────────────────────────────────────────────────────────────────────
sigma_se  = 1.9e-14 * 1e-4  # spin-exchange cross-section [m²]  (converted from cm²)
kappa_se  = -3.0e-19        # Hz·m³  (empirical, sign is negative for Rb HFS)

# ─────────────────────────────────────────────────────────────────────────────
# BLACKBODY RADIATION (BBR) SHIFT  [Bize1999]
# Δν_BBR / ν₀ = -β_BBR * (T/T_ref)^4 * [1 + ε*(T/T_ref)^2]
# Note: tiny correction, included for completeness
# ─────────────────────────────────────────────────────────────────────────────
T_ref_BBR   = 300.0     # [K]
beta_BBR    = 1.26e-14  # (dimensionless at T_ref)  [Bize1999]
eps_BBR     = 0.013     # dynamic correction factor [Bize1999]

# ─────────────────────────────────────────────────────────────────────────────
# DIFFUSION COEFFICIENTS  (at STP: T=273.15 K, P=1 atm)  [Danos1958 + lit.]
# D(T,P) = D0 * (T/273.15)^1.5 * (1 atm / P)
# ─────────────────────────────────────────────────────────────────────────────
D0_Rb_N2 = 0.154e-4    # [m²/s] at STP  (0.154 cm²/s·atm)
D0_Rb_Ar = 0.320e-4    # [m²/s] at STP  (0.320 cm²/s·atm)

# ─────────────────────────────────────────────────────────────────────────────
# SATURATION INTENSITY — D1 line  [Steck2021, Table 7]
# These are the correct multi-level values including degeneracy factors.
# I_sat_D1_isotropic : unpolarized / isotropic pump, mF-averaged [Steck2021 Table 7]
# I_sat_D1_lin       : linearly polarised light (lin||lin CPT geometry) [Steck2021 Table 7]
# ─────────────────────────────────────────────────────────────────────────────
I_sat_D1_isotropic = 44.84    # [W/m²]  =  4.484 mW/cm²  [Steck2021, Table 7]
I_sat_D1_lin       = 16.70    # [W/m²]  =  1.669 mW/cm²  [Steck2021, Table 7]
# Default used in Rabi formula: lin‖lin polarisation geometry for CPT
I_sat_D1_CPT       = I_sat_D1_lin

# ─────────────────────────────────────────────────────────────────────────────
# LIGHT SHIFT (AC STARK SHIFT) — intensity coefficient
# Δν_LS ≈ alpha_LS * I_total   (linear approximation for moderate power)
# alpha_LS depends on detuning; typical range for D1 CPT:
# ─────────────────────────────────────────────────────────────────────────────
alpha_LS_typ = -1.0e-3   # [Hz / (μW/cm²)] — representative value; sign can vary

# ─────────────────────────────────────────────────────────────────────────────
# BAROMETRIC PRESSURE COEFFICIENT  [Vanier2005, sec. 6.7]
# Δν_baro = kappa_baro * ΔP_atm   (cell enclosed, mechanically coupled)
# Typical: ~10^-12 per mbar for hard-sealed glass cells
# ─────────────────────────────────────────────────────────────────────────────
kappa_baro = 1.0e-12   # [fractional / mbar]  → multiply by nu_hfs for Hz/mbar

# ─────────────────────────────────────────────────────────────────────────────
# OPTICAL PRESSURE BROADENING COEFFICIENTS  [Vanier2005, sec. 6.2]
# Lorentzian HWHM broadening of the D1 optical line by buffer gas collisions.
# Γ_opt_broad(T, P) ≈ 2π × [γ_N2 × P_N2 + γ_Ar × P_Ar]  [in rad/s]
# Temperature dependence ~(T/T0)^0.3 is weak; ignored here (< 5% over 20–80 °C).
# ─────────────────────────────────────────────────────────────────────────────
gamma_opt_N2_Hz_Torr = 17.0e6   # [Hz/Torr]  HWHM optical broadening by N₂  [Vanier2005]
gamma_opt_Ar_Hz_Torr =  8.0e6   # [Hz/Torr]  HWHM optical broadening by Ar  [Vanier2005]

# ─────────────────────────────────────────────────────────────────────────────
# N₂ EXCITED-STATE QUENCHING RATE COEFFICIENT  [Rotondaro1997]
# The excited 5²P₁/₂ state is collisionally quenched by N₂ at rate:
#   Γ_quench = k_Q_N2 × n_N2   [rad/s]
# k_Q_N2 is the bimolecular quenching rate constant at ~60 °C.
# Quenching dramatically increases Γ_hom and reduces optical coherence lifetime.
# ─────────────────────────────────────────────────────────────────────────────
k_Q_N2 = 1.4e-10 * 1e-6  # [m³/s]  (converted from 1.4e-10 cm³/s) [Rotondaro1997]

# ─────────────────────────────────────────────────────────────────────────────
# WALL COLLISION PARAMETERS (no-buffer-gas / anti-relaxation coating cells)
# [Bouchiat1966; Straessle2014]
# P_depol_bare    : depolarization probability per wall collision, bare glass
# P_depol_paraffin: depolarization probability per wall collision, paraffin coating
# P_depol_PDMS    : depolarization probability per wall collision, PDMS/OTS coating
# ─────────────────────────────────────────────────────────────────────────────
P_depol_bare     = 1.0      # bare glass: each collision destroys coherence
P_depol_paraffin = 1e-3     # paraffin-coated: ~1 in 1000 collisions depolarises
P_depol_PDMS     = 1e-4     # PDMS / OTS coating: ~1 in 10 000 collisions
