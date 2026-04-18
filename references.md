# References for CPT Clock Performance Model

All analytical formulas used in the code are cited to primary literature below.
Each module lists the specific equation numbers where applicable.

---

## Physical Constants & Rb-87 Data

- **[Steck2021]** D. A. Steck, *Rubidium 87 D Line Data*, revision 2.2.2, 2021.
  URL: <http://steck.us/alkalidata>
  - ν₀ = 6,834,682,610.904 Hz (Table 1)
  - Γ_D1 = 2π × 5.746 MHz
  - Vapour pressure: log₁₀(P/Pa) = A − B/T (Table 2, Nesmeyanov relation)

---

## Vapour Pressure & Atom Density (`rb_vapor.py`)

- **[Steck2021]** Table 2 — Nesmeyanov relation coefficients for solid and liquid Rb.
- **[Alcock1984]** C. B. Alcock, V. P. Itkin, M. K. Horrigan, *Can. Metall. Q.* **23**, 309 (1984).
  Cross-check for alkali vapour pressures.

---

## Cell Model — Diffusion & Decoherence (`cell_model.py`)

- **[Bouchiat1966]** M. A. Bouchiat & J. Brossel, *Phys. Rev.* **147**, 41 (1966).
  Diffusion-limited relaxation in cylindrical cells: γ_diff = D·[(2.405/R)² + (π/L)²]

- **[Beverini1971]** N. Beverini, P. Minguzzi, F. Strumia, *Phys. Rev. A* **4**, 550 (1971).
  Transit-time broadening in buffer-gas cells.

- **[Franzen1959]** W. Franzen, *Phys. Rev.* **115**, 850 (1959).
  Cross-check for early buffer gas diffusion formulations.

- **[Franz1976]** F. A. Franz & C. Volk, *Phys. Rev. A* **14**, 1711 (1976).
  Rb diffusion constants in buffer gases and spin-relaxation rates.

- **[Blanc1908]** *Blanc's law* for diffusion in gas mixtures: 1/D_eff = Σ_i x_i/D_i.

- **[Kozlova2011]** O. Kozlova, S. Guérandel, E. de Clercq, *Proc. European Freq. Time Forum* (2011).
  Buffer-gas collisional broadening coefficients:
  γ(N₂) ≈ 143.6 Hz/Torr, γ(Ar) ≈ 42.3 Hz/Torr.

- **[Vanier2005]** J. Vanier & C. Audoin, *The Quantum Physics of Atomic Frequency Standards*, IOP (2005).
  sec. 3.4 (diffusion), sec. 3.6 (spin-exchange relaxation).

---

## VCSEL Modulation Spectrum (`vcsel_model.py`)

- **[Zhu1993]** M. Zhu & L. S. Cutler, *Proc. 25th PTTI* (1993). Original CW-CPT concept.

- **[Levi2000]** F. Levi, L. Lorini, D. Calonico, A. Godone, *Eur. Phys. J. D* **12**, 53 (2000).
  VCSEL modulation spectrum; optical sideband power fractions J_k²(m).

- **[Shah2006]** V. Shah & J. Kitching, *Adv. At. Mol. Opt. Phys.* **59** (2010).
  Light-shift cancellation at optimum modulation index m ≈ 2.4.

---

## CPT Signal — Lineshape, Contrast, Linewidth (`cpt_signal.py`)

- **[Wynands1999]** R. Wynands & A. Nagel, *Appl. Phys. B* **68**, 1 (1999).
  Analytical 3-level Λ model; Lorentzian lineshape eq. 5; contrast eq. 8; linewidth eq. 7.

- **[Vanier2005]** sec. 5.2 — steady-state density matrix for CPT signal.

- **[Knappe2004]** S. Knappe, V. Shah, P. D. D. Schwindt, L. Hollberg, J. Kitching,
  *Opt. Lett.* **29**, 388 (2004). CSAC signal contrast C_max ≈ 10%.

- **[Levi2000]** Discriminator slope: D = C·T_bg / (2·γ_CPT). [Vanier2005, sec. 5.3]

---

## Systematic Frequency Shifts (`frequency_shifts.py`)

### Buffer Gas Shift

- **[Kozlova2011]** Table 1: β₀, β₁, β₂ coefficients for N₂ and Ar.
  N₂: β₀ = +543 Hz/Torr, β₁ = +0.36 Hz/Torr/K.
  Ar: β₀ = −54.5 Hz/Torr, β₁ = −0.29 Hz/Torr/K.

- **[Vanier2005]** sec. 6.3: general buffer-gas shift theory.

- **[Inversion temperature]** At T_inv, dΔν/dT = 0. For P_Ar/P_N₂ ≈ 1.6: T_inv ≈ 60 °C.
  Derived from Kozlova2011 coefficients; cross-checked with [Knappe2004] and [Boudot2011].

- **[Boudot2011]** R. Boudot et al., *Opt. Express* **19**, 3106 (2011).
  Macroscopic dark-line resonances in macroscopic ... (DOI: `10.1364/OE.19.003106`)

### Light Shift (AC Stark Shift)

- **[Levi2000]** eq. 10–11: light shift from all VCSEL sidebands; semi-empirical α_eff(m).

- **[Shah2006]** Modulation index tuning to suppress light shift; zero-crossing at m ≈ 2.4.

- **[Vanier2005]** sec. 5.4: detuning-dependent light shift.

### Second-Order Zeeman Shift

- **[Vanier2005]** eq. 3.48: Δν_Z2 = K_Z·B² = 575.15 Hz/G² for ⁸⁷Rb (0–0 transition).
  Derived from the Breit-Rabi formula for the ⁸⁷Rb ground state.

### Spin-Exchange Shift

- **[Allard2004]** F. Allard, I. Maksimovic, M. Guerlin, & C. J. Bordé,
  *Phys. Rev. A* **70**, 012513 (2004). κ_se and its sign for Rb HFS.

### Blackbody Radiation Shift

- **[Bize1999]** S. Bize, Y. Sortais, M. S. Santos, C. Mandache, A. Clairon, C. Salomon,
  *Europhys. Lett.* **45**, 558 (1999).
  β_BBR = 1.26 × 10⁻¹⁴, ε = 0.013 at T_ref = 300 K.

### Barometric Shift

- **[Vanier2005]** sec. 6.7: external pressure coupling through cell walls.
  κ_baro ≈ 10⁻¹² per mbar (glass cell estimate).

---

## Noise Sources (`noise_budget.py`)

### Photon Shot Noise

- **[Wynands1999]** eq. 12.
- **[Vanier2005]** sec. 5.3.2.
  σ_y,shot = (1/ν₀)·(γ_CPT/2)·√(hν/ηP_det)·τ^(-1/2)

### Laser RIN

- **[Vanier2005]** sec. 5.3.3.
  σ_y,RIN = (1/ν₀)·(γ_CPT/2)·√RIN·τ^(-1/2)/C

### FM Noise (FM→AM Conversion)

- **[Audoin1998]** C. Audoin, G. Santarelli, A. Makdissi, A. Clairon,
  *IEEE Trans. UFFC* **45**, 877 (1998). FM-to-AM conversion via absorption slope.
  (See also Dick effect equations: DOI: `10.1109/58.710593`)

### PM-AM Noise (Laser Phase-to-Amplitude)

- **[Camparo1999]** J. C. Camparo & J. G. Coffer, *Phys. Rev. A* **59**, 728 (1999).
  Conversion of laser phase noise to amplitude noise via dispersive atomic line.

### Dick Effect

- **[Santarelli1998]** G. Santarelli, C. Audoin, A. Makdissi, P. Laurent, G. J. Dick, A. Clairon,
  *IEEE Trans. UFFC* **45**, 887 (1998).
  σ_y,Dick = √(S_φ(f_servo))/ν₀ · τ^(-1/2)

### Temperature Fluctuation Noise

- **[Vanier2005]** sec. 6.4: random-walk temperature model for oven servo.

### Magnetic Field Fluctuation Noise

- **[Vanier2005]** sec. 6.5: σ_y,B = 2·K_Z·B·σ_B/ν₀.

### Electronics / Detection Noise

- **[Knappe2004]**: NEP-limited detection in miniature CPT clocks.
- **[Vanier2005]** sec. 5.3.4.

### Long-Term Drift

- **[Vanier2005]** sec. 6.6: linear drift model; σ_y,drift = d·τ/√3.

---

## Allan Deviation (`allan_deviation.py`)

- **[Allan1966]** D. W. Allan, *Proc. IEEE* **54**, 221 (1966). Definition of ADEV.
- **[Riley2008]** W. J. Riley, *Handbook of Frequency Stability Analysis*,
  NIST Special Publication 1065 (2008). Noise types and slope identification.
  URL: <http://tf.nist.gov/timefreq/general/pdf/868.pdf>
- **[Vanier2005]** sec. 5.2: clock stability metrics.

---

## Environmental Sweeps (`environment_sweep.py`)

- **[Kozlova2011]** Temperature sweep of buffer gas shift.
- **[Vanier2005]** sec. 6.3: optimum operating temperature.
- **[Knappe2004]** Ar/N₂ mixture optimisation for CSAC class clocks.
