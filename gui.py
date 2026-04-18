"""
gui.py
======
Tkinter + Matplotlib GUI for the CPT clock performance model.

Layout:
  Left  — scrollable parameter entry panel (groups: Cell, Laser, Electronics,
           Environment, Noise) + Run button.
  Right — Matplotlib figure with 6 tabbed plot panels.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import traceback

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.gridspec import GridSpec

from constants  import nu_hfs
from cell_model import total_ground_decoherence
from cpt_signal import (cpt_linewidth_Hz, cpt_contrast, lineshape,
                         rabi_frequency, discriminator_slope, compute_cpt_signal)
from vcsel_model import sideband_powers
from frequency_shifts import total_shift_budget
from noise_budget import total_noise_budget, time_error_prediction
from allan_deviation import tau_array, compute_allan_deviation, sensitivity_table
from environment_sweep import temperature_sweep, pressure_ratio_sweep

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────────────────
BG_DARK   = "#0f1117"
BG_PANEL  = "#1a1d27"
BG_GROUP  = "#22263a"
FG_TEXT   = "#e8eaf6"
FG_DIM    = "#8a8fa8"
ACC_BLUE  = "#4fc3f7"
ACC_CYAN  = "#00e5ff"
ACC_GREEN = "#69f0ae"
ACC_GOLD  = "#ffd740"
ACC_RED   = "#ff5252"
ACC_PURPLE= "#ce93d8"
ENTRY_BG  = "#2c3150"
ENTRY_FG  = "#e8eaf6"
BTN_BG    = "#1565c0"
BTN_FG    = "#ffffff"
BTN_ACT   = "#1976d2"

NOISE_COLORS = {
    "sigma_shot":    ACC_CYAN,
    "sigma_rin":    ACC_GREEN,
    "sigma_fm":     ACC_GOLD,
    "sigma_flicker": "#ba68c8",
    "sigma_dick":   ACC_PURPLE,
    "sigma_T":      "#ff8a65",
    "sigma_B":      "#80deea",
    "sigma_elec":   "#b0bec5",
    "sigma_drift":  ACC_RED,
    "sigma_total":  "#ffffff",
}
NOISE_LABELS = {
    "sigma_shot":    "Shot noise",
    "sigma_rin":    "Laser RIN",
    "sigma_fm":     "FM noise",
    "sigma_flicker": "Flicker (1/f)",
    "sigma_dick":   "Dick effect",
    "sigma_T":      "Temp. fluc.",
    "sigma_B":      "B-field fluc.",
    "sigma_elec":   "Electronics",
    "sigma_drift":  "Aging drift",
    "sigma_total":  "Total",
}

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETER DEFINITIONS
# Each entry: (key, label, default_value, unit, group)
# ─────────────────────────────────────────────────────────────────────────────
PARAMS = [
    # Cell / Atomic  — GC25075-RB: 75 mm long × 25.4 mm OD → R = 12.7 mm
    ("T_C",            "Cell temperature",       45.0,   "°C",        "Cell"),
    ("P_N2_Torr",      "N₂ pressure",            0.0,    "Torr",      "Cell"),
    ("P_Ar_Torr",      "Ar pressure",            0.0,    "Torr",      "Cell"),
    ("cell_R_mm",      "Cell radius",            12.7,   "mm",        "Cell"),   # GC25075-RB OD=25.4mm → R=12.7mm
    ("cell_L_mm",      "Cell length",            75.0,   "mm",        "Cell"),   # GC25075-RB
    ("B_mG",           "C-field (B₀)",           50.0,   "mGauss",    "Cell"),
    # Laser / Optical  — C171TMD-B collimation, θ_1/e²=14.4°, w=1.55mm → d=3.1mm
    ("P_total_uW",     "Total laser power",      25.0,   "μW",        "Laser"),  # power at cell (notes §4.2)
    ("mod_index",      "Mod. index m",           1.85,   "—",         "Laser"),
    ("beam_diam_mm",   "Beam diameter",          3.1,    "mm",        "Laser"),  # C171TMD-B: f=6.2mm, θ=14.4° (notes §3)
    ("laser_detuning_MHz", "Laser detuning",     0.0,    "MHz",       "Laser"),
    ("laser_lw_MHz",   "VCSEL linewidth",        50.0,   "MHz",       "Laser"),
    # Electronics  — PDA36A2 + ADF4356 eval board + Red Pitaya 125-14
    ("eta_QE",         "Photodiode QE (η)",      0.70,   "—",         "Electronics"),  # R=0.45 A/W @795nm → η≈0.70
    ("RIN_dBHz",       "Laser RIN",              -125.0, "dBc/Hz",    "Electronics"),  # VCSEL typ. (notes §6.2)
    ("LO_PN_dBcHz",    "LO phase noise",         -115.0, "dBc/Hz",    "Electronics"),  # ADF4356 @100kHz offset
    ("servo_bw_Hz",    "FM servo freq.",          100.0,  "Hz",        "Electronics"),
    ("NEP_W_rtHz",     "Photodetector NEP",       3.2e-12,"W/√Hz",    "Electronics"),  # PDA36A2 ~50dB gain (notes §6.3)
    # Environment / Noise
    ("sigma_T_mK",     "Temp. stability",         1.0,   "mK",        "Environment"),  # ±1 mK cell control
    ("tau_thermal_s",  "Thermal time const.",     10.0,  "s",         "Environment"),
    ("sigma_B_uG",     "B-field noise σ_B",       5.0,   "μGauss",    "Environment"),
    ("Delta_P_atm_mbar","Atm. press. deviation",   0.0,  "mbar",      "Environment"),
    ("glass_thickness_mm", "Glass thickness",     0.5,   "mm",        "Environment"),
    ("vcsel_aging_rate",   "Laser aging rate",    1.0,   "ppm/day",   "Environment"),
    ("rb_reaction_rate",   "Rb reaction rate",    0.5,   "ppm/day",   "Environment"),
]

# ─────────────────────────────────────────────────────────────────────────────

def params_from_entries(entries: dict) -> dict:
    """Read GUI entries and convert to internal units."""
    p = {}
    for key, _, default, _, _ in PARAMS:
        try:
            p[key] = float(entries[key].get())
        except (ValueError, KeyError):
            p[key] = default

    # Unit conversions
    p["T_K"]         = p["T_C"] + 273.15
    p["B_Gauss"]     = p["B_mG"] * 1e-3
    p["cell_R_m"]    = p["cell_R_mm"] * 1e-3
    p["cell_L_m"]    = p["cell_L_mm"] * 1e-3
    
    # Laser Linewidth derived units
    if "laser_lw_MHz" in p:
        p["laser_lw_kHz"] = p["laser_lw_MHz"] * 1e3
        p["laser_linewidth_Hz"] = p["laser_lw_MHz"] * 1e6
        
    return p


# ─────────────────────────────────────────────────────────────────────────────
# STYLE HELPER
# ─────────────────────────────────────────────────────────────────────────────

def style_ax(ax, title="", xlabel="", ylabel="", grid=True):
    ax.set_facecolor("#12151f")
    ax.tick_params(colors=FG_DIM, labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#3a3f5c")
    if title:
        ax.set_title(title, color=FG_TEXT, fontsize=12, fontweight="bold", pad=7)
    if xlabel:
        ax.set_xlabel(xlabel, color=FG_DIM, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=FG_DIM, fontsize=10)
    if grid:
        ax.grid(True, color="#2a2f4a", linewidth=0.5, linestyle="--")


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_lineshape(ax, params, cell_result, cpt_result):
    ax.cla()
    style_ax(ax, "CPT Dark Resonance Lineshape",
             r"Two-photon Detuning  $\delta$  [Hz]",
             "Normalised Transmission")

    gamma_CPT = cpt_result["gamma_CPT_Hz"]
    C         = cpt_result["contrast"]

    delta = np.linspace(-3 * gamma_CPT, 3 * gamma_CPT, 1000)
    B_G = params.get("B_Gauss", 0.0)
    T_curve = lineshape(delta, gamma_CPT, C, B_G=B_G)
    ax.plot(delta, T_curve, color=ACC_CYAN, lw=1.8, label="CPT signal")

    # Annotate linewidth (half-maximum of the central clock resonance dip)
    T_baseline = 1.0
    T_min = float(lineshape(0.0, gamma_CPT, C, B_G=B_G))
    half_dip = (T_baseline + T_min) / 2.0
    ax.axhline(half_dip, color=ACC_GOLD, lw=0.7, ls="--", alpha=0.6)
    ax.annotate("", xy=(gamma_CPT / 2, half_dip),
                xytext=(-gamma_CPT / 2, half_dip),
                arrowprops=dict(arrowstyle="<->", color=ACC_GOLD, lw=1.0))
    ax.text(0, half_dip + 0.0005, f"FWHM = {gamma_CPT:.1f} Hz",
            ha="center", va="bottom", color=ACC_GOLD, fontsize=9)

    # Info box — adapts to both buffer-gas (diffusion) and no-buffer-gas (ballistic) regimes
    regime = cell_result.get("regime", "diffusion")
    gamma2 = cell_result["gamma2_total_Hz"]
    gamma_se = cell_result.get("gamma_se_Hz", 0.0)

    if regime == "ballistic":
        gamma_wall    = cell_result.get("gamma_wall_Hz", 0.0)
        gamma_transit = cell_result.get("gamma_transit_Hz", 0.0)
        info = (r"$\gamma_{CPT}$" + f" = {gamma_CPT:.1f} Hz\n"
                f"Contrast = {C * 100:.2f}%\n"
                r"$\gamma_2$" + f" = {gamma2:.1f} Hz  [ballistic]\n"
                f"  wall:    {gamma_wall:.1f} Hz\n"
                f"  s-e:     {gamma_se:.1f} Hz\n"
                f"  transit: {gamma_transit:.0f} Hz  (inhomog.)")
    else:
        gamma_diff = cell_result.get("gamma_diff_Hz", 0.0)
        gamma_bg   = cell_result.get("gamma_bg_Hz", 0.0)
        info = (r"$\gamma_{CPT}$" + f" = {gamma_CPT:.1f} Hz\n"
                f"Contrast = {C * 100:.2f}%\n"
                r"$\gamma_2$" + f" = {gamma2:.1f} Hz  [diffusion]\n"
                f"  diff:  {gamma_diff:.1f} Hz\n"
                f"  s-e:   {gamma_se:.1f} Hz\n"
                f"  buf.:  {gamma_bg:.1f} Hz")

    ax.text(0.98, 0.97, info, transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color=FG_DIM,
            bbox=dict(boxstyle="round,pad=0.4", fc=BG_GROUP, ec="#3a3f5c", alpha=0.9))
    ax.legend(fontsize=9, facecolor=BG_GROUP, edgecolor="#3a3f5c",
              labelcolor=FG_TEXT, loc="lower left")


def plot_shift_budget(ax, shifts):
    ax.cla()
    style_ax(ax, "Systematic Frequency Shift Budget",
             "", r"Shift  $\Delta\nu$  [Hz]")

    labels = ["Buffer Gas", "Light Shift", r"2nd-ord. Zeeman",
              "Spin-Exchange", "BBR", "Barometric", "TOTAL"]
    values = [
        shifts["dnu_buffer_gas_Hz"],
        shifts["dnu_light_shift_Hz"],
        shifts["dnu_zeeman_Hz"],
        shifts["dnu_spin_exchange_Hz"],
        shifts["dnu_BBR_Hz"],
        shifts["dnu_barometric_Hz"],
        shifts["dnu_total_Hz"],
    ]
    colors = [ACC_CYAN, ACC_GOLD, ACC_GREEN, ACC_PURPLE,
              "#ff8a65", "#80deea", ACC_RED]

    x     = np.arange(len(labels))
    bars  = ax.bar(x, values, color=colors, width=0.55, edgecolor="#0f1117",
                   linewidth=0.6, alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.axhline(0, color=FG_DIM, lw=0.5)

    for bar, v in zip(bars, values):
        if abs(v) > 1e-6:
            va_str = "bottom" if v >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, v,
                    f"{v:.3g}", ha="center", va=va_str, fontsize=8, color=FG_TEXT)

    # Temperature sensitivity annotation
    ddT = shifts["ddnu_bg_dT_Hz_K"]
    T_inv_C = shifts["T_inversion_C"]
    info = (r"$d\nu/dT$ (buf.gas) = " + f"{ddT:.3f} Hz/K\n"
            r"$T_{inv}$ = " + f"{T_inv_C:.1f} \u00b0C")
    ax.text(0.98, 0.02, info, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color=FG_DIM,
            bbox=dict(boxstyle="round,pad=0.4", fc=BG_GROUP, ec="#3a3f5c", alpha=0.9))


def plot_allan_deviation(ax, adev_result):
    ax.cla()
    style_ax(ax, r"Allan Deviation  $\sigma_y(\tau)$",
             r"Averaging Time  $\tau$  [s]",
             r"Fractional Frequency Instability  $\sigma_y$")
    ax.set_xscale("log")
    ax.set_yscale("log")

    tau = adev_result["tau"]
    for key, color in NOISE_COLORS.items():
        arr = adev_result.get(key)
        if arr is None:
            continue
        lw  = 2.2 if key == "sigma_total" else 0.9
        ls  = "-"  if key == "sigma_total" else "--"
        alpha = 1.0 if key == "sigma_total" else 0.72
        ax.plot(tau, arr, color=color, lw=lw, ls=ls, alpha=alpha,
                label=NOISE_LABELS[key])

    # Reference guide line: tau^(-1/2)
    tau_ref = np.array([tau[0], tau[-1]])
    val_ref = adev_result["sigma_total"]
    ref_mid = val_ref[len(val_ref) // 4]
    ax.plot(tau_ref,
            ref_mid * (tau_ref / tau[len(tau) // 4]) ** (-0.5),
            color=FG_DIM, lw=0.6, ls=":", alpha=0.4,
            label=r"$\tau^{-1/2}$ guide")

    ax.legend(fontsize=9, facecolor=BG_GROUP, edgecolor="#3a3f5c",
              labelcolor=FG_TEXT, ncol=2, loc="best")

    sigma_1s = adev_result["sigma_1s"]
    ax.text(0.02, 0.97,
            r"$\sigma_y$(1 s) = " + f"{sigma_1s:.2e}",
            transform=ax.transAxes, va="top", fontsize=10,
            color=ACC_CYAN,
            bbox=dict(boxstyle="round,pad=0.3", fc=BG_GROUP, ec="#3a3f5c", alpha=0.9))


def plot_noise_budget_bar(ax, params):
    ax.cla()
    style_ax(ax, r"Noise Budget at  $\tau$ = 1 s",
             "Noise Source", r"$\sigma_y$(1 s)")

    table = sensitivity_table(params)
    names  = [NOISE_LABELS.get(k, k) for k, _ in table if k != "sigma_total"]
    values = [v for k, v in table if k != "sigma_total"]
    colors = [NOISE_COLORS.get(k, "#aaaaaa") for k, _ in table if k != "sigma_total"]

    x = np.arange(len(names))
    ax.bar(x, values, color=colors, width=0.6, edgecolor="#0f1117",
           linewidth=0.5, alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax.set_yscale("log")

    for xi, v in zip(x, values):
        if v > 0:
            ax.text(xi, v * 1.1, f"{v:.1e}", ha="center", fontsize=8, color=FG_TEXT)


def plot_temperature_sweep(ax, params):
    ax.cla()
    # Twin axis
    ax2 = ax.twinx()
    style_ax(ax, r"Temperature Sensitivity Sweep",
             "Cell Temperature  [\u00b0C]",
             r"Frequency Shift  $\Delta\nu$  [Hz]")

    sw = temperature_sweep(params)

    ax.plot(sw["T_C"], sw["dnu_bg_Hz"], color=ACC_CYAN, lw=1.5,
            label="Buffer gas shift")
    ax.plot(sw["T_C"], sw["dnu_total_Hz"], color=ACC_RED, lw=1.5,
            ls="--", label="Total shift")
    ax.axhline(0, color=FG_DIM, lw=0.5)

    # Mark inversion temperature
    T_inv_C = params.get("_T_inv_C", None)
    if T_inv_C and not np.isnan(T_inv_C):
        ax.axvline(T_inv_C, color=ACC_GOLD, lw=0.8, ls="--", alpha=0.6)
        ax.text(T_inv_C + 0.5, 0,
                f"$T_{{inv}}$={T_inv_C:.1f}\u00b0C", color=ACC_GOLD, fontsize=8)

    ax2.plot(sw["T_C"], sw["sigma_1s"], color=ACC_GREEN, lw=1.0,
             ls=":", alpha=0.85, label=r"$\sigma_y$(1 s)")
    ax2.set_ylabel(r"$\sigma_y$ (1 s)", color=ACC_GREEN, fontsize=10)
    ax2.tick_params(axis="y", colors=ACC_GREEN, labelsize=9)
    ax2.set_yscale("log")
    ax2.spines["right"].set_edgecolor(ACC_GREEN)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              fontsize=9, facecolor=BG_GROUP, edgecolor="#3a3f5c",
              labelcolor=FG_TEXT, loc="upper right")


def plot_pressure_ratio_sweep(ax, params):
    ax.cla()
    ax2 = ax.twinx()
    style_ax(ax, r"$Ar/N_2$ Pressure Ratio Optimisation",
             r"$P_{Ar} / P_{N_2}$  (at fixed total pressure)",
             r"$d\nu/dT$  [Hz/K]  (buffer gas)")

    sw = pressure_ratio_sweep(params)

    # Clip T_inv to ±200 °C to suppress the asymptote spike
    T_inv_clipped = np.clip(sw["T_inv_C"], -200.0, 200.0)

    ax.plot(sw["ratio"], sw["ddnu_dT_Hz_K"], color=ACC_CYAN, lw=1.5,
            label=r"$d\nu/dT$")
    ax.axhline(0, color=ACC_GOLD, lw=1.0, ls="--", label="Zero T-coeff.")

    ax2.plot(sw["ratio"], T_inv_clipped, color=ACC_GREEN, lw=1.0,
             ls=":", alpha=0.85, label=r"$T_{inv}$")
    ax2.axhline(params["T_C"], color=ACC_RED, lw=0.7, ls="--", alpha=0.6,
                label=f"$T_{{op}}$ = {params['T_C']:.0f} \u00b0C")
    ax2.set_ylabel(r"Inversion Temp.  $T_{inv}$  [\u00b0C]", color=ACC_GREEN, fontsize=10)
    ax2.tick_params(axis="y", colors=ACC_GREEN, labelsize=9)
    ax2.set_ylim(-200, 200)   # locked to ±200 °C — suppress divergence
    ax2.spines["right"].set_edgecolor(ACC_GREEN)

    # Mark operating ratio + zero-crossing
    op_ratio = params["P_Ar_Torr"] / max(params["P_N2_Torr"], 1e-9)
    ax.axvline(op_ratio, color=FG_DIM, lw=0.8, ls=":", alpha=0.6)
    ax.text(op_ratio + 0.04, ax.get_ylim()[0] * 0.85 if ax.get_ylim()[0] < 0 else 0.1,
            f"Current\n{op_ratio:.2f}", color=FG_DIM, fontsize=8)

    # Annotate the zero-crossing ratio
    zero_idx = np.where(np.diff(np.sign(sw["ddnu_dT_Hz_K"])))[0]
    if len(zero_idx) > 0:
        r_zero = float(sw["ratio"][zero_idx[0]])
        ax.axvline(r_zero, color=ACC_GOLD, lw=0.7, ls="-.", alpha=0.5)
        ax.text(r_zero + 0.04, ax.get_ylim()[1] * 0.8 if len(ax.get_ylim()) else 0.5,
                f"Zero-crossing\n{r_zero:.2f}",
                color=ACC_GOLD, fontsize=8)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              fontsize=9, facecolor=BG_GROUP, edgecolor="#3a3f5c",
              labelcolor=FG_TEXT, loc="upper right")

def plot_holdover(ax, params, adev_result):
    ax.cla()
    style_ax(ax, r"Time Error Prediction (Holdover)",
             r"Holdover Time  $\tau$  [s]",
             r"Absolute Phase/Time Error  $\Delta T$  [s]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    tau = adev_result["tau"]
    from noise_budget import helium_permeation_drift, vcsel_aging_drift, rb_reactivity_drift
    if "glass_thickness_mm" in params:
        drift_he = helium_permeation_drift(
            params["cell_R_m"], params["cell_L_m"], params["glass_thickness_mm"]*1e-3, params["T_K"]
        )
        drift_vcsel = vcsel_aging_drift(params.get("vcsel_aging_rate", 1.0))
        drift_rb = rb_reactivity_drift(params.get("rb_reaction_rate", 0.5))
        drift = (drift_he + drift_vcsel + drift_rb) / 86400.0  # s^-1
    else:
        drift = params.get("drift_rate_frac_per_day", 3e-11) / 86400.0
    
    # Generic TDEV integration approx: sigma_x = tau * sigma_y / sqrt(3)
    sigma_x_approx = tau * adev_result["sigma_total"] / np.sqrt(3.0)
    
    delta_T = time_error_prediction(tau, T0_offset_s=0.0, fractional_freq_offset=0.0,
                                    drift_rate_A=drift, sigma_x_tau=sigma_x_approx)

    ax.plot(tau, delta_T, color=ACC_CYAN, lw=2.0, label=r"Time Error $\Delta T$")
    
    # 1-day intercept
    if tau[-1] >= 86400:
        val_1day = float(np.interp(86400, tau, delta_T))
        ax.axvline(86400, color=ACC_GOLD, lw=1.0, ls="--", alpha=0.6)
        ax.text(86400 * 0.9, val_1day, rf" 1 day: {val_1day*1e6:.2f} $\mu$s ",
                color=ACC_GOLD, fontsize=9, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc=BG_GROUP, ec="#3a3f5c", alpha=0.9))

    ax.grid(True, which="both", color="#2a2f4a", linewidth=0.4, linestyle=":")
    ax.legend(fontsize=9, facecolor=BG_GROUP, edgecolor="#3a3f5c",
              labelcolor=FG_TEXT, loc="upper left")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP CLASS
# ─────────────────────────────────────────────────────────────────────────────

class CPTClockApp:

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("CPT Atomic Clock Performance Model  —  ⁸⁷Rb (buffer-gas & no-buffer-gas)")
        root.configure(bg=BG_DARK)
        root.geometry("1440x860")
        root.minsize(900, 600)

        self._style_ttk()
        self._build_layout()

    # ── TTK Styling ────────────────────────────────────────────────────────

    def _style_ttk(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame",      background=BG_DARK)
        style.configure("Group.TFrame",background=BG_GROUP)
        style.configure("TLabel",      background=BG_GROUP,  foreground=FG_TEXT,
                         font=("Segoe UI", 9))
        style.configure("Dim.TLabel",  background=BG_GROUP,  foreground=FG_DIM,
                         font=("Segoe UI", 8))
        style.configure("Header.TLabel", background=BG_GROUP, foreground=ACC_CYAN,
                         font=("Segoe UI", 9, "bold"))
        style.configure("TNotebook",   background=BG_DARK,   borderwidth=0)
        style.configure("TNotebook.Tab", background=BG_PANEL, foreground=FG_DIM,
                         padding=[10, 4], font=("Segoe UI", 8))
        style.map("TNotebook.Tab",
                  background=[("selected", BG_GROUP)],
                  foreground=[("selected", ACC_CYAN)])
        style.configure("Run.TButton", background=BTN_BG, foreground=BTN_FG,
                         font=("Segoe UI", 10, "bold"), padding=8, relief="flat")
        style.map("Run.TButton",
                  background=[("active", BTN_ACT), ("pressed", ACC_BLUE)])

    # ── Layout ─────────────────────────────────────────────────────────────

    def _build_layout(self):
        # Root paned window: left panel + right plots
        pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL,
                              bg=BG_DARK, sashwidth=4, sashrelief=tk.FLAT,
                              sashpad=2)
        pane.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # ── LEFT PANEL ──────────────────────────────────────────────────────
        left_outer = ttk.Frame(pane, style="TFrame")
        pane.add(left_outer, minsize=260, width=290, stretch="never")

        # Title
        hdr = tk.Label(left_outer, text="⁸⁷Rb CPT Clock Model",
                       bg=BG_PANEL, fg=ACC_CYAN,
                       font=("Segoe UI", 11, "bold"), pady=8)
        hdr.pack(fill=tk.X)

        sub = tk.Label(left_outer, text="Parameters",
                       bg=BG_PANEL, fg=FG_DIM,
                       font=("Segoe UI", 8))
        sub.pack(fill=tk.X)

        # Scrollable canvas for parameters
        canvas_frame = tk.Frame(left_outer, bg=BG_PANEL)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(canvas_frame, bg=BG_PANEL,
                           highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical",
                                  command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scroll_frame = tk.Frame(canvas, bg=BG_PANEL)
        canvas_window = canvas.create_window((0, 0), window=self.scroll_frame,
                                             anchor="nw")

        def _on_frame_configure(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
        def _on_canvas_configure(e):
            canvas.itemconfig(canvas_window, width=e.width)
        self.scroll_frame.bind("<Configure>", _on_frame_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(e):
            canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Build parameter entries grouped
        self.entries = {}
        self._build_param_groups()

        # Run button + status
        btn_frame = tk.Frame(left_outer, bg=BG_DARK, pady=6)
        btn_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_var = tk.StringVar(value="Ready")
        status_lbl = tk.Label(btn_frame, textvariable=self.status_var,
                              bg=BG_DARK, fg=FG_DIM,
                              font=("Segoe UI", 7), wraplength=260)
        status_lbl.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=2)

        run_btn = ttk.Button(btn_frame, text="▶   Run  Simulation",
                             style="Run.TButton", command=self.run)
        run_btn.pack(fill=tk.X, padx=6, pady=(0, 4))

        # ── RIGHT PANEL ─────────────────────────────────────────────────────
        right_frame = ttk.Frame(pane, style="TFrame")
        pane.add(right_frame, stretch="always")

        notebook = ttk.Notebook(right_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        tab_defs = [
            ("CPT Signal",      "tab_signal"),
            ("Shift Budget",    "tab_shifts"),
            ("Allan Dev.",      "tab_allan"),
            ("Noise Budget",    "tab_noise"),
            ("T Sweep",         "tab_tsweep"),
            ("P Ratio Opt.",    "tab_pratio"),
            ("Holdover",        "tab_holdover"),
        ]
        self.axes = {}
        self.figs = {}
        self.canvases = {}

        for label, attr in tab_defs:
            frame = ttk.Frame(notebook, style="TFrame")
            notebook.add(frame, text=f"  {label}  ")

            fig = plt.Figure(figsize=(8, 5), facecolor=BG_DARK,
                             tight_layout=True)
            ax  = fig.add_subplot(111)
            ax.set_facecolor("#12151f")
            for spine in ax.spines.values():
                spine.set_edgecolor("#3a3f5c")
            ax.tick_params(colors=FG_DIM)

            canvas_mpl = FigureCanvasTkAgg(fig, master=frame)
            canvas_mpl.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            toolbar_frame = tk.Frame(frame, bg=BG_DARK)
            toolbar_frame.pack(fill=tk.X, side=tk.BOTTOM)
            tb = NavigationToolbar2Tk(canvas_mpl, toolbar_frame)
            tb.config(bg=BG_DARK)
            tb.update()
            # Dark toolbar buttons
            for child in tb.winfo_children():
                try:
                    child.config(bg=BG_DARK, fg=FG_TEXT)
                except Exception:
                    pass

            setattr(self, attr + "_fig", fig)
            setattr(self, attr + "_ax",  ax)
            setattr(self, attr + "_canvas", canvas_mpl)

        self._annotate_empty_plots()

    def _build_param_groups(self):
        """Create grouped, labeled entry fields."""
        groups = {}
        for key, label, default, unit, group in PARAMS:
            groups.setdefault(group, []).append((key, label, default, unit))

        for group_name, items in groups.items():
            # Group header
            grp_outer = tk.Frame(self.scroll_frame, bg=BG_PANEL)
            grp_outer.pack(fill=tk.X, padx=5, pady=(6, 0))

            hdr = tk.Label(grp_outer, text=f"  ◈  {group_name}",
                           bg=BG_PANEL, fg=ACC_CYAN,
                           font=("Segoe UI", 8, "bold"), anchor="w",
                           pady=4)
            hdr.pack(fill=tk.X)

            grp = tk.Frame(grp_outer, bg=BG_GROUP, padx=6, pady=4)
            grp.pack(fill=tk.X)
            grp.columnconfigure(0, weight=3)
            grp.columnconfigure(1, weight=2)
            grp.columnconfigure(2, weight=1)

            for row_idx, (key, label, default, unit) in enumerate(items):
                lbl = tk.Label(grp, text=label, bg=BG_GROUP, fg=FG_TEXT,
                               font=("Segoe UI", 8), anchor="w")
                lbl.grid(row=row_idx, column=0, sticky="w", pady=2, padx=2)

                var = tk.StringVar(value=str(default))
                ent = tk.Entry(grp, textvariable=var, width=11,
                               bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=FG_TEXT,
                               relief=tk.FLAT, font=("Segoe UI", 8),
                               highlightthickness=1,
                               highlightcolor=ACC_BLUE,
                               highlightbackground="#3a3f5c")
                ent.grid(row=row_idx, column=1, sticky="ew", pady=2, padx=4)

                unit_lbl = tk.Label(grp, text=unit, bg=BG_GROUP, fg=FG_DIM,
                                    font=("Segoe UI", 7), anchor="w")
                unit_lbl.grid(row=row_idx, column=2, sticky="w", padx=2)

                self.entries[key] = var

    def _annotate_empty_plots(self):
        """Put placeholder text on all plots before first run."""
        for attr in ["tab_signal", "tab_shifts", "tab_allan",
                     "tab_noise", "tab_tsweep", "tab_pratio", "tab_holdover"]:
            ax  = getattr(self, attr + "_ax")
            fig = getattr(self, attr + "_fig")
            ax.set_facecolor("#12151f")
            ax.text(0.5, 0.5,
                    "Press  ▶ Run  to compute",
                    transform=ax.transAxes,
                    ha="center", va="center",
                    color=FG_DIM, fontsize=13,
                    fontstyle="italic")
            for spine in ax.spines.values():
                spine.set_edgecolor("#3a3f5c")
            ax.tick_params(colors=FG_DIM)
            getattr(self, attr + "_canvas").draw()

    # ── RUN ────────────────────────────────────────────────────────────────

    def run(self):
        self.status_var.set("⏳  Computing…")
        self.root.update_idletasks()

        try:
            self._do_run()
            self.status_var.set("✓  Done")
        except Exception as exc:
            self.status_var.set(f"✗  Error: {exc}")
            messagebox.showerror("Simulation Error",
                                 traceback.format_exc())

    def _do_run(self):
        p = params_from_entries(self.entries)

        # ── 1. Cell decoherence  (dispatches to diffusion or ballistic regime) ──
        cell = total_ground_decoherence(
            p["cell_R_m"], p["cell_L_m"],
            p["P_N2_Torr"], p["P_Ar_Torr"], p["T_K"],
            beam_diam_m=p["beam_diam_mm"] * 1e-3)

        # ── 2. CPT signal ────────────────────────────────────────────────
        p["gamma2_Hz"] = cell["gamma2_total_Hz"]
        cpt_res = compute_cpt_signal(p)
        gamma_CPT = cpt_res["gamma_CPT_Hz"]
        C_cpt     = cpt_res["contrast"]

        P_det_W = p["P_total_uW"] * 1e-6 * (1.0 - C_cpt)   # transmitted power

        # ── 3. Frequency shifts ──────────────────────────────────────────
        shifts = total_shift_budget(
            p["T_K"], p["P_N2_Torr"], p["P_Ar_Torr"],
            p["B_Gauss"], p["P_total_uW"],
            p["beam_diam_mm"], p["mod_index"],
            p.get("laser_detuning_MHz", 0.0),
            p.get("Delta_P_atm_mbar", 0.0))
        p["_T_inv_C"] = shifts["T_inversion_C"]

        # ── 4. Noise budget params ───────────────────────────────────────
        noise_p = dict(p)
        noise_p["gamma_CPT_Hz"]  = gamma_CPT
        noise_p["contrast"]      = C_cpt
        noise_p["P_det_W"]       = P_det_W
        noise_p["ddnu_dT_Hz_K"]  = shifts["ddnu_bg_dT_Hz_K"]

        # ── 5. Allan deviation ───────────────────────────────────────────
        tau = tau_array(1.0, 1e6, 25)
        adev = compute_allan_deviation(tau, noise_p)

        # ── 6. Plot ──────────────────────────────────────────────────────
        plot_lineshape(self.tab_signal_ax, p, cell, cpt_res)
        self.tab_signal_canvas.draw()

        plot_shift_budget(self.tab_shifts_ax, shifts)
        self.tab_shifts_canvas.draw()

        plot_allan_deviation(self.tab_allan_ax, adev)
        self.tab_allan_canvas.draw()

        plot_noise_budget_bar(self.tab_noise_ax, noise_p)
        self.tab_noise_canvas.draw()

        plot_temperature_sweep(self.tab_tsweep_ax, p)
        self.tab_tsweep_canvas.draw()

        plot_pressure_ratio_sweep(self.tab_pratio_ax, p)
        self.tab_pratio_canvas.draw()
        
        plot_holdover(self.tab_holdover_ax, p, adev)
        self.tab_holdover_canvas.draw()

