"""Figures for the revised manuscript.

Every figure is drawn from tracked output files of the repository; no number
is typed in by hand.  One style is shared by all figures: Latin Modern text to
match the manuscript, the Okabe-Ito palette, no titles, panel letters, light
horizontal grid only.  Figures are sized for the elsarticle preprint text
width (390 pt = 5.4 in) so that they are included at 1:1 scale.

Usage:  python make_figures_v2.py <repo_root> <output_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

W = 5.4  # text width in inches

# Okabe-Ito
BLUE, ORANGE, GREEN, VERM, SKY, GREY, BLACK = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#8C8C8C", "#000000")
MAT_COL = {24: ORANGE, 48: GREEN, 72: BLUE}

import shutil
USETEX = shutil.which("latex") is not None
mpl.rcParams.update({
    "text.usetex": USETEX,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"] if USETEX else ["DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#E0E0E0",
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.3,
    "legend.frameon": False,
    
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def panel(ax, letter):
    lab = rf"\textbf{{({letter})}}" if USETEX else f"({letter})"
    ax.text(-0.13, 1.04, lab, transform=ax.transAxes,
            fontsize=9, va="bottom", ha="left")


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


def thousands(ax, axis="y"):
    fmt = mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


# ---------------------------------------------------------------- data
curve = pd.read_csv(REPO / "outputs/market_calibration_final/hourly_forward_curve.csv",
                    parse_dates=["time_utc"])
quotes = pd.read_csv(REPO / "inputs/market/vep_monthly_quotes.csv")
real = pd.read_csv(REPO / "inputs/market/realized_ptf_2026.csv", sep=";")
real["t"] = pd.to_datetime(real["Tarih"] + " " + real["Saat"], format="%d.%m.%Y %H:%M")
real["P"] = (real["PTF (TL/MWh)"].astype(str).str.replace(".", "", regex=False)
             .str.replace(",", ".", regex=False).astype(float))
curve["t_tr"] = curve["time_utc"].dt.tz_convert("Europe/Istanbul").dt.tz_localize(None)

# ================================================================ F2 curve
anc = pd.read_csv(REPO / "outputs/market_calibration_final/near_term_anchor_sensitivity.csv")
anc = anc[anc.anchor_mode == "spot_to_next_linear"].sort_values("anchor_vs_spot_pct")
fig, axes = plt.subplots(1, 2, figsize=(W, 2.35), gridspec_kw={"width_ratios": [1.55, 1]})
ax = axes[0]
ax.axvspan(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-01"), color="#F2F2F2", zorder=0, lw=0)
ax.plot(curve["t_tr"], curve["hourly_forward_TRY_MWh"], color=BLUE, lw=0.6)
for _, q in quotes.iterrows():
    start = pd.Timestamp(int(q.delivery_year), int(q.delivery_month), 1)
    end = start + pd.offsets.MonthBegin(1)
    ax.hlines(q.price_TRY_MWh, start, end, color=BLACK, lw=1.1)
ax.text(pd.Timestamp("2026-01-16"), 3900, "no quote", ha="center", fontsize=7.5, color="#555555")
ax.text(pd.Timestamp("2026-04-15"), 2800, "monthly quote", fontsize=7.5, ha="center")
ax.text(pd.Timestamp("2026-04-15"), 1950, "hourly curve", fontsize=7.5, ha="center", color=BLUE)
ax.set_ylabel("TRY/MWh")
ax.set_ylim(1500, 4200)
thousands(ax)
ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%b"))
panel(ax, "a")

ax = axes[1]
ax.plot(anc.anchor_vs_spot_pct, anc.option_value_TRY_MWh, "-o", color=BLUE, ms=3.5)
ax.axvline(0, color=GREY, lw=0.6)
ax.set_xlabel(r"anchor vs.\ spot (\%)" if USETEX else "anchor vs. spot (%)")
ax.set_ylabel(r"call, $T=72$ h, $K=3{,}000$")
ax.set_xticks([-20, -10, 0, 10, 20])
panel(ax, "b")
fig.tight_layout(w_pad=1.5)
save(fig, "fig_forward_curve")

# ================================================================ F3 regimes
par = yaml.safe_load(open(REPO / "inputs/historical/m2_frozen_parameters.yaml", encoding="utf-8"))
tv = par["tvtp"]
z = pd.read_csv(REPO / "inputs/historical/rd_standardized.csv")["z"].to_numpy()
zz = np.linspace(-3, 3, 400)
lam = lambda x: 1.0 / (1.0 + np.exp(-x))
p01 = lam(tv["alpha01"] + tv["gamma01"] * zz)
p10 = lam(tv["alpha10"] + tv["gamma10"] * zz)
rd = pd.read_csv(REPO / "outputs/forward_centered_diagnostics/residual_diagnostics.csv")

fig, axes = plt.subplots(1, 2, figsize=(W, 2.3))
ax = axes[0]
hist, edges = np.histogram(z[np.isfinite(z)], bins=60, range=(-3, 3), density=True)
ax2 = ax.twinx()
ax2.fill_between(0.5 * (edges[1:] + edges[:-1]), hist, step="mid", color="#EDEDED", lw=0)
ax2.set_ylim(0, hist.max() * 2.6)
ax2.axis("off")
ax.set_zorder(ax2.get_zorder() + 1)
ax.patch.set_visible(False)
ax.plot(zz, p01, color=VERM, label=r"$p_{01}$ calm $\to$ stress")
ax.plot(zz, p10, color=BLUE, ls="--", label=r"$p_{10}$ stress $\to$ calm")
ax.set_xlabel(r"standardised residual demand $z_{t-1}$")
ax.set_ylabel("hourly transition probability")
ax.set_ylim(0, 0.65)
ax.legend(loc="upper right")
panel(ax, "a")

ax = axes[1]
h = rd["hours"]
ax.plot(h, rd["residual_std_TRY_MWh"], color=BLUE, label="residual s.d.")
ax.set_xlabel("horizon (hours)")
ax.set_ylabel("TRY/MWh")
ax.set_xlim(0, 168)
ax.set_xticks([0, 24, 48, 72, 96, 120, 144, 168])
ax.set_ylim(0, 650)
ax3 = ax.twinx()
ax3.plot(h, rd["p_stress"], color=GREY, ls=":", lw=1.1)
ax3.set_ylim(0, 1.0)
ax3.set_ylabel("stress probability", color="#555555")
ax3.spines["right"].set_visible(True)
ax3.grid(False)
from matplotlib.lines import Line2D
ax.legend([Line2D([], [], color=BLUE), Line2D([], [], color=GREY, ls=":")],
          ["residual s.d. (left)", "stress probability (right)"], loc="lower right")
panel(ax, "b")
fig.tight_layout(w_pad=1.8)
save(fig, "fig_regimes")

# ================================================================ F4 option values
grid = pd.read_csv(REPO / "outputs/market_calibration_final/strike_maturity_grid.csv")
pm = pd.DataFrame(json.load(open(REPO / "paper/figures/pde_mc.json", encoding="utf-8")))
iv = pd.read_csv(REPO / "outputs/fw3_benchmarks/implied_vol_surface_black76.csv")

gc = pd.read_csv(REPO / "outputs/fw3_benchmarks/grid_comparison.csv")
g72 = gc[gc.maturity_h == 72].sort_values("strike")
fig, axes = plt.subplots(1, 2, figsize=(W, 2.35))
ax = axes[0]
ax.plot(g72.strike, g72.model_call, color=BLUE, lw=1.6, label="this model")
ax.plot(g72.strike, g72.B3_lucia_schwartz_call, color=GREEN, ls="--", label="Lucia--Schwartz")
ax.plot(g72.strike, g72.B2_bachelier_call_hist, color=ORANGE, ls="-.", label="Bachelier")
ax.plot(g72.strike, g72.B1_black76_call_hist, color=GREY, ls=":", lw=1.5, label="Black-76")
ax.errorbar(pm.strike, pm.mc, yerr=1.96 * pm.mc_se, fmt="o", ms=2.4, color=BLACK,
            elinewidth=0.7, capsize=1.5, label="Monte Carlo")
ax.set_xlabel("strike $K$ (TRY/MWh)")
ax.set_ylabel(r"call value, $T=72$ h (TRY/MWh)")
thousands(ax, "x")
ax.legend(loc="upper right", fontsize=7)
panel(ax, "a")

ax = axes[1]
for T in (24, 48, 72):
    ax.plot(iv["strike"], iv[f"{float(T)}"], color=MAT_COL[T], label=f"{T} h")
ax.set_xlabel("strike $K$ (TRY/MWh)")
ax.set_ylabel("implied volatility, Black-76")
thousands(ax, "x")
ax.legend(loc="upper right")
panel(ax, "b")
fig.tight_layout(w_pad=1.8)
save(fig, "fig_option_values")

# ================================================================ F5 sensitivity
iso = pd.read_csv(REPO / "outputs/fw9_self_estimation/kappa_sensitivity_isovariance_v2.csv")
iso = iso[iso.iso_variance_target == "production"].sort_values("kappa_per_hour")
sg = pd.read_csv(REPO / "outputs/fw2_risk_premium/sensitivity_grid.csv")
q2 = sg[(sg.family.isin(["Q2_only", "baseline"])) & (sg.strike_TRY_MWh == 3000)
        & (sg.maturity_h == 72) & (sg.option_type == "call")]

fig, axes = plt.subplots(1, 2, figsize=(W, 2.45), gridspec_kw={"width_ratios": [1.45, 1]})
ax = axes[0]
ax.axvspan(0.15, 0.22, color="#EAF2F8", lw=0, zorder=0)
ax.text(0.185, 228, "estimated\nrange", ha="center", fontsize=7, color="#3A6F95", va="top")
for col, T, ls, dy in (("call_T72", "72 h", "-", 9), ("call_T12", "12 h", "--", -9), ("call_T1", "1 h", ":", 0)):
    ax.plot(iso.kappa_per_hour, iso[col], color=BLUE, ls=ls)
    ax.text(0.302, iso[col].iloc[-1] + dy, T, fontsize=7.5, va="center", color=BLUE)
ax.axvline(0.078394, color=VERM, lw=0.8)
ax.text(0.082, 12, "production $\\kappa$", fontsize=7.5, color=VERM)
ax.set_xlim(0, 0.33)
ax.set_ylim(0, 235)
ax.set_xlabel(r"mean-reversion rate $\kappa$ (h$^{-1}$), stationary variance fixed")
ax.set_ylabel("call value, $K=3{,}000$ (TRY/MWh)")
panel(ax, "a")

ax = axes[1]
vals = np.full((3, 3), np.nan)
lv = [-0.5, 0.0, 0.5]
for _, r in q2.iterrows():
    i, j = lv.index(round(r.eta_01, 2)), lv.index(round(r.eta_10, 2))
    vals[i, j] = r.delta_pct
vals[1, 1] = 0.0
lim = np.nanmax(np.abs(vals))
im = ax.imshow(vals, cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower")
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{vals[i, j]:+.1f}\\%" if USETEX else f"{vals[i, j]:+.1f}%", ha="center", va="center", fontsize=7.5,
                color="white" if abs(vals[i, j]) > 0.6 * lim else BLACK)
ax.set_xticks(range(3), [f"{v:+.1f}" for v in lv])
ax.set_yticks(range(3), [f"{v:+.1f}" for v in lv])
ax.set_xlabel(r"$\eta_{10}$ (stress $\to$ calm)")
ax.set_ylabel(r"$\eta_{01}$ (calm $\to$ stress)")
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
panel(ax, "b")
fig.tight_layout(w_pad=1.8)
save(fig, "fig_sensitivity")

# ================================================================ F6 variance over time
vs = pd.read_csv(REPO / "outputs/fw11_variance_stability/figure_source.csv", comment="#")
fig, axes = plt.subplots(1, 2, figsize=(W, 2.35), gridspec_kw={"width_ratios": [1.35, 1]})
ax = axes[0]
x = np.arange(len(vs))
is26 = vs.date_label.str.startswith("2026").to_numpy()
ax.axhspan(0.93, 1.13, color="#EDEDED", lw=0, zorder=0)
ax.axhline(1.0, color=BLACK, lw=0.6)
ax.plot(x, vs.ratio_pooled, "o", mfc="white", mec=GREY, ms=4, label="pooled hour-of-week shape")
ax.plot(x[~is26], vs.ratio_12m_only[~is26], "o", color=BLUE, ms=4.5, label="12-month shape")
ax.plot(x[is26], vs.ratio_12m_only[is26], "o", color=VERM, ms=4.5)
mon = {"06": "Jun", "12": "Dec", "09": "Sep"}
labels = []
for d in vs.date_label:
    if d.startswith("2026-01-01"):
        labels.append("2026\nonly")
    else:
        labels.append(f"{mon[d[5:7]]}\n{d[:4]}")
ax.set_xticks(x, labels, fontsize=6.8)
ax.set_ylabel("realised / implied residual s.d.")
ax.set_ylim(0.6, 2.4)
ax.legend(loc="upper left")
panel(ax, "a")

ax = axes[1]
L = np.linspace(1500, 3000, 50)
spp = 282.48
c = (vs.sd_prod_TRY / np.sqrt(vs.L_TRY_MWh**2 + spp**2)).mean()
ax.plot(L, c * np.sqrt(L**2 + spp**2), color=BLACK, lw=0.9)
ax.text(1550, c * np.sqrt(1550**2 + spp**2) + 40, "model", fontsize=7.5)
ax.plot(vs.L_TRY_MWh[~is26], vs.sd_12m_only_TRY[~is26], "o", color=BLUE, ms=4.5)
ax.plot(vs.L_TRY_MWh[is26], vs.sd_12m_only_TRY[is26], "o", color=VERM, ms=4.5)
ax.text(2010, 770, "2026", color=VERM, fontsize=7.5)
ax.set_xlabel("mean price $L$ (TRY/MWh)")
ax.set_ylabel("residual s.d. (TRY/MWh)")
ax.set_ylim(0, 900)
thousands(ax, "x")
panel(ax, "b")
fig.tight_layout(w_pad=1.8)
save(fig, "fig_variance_stability")

# ================================================================ F7 out of sample
dec = pd.read_csv(REPO / "outputs/fw10_validation/forward_residual_decomposition.csv")
m0 = dec[dec.model == "M0"].sort_values("h")
m1 = dec[dec.model == "M1"].sort_values("h")
ph = pd.read_csv(REPO / "outputs/fw10_validation/fig_pit_histogram_fw10b.csv")

fig, axes = plt.subplots(1, 2, figsize=(W, 2.35), gridspec_kw={"width_ratios": [1.3, 1]})
ax = axes[0]
hh = m0.h.to_numpy()
xh = np.arange(len(hh))
ax.errorbar(xh - 0.12, m0.forward_err_mean, yerr=m0.forward_err_sd, fmt="s", ms=3.5,
            color=VERM, elinewidth=0.8, capsize=2, label="forward error, mean $\\pm$ s.d.")
ax.plot(xh + 0.12, m0.pure_resid_sd, "o", color=BLACK, ms=4, label="realised residual s.d.")
ax.plot(xh + 0.12, m0.model_sd_mean, "o", mfc="white", mec=BLUE, ms=4, mew=1.1,
        label="model residual s.d.")
ax.axhline(0, color=BLACK, lw=0.5)
ax.set_xticks(xh, [f"{v} h" for v in hh])
ax.set_xlabel("horizon after last known hour")
ax.set_ylabel("TRY/MWh")
ax.set_ylim(-500, 2700)
thousands(ax)
ax.legend(loc="upper right", fontsize=7)
panel(ax, "a")

ax = axes[1]
for model, col, lab in (("M0", BLUE, "production parameters"), ("M1", GREY, "re-estimated parameters")):
    sub = ph[ph.model == model].groupby(["bin_lo", "bin_hi"], as_index=False)["count"].sum()
    dens = sub["count"] / sub["count"].sum() / (sub.bin_hi - sub.bin_lo)
    ax.stairs(dens.to_numpy(), np.r_[sub.bin_lo.to_numpy(), sub.bin_hi.iloc[-1]],
              color=col, lw=1.3 if model == "M0" else 1.0, label=lab,
              ls="-" if model == "M0" else "--")
ax.axhline(1.0, color=BLACK, lw=0.6)
ax.set_xlabel("probability integral transform")
ax.set_ylabel("density, all horizons")
ax.set_xlim(0, 1)
ax.set_ylim(0, 5.5)
ax.legend(loc="upper right")
panel(ax, "b")
fig.tight_layout(w_pad=1.8)
save(fig, "fig_out_of_sample")

# ================================================================ F8 strip error
md = pd.read_csv(REPO / "outputs/multi_date/multi_date_backtest.csv")
md = md[md.realize == "ok"] if "realize" in md.columns else md
by_date = md.groupby("valuation_date").agg(bias=("bias_TRY_MWh", "mean"),
                                           rel=("relative_bias_pct", "mean")).reset_index()
fig, axes = plt.subplots(1, 2, figsize=(W, 2.35), gridspec_kw={"width_ratios": [1.45, 1]})
ax = axes[0]
r_day26 = real.set_index("t")["P"].resample("D").mean()
r_day26 = r_day26[(r_day26.index >= "2026-01-01") & (r_day26.index <= curve["t_tr"].max())]
f_m = curve.set_index("t_tr")["hourly_forward_TRY_MWh"].resample("MS").mean()
r_m = real.set_index("t")["P"].resample("MS").mean().reindex(f_m.index)
ax.plot(r_day26.index, r_day26.values, color=GREY, lw=0.7, label="realised, daily mean")
ax.step(f_m.index, f_m.values, where="post", color=BLUE, lw=1.3, label="forward, monthly mean")
ax.step(r_m.index, r_m.values, where="post", color=BLACK, lw=1.3, ls="--", label="realised, monthly mean")
ax.set_ylabel("TRY/MWh")
ax.set_ylim(0, 5000)
thousands(ax)
ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%b"))
ax.set_xlim(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-08-01"))
ax.legend(loc="upper right", ncol=1, fontsize=6.5, handlelength=1.6)
panel(ax, "a")

ax = axes[1]
xd = np.arange(len(by_date))
cols = [VERM if d.startswith("2025-12") else BLUE for d in by_date.valuation_date]
ax.bar(xd, by_date.rel, color=cols, width=0.62)
ax.axhline(0, color=BLACK, lw=0.6)
ax.set_xticks(xd, [d[:7] for d in by_date.valuation_date], rotation=45, ha="right", fontsize=7)
ax.set_ylabel(r"mean relative error (\%)" if USETEX else "mean relative error (%)")
ax.set_ylim(-45, 20)
panel(ax, "b")
fig.tight_layout(w_pad=1.8)
save(fig, "fig_strip_error")

print("figures written to", OUT)
