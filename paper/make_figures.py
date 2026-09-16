"""Regenerate all manuscript figures at publication quality.

Sized for a single-column elsarticle preprint: 5.5 in wide, 9-10 pt labels,
included at \textwidth so almost no downscaling occurs.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import os

REPO = "/home/claude/repo_current"
OUT = "/home/claude/paper/figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 9.5,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.5,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
W = 5.5


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- Fig 2
# Hourly forward curve with monthly quotes and the anchored January window
curve = pd.read_csv(f"{REPO}/outputs/market_calibration_final/hourly_forward_curve.csv",
                    parse_dates=["time_utc"])
quotes = pd.read_csv(f"{REPO}/inputs/market/vep_monthly_quotes.csv")

fig, ax = plt.subplots(figsize=(W, 3.2))
anch = curve[curve.near_term_anchor_flag]
ax.axvspan(curve.time_utc.iloc[0], anch.time_utc.iloc[-1],
           color="0.88", zorder=0)
ax.text(anch.time_utc.iloc[len(anch) // 2], 3620, "unquoted\n(anchored)",
        ha="center", va="top", fontsize=8, color="0.35")
ax.plot(curve.time_utc, curve.hourly_forward_TRY_MWh, color="#1f4e79",
        lw=1.2, label=r"fitted hourly curve $F(t)$")
for _, q in quotes.iterrows():
    m = curve[curve.delivery_month == f"{int(q.delivery_year)}-{int(q.delivery_month):02d}"]
    if len(m):
        ax.hlines(q.price_TRY_MWh, m.time_utc.iloc[0], m.time_utc.iloc[-1],
                  color="#c0392b", lw=2.0, zorder=3,
                  label="observed VEP monthly quote" if q.contract_name == "EBM0226" else None)
ax.set_xlabel("Delivery date, 2026")
ax.set_ylabel("TRY/MWh")
ax.set_ylim(2150, 3700)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.legend(frameon=False, loc="lower left")
save(fig, "fwd_curve")

# ---------------------------------------------------------------- Fig 3/4
grid = pd.read_csv(f"{REPO}/outputs/market_calibration_final/strike_maturity_grid.csv")
mats = sorted(grid.maturity_h.unique())
strikes = sorted(grid.strike_TRY_MWh.unique())

fig, (axA, axB) = plt.subplots(2, 1, figsize=(W, 6.0))

# (a) strike sections
cmap = plt.cm.viridis(np.linspace(0.05, 0.88, len(mats)))
for c, m in zip(cmap, mats):
    ssub = grid[grid.maturity_h == m].sort_values("strike_TRY_MWh")
    axA.plot(ssub.strike_TRY_MWh, ssub.call_TRY_MWh, color=c, marker="o", ms=3,
             label=f"$T={int(m)}$ h")
fmin, fmax = grid.F_T_TRY_MWh.min(), grid.F_T_TRY_MWh.max()
axA.axvspan(fmin, fmax, color="0.55", alpha=0.35, zorder=0)
axA.annotate("expiry-hour $F(T)$", xy=((fmin + fmax) / 2, 250),
             xytext=(2040, 120), fontsize=8.5, color="0.25",
             arrowprops=dict(arrowstyle="->", color="0.45", lw=0.8))
axA.set_xlabel("Strike (TRY/MWh)")
axA.set_ylabel("Call value (TRY/MWh)")
axA.legend(frameon=False, ncol=3, fontsize=7.8, loc="upper right")
axA.set_ylim(-45, 1180)
axA.set_title("(a) sections at fixed maturity", fontsize=9.5, loc="left")

# (b) maturity sections, representative strikes
sub = [2000, 2400, 2800, 3000, 3400, 4000]
cmap = plt.cm.viridis(np.linspace(0.05, 0.88, len(sub)))
for c, k in zip(cmap, sub):
    ssub = grid[grid.strike_TRY_MWh == k].sort_values("maturity_h")
    atm = abs(k - 3000) < 1
    axB.plot(ssub.maturity_h, ssub.call_TRY_MWh,
             color="black" if atm else c, marker="o", ms=3,
             lw=2.2 if atm else 1.3,
             label=f"$K={int(k)}$" + (" (ATM)" if atm else ""))
axB.set_xscale("log")
axB.set_xticks(mats)
axB.set_xticklabels([str(int(m)) for m in mats])
axB.set_xlabel("Maturity (hours, log scale)")
axB.set_ylabel("Call value (TRY/MWh)")
axB.set_ylim(-45, 1330)
axB.legend(frameon=False, ncol=3, fontsize=7.8, loc="upper center")
axB.set_title("(b) sections at fixed strike", fontsize=9.5, loc="left")

fig.subplots_adjust(hspace=0.34)
save(fig, "option_surface")

# ---------------------------------------------------------------- Fig 5
rd = pd.read_csv(f"{REPO}/outputs/forward_centered_diagnostics/residual_diagnostics.csv")
fig, ax = plt.subplots(figsize=(W, 3.0))
ax.plot(rd.hours, rd.residual_std_TRY_MWh, color="#1f4e79",
        label=r"production, $\kappa=0.0784$ h$^{-1}$")
arch_h = np.array([24, 72, 168, 336])
arch_sd = np.array([1033.5, 1813.5, 2736.0, 3756.0])
ax.plot(arch_h, arch_sd, "s--", color="#c0392b", ms=4, lw=1.2,
        label=r"archived, $\kappa=4.11\times10^{-6}$ h$^{-1}$")
ax.set_xlabel("Horizon (hours)")
ax.set_ylabel("Residual SD (TRY/MWh)")
ax.set_xlim(0, 730)
ax.legend(frameon=False, loc="center right")
save(fig, "residual_sd")

# ---------------------------------------------------------------- Fig 6
an = pd.read_csv(f"{REPO}/outputs/market_calibration_final/near_term_anchor_sensitivity.csv")
an = an.sort_values("january_anchor_TRY_MWh")
fig, ax = plt.subplots(figsize=(W, 3.1))
ax.plot(an.january_anchor_TRY_MWh, an.option_value_TRY_MWh,
        "o-", color="#1f4e79", ms=5)
base = an[abs(an.anchor_vs_spot_pct) < 1e-6]
ax.plot(base.january_anchor_TRY_MWh, base.option_value_TRY_MWh, "o",
        color="#c0392b", ms=8, zorder=5, label="production anchor (observed spot)")
for _, r in an.iterrows():
    ax.annotate(f"{r.option_value_pct_vs_mid:+.0f}%",
                (r.january_anchor_TRY_MWh, r.option_value_TRY_MWh),
                textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=8, color="0.3")
ax.set_xlabel("Assumed January 2026 anchor level (TRY/MWh)")
ax.set_ylabel("72 h $K=3{,}000$ call (TRY/MWh)")
ax.set_ylim(-40, 620)
ax.legend(frameon=False, loc="upper left")
save(fig, "anchor_sensitivity")

# ---------------------------------------------------------------- Fig 7
lg = pd.read_csv(f"{REPO}/outputs/market_calibration_final/legacy_vs_forward_centered.csv")
fig, ax = plt.subplots(figsize=(W, 3.0))
ax.axhline(0, color="0.6", lw=0.8)
ax.plot(lg.horizon_hours, lg.forward_centered_TRY_MWh, "o-", color="#1f4e79",
        ms=5, label=r"centred: $\mathbb{E}^{\mathbb{Q}}[P_T]=F(T)$")
ax.plot(lg.horizon_hours, lg.legacy_analytic_TRY_MWh, "s--", color="#c0392b",
        ms=5, label="multiplicative (archived specification)")
ax.set_xlabel("Horizon (hours)")
ax.set_ylabel(r"Implied $\mathbb{E}[P_T]$ (TRY/MWh)")
ax.set_xticks(lg.horizon_hours)
ax.legend(frameon=False, loc="center right")
save(fig, "legacy_vs_centered")

# ---------------------------------------------------------------- Fig 8
sw = pd.read_csv(f"{REPO}/outputs/scenario_sweep/rd_scenario_sweep.csv")
fig, ax = plt.subplots(figsize=(W, 3.0))
ax.plot(sw.rd_offset_sigma, sw.call_value_TRY_MWh, "o-", color="#1f4e79", ms=4.5)
b = sw[sw.rd_offset_sigma == 0]
ax.plot(b.rd_offset_sigma, b.call_value_TRY_MWh, "o", color="#c0392b", ms=8,
        zorder=5, label="production scenario")
ax2 = ax.twinx()
ax2.plot(sw.rd_offset_sigma, sw.residual_sd_TRY_MWh, "^:", color="0.45",
         ms=4, lw=1.1, label="terminal residual SD")
ax2.set_ylabel("Residual SD (TRY/MWh)", color="0.35")
ax2.tick_params(axis="y", colors="0.35")
ax2.grid(False)
ax.set_xlabel(r"Residual-demand offset ($\sigma$ units)")
ax.set_ylabel("72 h $K=3{,}000$ call (TRY/MWh)")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, frameon=False, loc="lower left")
save(fig, "rd_sweep")

# ---------------------------------------------------------------- Fig 9
real = pd.read_csv(f"{REPO}/inputs/market/realized_ptf_2026.csv", sep=";",
                   encoding="utf-8-sig")
real.columns = [c.strip() for c in real.columns]
pcol = [c for c in real.columns if "TL/MWh" in c][0]
real["price"] = (real[pcol].astype(str).str.replace(".", "", regex=False)
                 .str.replace(",", ".", regex=False).astype(float))
real["ts_local"] = pd.to_datetime(real["Tarih"] + " " + real["Saat"],
                                  format="%d.%m.%Y %H:%M")
real["time_utc"] = (real["ts_local"].dt.tz_localize("Europe/Istanbul",
                                                    ambiguous="NaT",
                                                    nonexistent="NaT")
                    .dt.tz_convert("UTC"))
merged = pd.merge(curve[["time_utc", "hourly_forward_TRY_MWh"]],
                  real[["time_utc", "price"]], on="time_utc", how="inner")
merged["day"] = merged.time_utc.dt.floor("D")
daily = merged.groupby("day").agg(realized=("price", "mean"),
                                  forward=("hourly_forward_TRY_MWh", "mean"))

mo = pd.read_csv(f"{REPO}/outputs/market_calibration_final/realized_2026_backtest.monthly.csv")
labels = [pd.to_datetime(m).strftime("%b") for m in mo.month]
xb = np.arange(len(mo))

fig, (axA, axB) = plt.subplots(2, 1, figsize=(W, 5.6),
                               gridspec_kw=dict(height_ratios=[1.25, 1]))

axA.fill_between(daily.index, daily.realized, daily.forward,
                 where=daily.forward >= daily.realized,
                 color="#c0392b", alpha=0.13, linewidth=0)
axA.plot(daily.index, daily.forward, color="#c0392b", lw=1.6,
         label="forward curve, daily mean")
axA.plot(daily.index, daily.realized, color="#1f4e79", lw=1.0,
         label="realised PTF, daily mean")
axA.set_xlabel("Delivery date, 2026")
axA.set_ylabel("TRY/MWh")
axA.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
axA.xaxis.set_major_locator(mdates.MonthLocator())
axA.set_ylim(0, 3750)
axA.legend(frameon=False, loc="lower left", fontsize=8)
axA.set_title("(a) daily trajectory", fontsize=9.5, loc="left")

cols = ["#5b8db8" if a else "#c0392b" for a in mo.near_term_anchor]
axB.bar(xb, mo.mean_bias_realized_minus_forward, color=cols, width=0.66)
for i, (b, r) in enumerate(zip(mo.mean_bias_realized_minus_forward,
                               mo.model_over_realized_ratio)):
    axB.text(i, b - 95, f"{r:.2f}", ha="center", fontsize=8, color="0.25")
axB.axhline(0, color="0.3", lw=0.8)
axB.set_xticks(xb); axB.set_xticklabels(labels)
axB.set_xlabel("Delivery month, 2026")
axB.set_ylabel("Mean bias (TRY/MWh)")
axB.set_ylim(-2250, 250)
from matplotlib.patches import Patch
axB.legend(handles=[Patch(color="#5b8db8", label="anchored (unquoted)"),
                    Patch(color="#c0392b", label="VEP-quoted")],
           frameon=False, loc="lower left", fontsize=8)
axB.text(0.99, 0.90, "annotations: model/realised dispersion ratio",
         transform=axB.transAxes, ha="right", fontsize=7.5, color="0.35")
axB.set_title("(b) monthly bias", fontsize=9.5, loc="left")

fig.subplots_adjust(hspace=0.42)
save(fig, "backtest")

print("done")

# ---------------------------------------------------------------- Fig: TVTP
import yaml
par = yaml.safe_load(open(f"{REPO}/inputs/historical/m2_frozen_parameters.yaml"))
tv = par["tvtp"]
a01, g01 = float(tv["alpha01"]), float(tv["gamma01"])
a10, g10 = float(tv["alpha10"]), float(tv["gamma10"])
z = pd.read_csv(f"{REPO}/inputs/historical/rd_standardized.csv")["z"].values
zz = np.linspace(-3.2, 3.2, 400)
L = lambda x: 1.0 / (1.0 + np.exp(-x))

fig, ax = plt.subplots(figsize=(W, 3.2))
axh = ax.twinx()
axh.hist(z, bins=90, range=(-3.2, 3.2), color="0.82", zorder=0)
axh.set_yticks([]); axh.grid(False)
axh.set_ylim(0, np.histogram(z, bins=90, range=(-3.2, 3.2))[0].max() * 3.4)
ax.set_zorder(axh.get_zorder() + 1); ax.patch.set_visible(False)
ax.plot(zz, L(a01 + g01 * zz), color="#c0392b", lw=1.8,
        label=r"$p_{01}(z)$: calm $\rightarrow$ stress")
ax.plot(zz, L(a10 + g10 * zz), color="#1f4e79", lw=1.8,
        label=r"$p_{10}(z)$: stress $\rightarrow$ calm")
ax.axvline(0, color="0.5", lw=0.8, ls=":")
ax.set_xlabel(r"Standardised residual demand $z_{t-1}$")
ax.set_ylabel("Hourly transition probability")
ax.set_xlim(-3.2, 3.2); ax.set_ylim(0, 0.45)
ax.legend(frameon=False, loc="upper right")
ax.text(-3.05, 0.055, "shaded: sample\ndistribution of $z$",
        fontsize=7.5, color="0.45", va="bottom")
save(fig, "tvtp_mechanism")

# ---------------------------------------------------------------- Fig: PDE/MC
import json
pm = pd.DataFrame(json.load(open(f"{OUT}/pde_mc.json")))
fig, (axa, axb) = plt.subplots(2, 1, figsize=(W, 4.0), sharex=True,
                               gridspec_kw=dict(height_ratios=[2.1, 1],
                                                hspace=0.13))
axa.plot(pm.strike, pm.pde, color="#1f4e79", lw=1.6, label="PDE (finite difference)")
axa.errorbar(pm.strike, pm.mc, yerr=1.96 * pm.mc_se, fmt="o", ms=4.5,
             color="#c0392b", capsize=3, lw=1.1,
             label=r"Monte Carlo, $4\times10^{4}$ paths (95% CI)")
axa.set_ylabel("Call value (TRY/MWh)")
axa.legend(frameon=False)
d = 100 * (pm.mc - pm.pde) / pm.pde
axb.axhline(0, color="0.5", lw=0.8)
axb.plot(pm.strike, d, "o-", color="0.35", ms=4)
axb.set_ylabel("MC $-$ PDE (%)")
axb.set_xlabel("Strike (TRY/MWh)")
axb.set_ylim(-3.6, 0.7)
save(fig, "pde_mc")
print("extra figures done")

# ---------------------------------------------------------------- Fig: multi-date
dates = ["2022-12-31", "2023-06-30", "2023-12-31", "2024-06-30",
         "2024-12-31", "2025-06-30", "2025-12-31"]
relbias = [-39.8, -36.5, -24.5, -6.2, 7.2, 11.4, -39.1]
mae_h = [228.9, 675.0, 782.2, 1120.8, 970.8, 869.4, 708.4]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(W, 2.85),
                               gridspec_kw=dict(width_ratios=[1.25, 1]))

cols = ["#c0392b" if d == "2025-12-31" else "#5b8db8" for d in dates]
xp = np.arange(len(dates))
axA.bar(xp, relbias, color=cols, width=0.68)
axA.axhline(0, color="0.3", lw=0.8)
axA.set_xticks(xp)
axA.set_xticklabels([d[2:7] for d in dates], rotation=45, ha="right", fontsize=8)
axA.set_ylabel("Mean relative bias (%)")
axA.set_xlabel("Valuation date")
axA.set_ylim(-52, 20)
axA.set_title("(a) by valuation date", fontsize=9.5, loc="left")
from matplotlib.patches import Patch as _P
axA.legend(handles=[_P(color="#c0392b", label="date used in Sec. 7.4")],
           frameon=False, loc="lower left", fontsize=7.5)

axB.plot(np.arange(1, 8), mae_h, "o-", color="#1f4e79", ms=5)
axB.plot([1], [mae_h[0]], "o", color="#c0392b", ms=7, zorder=5)
axB.annotate("anchored\nmonth", xy=(1, mae_h[0]), xytext=(1.5, 430),
             fontsize=7.5, color="#c0392b",
             arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.8))
axB.set_xlabel("Horizon (months)")
axB.set_ylabel("MAE (TRY/MWh)")
axB.set_xticks(range(1, 8))
axB.set_ylim(0, 1250)
axB.set_title("(b) by delivery horizon", fontsize=9.5, loc="left")

fig.subplots_adjust(wspace=0.42)
save(fig, "multi_date")
print("multi-date figure done")
