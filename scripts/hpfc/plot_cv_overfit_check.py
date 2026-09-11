"""Visual overfit check for the recency-weighted HPFC shape model.

Usage (repository root, venv active):

    python scripts/hpfc/plot_cv_overfit_check.py

Reads only already-computed real outputs -- nothing is re-fitted here:
  outputs/hpfc/hpfc_cv_results.csv      rolling-origin CV scores (K, H, test year)
  outputs/hpfc/hpfc_selection.json      selected (H*, K*), best score, equal-weight score
  outputs/hpfc/hpfc_profile_2026.csv    fitted shape by month x day type x hour
  outputs/hpfc/hpfc_eval_2026.csv       2026 out-of-sample monthly evaluation
  inputs/market/realized_ptf_2026.csv   realised EPIAS PTF 2026 (the series fit_hpfc.py
                                        scores against; ptf_raw/ stops at 2025)

Left panel : mean out-of-sample ratio RMSE vs half-life H, one line per K, the
             selected (H*, K*) starred, the equal-weight baseline as a reference
             line and the tie-tolerance band that produced the choice.
Right panel: July 2026 (never seen by the fit) realised average hourly profile
             vs the fitted HPFC shape, both normalised to mean 1 over the month
             exactly as in hpfc._ratio_frame / ShapeModel.shape.

Writes outputs/hpfc/cv_overfit_check.png and prints every plotted number.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from pde_option_model.hpfc import TURKEY_TZ, day_type, turkish_holidays   # noqa: E402
from pde_option_model.premium import load_epias_ptf_csv                   # noqa: E402

OUT = REPO / "outputs/hpfc"
TIE_TOL = 0.005          # select_half_life(tie_tol=0.005): relative tolerance on the mean score
MONTH = 7                # July 2026: strongest midday solar dip, 59.8 % variance reduction
DAY_TYPE_NAMES = {0: "working day", 1: "Saturday", 2: "Sunday / holiday"}

# reference categorical palette, fixed slot order (dataviz skill, validated instance)
SERIES = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d3"


def main() -> int:
    sel = json.loads((OUT / "hpfc_selection.json").read_text(encoding="utf-8"))
    cv = pd.read_csv(OUT / "hpfc_cv_results.csv")
    prof = pd.read_csv(OUT / "hpfc_profile_2026.csv")
    ev = pd.read_csv(OUT / "hpfc_eval_2026.csv")

    # ---------------- left panel data: mean CV score per (K, H) ----------------
    H_star, K_star = float(sel["half_life_years"]), int(sel["n_harmonics"])
    agg = (cv.groupby(["n_harmonics", "half_life_years"])["ratio_rmse"].mean()
           .unstack(0))                                  # rows H (inf last), cols K
    H_vals = list(agg.index)                             # sorted; np.inf sorts last
    xpos = np.arange(len(H_vals))                        # categorical x, inf at the end
    rmse_sel = float(agg.loc[H_star, K_star])
    rmse_eq = float(agg.loc[np.inf, K_star])
    best_score = float(agg.min().min())
    best_K = int(agg.min().idxmin())
    best_H = float(agg[best_K].idxmin())
    tie_thr = best_score * (1 + TIE_TOL)
    improvement_pct = 100.0 * (rmse_eq - rmse_sel) / rmse_eq

    assert abs(rmse_sel - sel["mean_ratio_rmse"]) < 1e-12, "CSV mean != selection.json"
    assert abs(rmse_eq - sel["equal_weight_ratio_rmse"]) < 1e-12
    assert abs(best_score - sel["best_score"]) < 1e-12

    # ---------------- right panel data: July 2026 realised vs fitted ----------------
    realized = load_epias_ptf_csv(REPO / "inputs/market/realized_ptf_2026.csv")
    loc = realized.tz_convert(TURKEY_TZ)
    jul = loc[(loc.index.year == 2026) & (loc.index.month == MONTH)]
    n_hours = int(jul.notna().sum())
    m_mean = float(jul.mean())
    r = jul / m_mean                                              # same as hpfc._ratio_frame
    dt = day_type(jul.index.tz_localize(None), turkish_holidays([2026]))
    real_prof = (pd.DataFrame(dict(day_type=dt, hour=jul.index.hour, r=r.to_numpy()))
                 .groupby(["day_type", "hour"])["r"].mean().unstack(0))
    fit_prof = (prof[prof["month"] == MONTH]
                .pivot(index="hour", columns="day_type", values="shape"))
    n_days = pd.Series(dt, index=jul.index).groupby(jul.index.normalize()).first().value_counts()
    ev_row = ev[ev["month"] == f"2026-{MONTH:02d}"].iloc[0]
    # July-wide ratio RMSE of the fitted shape against every realised hour (all day types)
    fit_hourly = fit_prof.to_numpy()[jul.index.hour, dt]
    jul_ratio_rmse = float(np.sqrt(np.nanmean((r.to_numpy() - fit_hourly) ** 2)))

    # ---------------- console: every number that appears in the figure ----------------
    print("=== LEFT PANEL (outputs/hpfc/hpfc_cv_results.csv, hpfc_selection.json) ===")
    print(f"selected H* = {H_star} y, K* = {K_star}")
    print(f"mean OOS ratio RMSE at (H*, K*)          = {rmse_sel:.6f}")
    print(f"equal-weight baseline (H = inf, K = {K_star})   = {rmse_eq:.6f}")
    print(f"improvement from recency weighting       = {improvement_pct:.2f} %")
    print(f"single best point: H = {best_H} y, K = {best_K}, RMSE = {best_score:.6f}")
    print(f"tie rule: keep configs with RMSE <= best*(1+{TIE_TOL}) = {tie_thr:.6f}; "
          f"then longest H, then fewest K")
    print("mean OOS ratio RMSE table (rows H years, cols K):")
    print(agg.round(6).to_string())
    print()
    print("=== RIGHT PANEL (inputs/market/realized_ptf_2026.csv, hpfc_profile_2026.csv) ===")
    print(f"July 2026: n = {n_hours} realised hours, monthly mean = {m_mean:.2f} TRY/MWh, "
          f"days by type = {dict(sorted(n_days.items()))}")
    print(f"eval row (hpfc_eval_2026.csv): rmse_smooth_demeaned={ev_row['rmse_smooth_demeaned']}, "
          f"rmse_hpfc_demeaned={ev_row['rmse_hpfc_demeaned']}, "
          f"var_reduction_pct={ev_row['var_reduction_pct']}")
    print(f"July 2026 hourly ratio RMSE, fitted shape vs realised (all hours) = {jul_ratio_rmse:.4f}")
    table = pd.concat({"realised": real_prof, "fitted": fit_prof}, axis=1)
    table.columns = [f"{a}_{DAY_TYPE_NAMES[b]}" for a, b in table.columns]
    print(table.round(4).to_string())

    # ---------------- figure ----------------
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.8), dpi=150,
                                   gridspec_kw=dict(wspace=0.26))
    fig.patch.set_facecolor("#fcfcfb")

    # --- left: CV curve per K ---
    for ax in (ax1, ax2):
        ax.set_facecolor("#fcfcfb")
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    ax1.axhspan(best_score, tie_thr, color=INK2, alpha=0.10, lw=0, zorder=0)
    ax1.axhline(rmse_eq, color=INK2, linestyle="--", linewidth=1.2, zorder=1)
    ax1.text(xpos[0] - 0.1, rmse_eq + 0.0006,
             f"equal-weight baseline (H→∞, K={K_star}): {rmse_eq:.4f}",
             color=INK2, fontsize=8, va="bottom", ha="left")
    for K in agg.columns:
        ax1.plot(xpos, agg[K].to_numpy(), color=SERIES[int(K)], linewidth=2,
                 marker="o", markersize=4.5, markeredgecolor="#fcfcfb", markeredgewidth=1,
                 label=f"K = {K}", zorder=3)
    # direct labels at the right end, pushed apart so K=2 / K=3 (0.0002 apart) do not collide
    ends = sorted((float(agg[K].iloc[-1]), int(K)) for K in agg.columns)
    ypos = [e[0] for e in ends]
    for i in range(1, len(ypos)):
        ypos[i] = max(ypos[i], ypos[i - 1] + 0.0011)
    for (yv, K), yl in zip(ends, ypos):
        ax1.text(xpos[-1] + 0.18, yl, f"K={K}", color=INK, fontsize=8, va="center", ha="left")
    ax1.plot(xpos[H_vals.index(best_H)], best_score, marker="o", markersize=11,
             markerfacecolor="none", markeredgecolor=INK, markeredgewidth=1.2, zorder=4)
    ax1.annotate(f"single best point\nH={best_H}, K={best_K}: {best_score:.4f}",
                 (xpos[H_vals.index(best_H)], best_score), xytext=(-8, -30),
                 textcoords="offset points", ha="right", fontsize=8, color=INK,
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    xs = xpos[H_vals.index(H_star)]
    ax1.plot(xs, rmse_sel, marker="*", markersize=17, color=SERIES[K_star],
             markeredgecolor=INK, markeredgewidth=1, zorder=5)
    ax1.annotate(f"selected H*={H_star} y, K*={K_star}\nRMSE = {rmse_sel:.4f} "
                 f"(−{improvement_pct:.1f} % vs equal weights)",
                 (xs, rmse_sel), xytext=(14, -38), textcoords="offset points",
                 ha="left", fontsize=8.5, color=INK, fontweight="bold",
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax1.text(0.02, 0.98,
             "Selection rule (hpfc.select_half_life, tie_tol = 0.5 %):\n"
             f"keep every (H, K) with RMSE ≤ best × 1.005 = {tie_thr:.4f} (grey band),\n"
             "then take the LONGEST half-life, then the FEWEST harmonics.\n"
             "→ not the single minimum; the flattest/simplest config in the tie set.",
             transform=ax1.transAxes, fontsize=7.8, va="top", ha="left", color=INK,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=GRID))
    ax1.set_xticks(xpos)
    ax1.set_xticklabels([("∞ (equal)" if not np.isfinite(h) else f"{h:g}") for h in H_vals])
    ax1.set_xlim(xpos[0] - 0.4, xpos[-1] + 0.9)
    ax1.set_ylim(0.190, 0.2155)
    ax1.set_xlabel("half-life H (years)  —  weight w = 2^(−age/H)")
    ax1.set_ylabel("mean out-of-sample ratio RMSE (test years 2023, 2024, 2025)")
    ax1.set_title("Rolling-origin CV: score vs half-life, one line per Fourier order K",
                  fontsize=10, loc="left", color=INK)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.40, 0.68), frameon=False, fontsize=8,
               title="harmonics", title_fontsize=8, ncol=3)

    # --- right: July 2026 realised vs fitted ---
    hours = np.arange(24)
    for d, col in ((0, SERIES[1]), (2, SERIES[2])):
        ax2.plot(hours, fit_prof[d].to_numpy(), color=col, linewidth=2.2,
                 label=f"fitted HPFC shape — {DAY_TYPE_NAMES[d]}", zorder=3)
        ax2.plot(hours, real_prof[d].to_numpy(), color=col, linewidth=1, linestyle="--",
                 marker="o", markersize=5, markerfacecolor="#fcfcfb", markeredgewidth=1.4,
                 label=f"realised July 2026 — {DAY_TYPE_NAMES[d]} ({int(n_days[d])} days)",
                 zorder=4)
    ax2.axhline(1.0, color=INK2, linewidth=0.8, linestyle=":", zorder=1)
    ax2.set_xticks(hours[::2])
    ax2.set_xlim(-0.5, 23.5)
    ax2.set_xlabel("hour of day (Europe/Istanbul)")
    ax2.set_ylabel("price ÷ July-2026 monthly mean  (mean = 1 over the month)")
    ax2.set_ylim(0, 2.0)
    ax2.set_title("True out-of-sample: July 2026 realised vs fitted shape (fit cut-off 2025-12-31)",
                  fontsize=10, loc="left", color=INK)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=7.8,
               ncol=2)
    ax2.text(0.98, 0.98,
             f"July 2026, n = {n_hours} h, mean {m_mean:,.0f} TRY/MWh\n"
             f"hourly ratio RMSE fitted vs realised = {jul_ratio_rmse:.4f}\n"
             f"within-month RMSE: smooth {ev_row['rmse_smooth_demeaned']:.0f} → "
             f"HPFC {ev_row['rmse_hpfc_demeaned']:.0f} TRY/MWh "
             f"(−{ev_row['var_reduction_pct']:.1f} %)",
             transform=ax2.transAxes, fontsize=7.8, va="top", ha="right", color=INK,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=GRID))

    fig.suptitle("Recency-weighted HPFC shape — overfit check on real EPİAŞ PTF "
                 "(CV 2023-25 on 2019-25 history; 2026 never used in fitting)",
                 fontsize=11.5, color=INK, x=0.01, ha="left", y=0.995)
    fig.text(0.01, 0.012,
             "sources: outputs/hpfc/hpfc_cv_results.csv, hpfc_selection.json, "
             "hpfc_profile_2026.csv, hpfc_eval_2026.csv; inputs/market/realized_ptf_2026.csv",
             fontsize=7, color=INK2)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.90, bottom=0.19)
    out = OUT / "cv_overfit_check.png"
    fig.savefig(out, facecolor=fig.get_facecolor())
    print(f"\nsaved: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
