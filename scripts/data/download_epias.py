"""Download EPİAŞ Şeffaflık data with the `eptr2` client.

Requires a (free) EPİAŞ Şeffaflık Platformu account.  Put the credentials in a
`.env` file in the repository root (never commit it; `.env` is git-ignored by
this script's instructions):

    EPTR_USERNAME=you@example.com
    EPTR_PASSWORD=yourpassword

Usage (from the repository root, venv active):

    pip install eptr2 python-dotenv
    python scripts/data/download_epias.py ptf --start 2019 --end 2025
    python scripts/data/download_epias.py list-calls --grep vep
    python scripts/data/download_epias.py probe --call <key> --start-date 2025-12-01 --end-date 2025-12-02

`ptf` writes one file per year to inputs/historical/ptf_raw/ptf_<YYYY>.csv in the
SAME format as the Şeffaflık web export (Tarih;Saat;PTF (TL/MWh);PTF (USD/MWh);
PTF (EUR/MWh), Turkish number format), so pde_option_model.premium.load_epias_ptf_csv
reads both the same way.  Downloads are chunked by month.

NOTE: written against the public eptr2 README (call key "mcp"/"ptf", dates
"YYYY-MM-DD"); the column names of the response are detected defensively and
printed on first use.  If detection fails the script stops and prints the
columns it received -- send that output back.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def _client():
    try:
        from eptr2 import EPTR2
    except ImportError:
        sys.exit("eptr2 is not installed:  python -m pip install eptr2 python-dotenv")
    env = REPO / ".env"
    if not env.exists():
        sys.exit(f"credentials file not found: {env}\n"
                 "create it with EPTR_USERNAME=... and EPTR_PASSWORD=... lines")
    import os
    os.chdir(REPO)                      # eptr2 reads .env from the working dir
    return EPTR2(use_dotenv=True, recycle_tgt=True)


def _pick(cols, *cands):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None


def _fmt_tr(x: pd.Series) -> pd.Series:
    """2799.98 -> '2.799,98' (Şeffaflık web-export style)."""
    return x.map(lambda v: "" if pd.isna(v) else
                 f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))


def download_ptf(start_year: int, end_year: int, outdir: Path, pause: float = 1.0) -> None:
    eptr = _client()
    outdir.mkdir(parents=True, exist_ok=True)
    shown = False
    for year in range(start_year, end_year + 1):
        frames = []
        for m in range(1, 13):
            s = pd.Timestamp(year=year, month=m, day=1)
            e = s + pd.offsets.MonthEnd(1)
            for attempt in range(3):
                try:
                    res = eptr.call("mcp", start_date=s.strftime("%Y-%m-%d"),
                                    end_date=e.strftime("%Y-%m-%d"))
                    break
                except Exception as exc:          # network / auth hiccup
                    print(f"  {s:%Y-%m} attempt {attempt + 1} failed: {exc}")
                    time.sleep(3 * (attempt + 1))
            else:
                sys.exit(f"giving up on {s:%Y-%m}")
            df = pd.DataFrame(res)
            if not shown:
                print("response columns:", list(df.columns))
                print(df.head(3).to_string())
                shown = True
            frames.append(df)
            time.sleep(pause)
        df = pd.concat(frames, ignore_index=True)
        c_date = _pick(df.columns, "date", "tarih")
        c_try = _pick(df.columns, "price", "ptf", "mcp")
        c_usd = _pick(df.columns, "priceUsd", "price_usd", "ptfUsd")
        c_eur = _pick(df.columns, "priceEur", "price_eur", "ptfEur")
        if c_date is None or c_try is None:
            sys.exit(f"could not identify date/price columns in {list(df.columns)}")
        ts = pd.to_datetime(df[c_date])
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert("Europe/Istanbul").dt.tz_localize(None)
        out = pd.DataFrame({
            "Tarih": ts.dt.strftime("%d.%m.%Y"),
            "Saat": ts.dt.strftime("%H:%M"),
            "PTF (TL/MWh)": _fmt_tr(pd.to_numeric(df[c_try], errors="coerce")),
            "PTF (USD/MWh)": _fmt_tr(pd.to_numeric(df[c_usd], errors="coerce")) if c_usd else "",
            "PTF (EUR/MWh)": _fmt_tr(pd.to_numeric(df[c_eur], errors="coerce")) if c_eur else "",
        })
        out = out.drop_duplicates(subset=["Tarih", "Saat"], keep="last")
        path = outdir / f"ptf_{year}.csv"
        out.to_csv(path, sep=";", index=False, encoding="utf-8")
        n_exp = (366 if pd.Timestamp(year=year, month=12, day=31).dayofyear == 366 else 365) * 24
        print(f"{path.name}: {len(out)} rows (expected {n_exp})")


def list_calls(grep: str | None) -> None:
    eptr = _client()
    calls = eptr.get_available_calls()
    items = list(calls) if not isinstance(calls, dict) else list(calls.keys())
    if grep:
        items = [c for c in items if grep.lower() in str(c).lower()]
    print("\n".join(map(str, items)) or "(no match)")


def probe(call: str, start_date: str, end_date: str) -> None:
    eptr = _client()
    res = eptr.call(call, start_date=start_date, end_date=end_date)
    df = pd.DataFrame(res)
    print("columns:", list(df.columns))
    print(df.head(10).to_string())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ptf")
    p.add_argument("--start", type=int, default=2019)
    p.add_argument("--end", type=int, default=2025)
    p.add_argument("--outdir", type=Path, default=REPO / "inputs" / "historical" / "ptf_raw")
    q = sub.add_parser("list-calls")
    q.add_argument("--grep")
    r = sub.add_parser("probe")
    r.add_argument("--call", required=True)
    r.add_argument("--start-date", required=True)
    r.add_argument("--end-date", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "ptf":
        download_ptf(a.start, a.end, a.outdir)
    elif a.cmd == "list-calls":
        list_calls(a.grep)
    else:
        probe(a.call, a.start_date, a.end_date)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
