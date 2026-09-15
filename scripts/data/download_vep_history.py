"""Download EPİAŞ VEP (PFM) Daily Index Price history with the `eptr2` client.

The VEP "Daily Index Price" (Günlük Gösterge Fiyatı, GGF) is the daily reference
price EPİAŞ publishes at 16:45 on business days for every contract open for
trading.  It is the quote source behind `inputs/market/vep_monthly_quotes.csv`
(EBM0226 = 2900.99 TRY/MWh on 2025-12-31).

This script downloads the RAW responses only -- no parsing, no reshaping, no
derived files.  Everything downstream (contract filtering, per-valuation-date
quote sets, forward-curve construction) is done from these raw files so that the
network step and the modelling step stay independent and auditable.

Endpoints used (from the eptr2 API schema, category VEP):

    vep-ggf                POST electricity-service/v1/markets/pfm/data/ggf
    vep-contract-price-summary
                           POST .../pfm/data/contract-price-summary
    vep-ggf-period         POST .../pfm/data/ggf-delivery-period-list
    vep-load-types         POST .../pfm/data/load-type-list

Usage (repo root, venv active, `.env` holding EPTR_USERNAME / EPTR_PASSWORD):

    pip install eptr2 python-dotenv
    python scripts/data/download_vep_history.py ggf --start 2022-01 --end 2026-09
    python scripts/data/download_vep_history.py summary --start 2022-01 --end 2026-09
    python scripts/data/download_vep_history.py meta

Output (all raw, git-ignored data, TRY/MWh as returned by EPİAŞ):

    inputs/market/historical_vep/raw/vep_ggf_<YYYY-MM>.json
    inputs/market/historical_vep/raw/vep_contract_price_summary_<YYYY-MM>.json
    inputs/market/historical_vep/raw/vep_meta_<call>.json
    inputs/market/historical_vep/raw/_download_log.json

Each output file stores the response verbatim under "response" together with the
request parameters and a UTC download timestamp, so a later run can be diffed
against an earlier one.  Chunks that return no rows are recorded with
``"n_rows": 0`` and are NOT deleted -- an empty month is a finding, not an error.

Re-running skips chunks that already exist unless --overwrite is given.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
OUTDIR = REPO / "inputs" / "market" / "historical_vep" / "raw"

# Contracts used by the forward-curve pipeline: monthly baseload, EBM<MM><YY>.
MONTHLY_BASELOAD_PREFIX = "EBM"

# Known anchor from inputs/market/vep_monthly_quotes.csv; used as a sanity check.
ANCHOR = {"date": "2025-12-31", "contract": "EBM0226", "price_TRY_MWh": 2900.99}


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


def _month_chunks(start: str, end: str) -> List[tuple[str, str, str]]:
    """[(tag 'YYYY-MM', first day, last day), ...] inclusive of both endpoints."""
    import pandas as pd
    s = pd.Period(start, freq="M")
    e = pd.Period(end, freq="M")
    if e < s:
        sys.exit(f"--end {end} is before --start {start}")
    out = []
    p = s
    while p <= e:
        out.append((str(p),
                    p.start_time.strftime("%Y-%m-%d"),
                    p.end_time.strftime("%Y-%m-%d")))
        p += 1
    return out


def _rows(res: Any) -> List[Dict[str, Any]]:
    """Normalise an eptr2 response into a list of record dicts."""
    if res is None:
        return []
    if isinstance(res, list):
        return [r for r in res if isinstance(r, dict)]
    if isinstance(res, dict):
        for key in ("items", "body", "content", "data", "result"):
            if key in res:
                return _rows(res[key])
        return [res]
    try:                                    # a DataFrame-like response
        return res.to_dict(orient="records")
    except Exception:
        return []


def _write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)


def _call_with_retry(eptr, call: str, params: Dict[str, Any],
                     attempts: int = 3) -> Any:
    last: Optional[Exception] = None
    for i in range(attempts):
        try:
            return eptr.call(call, **params)
        except Exception as exc:
            last = exc
            print(f"    attempt {i + 1}/{attempts} failed: {exc}")
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"{call} {params}: giving up after {attempts} attempts: {last}")


def download_chunked(call: str, file_stem: str, start: str, end: str,
                     overwrite: bool, pause: float) -> None:
    eptr = _client()
    chunks = _month_chunks(start, end)
    log: List[Dict[str, Any]] = []
    shown = False
    print(f"{call}: {len(chunks)} monthly chunks {chunks[0][0]} .. {chunks[-1][0]}")
    for tag, first, last in chunks:
        path = OUTDIR / f"{file_stem}_{tag}.json"
        if path.exists() and not overwrite:
            print(f"  {tag}: exists, skipped")
            log.append({"chunk": tag, "status": "skipped_existing", "file": path.name})
            continue
        params = {"start_date": first, "end_date": last}
        res = _call_with_retry(eptr, call, params)
        rows = _rows(res)
        if rows and not shown:
            print(f"  response keys: {sorted(rows[0].keys())}")
            print(f"  first record : {json.dumps(rows[0], ensure_ascii=False)}")
            shown = True
        _write(path, {
            "call": call,
            "params": params,
            "downloaded_utc": datetime.now(timezone.utc).isoformat(),
            "n_rows": len(rows),
            "response": rows,
        })
        n_ebm = sum(1 for r in rows
                    if str(r.get("contractName", r.get("contract", "")))
                    .upper().startswith(MONTHLY_BASELOAD_PREFIX))
        print(f"  {tag}: {len(rows)} rows ({n_ebm} {MONTHLY_BASELOAD_PREFIX}*) -> {path.name}")
        log.append({"chunk": tag, "status": "ok", "n_rows": len(rows),
                    "n_monthly_baseload": n_ebm, "file": path.name})
        time.sleep(pause)

    logpath = OUTDIR / "_download_log.json"
    prev: List[Dict[str, Any]] = []
    if logpath.exists():
        try:
            prev = json.load(open(logpath, encoding="utf-8")).get("runs", [])
        except Exception:
            prev = []
    prev.append({"call": call, "start": start, "end": end,
                 "downloaded_utc": datetime.now(timezone.utc).isoformat(),
                 "chunks": log})
    _write(logpath, {"runs": prev})
    print(f"\nlog -> {logpath}")
    _anchor_check(file_stem)


def _anchor_check(file_stem: str) -> None:
    """Verify the one VEP number the repo already knows, if it was downloaded."""
    path = OUTDIR / f"{file_stem}_{ANCHOR['date'][:7]}.json"
    if not path.exists():
        return
    rows = json.load(open(path, encoding="utf-8")).get("response", [])
    hits = []
    for r in rows:
        name = str(r.get("contractName", r.get("contract", ""))).upper()
        if name != ANCHOR["contract"]:
            continue
        d = str(r.get("date", r.get("gunlukGostergeFiyatiTarihi", "")))[:10]
        if d == ANCHOR["date"]:
            hits.append(r)
    if not hits:
        print(f"ANCHOR CHECK: no {ANCHOR['contract']} row found for {ANCHOR['date']} "
              f"in {path.name} -- check the date/contract field names above.")
        return
    print(f"ANCHOR CHECK: {ANCHOR['contract']} on {ANCHOR['date']} -> "
          f"{json.dumps(hits[0], ensure_ascii=False)}")
    print(f"              expected price {ANCHOR['price_TRY_MWh']} TRY/MWh")


def download_meta() -> None:
    """One-off descriptive calls: load types and GGF delivery periods."""
    eptr = _client()
    for call, params in (
        ("vep-load-types", {"start_date": "2022-01-01", "end_date": "2026-09-30"}),
        ("vep-ggf-period", {"start_date": "2022-01-01", "end_date": "2026-09-30"}),
        ("vep-delivery-period-list", {"start_date": "2022-01-01", "end_date": "2026-09-30"}),
        ("vep-delivery-year-list", {"start_date": "2022-01-01", "end_date": "2026-09-30"}),
    ):
        try:
            res = _call_with_retry(eptr, call, params, attempts=2)
        except Exception as exc:
            print(f"{call}: FAILED ({exc})")
            continue
        rows = _rows(res)
        path = OUTDIR / f"vep_meta_{call.replace('-', '_')}.json"
        _write(path, {"call": call, "params": params,
                      "downloaded_utc": datetime.now(timezone.utc).isoformat(),
                      "n_rows": len(rows), "response": rows})
        print(f"{call}: {len(rows)} rows -> {path.name}")
        if rows:
            print(f"  {json.dumps(rows[0], ensure_ascii=False)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("ggf", help="VEP Daily Index Price (Günlük Gösterge Fiyatı)")
    g.add_argument("--start", default="2022-01", help="first month, YYYY-MM")
    g.add_argument("--end", default="2026-09", help="last month, YYYY-MM")
    g.add_argument("--overwrite", action="store_true")
    g.add_argument("--pause", type=float, default=1.0)

    s = sub.add_parser("summary", help="VEP contract price summary (first/high/low/last/DIP)")
    s.add_argument("--start", default="2022-01")
    s.add_argument("--end", default="2026-09")
    s.add_argument("--overwrite", action="store_true")
    s.add_argument("--pause", type=float, default=1.0)

    sub.add_parser("meta", help="load types and delivery-period listings")

    a = ap.parse_args(argv)
    if a.cmd == "ggf":
        download_chunked("vep-ggf", "vep_ggf", a.start, a.end, a.overwrite, a.pause)
    elif a.cmd == "summary":
        download_chunked("vep-contract-price-summary", "vep_contract_price_summary",
                         a.start, a.end, a.overwrite, a.pause)
    else:
        download_meta()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
