"""PoC: download the full daily OHLCV history of one Tadawul ticker from the SAHMK API.

API: GET https://api.sahmk.sa/api/v1/historical/{symbol}/  (docs: https://www.sahmk.sa/api-docs.md)
Needs a Starter+ plan; the key is read from the SAHMK_API_KEY environment variable.
Pages of up to 2000 bars are requested until has_more is false. Rows are saved untouched to
PoC/data/<symbol>_1d.csv, with the response metadata in <symbol>_1d.json.
Usage: SAHMK_API_KEY=... python PoC/sahmk_history.py [SYMBOL] [--from YYYY-MM-DD]
"""
import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.sahmk.sa/api/v1"
PAGE = 2000  # the API's maximum limit
OUT = Path(__file__).resolve().parent / "data"


def get(session: requests.Session, path: str, params: dict) -> dict:
    """One GET; waits and retries on 429, exits with the API's message on any other error."""
    for _ in range(5):
        r = session.get(f"{BASE_URL}{path}", params=params, timeout=30)
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 5))
            print(f"  429 rate limited, retrying in {wait}s")
            time.sleep(wait)
            continue
        if not r.ok:
            sys.exit(f"HTTP {r.status_code}: {r.text[:300]}")
        print(f"  {r.url.split('?')[1]}  daily quota left {r.headers.get('X-RateLimit-Remaining', '?')}")
        return r.json()
    sys.exit("still rate limited after 5 tries")


def history(symbol: str, start: str, end: str) -> tuple[pd.DataFrame, dict]:
    """Every daily bar for symbol between start and end, oldest first, and the first page's metadata."""
    key = os.environ.get("SAHMK_API_KEY")
    if not key:
        sys.exit("set SAHMK_API_KEY first")
    session = requests.Session()
    session.headers["X-API-Key"] = key
    rows, offset, meta = [], 0, None
    while True:
        page = get(session, f"/historical/{symbol}/",
                   {"interval": "1d", "from": start, "to": end, "limit": PAGE, "offset": offset})
        meta = meta or {k: v for k, v in page.items() if k != "data"}
        rows += page["data"]
        if not page["has_more"]:
            break
        offset += page["count"]
    df = pd.DataFrame(rows)
    assert len(df) == meta["total"], f"got {len(df)} rows, API reports {meta['total']}"
    return df, meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("symbol", nargs="?", default="2140", help="Tadawul symbol (default 2140, ADC)")
    p.add_argument("--from", dest="start", default="2000-01-01", help="earliest date to ask for")
    args = p.parse_args()

    end = str(date.today())
    print(f"SAHMK {args.symbol} 1d {args.start} .. {end}")
    df, meta = history(args.symbol, args.start, end)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / f"{args.symbol}_1d.csv", index=False)
    (OUT / f"{args.symbol}_1d.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    d = pd.to_datetime(df["date"])
    print(f"{len(df)} rows {d.min().date()} .. {d.max().date()}, columns {list(df.columns)}")
    print("rows by weekday:", d.dt.day_name().value_counts().to_dict())
    print(f"saved {OUT / f'{args.symbol}_1d.csv'}")


if __name__ == "__main__":
    main()
