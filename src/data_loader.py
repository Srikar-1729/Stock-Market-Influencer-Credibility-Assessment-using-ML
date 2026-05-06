from __future__ import annotations

import argparse
import time
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

from .config import DEFAULT_TICKERS, RAW_DIR, REMAINING_NIFTY50_TICKERS, ensure_dirs


def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance can return MultiIndex columns (field, ticker) even for a single ticker.
    This flattens them to single-level columns like "Open", "Close", etc.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        # Prefer the first level (field names).
        df.columns = [str(c[0]) for c in df.columns.to_list()]
    return df


def _parse_end(end: str) -> str:
    if end.lower() == "today":
        return date.today().isoformat()
    return end


def download_ticker(
    ticker: str,
    start: str = "2010-01-01",
    end: str = "today",
    interval: str = "1d",
    out_dir: Path = RAW_DIR,
    max_retries: int = 3,
    retry_delay_sec: float = 2.0,
    overwrite: bool = True,
) -> Path:
    """
    Download OHLCV data from Yahoo Finance via yfinance and store as CSV.
    """
    ensure_dirs()
    end = _parse_end(end)
    out_path = out_dir / f"{ticker.replace('.', '_')}.csv"
    if out_path.exists() and not overwrite:
        print(f"[SKIP] {ticker}: {out_path.name} already exists")
        return out_path

    last_error: Exception | None = None
    df = pd.DataFrame()
    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=False,
                progress=False,
            )
            if not df.empty:
                break
            last_error = ValueError(f"No data returned for ticker={ticker}")
        except Exception as exc:
            last_error = exc

        if attempt < max_retries:
            wait_for = retry_delay_sec * attempt
            print(f"[WARN] {ticker}: download attempt {attempt}/{max_retries} failed. Retrying in {wait_for:.1f}s...")
            time.sleep(wait_for)

    if df.empty:
        msg = f"Failed to download ticker={ticker} after {max_retries} attempts."
        if last_error is not None:
            msg = f"{msg} Last error: {last_error}"
        raise RuntimeError(msg)

    df = _flatten_yfinance_columns(df)
    df = df.reset_index()
    df.columns = [str(c).replace(" ", "_") for c in df.columns]

    # Standardize columns.
    rename = {}
    if "Date" not in df.columns and "Datetime" in df.columns:
        rename["Datetime"] = "Date"
    df = df.rename(columns=rename)

    ordered_cols = ["Date", "Adj_Close", "Close", "High", "Low", "Open", "Volume"]
    missing = set(ordered_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for {ticker}: {sorted(missing)}. Got: {list(df.columns)}")

    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)
    df = df[ordered_cols]

    df.to_csv(out_path, index=False)
    return out_path


def download_universe(
    tickers: Iterable[str],
    start: str = "2010-01-01",
    end: str = "today",
    interval: str = "1d",
    max_retries: int = 3,
    retry_delay_sec: float = 2.0,
    fail_fast: bool = False,
    skip_existing: bool = False,
) -> list[Path]:
    paths: list[Path] = []
    for t in tickers:
        try:
            paths.append(
                download_ticker(
                    t,
                    start=start,
                    end=end,
                    interval=interval,
                    max_retries=max_retries,
                    retry_delay_sec=retry_delay_sec,
                    overwrite=not skip_existing,
                )
            )
            print(f"[OK] {t}")
        except Exception as exc:
            print(f"[WARN] Skipping {t}: {exc}")
            if fail_fast:
                raise
    return paths


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Yahoo Finance OHLCV CSVs.")
    p.add_argument("--start", type=str, default="2010-01-01")
    p.add_argument("--end", type=str, default="today")
    p.add_argument("--interval", type=str, default="1d")
    p.add_argument("--tickers", type=str, nargs="*", default=None, help="Override tickers list.")
    p.add_argument("--retries", type=int, default=3, help="Retry count per ticker.")
    p.add_argument("--retry_delay", type=float, default=2.0, help="Base retry delay in seconds.")
    p.add_argument("--fail_fast", action="store_true", help="Stop immediately on first failed ticker.")
    p.add_argument(
        "--remaining_nifty50",
        action="store_true",
        help="Download only the remaining NIFTY 50 tickers not in DEFAULT_TICKERS.",
    )
    p.add_argument("--skip_existing", action="store_true", help="Do not overwrite CSV files that already exist.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.tickers:
        tickers = args.tickers
    elif args.remaining_nifty50:
        tickers = REMAINING_NIFTY50_TICKERS
    else:
        tickers = DEFAULT_TICKERS
    start_ts = datetime.now()
    paths = download_universe(
        tickers,
        start=args.start,
        end=args.end,
        interval=args.interval,
        max_retries=max(1, int(args.retries)),
        retry_delay_sec=max(0.0, float(args.retry_delay)),
        fail_fast=bool(args.fail_fast),
        skip_existing=bool(args.skip_existing),
    )
    dur = datetime.now() - start_ts
    total = len(tickers)
    print(f"Downloaded {len(paths)}/{total} tickers in {dur}. Saved to {RAW_DIR}")
    if len(paths) == 0:
        raise SystemExit("No ticker data could be downloaded. Check network/DNS and retry.")


if __name__ == "__main__":
    main()

