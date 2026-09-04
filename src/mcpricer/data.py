"""Market data loading.

Three daily FRED series, cached as CSV under data/ so every backtest number in
this repo is reproducible without network access:

    SP500   S&P 500 index level
    VIXCLS  CBOE volatility index, the market's 30-day implied volatility
    DGS1MO  1-month Treasury constant maturity yield, used as the risk-free rate
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

__all__ = ["DATA_DIR", "SERIES", "load_market_data", "download_series"]

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
SERIES = {"spx": "SP500", "vix": "VIXCLS", "rate": "DGS1MO"}


def download_series(series_id: str, destination: Path, timeout: int = 30) -> Path:
    """Refresh one cached FRED series. Not called during tests."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(
        FRED_CSV.format(series_id=series_id), timeout=timeout
    ) as response:
        destination.write_bytes(response.read())
    return destination


def load_market_data(data_dir: Path | None = None) -> pd.DataFrame:
    """Daily spot, implied volatility and risk-free rate on a common calendar.

    Returns a frame indexed by date with columns spot, iv, rate. Volatility and
    rate are converted from percentage points to decimals. Rows where any
    series is missing are dropped, so the index is the intersection of the
    three calendars.
    """
    directory = Path(data_dir) if data_dir is not None else DATA_DIR
    frames = {}
    for name, series_id in SERIES.items():
        path = directory / f"{series_id.lower()}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"missing cached series {path}; run download_series({series_id!r}, ...)"
            )
        frame = pd.read_csv(path, na_values=["."])
        frame.columns = ["date", name]
        frames[name] = frame.assign(date=pd.to_datetime(frame["date"]))

    merged = frames["spx"]
    for name in ("vix", "rate"):
        merged = merged.merge(frames[name], on="date", how="inner")

    merged = merged.dropna().sort_values("date").set_index("date")
    return merged.rename(columns={"spx": "spot", "vix": "iv"}).assign(
        iv=lambda f: f["iv"] / 100.0, rate=lambda f: f["rate"] / 100.0
    )
