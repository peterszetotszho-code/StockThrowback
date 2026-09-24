"""Historical price data acquisition and local parquet caching via yfinance."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "outputs" / "cache"
PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _cache_path(ticker: str, start: str, end: str, cache_dir: Path) -> Path:
    """Return a deterministic parquet path for a (ticker, range) request."""
    safe = ticker.replace(".", "_").replace("^", "_").replace(":", "_")
    return cache_dir / f"{safe}_{start}_{end}.parquet"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten a possible MultiIndex column layout and keep only OHLCV columns."""
    cols = df.columns
    if isinstance(cols, pd.MultiIndex):
        df = df.copy()
        # Locate the index level that holds the OHLCV price labels.
        for level in range(cols.nlevels):
            level_names = {str(c) for c in cols.get_level_values(level)}
            if all(c in level_names for c in PRICE_COLUMNS):
                df.columns = cols.get_level_values(level)
                break
        else:
            raise ValueError("Could not locate OHLCV columns in MultiIndex data.")
    missing = [c for c in PRICE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Downloaded data is missing required columns: {missing}")
    return df[PRICE_COLUMNS]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-trading rows and fill minor gaps.

    Rows where every column is NaN are dropped. OHLC gaps are forward-filled;
    Volume gaps are treated as zero-volume (suspension) days.
    """
    before = len(df)
    df = df.dropna(how="all")
    df[["Open", "High", "Low", "Close"]] = df[["Open", "High", "Low", "Close"]].ffill()
    df["Volume"] = df["Volume"].fillna(0)
    df = df.dropna(subset=["Close"])
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d rows with no traded price (suspension/missing).", dropped)
    df["Volume"] = df["Volume"].astype("int64")
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df.sort_index()


def fetch_stock_data(
    ticker: str,
    start: str,
    end: str,
    cache_dir: Path | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Download OHLCV data for a ticker, using a local parquet cache when possible.

    Args:
        ticker: Yahoo Finance ticker symbol, e.g. "0700.HK" or "AAPL".
        start: Inclusive start date in "YYYY-MM-DD" format.
        end: Inclusive end date in "YYYY-MM-DD" format.
        cache_dir: Directory for parquet cache files. Defaults to ``outputs/cache``.
        use_cache: When True, return a cached snapshot if one exists.

    Returns:
        A DataFrame indexed by ``Date`` with columns Open/High/Low/Close/Volume.

    Raises:
        ValueError: If the cleaned result is empty (bad ticker or range).
        RuntimeError: If the download fails.
    """
    import yfinance as yf  # Local import keeps the module import-light.

    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    path = _cache_path(ticker, start, end, cache_dir)

    if use_cache and path.exists():
        logger.info("Cache hit for %s (%s -> %s): %s", ticker, start, end, path)
        return pd.read_parquet(path)

    logger.info("Downloading %s (%s -> %s) ...", ticker, start, end)
    try:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    except Exception as exc:  # noqa: BLE001 - surface any download failure
        raise RuntimeError(f"Failed to download {ticker}: {exc}") from exc

    if raw is None or raw.empty:
        raise ValueError(f"No data returned for {ticker} in [{start}, {end}]. Check the ticker/range.")

    df = _clean(_normalize_columns(raw))
    if df.empty:
        raise ValueError(f"Cleaned data for {ticker} is empty.")

    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    logger.info("Cached %d rows to %s", len(df), path)
    return df


def get_company_name(ticker: str) -> str | None:
    """Return a ticker's company name via yfinance, or None if unavailable.

    This is a best-effort lookup (network); failures return None rather than
    raising, so callers can degrade gracefully.
    """
    import yfinance as yf  # Local import keeps the module import-light.

    try:
        info = yf.Ticker(ticker).info
        # Prefer the full legal name for display (e.g. "Tencent Holdings
        # Limited") over the short ticker alias ("TENCENT" / "BABA-W").
        name = info.get("longName") or info.get("shortName") or None
        return name.strip() if name else None
    except Exception as exc:  # noqa: BLE001 - network / invalid ticker
        logger.warning("Could not resolve company name for %s: %s", ticker, exc)
        return None
