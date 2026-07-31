"""
21 EMA + 50 EMA Base Breakout Swing Trading Scanner
===================================================
Single-file professional Streamlit application for NSE stocks.

Strategy pipeline:
    1. EMA-21 crosses above EMA-50            -> setup begins (NO buy signal)
    2. Price advances 8-15% (configurable)    -> setup confirmed, else rejected
    3. Tight sideways base forms              -> 5-30 candles, limited width,
       >= 2 resistance touches, price above EMA-50, EMA-21 > EMA-50 throughout
    4. BUY only on close > resistance + buffer with volume expansion,
       close > EMA-21, EMA-21 > EMA-50, HH-HL trend intact
    5. Repeated breakouts (2, 3, 4 ...) after pullbacks near the EMA-21

Scanning, historical scanning, filtering, backtesting and CSV / Excel export
only. No charts or graphical plotting anywhere in the application.

Install:  pip install streamlit pandas numpy yfinance openpyxl requests
Run:      streamlit run app.py
"""

from __future__ import annotations

import io
import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ema_base_breakout")

APP_TITLE = "21 EMA + 50 EMA Base Breakout Scanner"
YF_SUFFIX = ".NS"
DOWNLOAD_CHUNK = 40
MAX_RETRIES = 3
PAD_DAYS = 320  # extra calendar days downloaded for EMA warm-up

INDEX_URLS: Dict[str, str] = {
    "NIFTY50": "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
    "NIFTY100": "https://archives.nseindia.com/content/indices/ind_nifty100list.csv",
    "NIFTY200": "https://archives.nseindia.com/content/indices/ind_nifty200list.csv",
    "NIFTY500": "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
}

NIFTY50_FALLBACK: List[str] = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BHARTIARTL",
    "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "ETERNAL",
    "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO",
    "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDUSINDBK", "INFY",
    "ITC", "JIOFIN", "JSWSTEEL", "KOTAKBANK", "LT",
    "M&M", "MARUTI", "NESTLEIND", "NTPC", "ONGC",
    "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SHRIRAMFIN",
    "SUNPHARMA", "TATACONSUM", "TATAMOTORS", "TATASTEEL", "TCS",
    "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO",
]


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StrategyParams:
    """All strategy parameters (hashable so Streamlit can cache on it)."""
    fast_ema: int = 21
    slow_ema: int = 50
    min_advance_pct: float = 8.0
    min_base_len: int = 5
    max_base_len: int = 30
    max_base_width_pct: float = 8.0
    min_res_touches: int = 2
    touch_tolerance_pct: float = 1.0
    breakout_buffer_pct: float = 0.20
    vol_sma_len: int = 20
    vol_multiplier: float = 1.5
    require_hh_hl: bool = True
    require_price_above_fast: bool = True
    require_fast_above_slow: bool = True
    pullback_proximity_pct: float = 2.0


@dataclass(frozen=True)
class FilterParams:
    """Result filters. A value of 0 disables the respective numeric filter."""
    min_price: float = 0.0
    max_price: float = 0.0
    min_volume: float = 0.0
    min_avg_volume: float = 0.0
    min_rel_volume: float = 0.0
    req_price_above_fast: bool = False
    req_fast_above_slow: bool = False
    max_dist_from_fast: float = 0.0
    max_gap_pct: float = 0.0
    breakout_filter: str = "All"


@dataclass(frozen=True)
class BacktestParams:
    """Portfolio backtest parameters."""
    initial_capital: float = 1_000_000.0
    position_size_pct: float = 10.0
    max_positions: int = 10
    stop_loss_pct: float = 5.0
    target_pct: float = 20.0
    trailing_stop_pct: float = 0.0
    exit_on_fast_close: bool = True
    exit_on_slow_close: bool = False
    max_holding_days: int = 60
    commission_pct: float = 0.10
    slippage_pct: float = 0.05


# ---------------------------------------------------------------------------
# Symbol universes
# ---------------------------------------------------------------------------

@st.cache_data(ttl=24 * 3600, show_spinner=False)
def fetch_index_symbols(index_name: str) -> pd.DataFrame:
    """Download an NSE index constituent list (Symbol + Company Name).

    Falls back to a static NIFTY50 list when the archive is unreachable.
    """
    url = INDEX_URLS[index_name]
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, headers=headers, timeout=20)
            resp.raise_for_status()
            raw = pd.read_csv(io.StringIO(resp.text))
            sym_col = "Symbol" if "Symbol" in raw.columns else raw.columns[0]
            name_col = ("Company Name" if "Company Name" in raw.columns
                        else sym_col)
            out = pd.DataFrame({
                "Symbol": raw[sym_col].astype(str).str.strip().str.upper(),
                "Name": raw[name_col].astype(str).str.strip(),
            }).dropna().drop_duplicates(subset="Symbol")
            if not out.empty:
                return out.reset_index(drop=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Index fetch %s attempt %d failed: %s",
                           index_name, attempt + 1, exc)
            time.sleep(1.2 * (attempt + 1))
    logger.error("Falling back to static NIFTY50 list for %s", index_name)
    return pd.DataFrame({"Symbol": NIFTY50_FALLBACK,
                         "Name": NIFTY50_FALLBACK})


def load_custom_symbols(uploaded) -> pd.DataFrame:
    """Parse a user-uploaded CSV/Excel symbol file into Symbol/Name columns."""
    try:
        if uploaded.name.lower().endswith((".xlsx", ".xls")):
            raw = pd.read_excel(uploaded)
        else:
            raw = pd.read_csv(uploaded)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read the uploaded file: {exc}")
        return pd.DataFrame(columns=["Symbol", "Name"])
    if raw.empty:
        return pd.DataFrame(columns=["Symbol", "Name"])
    cols = {c.lower().strip(): c for c in raw.columns}
    sym_col = cols.get("symbol", raw.columns[0])
    name_col = cols.get("name", cols.get("company name", sym_col))
    syms = (raw[sym_col].astype(str).str.strip().str.upper()
            .str.replace(".NS", "", regex=False))
    out = pd.DataFrame({"Symbol": syms,
                        "Name": raw[name_col].astype(str).str.strip()})
    out = out[out["Symbol"].str.len() > 0].drop_duplicates(subset="Symbol")
    return out.reset_index(drop=True)


def get_universe(choice: str, uploaded) -> pd.DataFrame:
    """Resolve the selected universe to a Symbol/Name dataframe."""
    if choice == "Custom Upload":
        if uploaded is None:
            st.warning("Upload a CSV/Excel file with a Symbol column.")
            return pd.DataFrame(columns=["Symbol", "Name"])
        return load_custom_symbols(uploaded)
    return fetch_index_symbols(choice)


# ---------------------------------------------------------------------------
# Price data download (multi-threaded, cached, with retries)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def download_batch(symbols: Tuple[str, ...], start_s: str, end_s: str,
                   max_workers: int = 12
                   ) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """Download OHLCV for a batch of NSE symbols via yfinance.

    Returns (data_by_symbol, failed_symbols). Each request retries up to
    MAX_RETRIES times. Results are cached by Streamlit for one hour.
    """

    def _one(sym: str) -> Optional[pd.DataFrame]:
        ticker = sym if sym.endswith(YF_SUFFIX) else sym + YF_SUFFIX
        for attempt in range(MAX_RETRIES):
            try:
                df = yf.download(ticker, start=start_s, end=end_s,
                                 progress=False, auto_adjust=False,
                                 threads=False)
                if df is not None and not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                    df = df.dropna()
                    try:
                        df.index = pd.to_datetime(df.index).tz_localize(None)
                    except TypeError:
                        df.index = pd.to_datetime(df.index)
                    if len(df) > 0:
                        return df
            except Exception as exc:  # noqa: BLE001
                logger.warning("Download %s attempt %d failed: %s",
                               sym, attempt + 1, exc)
            time.sleep(0.5 * (attempt + 1))
        return None

    data: Dict[str, pd.DataFrame] = {}
    failed: List[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_one, s): s for s in symbols}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                df = fut.result()
            except Exception:  # noqa: BLE001
                df = None
            if df is None or len(df) < 60:
                failed.append(sym)
            else:
                data[sym] = df
    return data, failed


def download_universe(symbols: List[str], start_s: str, end_s: str,
                      max_workers: int
                      ) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """Chunked download with a progress bar (each chunk is cached)."""
    data: Dict[str, pd.DataFrame] = {}
    failed: List[str] = []
    chunks = [symbols[i:i + DOWNLOAD_CHUNK]
              for i in range(0, len(symbols), DOWNLOAD_CHUNK)]
    prog = st.progress(0.0, text="Downloading price data ...")
    for ci, chunk in enumerate(chunks):
        d, f = download_batch(tuple(chunk), start_s, end_s, max_workers)
        data.update(d)
        failed.extend(f)
        prog.progress((ci + 1) / len(chunks),
                      text=f"Downloaded {len(data)} / {len(symbols)} symbols")
    prog.empty()
    return data, failed


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def add_indicators(df: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    """Attach fast/slow EMAs and the volume SMA to a price dataframe."""
    out = df.copy()
    out["EMA_F"] = out["Close"].ewm(span=p.fast_ema, adjust=False).mean()
    out["EMA_S"] = out["Close"].ewm(span=p.slow_ema, adjust=False).mean()
    out["VOL_SMA"] = out["Volume"].rolling(p.vol_sma_len).mean()
    return out


# ---------------------------------------------------------------------------
# Strategy engine
# ---------------------------------------------------------------------------

def _pivot_hh_hl_ok(h: np.ndarray, l: np.ndarray, i: int,
                    lookback: int = 60, wing: int = 3,
                    tol_pct: float = 3.0) -> bool:
    """Check that the recent swing structure keeps Higher Highs / Higher Lows.

    Uses simple fractal pivots (wing bars on each side) within `lookback`
    bars ending at index i. A small tolerance keeps flat-base micro-pivots
    from invalidating an intact trend. Returns True when there are not
    enough pivots to judge.
    """
    s = max(0, i - lookback)
    hh = h[s:i + 1]
    ll = l[s:i + 1]
    n = len(hh)
    piv_h: List[float] = []
    piv_l: List[float] = []
    for j in range(wing, n - wing):
        win_h = hh[j - wing:j + wing + 1]
        win_l = ll[j - wing:j + wing + 1]
        if hh[j] >= win_h.max():
            piv_h.append(float(hh[j]))
        if ll[j] <= win_l.min():
            piv_l.append(float(ll[j]))
    k = 1 - tol_pct / 100.0
    ok_h = len(piv_h) < 2 or piv_h[-1] >= piv_h[-2] * k
    ok_l = len(piv_l) < 2 or piv_l[-1] >= piv_l[-2] * k
    return ok_h and ok_l


def _base_ok(h: np.ndarray, l: np.ndarray, c: np.ndarray,
             ef: np.ndarray, es: np.ndarray, s: int, e: int,
             p: StrategyParams) -> Optional[Dict[str, float]]:
    """Validate a candidate base window [s, e] (inclusive).

    Conditions: width within limit, minimum resistance touches, every close
    above the slow EMA and fast EMA above slow EMA for the whole window.
    """
    hh = float(h[s:e + 1].max())
    ll = float(l[s:e + 1].min())
    if ll <= 0:
        return None
    width = (hh - ll) / ll * 100.0
    if width > p.max_base_width_pct:
        return None
    tol_level = hh * (1 - p.touch_tolerance_pct / 100.0)
    touches = int((h[s:e + 1] >= tol_level).sum())
    if touches < p.min_res_touches:
        return None
    if not (c[s:e + 1] > es[s:e + 1]).all():
        return None
    if not (ef[s:e + 1] > es[s:e + 1]).all():
        return None
    return {"res": hh, "low": ll, "len": float(e - s + 1),
            "touches": float(touches), "width": width}


def scan_symbol(df: pd.DataFrame, p: StrategyParams
                ) -> Tuple[List[dict], Optional[dict]]:
    """Run the full breakout pipeline on one symbol.

    Returns (events, active): `events` is every historical breakout
    (with breakout numbers per crossover segment); `active` describes the
    currently-live crossover segment (if the fast EMA is still above the
    slow EMA on the latest bar).
    """
    n = len(df)
    if n < p.slow_ema + 20:
        return [], None
    c = df["Close"].to_numpy(dtype=float)
    h = df["High"].to_numpy(dtype=float)
    l = df["Low"].to_numpy(dtype=float)
    v = df["Volume"].to_numpy(dtype=float)
    ef = df["EMA_F"].to_numpy(dtype=float)
    es = df["EMA_S"].to_numpy(dtype=float)
    vs = df["VOL_SMA"].to_numpy(dtype=float)
    dates = df.index

    cross_up = np.where((ef[1:] > es[1:]) & (ef[:-1] <= es[:-1]))[0] + 1
    events: List[dict] = []
    active: Optional[dict] = None

    for cu in cross_up:
        below = np.where(ef[cu + 1:] < es[cu + 1:])[0]
        seg_end = int(cu + below[0]) if len(below) else n - 1
        cross_close = float(c[cu])
        if cross_close <= 0:
            continue

        adv_hits = np.where(
            c[cu + 1:seg_end + 1] >= cross_close * (1 + p.min_advance_pct / 100.0)
        )[0]
        adv_i: Optional[int] = int(cu + 1 + adv_hits[0]) if len(adv_hits) else None

        seg_events = 0
        last_bo: Optional[int] = None
        pulled = True

        if adv_i is not None:
            i = max(adv_i + 1, int(cu) + 1 + p.min_base_len)
            while i <= seg_end:
                if seg_events >= 1 and not pulled:
                    if l[i] <= ef[i] * (1 + p.pullback_proximity_pct / 100.0):
                        pulled = True
                    else:
                        i += 1
                        continue
                min_start = (last_bo + 1) if last_bo is not None else int(cu) + 1
                found_base: Optional[Dict[str, float]] = None
                for L in range(p.min_base_len, p.max_base_len + 1):
                    s = i - L
                    if s < min_start:
                        break
                    b = _base_ok(h, l, c, ef, es, s, i - 1, p)
                    if b is not None:
                        found_base = b
                        break
                if found_base is not None:
                    bo_price = found_base["res"] * (1 + p.breakout_buffer_pct / 100.0)
                    vol_ok = (np.isfinite(vs[i]) and vs[i] > 0
                              and v[i] > vs[i] * p.vol_multiplier)
                    cond = bool(c[i] > bo_price and vol_ok)
                    if cond and p.require_price_above_fast:
                        cond = bool(c[i] > ef[i])
                    if cond and p.require_fast_above_slow:
                        cond = bool(ef[i] > es[i])
                    if cond and p.require_hh_hl:
                        cond = _pivot_hh_hl_ok(h, l, i)
                    if cond:
                        seg_events += 1
                        events.append({
                            "idx": int(i),
                            "Signal Date": dates[i],
                            "Close": float(c[i]),
                            "EMA_F": float(ef[i]),
                            "EMA_S": float(es[i]),
                            "Resistance": found_base["res"],
                            "Base Low": found_base["low"],
                            "Base Length": int(found_base["len"]),
                            "Breakout Price": float(bo_price),
                            "Breakout Number": int(seg_events),
                            "Volume": float(v[i]),
                            "Volume SMA": float(vs[i]),
                            "Volume Ratio": float(v[i] / vs[i]),
                            "Days Since EMA Crossover": int(i - cu),
                            "Gain Since EMA Crossover %":
                                float((c[i] / cross_close - 1) * 100.0),
                            "Cross Idx": int(cu),
                        })
                        last_bo = int(i)
                        pulled = False
                i += 1

        if len(below) == 0:  # crossover segment still live on the last bar
            active = {
                "cross_idx": int(cu),
                "cross_close": cross_close,
                "adv_idx": adv_i,
                "seg_events": seg_events,
                "last_bo": last_bo,
            }
    return events, active


def build_current_row(sym: str, name: str, df: pd.DataFrame,
                      p: StrategyParams, events: List[dict],
                      active: Optional[dict]) -> dict:
    """Assemble one row of the current-day scanner table."""
    n = len(df)
    price = float(df["Close"].iloc[-1])
    ef = float(df["EMA_F"].iloc[-1])
    es = float(df["EMA_S"].iloc[-1])
    vol = float(df["Volume"].iloc[-1])
    vs_raw = df["VOL_SMA"].iloc[-1]
    vs = float(vs_raw) if pd.notna(vs_raw) else np.nan
    vol_ratio = vol / vs if vs and vs > 0 else np.nan
    gap = np.nan
    if n >= 2 and float(df["Close"].iloc[-2]) > 0:
        gap = (float(df["Open"].iloc[-1]) / float(df["Close"].iloc[-2]) - 1) * 100.0

    row = {
        "Symbol": sym,
        "Company Name": name,
        "Current Price": price,
        "EMA_F": ef,
        "EMA_S": es,
        "EMA Distance %": (ef - es) / es * 100.0 if es else np.nan,
        "Base Length": np.nan,
        "Resistance": np.nan,
        "Base Low": np.nan,
        "Breakout Price": np.nan,
        "Breakout Number": np.nan,
        "Volume": vol,
        "Volume SMA": vs,
        "Volume Ratio": vol_ratio,
        "Days Since EMA Crossover": np.nan,
        "Gain Since EMA Crossover %": np.nan,
        "Gap %": gap,
        "Signal Date": pd.NaT,
        "Signal": "No Signal",
    }
    if active is None:
        return row

    row["Days Since EMA Crossover"] = int(n - 1 - active["cross_idx"])
    row["Gain Since EMA Crossover %"] = (price / active["cross_close"] - 1) * 100.0

    last_ev = events[-1] if events else None
    if last_ev is not None and last_ev["idx"] == n - 1:
        row.update({
            "Base Length": last_ev["Base Length"],
            "Resistance": last_ev["Resistance"],
            "Base Low": last_ev["Base Low"],
            "Breakout Price": last_ev["Breakout Price"],
            "Breakout Number": last_ev["Breakout Number"],
            "Signal Date": last_ev["Signal Date"],
            "Signal": "BUY",
        })
        return row

    if active["adv_idx"] is not None:  # look for a live base ending today
        h = df["High"].to_numpy(dtype=float)
        l = df["Low"].to_numpy(dtype=float)
        c = df["Close"].to_numpy(dtype=float)
        efa = df["EMA_F"].to_numpy(dtype=float)
        esa = df["EMA_S"].to_numpy(dtype=float)
        min_start = (active["last_bo"] + 1 if active["last_bo"] is not None
                     else active["cross_idx"] + 1)
        for L in range(p.min_base_len, p.max_base_len + 1):
            s = n - L
            if s < min_start:
                break
            b = _base_ok(h, l, c, efa, esa, s, n - 1, p)
            if b is not None:
                row.update({
                    "Base Length": int(b["len"]),
                    "Resistance": b["res"],
                    "Base Low": b["low"],
                    "Breakout Price": b["res"] * (1 + p.breakout_buffer_pct / 100.0),
                    "Breakout Number": active["seg_events"] + 1,
                })
                break
    return row


# ---------------------------------------------------------------------------
# Scanners
# ---------------------------------------------------------------------------

def run_current_scan(universe: pd.DataFrame, p: StrategyParams,
                     lookback_days: int, max_workers: int
                     ) -> Tuple[pd.DataFrame, dict, List[str]]:
    """Scan the latest trading day for every symbol in the universe."""
    t0 = time.time()
    end = date.today() + timedelta(days=1)
    start = date.today() - timedelta(days=lookback_days + PAD_DAYS)
    symbols = universe["Symbol"].tolist()
    names = dict(zip(universe["Symbol"], universe["Name"]))
    data, failed = download_universe(symbols, start.isoformat(),
                                     end.isoformat(), max_workers)
    rows: List[dict] = []
    total_events = 0
    prog = st.progress(0.0, text="Scanning ...")
    done = 0
    for sym, df in data.items():
        try:
            dfx = add_indicators(df, p)
            events, active = scan_symbol(dfx, p)
            total_events += len(events)
            rows.append(build_current_row(sym, names.get(sym, sym),
                                          dfx, p, events, active))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Scan failed for %s: %s", sym, exc)
            failed.append(sym)
        done += 1
        if done % 20 == 0 or done == len(data):
            prog.progress(done / max(len(data), 1),
                          text=f"Scanned {done} / {len(data)} symbols")
    prog.empty()
    res = pd.DataFrame(rows)
    metrics = {
        "scanned": len(data),
        "valid_setups": int(res["Resistance"].notna().sum()) if not res.empty else 0,
        "todays_breakouts": int((res["Signal"] == "BUY").sum()) if not res.empty else 0,
        "historical_breakouts": total_events,
        "avg_base_len": float(res["Base Length"].mean()) if not res.empty else np.nan,
        "avg_vol_ratio": float(res["Volume Ratio"].mean()) if not res.empty else np.nan,
        "avg_gain": float(res["Gain Since EMA Crossover %"].mean()) if not res.empty else np.nan,
        "scan_time": time.time() - t0,
    }
    return res, metrics, failed


def run_historical_scan(universe: pd.DataFrame, p: StrategyParams,
                        start_d: date, end_d: date, max_workers: int
                        ) -> Tuple[pd.DataFrame, List[str]]:
    """Return every historical breakout between start_d and end_d."""
    symbols = universe["Symbol"].tolist()
    names = dict(zip(universe["Symbol"], universe["Name"]))
    dl_start = (start_d - timedelta(days=PAD_DAYS)).isoformat()
    dl_end = (end_d + timedelta(days=1)).isoformat()
    data, failed = download_universe(symbols, dl_start, dl_end, max_workers)
    rows: List[dict] = []
    prog = st.progress(0.0, text="Scanning history ...")
    done = 0
    lo = pd.Timestamp(start_d)
    hi = pd.Timestamp(end_d)
    for sym, df in data.items():
        try:
            dfx = add_indicators(df, p)
            events, _ = scan_symbol(dfx, p)
            for ev in events:
                d = pd.Timestamp(ev["Signal Date"])
                if lo <= d <= hi:
                    rows.append({
                        "Signal Date": d.date(),
                        "Symbol": sym,
                        "Company Name": names.get(sym, sym),
                        "Close": ev["Close"],
                        "EMA_F": ev["EMA_F"],
                        "EMA_S": ev["EMA_S"],
                        "Resistance": ev["Resistance"],
                        "Base Low": ev["Base Low"],
                        "Base Length": ev["Base Length"],
                        "Breakout Number": ev["Breakout Number"],
                        "Volume Ratio": ev["Volume Ratio"],
                        "Days Since EMA Crossover": ev["Days Since EMA Crossover"],
                        "Gain Since EMA Crossover %": ev["Gain Since EMA Crossover %"],
                    })
        except Exception as exc:  # noqa: BLE001
            logger.exception("Historical scan failed for %s: %s", sym, exc)
            failed.append(sym)
        done += 1
        if done % 20 == 0 or done == len(data):
            prog.progress(done / max(len(data), 1),
                          text=f"Scanned {done} / {len(data)} symbols")
    prog.empty()
    res = pd.DataFrame(rows)
    if not res.empty:
        res = res.sort_values("Signal Date").reset_index(drop=True)
    return res, failed


def apply_filters(df: pd.DataFrame, f: FilterParams, p: StrategyParams,
                  price_col: str) -> pd.DataFrame:
    """Apply the user-selected result filters to a scanner dataframe."""
    if df.empty:
        return df
    mask = pd.Series(True, index=df.index)
    price = df[price_col]
    if f.min_price > 0:
        mask &= price >= f.min_price
    if f.max_price > 0:
        mask &= price <= f.max_price
    if f.min_volume > 0 and "Volume" in df.columns:
        mask &= df["Volume"] >= f.min_volume
    if f.min_avg_volume > 0 and "Volume SMA" in df.columns:
        mask &= df["Volume SMA"] >= f.min_avg_volume
    if f.min_rel_volume > 0 and "Volume Ratio" in df.columns:
        mask &= df["Volume Ratio"] >= f.min_rel_volume
    if f.req_price_above_fast and "EMA_F" in df.columns:
        mask &= price > df["EMA_F"]
    if f.req_fast_above_slow and {"EMA_F", "EMA_S"} <= set(df.columns):
        mask &= df["EMA_F"] > df["EMA_S"]
    if f.max_dist_from_fast > 0 and "EMA_F" in df.columns:
        dist = (price - df["EMA_F"]).abs() / df["EMA_F"] * 100.0
        mask &= dist <= f.max_dist_from_fast
    if f.max_gap_pct > 0 and "Gap %" in df.columns:
        mask &= df["Gap %"].abs().fillna(0) <= f.max_gap_pct
    if f.breakout_filter != "All" and "Breakout Number" in df.columns:
        mask &= df["Breakout Number"] == int(f.breakout_filter)
    return df[mask].reset_index(drop=True)


def display_rename(df: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    """Rename generic EMA columns to their user-facing labels."""
    return df.rename(columns={"EMA_F": f"EMA{p.fast_ema}",
                              "EMA_S": f"EMA{p.slow_ema}"})


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------

def run_backtest_engine(events: List[dict], data: Dict[str, pd.DataFrame],
                        bt: BacktestParams, start_d: date, end_d: date
                        ) -> Tuple[pd.DataFrame, pd.Series]:
    """Event-driven portfolio backtest over all breakout signals.

    Entries at the next day's open (plus slippage); exits checked daily in
    priority order: stop / trailing stop, profit target, EMA-fast close,
    EMA-slow close, maximum holding days. Remaining positions are closed at
    the final available price. Returns (trade_log, daily_equity).
    """
    lo = pd.Timestamp(start_d)
    hi = pd.Timestamp(end_d)
    idx_map = {s: {d: i for i, d in enumerate(df.index)}
               for s, df in data.items()}
    entries_by_date: Dict[pd.Timestamp, List[dict]] = defaultdict(list)
    for ev in events:
        sym = ev["Symbol"]
        df = data[sym]
        i = ev["idx"]
        if i + 1 < len(df):
            entries_by_date[df.index[i + 1]].append(ev)

    all_dates = sorted({d for df in data.values()
                        for d in df.index if lo <= d <= hi})
    cash = float(bt.initial_capital)
    open_pos: List[dict] = []
    trades: List[dict] = []
    eq_idx: List[pd.Timestamp] = []
    eq_val: List[float] = []

    def _close(pos: dict, exit_date: pd.Timestamp, exit_px_raw: float,
               reason: str) -> None:
        nonlocal cash
        exit_px = exit_px_raw * (1 - bt.slippage_pct / 100.0)
        proceeds = pos["shares"] * exit_px
        exit_comm = proceeds * bt.commission_pct / 100.0
        cash += proceeds - exit_comm
        pnl = proceeds - exit_comm - pos["entry_cost"] - pos["entry_comm"]
        trades.append({
            "Symbol": pos["symbol"],
            "Entry Date": pos["entry_date"].date(),
            "Exit Date": exit_date.date(),
            "Entry Price": round(pos["entry_px"], 2),
            "Exit Price": round(exit_px, 2),
            "Shares": pos["shares"],
            "PnL": round(pnl, 2),
            "PnL %": round(pnl / (pos["entry_cost"] + pos["entry_comm"]) * 100.0, 2),
            "Holding Days": pos["bars"],
            "Breakout Number": pos["bnum"],
            "Exit Reason": reason,
        })

    for d in all_dates:
        still: List[dict] = []
        for pos in open_pos:
            sym = pos["symbol"]
            im = idx_map[sym]
            if d not in im:
                still.append(pos)
                continue
            row = data[sym].iloc[im[d]]
            pos["bars"] += 1
            o = float(row["Open"])
            hgh = float(row["High"])
            low = float(row["Low"])
            cls = float(row["Close"])
            base_stop = (pos["entry_px"] * (1 - bt.stop_loss_pct / 100.0)
                         if bt.stop_loss_pct > 0 else -np.inf)
            trail_stop = (pos["high_close"] * (1 - bt.trailing_stop_pct / 100.0)
                          if bt.trailing_stop_pct > 0 else -np.inf)
            stop = max(base_stop, trail_stop)
            tgt = (pos["entry_px"] * (1 + bt.target_pct / 100.0)
                   if bt.target_pct > 0 else np.inf)
            exit_px = None
            reason = ""
            if np.isfinite(stop) and low <= stop:
                exit_px = o if o < stop else stop
                reason = ("Trailing Stop" if trail_stop > base_stop
                          else "Stop Loss")
            elif np.isfinite(tgt) and hgh >= tgt:
                exit_px = o if o > tgt else tgt
                reason = "Profit Target"
            elif bt.exit_on_fast_close and cls < float(row["EMA_F"]):
                exit_px, reason = cls, "Close Below Fast EMA"
            elif bt.exit_on_slow_close and cls < float(row["EMA_S"]):
                exit_px, reason = cls, "Close Below Slow EMA"
            elif bt.max_holding_days > 0 and pos["bars"] >= bt.max_holding_days:
                exit_px, reason = cls, "Max Holding Days"
            if exit_px is not None:
                _close(pos, d, exit_px, reason)
            else:
                pos["high_close"] = max(pos["high_close"], cls)
                pos["last_close"] = cls
                still.append(pos)
        open_pos = still

        for ev in entries_by_date.get(d, []):
            if len(open_pos) >= bt.max_positions:
                break
            sym = ev["Symbol"]
            if any(pp["symbol"] == sym for pp in open_pos):
                continue
            row = data[sym].iloc[idx_map[sym][d]]
            o = float(row["Open"])
            if o <= 0:
                continue
            equity_now = cash + sum(pp["shares"] * pp["last_close"]
                                    for pp in open_pos)
            size = equity_now * bt.position_size_pct / 100.0
            px = o * (1 + bt.slippage_pct / 100.0)
            shares = int(size // px)
            cost = shares * px
            comm = cost * bt.commission_pct / 100.0
            if shares > 0 and cost + comm <= cash:
                cash -= cost + comm
                open_pos.append({
                    "symbol": sym, "entry_date": d, "entry_px": px,
                    "entry_cost": cost, "entry_comm": comm,
                    "shares": shares, "bars": 0,
                    "high_close": float(row["Close"]),
                    "last_close": float(row["Close"]),
                    "bnum": ev["Breakout Number"],
                })

        eq_idx.append(d)
        eq_val.append(cash + sum(pp["shares"] * pp["last_close"]
                                 for pp in open_pos))

    for pos in list(open_pos):
        _close(pos, all_dates[-1] if all_dates else pd.Timestamp(end_d),
               pos["last_close"], "Open At End")
    equity = pd.Series(eq_val, index=pd.DatetimeIndex(eq_idx), dtype=float)
    return pd.DataFrame(trades), equity


def backtest_stats(trades: pd.DataFrame, equity: pd.Series,
                   bt: BacktestParams) -> dict:
    """Compute headline statistics for a completed backtest."""
    out = {"Total Trades": 0, "Winning Trades": 0, "Losing Trades": 0,
           "Win Rate %": np.nan, "Profit Factor": np.nan,
           "Net Profit": 0.0, "CAGR %": np.nan, "Max Drawdown %": np.nan,
           "Average Gain %": np.nan, "Average Loss %": np.nan,
           "Average Holding Days": np.nan}
    if trades.empty:
        return out
    wins = trades[trades["PnL"] > 0]
    losses = trades[trades["PnL"] <= 0]
    gross_win = float(wins["PnL"].sum())
    gross_loss = abs(float(losses["PnL"].sum()))
    out["Total Trades"] = int(len(trades))
    out["Winning Trades"] = int(len(wins))
    out["Losing Trades"] = int(len(losses))
    out["Win Rate %"] = len(wins) / len(trades) * 100.0
    out["Profit Factor"] = (gross_win / gross_loss if gross_loss > 0
                            else np.inf)
    out["Net Profit"] = float(trades["PnL"].sum())
    if len(equity) > 1:
        days = (equity.index[-1] - equity.index[0]).days
        if days > 0 and equity.iloc[-1] > 0:
            out["CAGR %"] = ((equity.iloc[-1] / bt.initial_capital)
                             ** (365.0 / days) - 1) * 100.0
        out["Max Drawdown %"] = float(
            ((equity / equity.cummax()) - 1).min() * 100.0)
    if not wins.empty:
        out["Average Gain %"] = float(wins["PnL %"].mean())
    if not losses.empty:
        out["Average Loss %"] = float(losses["PnL %"].mean())
    out["Average Holding Days"] = float(trades["Holding Days"].mean())
    return out


def periodic_summary(trades: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Monthly ('M') or yearly ('Y') trade summary keyed by exit date."""
    if trades.empty:
        return pd.DataFrame()
    t = trades.copy()
    t["Exit Date"] = pd.to_datetime(t["Exit Date"])
    grp = t.groupby(t["Exit Date"].dt.to_period(freq))
    out = grp.agg(
        Trades=("PnL", "size"),
        Wins=("PnL", lambda s: int((s > 0).sum())),
        WinRate=("PnL", lambda s: round((s > 0).mean() * 100.0, 1)),
        NetPnL=("PnL", lambda s: round(float(s.sum()), 2)),
        AvgPnLPct=("PnL %", lambda s: round(float(s.mean()), 2)),
    ).reset_index()
    out = out.rename(columns={"Exit Date": "Period", "WinRate": "Win Rate %",
                              "NetPnL": "Net PnL", "AvgPnLPct": "Avg PnL %"})
    out["Period"] = out["Period"].astype(str)
    return out


def run_backtest(universe: pd.DataFrame, p: StrategyParams,
                 bt: BacktestParams, start_d: date, end_d: date,
                 max_workers: int) -> dict:
    """Download, scan and backtest the strategy across the universe."""
    symbols = universe["Symbol"].tolist()
    dl_start = (start_d - timedelta(days=PAD_DAYS)).isoformat()
    dl_end = (end_d + timedelta(days=1)).isoformat()
    data, failed = download_universe(symbols, dl_start, dl_end, max_workers)
    prepped: Dict[str, pd.DataFrame] = {}
    events: List[dict] = []
    lo = pd.Timestamp(start_d)
    hi = pd.Timestamp(end_d)
    prog = st.progress(0.0, text="Generating signals ...")
    done = 0
    for sym, df in data.items():
        try:
            dfx = add_indicators(df, p)
            prepped[sym] = dfx
            evs, _ = scan_symbol(dfx, p)
            for ev in evs:
                d = pd.Timestamp(ev["Signal Date"])
                if lo <= d <= hi:
                    e2 = dict(ev)
                    e2["Symbol"] = sym
                    events.append(e2)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Backtest scan failed for %s: %s", sym, exc)
            failed.append(sym)
        done += 1
        if done % 20 == 0 or done == len(data):
            prog.progress(done / max(len(data), 1),
                          text=f"Signals from {done} / {len(data)} symbols")
    prog.empty()
    events.sort(key=lambda e: pd.Timestamp(e["Signal Date"]))
    trades, equity = run_backtest_engine(events, prepped, bt, start_d, end_d)
    stats = backtest_stats(trades, equity, bt)
    return {"trades": trades, "equity": equity, "stats": stats,
            "monthly": periodic_summary(trades, "M"),
            "yearly": periodic_summary(trades, "Y"),
            "signals": len(events), "failed": failed}


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Encode a dataframe as UTF-8 CSV bytes."""
    return df.to_csv(index=False).encode("utf-8-sig")


def dfs_to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    """Write multiple dataframes to a single Excel workbook (openpyxl)."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, d in sheets.items():
            (d if not d.empty else pd.DataFrame({"Info": ["No data"]})
             ).to_excel(writer, sheet_name=name[:31], index=False)
    return buf.getvalue()


def export_buttons(df: pd.DataFrame, stem: str, key: str) -> None:
    """Render CSV and Excel download buttons for a result dataframe."""
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Export CSV", df_to_csv_bytes(df),
                       file_name=f"{stem}.csv", mime="text/csv",
                       key=f"{key}_csv", use_container_width=True)
    c2.download_button("⬇️ Export Excel",
                       dfs_to_excel_bytes({stem[:31]: df}),
                       file_name=f"{stem}.xlsx",
                       mime=("application/vnd.openxmlformats-officedocument"
                             ".spreadsheetml.sheet"),
                       key=f"{key}_xlsx", use_container_width=True)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def sidebar_controls() -> Tuple[str, object, StrategyParams, FilterParams,
                                int, int]:
    """Render the sidebar and return all user selections."""
    st.sidebar.title("⚙️ Configuration")
    universe = st.sidebar.selectbox(
        "Symbol Universe",
        ["NIFTY50", "NIFTY100", "NIFTY200", "NIFTY500", "Custom Upload"])
    uploaded = None
    if universe == "Custom Upload":
        uploaded = st.sidebar.file_uploader(
            "Upload symbols (CSV / Excel with a Symbol column)",
            type=["csv", "xlsx", "xls"])

    with st.sidebar.expander("Moving Averages", expanded=True):
        fast = st.number_input("Fast EMA", 2, 200, 21, key="fast_ema")
        slow = st.number_input("Slow EMA", 5, 400, 50, key="slow_ema")

    with st.sidebar.expander("Initial Advance"):
        adv = st.number_input("Required Advance After Crossover %",
                              8.0, 15.0, 8.0, 0.5,
                              help="Price must rise this much after the "
                                   "EMA crossover before a base is valid.")

    with st.sidebar.expander("Base Formation"):
        min_b = st.number_input("Minimum Base Candles", 3, 50, 5)
        max_b = st.number_input("Maximum Base Candles", 5, 120, 30)
        width = st.number_input("Maximum Base Width %", 1.0, 30.0, 8.0, 0.5)
        touches = st.number_input("Minimum Resistance Touches", 1, 10, 2)

    with st.sidebar.expander("Breakout"):
        buf = st.number_input("Breakout Buffer %", 0.0, 5.0, 0.20, 0.05)
        vlen = st.number_input("Volume SMA Length", 5, 100, 20)
        vmult = st.number_input("Volume Multiplier", 0.5, 10.0, 1.5, 0.1)

    with st.sidebar.expander("Trend Filter"):
        hhhl = st.checkbox("Require Higher High Higher Low", True)
        paf = st.checkbox("Price Above Fast EMA", True)
        fas = st.checkbox("Fast EMA Above Slow EMA", True)

    with st.sidebar.expander("Result Filters (0 = off)"):
        f_min_p = st.number_input("Minimum Price", 0.0, value=0.0)
        f_max_p = st.number_input("Maximum Price", 0.0, value=0.0)
        f_min_v = st.number_input("Minimum Volume", 0.0, value=0.0,
                                  step=10000.0)
        f_min_av = st.number_input("Minimum Average Volume", 0.0, value=0.0,
                                   step=10000.0)
        f_rel_v = st.number_input("Minimum Relative Volume", 0.0, value=0.0,
                                  step=0.1)
        f_paf = st.checkbox("Filter: Price Above Fast EMA", False)
        f_fas = st.checkbox("Filter: Fast EMA Above Slow EMA", False)
        f_dist = st.number_input("Maximum Distance From Fast EMA %",
                                 0.0, value=0.0, step=0.5)
        f_gap = st.number_input("Maximum Gap %", 0.0, value=0.0, step=0.5)
        f_bnum = st.selectbox("Breakout Number Filter",
                              ["All", "1", "2", "3"])

    with st.sidebar.expander("Performance"):
        workers = st.slider("Download Threads", 4, 24, 12)
        lookback = st.number_input("Current Scan Lookback (days)",
                                   250, 900, 420, 10)

    p = StrategyParams(
        fast_ema=int(fast), slow_ema=int(slow), min_advance_pct=float(adv),
        min_base_len=int(min_b), max_base_len=int(max_b),
        max_base_width_pct=float(width), min_res_touches=int(touches),
        breakout_buffer_pct=float(buf), vol_sma_len=int(vlen),
        vol_multiplier=float(vmult), require_hh_hl=bool(hhhl),
        require_price_above_fast=bool(paf),
        require_fast_above_slow=bool(fas))
    f = FilterParams(
        min_price=float(f_min_p), max_price=float(f_max_p),
        min_volume=float(f_min_v), min_avg_volume=float(f_min_av),
        min_rel_volume=float(f_rel_v), req_price_above_fast=bool(f_paf),
        req_fast_above_slow=bool(f_fas), max_dist_from_fast=float(f_dist),
        max_gap_pct=float(f_gap), breakout_filter=str(f_bnum))
    return universe, uploaded, p, f, int(workers), int(lookback)


def render_dashboard(m: dict) -> None:
    """Render the dashboard metric tiles."""
    r1 = st.columns(4)
    r1[0].metric("Stocks Scanned", m["scanned"])
    r1[1].metric("Valid Setups", m["valid_setups"])
    r1[2].metric("Today's Breakouts", m["todays_breakouts"])
    r1[3].metric("Historical Breakouts", m["historical_breakouts"])
    r2 = st.columns(4)
    r2[0].metric("Avg Base Length",
                 f"{m['avg_base_len']:.1f}" if pd.notna(m["avg_base_len"]) else "—")
    r2[1].metric("Avg Volume Ratio",
                 f"{m['avg_vol_ratio']:.2f}" if pd.notna(m["avg_vol_ratio"]) else "—")
    r2[2].metric("Avg Gain Since Crossover",
                 f"{m['avg_gain']:.1f}%" if pd.notna(m["avg_gain"]) else "—")
    r2[3].metric("Scan Time", f"{m['scan_time']:.1f}s")


def render_failed(failed: List[str], key: str) -> None:
    """Show failed downloads in a collapsible error log."""
    if failed:
        with st.expander(f"⚠️ {len(failed)} symbols failed "
                         f"(download/scan errors)"):
            st.write(", ".join(sorted(set(failed))))


def main() -> None:
    """Streamlit entry point."""
    st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide")
    st.title("📈 " + APP_TITLE)
    st.caption("EMA crossover → 8–15% advance → tight base → volume breakout. "
               "Scanning, historical scanning and backtesting only — no charts.")

    universe_choice, uploaded, p, filt, workers, lookback = sidebar_controls()
    if p.fast_ema >= p.slow_ema:
        st.sidebar.error("Fast EMA must be shorter than Slow EMA.")
    if p.min_base_len > p.max_base_len:
        st.sidebar.error("Minimum base candles exceed maximum.")

    tab_cur, tab_hist, tab_bt = st.tabs(
        ["🔍 Current Scanner", "🗓️ Historical Scanner", "🧪 Backtesting"])

    # ------------------------------------------------------------- current
    with tab_cur:
        if st.button("🚀 Run Current Scan", type="primary"):
            uni = get_universe(universe_choice, uploaded)
            if uni.empty:
                st.error("No symbols to scan.")
            else:
                res, metrics, failed = run_current_scan(uni, p, lookback,
                                                        workers)
                st.session_state["cur"] = {"df": res, "metrics": metrics,
                                           "failed": failed}
        cur = st.session_state.get("cur")
        if cur:
            render_dashboard(cur["metrics"])
            shown = apply_filters(cur["df"], filt, p, "Current Price")
            if not shown.empty:
                shown = shown.sort_values(
                    ["Signal", "Volume Ratio"],
                    ascending=[True, False]).reset_index(drop=True)
            st.subheader(f"Results ({len(shown)} stocks after filters)")
            st.dataframe(display_rename(shown.round(2), p),
                         use_container_width=True, hide_index=True,
                         height=480)
            export_buttons(display_rename(shown, p),
                           "current_scan", "cur")
            render_failed(cur["failed"], "cur")
        else:
            st.info("Set your parameters in the sidebar and run the scan.")

    # ---------------------------------------------------------- historical
    with tab_hist:
        c1, c2 = st.columns(2)
        h_start = c1.date_input("Start Date",
                                date.today() - timedelta(days=365),
                                key="h_start")
        h_end = c2.date_input("End Date", date.today(), key="h_end")
        if st.button("🕰️ Run Historical Scan", type="primary"):
            if h_start >= h_end:
                st.error("Start date must be before end date.")
            else:
                uni = get_universe(universe_choice, uploaded)
                if uni.empty:
                    st.error("No symbols to scan.")
                else:
                    res, failed = run_historical_scan(uni, p, h_start,
                                                      h_end, workers)
                    st.session_state["hist"] = {"df": res, "failed": failed}
        hist = st.session_state.get("hist")
        if hist:
            df_h = apply_filters(hist["df"], filt, p, "Close")
            st.metric("Historical Breakouts", len(df_h))
            st.dataframe(display_rename(df_h.round(2), p),
                         use_container_width=True, hide_index=True,
                         height=480)
            export_buttons(display_rename(df_h, p),
                           "historical_scan", "hist")
            render_failed(hist["failed"], "hist")

    # ------------------------------------------------------------ backtest
    with tab_bt:
        c1, c2, c3 = st.columns(3)
        b_start = c1.date_input("Start Date",
                                date.today() - timedelta(days=730),
                                key="b_start")
        b_end = c2.date_input("End Date", date.today(), key="b_end")
        capital = c3.number_input("Initial Capital (₹)", 10_000.0,
                                  value=1_000_000.0, step=50_000.0)
        c4, c5, c6 = st.columns(3)
        pos_size = c4.number_input("Position Size (% of equity)", 1.0, 100.0,
                                   10.0, 1.0)
        max_pos = c5.number_input("Maximum Open Positions", 1, 100, 10)
        sl = c6.number_input("Stop Loss % (0 = off)", 0.0, 50.0, 5.0, 0.5)
        c7, c8, c9 = st.columns(3)
        tgt = c7.number_input("Profit Target % (0 = off)", 0.0, 200.0,
                              20.0, 1.0)
        trail = c8.number_input("Trailing Stop % (0 = off)", 0.0, 50.0,
                                0.0, 0.5)
        max_hold = c9.number_input("Maximum Holding Days (0 = off)", 0, 500,
                                   60)
        c10, c11, c12 = st.columns(3)
        exit_fast = c10.checkbox("Exit on Close Below Fast EMA", True)
        exit_slow = c11.checkbox("Exit on Close Below Slow EMA", False)
        comm = c12.number_input("Commission % (per side)", 0.0, 2.0,
                                0.10, 0.01)
        slip = st.number_input("Slippage % (per side)", 0.0, 2.0, 0.05, 0.01)

        if st.button("🧪 Run Backtest", type="primary"):
            if b_start >= b_end:
                st.error("Start date must be before end date.")
            else:
                uni = get_universe(universe_choice, uploaded)
                if uni.empty:
                    st.error("No symbols to backtest.")
                else:
                    bt = BacktestParams(
                        initial_capital=float(capital),
                        position_size_pct=float(pos_size),
                        max_positions=int(max_pos),
                        stop_loss_pct=float(sl), target_pct=float(tgt),
                        trailing_stop_pct=float(trail),
                        exit_on_fast_close=bool(exit_fast),
                        exit_on_slow_close=bool(exit_slow),
                        max_holding_days=int(max_hold),
                        commission_pct=float(comm),
                        slippage_pct=float(slip))
                    result = run_backtest(uni, p, bt, b_start, b_end,
                                          workers)
                    st.session_state["bt"] = result
        btres = st.session_state.get("bt")
        if btres:
            s = btres["stats"]
            st.subheader("Performance Summary")
            r1 = st.columns(4)
            r1[0].metric("Total Trades", s["Total Trades"])
            r1[1].metric("Winning Trades", s["Winning Trades"])
            r1[2].metric("Losing Trades", s["Losing Trades"])
            r1[3].metric("Win Rate", f"{s['Win Rate %']:.1f}%"
                         if pd.notna(s["Win Rate %"]) else "—")
            r2 = st.columns(4)
            pf = s["Profit Factor"]
            r2[0].metric("Profit Factor",
                         "∞" if pf == np.inf else (f"{pf:.2f}"
                         if pd.notna(pf) else "—"))
            r2[1].metric("Net Profit", f"₹{s['Net Profit']:,.0f}")
            r2[2].metric("CAGR", f"{s['CAGR %']:.2f}%"
                         if pd.notna(s["CAGR %"]) else "—")
            r2[3].metric("Max Drawdown", f"{s['Max Drawdown %']:.2f}%"
                         if pd.notna(s["Max Drawdown %"]) else "—")
            r3 = st.columns(4)
            r3[0].metric("Average Gain", f"{s['Average Gain %']:.2f}%"
                         if pd.notna(s["Average Gain %"]) else "—")
            r3[1].metric("Average Loss", f"{s['Average Loss %']:.2f}%"
                         if pd.notna(s["Average Loss %"]) else "—")
            r3[2].metric("Avg Holding Days",
                         f"{s['Average Holding Days']:.1f}"
                         if pd.notna(s["Average Holding Days"]) else "—")
            r3[3].metric("Signals Generated", btres["signals"])

            st.subheader("Trade Log")
            st.dataframe(btres["trades"], use_container_width=True,
                         hide_index=True, height=380)
            cm, cy = st.columns(2)
            with cm:
                st.subheader("Monthly Summary")
                st.dataframe(btres["monthly"], use_container_width=True,
                             hide_index=True)
            with cy:
                st.subheader("Yearly Summary")
                st.dataframe(btres["yearly"], use_container_width=True,
                             hide_index=True)
            cbt1, cbt2 = st.columns(2)
            cbt1.download_button(
                "⬇️ Trade Log CSV", df_to_csv_bytes(btres["trades"]),
                file_name="backtest_trades.csv", mime="text/csv",
                key="bt_csv", use_container_width=True)
            cbt2.download_button(
                "⬇️ Full Report Excel",
                dfs_to_excel_bytes({"Trades": btres["trades"],
                                    "Monthly": btres["monthly"],
                                    "Yearly": btres["yearly"]}),
                file_name="backtest_report.xlsx",
                mime=("application/vnd.openxmlformats-officedocument"
                      ".spreadsheetml.sheet"),
                key="bt_xlsx", use_container_width=True)
            render_failed(btres["failed"], "bt")


if __name__ == "__main__":
    main()
