"""
NSE/BSE Buy-Signal Scanner & Portfolio Backtester
==================================================
Single-file Streamlit app (Python 3.11+).

Strategy (long-only)
--------------------
Step 1  Tracking      : When RSI(14) closes < 35 -> start a tracking session
                        (no buy yet). Only one active session per stock.
Step 2  Crossovers    : While tracking, record the FIRST occurrence of each
                        bullish EMA crossover (prev <= , curr > ), on close:
                          A) EMA10 crosses above EMA50
                          B) EMA21 crosses above EMA50
                          C) EMA10 crosses above EMA21
Step 3  Buy Signal    : Fire ONE buy only if all three crossovers occurred and
                        (max_pos - min_pos) <= 3 trading days apart. Buy on the
                        close of the candle where the final crossover completes.
Step 4  Reset         : After a buy (or after a definitive gap>3 failure) the
                        cycle ends; a fresh RSI<35 is required before tracking
                        again. No duplicate buys inside one tracking cycle.

Design guarantees
-----------------
* No repainting / no look-ahead: every decision at bar i uses only bars <= i,
  and crossovers use consecutive-close transitions only.
* The "3 trading days" gap is measured in bar (index) positions, not calendar
  days -> max(pos) - min(pos) <= 3.
* A gap>3 outcome is a terminal FAILURE for that cycle (otherwise an already
  completed crossover set would re-fire every bar); the stock then waits for a
  fresh RSI<35. This is the only deviation from "reset only after buy", made to
  keep the state machine well-defined.

Dependencies: streamlit, pandas, numpy, yfinance, openpyxl
"""

from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
RSI_PERIOD = 14
RSI_THRESHOLD = 35.0
EMA_FAST, EMA_MID, EMA_SLOW = 10, 21, 50
MAX_CROSS_GAP = 3          # trading-day gap allowed between crossovers
WARMUP_DAYS = 260          # calendar days of history fetched before start (EMA/RSI warmup)
MIN_BARS = 60              # minimum usable bars per symbol

# A representative NIFTY-200 universe (NSE tickers, '.NS' suffix added later).
NIFTY_200 = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR", "ITC",
    "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK", "BAJFINANCE", "ASIANPAINT",
    "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO", "ONGC", "NTPC",
    "POWERGRID", "NESTLEIND", "HCLTECH", "TATAMOTORS", "TATASTEEL", "JSWSTEEL",
    "ADANIENT", "ADANIPORTS", "COALINDIA", "GRASIM", "HINDALCO", "BAJAJFINSV",
    "BAJAJ-AUTO", "BRITANNIA", "CIPLA", "DIVISLAB", "DRREDDY", "EICHERMOT",
    "HEROMOTOCO", "INDUSINDBK", "M&M", "SBILIFE", "SHREECEM", "TECHM",
    "APOLLOHOSP", "BPCL", "HDFCLIFE", "TATACONSUM", "UPL", "PIDILITIND",
    "DABUR", "GODREJCP", "MARICO", "COLPAL", "BERGEPAINT", "HAVELLS",
    "SIEMENS", "DLF", "AMBUJACEM", "ACC", "BANKBARODA", "PNB", "CANBK",
    "IDFCFIRSTB", "FEDERALBNK", "BANDHANBNK", "AUBANK", "CHOLAFIN", "MUTHOOTFIN",
    "ICICIPRULI", "ICICIGI", "SBICARD", "HDFCAMC", "MFSL", "LICI",
    "GAIL", "IOC", "PETRONET", "IGL", "GUJGASLTD", "ATGL", "ADANIGREEN",
    "ADANIPOWER", "TATAPOWER", "TORNTPOWER", "NHPC", "SJVN", "POWERINDIA",
    "VEDL", "NMDC", "SAIL", "JINDALSTEL", "NATIONALUM", "APLAPOLLO",
    "PIIND", "SRF", "AARTIIND", "DEEPAKNTR", "TATACHEM", "NAVINFLUOR",
    "LUPIN", "AUROPHARMA", "BIOCON", "ALKEM", "TORNTPHARM", "GLENMARK",
    "ZYDUSLIFE", "IPCALAB", "LAURUSLABS", "ABBOTINDIA", "MANKIND",
    "LTIM", "PERSISTENT", "COFORGE", "MPHASIS", "OFSS", "LTTS",
    "TATAELXSI", "KPITTECH", "BSOFT", "CYIENT",
    "DMART", "TRENT", "NAUKRI", "ZOMATO", "PAGEIND", "JUBLFOOD",
    "VBL", "UNITDSPR", "UBL", "PGHH", "GILLETTE", "HONAUT",
    "ABB", "BEL", "HAL", "BHEL", "CUMMINSIND", "BOSCHLTD",
    "MRF", "BALKRISIND", "APOLLOTYRE", "ASHOKLEY", "TVSMOTOR", "BAJAJHLDNG",
    "MOTHERSON", "BHARATFORG", "SUNDRMFAST", "ESCORTS", "TIINDIA",
    "INDIGO", "INTERGLOBE", "CONCOR", "ADANIWILMAR", "PATANJALI",
    "PFC", "RECLTD", "IRFC", "IREDA", "IRCTC", "RVNL", "BEML",
    "LICHSGFIN", "MAXHEALTH", "FORTIS", "SYNGENE", "GLAND",
    "NAM-INDIA", "ANGELONE", "CDSL", "BSE", "MCX",
    "INDHOTEL", "DELHIVERY", "POLYCAB", "KEI", "DIXON", "AMBER",
    "ASTRAL", "SUPREMEIND", "KAJARIACER", "CENTURYPLY",
    "OBEROIRLTY", "GODREJPROP", "PRESTIGE", "PHOENIXLTD", "LODHA",
    "IDEA", "INDUSTOWER", "PERSISTENT", "ZENSARTECH",
    "CROMPTON", "WHIRLPOOL", "VOLTAS", "BLUESTARCO", "VGUARD",
    "CGPOWER", "THERMAX", "KALYANKJIL", "TITAGARH", "HINDPETRO",
    "MRPL", "MGL", "OIL", "CASTROLIND", "GNFC", "CHAMBLFERT",
    "COROMANDEL", "UPL", "BAYERCROP", "SUMICHEM", "RALLIS",
]
NIFTY_200 = sorted(set(NIFTY_200))  # de-dup


# --------------------------------------------------------------------------- #
# Indicators
# --------------------------------------------------------------------------- #
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Attach EMA10/21/50 and Wilder RSI(14) to a Close-only DataFrame."""
    close = df["Close"].astype(float)

    df["EMA10"] = close.ewm(span=EMA_FAST, adjust=False).mean()
    df["EMA21"] = close.ewm(span=EMA_MID, adjust=False).mean()
    df["EMA50"] = close.ewm(span=EMA_SLOW, adjust=False).mean()

    # Wilder's RSI (EMA smoothing with alpha = 1/period).
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean()
    avg_loss = loss.ewm(alpha=1.0 / RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    df["RSI"] = 100.0 - (100.0 / (1.0 + rs))
    df.loc[avg_loss == 0.0, "RSI"] = 100.0  # zero losses -> RSI 100
    return df


# --------------------------------------------------------------------------- #
# Data download (cached + parallel)
# --------------------------------------------------------------------------- #
def _normalize_symbol(sym: str) -> str:
    """Add '.NS' (NSE) if the user gave a bare symbol without an exchange suffix."""
    sym = str(sym).strip().upper()
    if not sym:
        return ""
    if "." not in sym:
        sym = sym + ".NS"
    return sym


@st.cache_data(show_spinner=False, ttl=3600)
def download_data(symbols: tuple[str, ...], start_iso: str, end_iso: str) -> dict[str, pd.DataFrame]:
    """
    Download daily OHLC (adjusted) closes for every symbol in parallel.
    Cached on (symbols, start, end). Returns {symbol: DataFrame[Close]}.
    """
    start = pd.to_datetime(start_iso)
    end = pd.to_datetime(end_iso) + timedelta(days=1)  # yfinance end is exclusive

    def fetch(sym: str):
        try:
            raw = yf.download(
                sym, start=start, end=end, progress=False,
                auto_adjust=True, threads=False,
            )
            if raw is None or raw.empty:
                return sym, None
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            if "Close" not in raw.columns:
                return sym, None
            keep = [c for c in ["Open", "High", "Low", "Close"] if c in raw.columns]
            out = raw[keep].dropna().copy()
            out.index = pd.to_datetime(out.index).tz_localize(None)
            return sym, out
        except Exception:
            return sym, None

    results: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=10) as pool:
        for sym, df in pool.map(fetch, symbols):
            if df is not None and len(df) >= MIN_BARS:
                results[sym] = df
    return results


# --------------------------------------------------------------------------- #
# Exit logic (spec left this open; configurable rules with swing defaults)
# --------------------------------------------------------------------------- #
def compute_exit(opens, highs, lows, closes, e10, e21, buy_pos,
                 target_pct, stop_pct, max_hold, use_ema_exit):
    """
    Forward-scan from the bar after entry and return (exit_pos, exit_price, reason).

    Price-level exits fill AT the level, not at the breaching close, so realized
    results respect the configured target/stop %s:
      * Target  -> fill at entry*(1+target%); if the bar GAPS open above it, fill
                   at the open (you cannot get filled better than the open).
      * Stop    -> fill at entry*(1-stop%); if the bar GAPS open below it, fill at
                   the open (a stop cannot protect against an overnight gap — that
                   slippage is real, not a bug).
    Within one bar the stop is assumed hit before the target (conservative). EMA
    and max-holding exits are not price levels, so they fill at that bar's close.
    Uses only current-bar OHLC -> no look-ahead. Falls back to 'Open (End)'.
    """
    n = len(closes)
    entry = closes[buy_pos]
    target_price = entry * (1.0 + target_pct / 100.0)
    stop_price = entry * (1.0 - stop_pct / 100.0)

    for j in range(buy_pos + 1, n):
        o, h, l, c = opens[j], highs[j], lows[j], closes[j]
        if np.isnan(c):
            continue
        hold = j - buy_pos

        # 1) Opening gaps straight through a level -> realistic fill at the open.
        if stop_pct > 0 and o <= stop_price:
            return j, float(o), "Stop Loss (gap)"
        if target_pct > 0 and o >= target_price:
            return j, float(o), "Target (gap)"

        # 2) Intrabar touch -> fill exactly at the level (stop prioritized).
        if stop_pct > 0 and l <= stop_price:
            return j, float(stop_price), "Stop Loss"
        if target_pct > 0 and h >= target_price:
            return j, float(target_price), "Target"

        # 3) Non-level exits fill at the bar close.
        if use_ema_exit and not np.isnan(e10[j]) and not np.isnan(e21[j]) and e10[j] < e21[j]:
            return j, float(c), "EMA10<EMA21"
        if max_hold > 0 and hold >= max_hold:
            return j, float(c), "Max Holding"

    return n - 1, float(closes[n - 1]), "Open (End)"


# --------------------------------------------------------------------------- #
# Core state machine: scan one stock
# --------------------------------------------------------------------------- #
def scan_stock(symbol, df, exit_cfg):
    """
    Run the tracking / crossover / buy state machine over one prepared DataFrame.
    Returns dict: {trades, failed, tracking}.
      * trades  : completed buy signals (with crossover dates + precomputed exit)
      * failed  : cycles where all crossovers occurred but the gap exceeded 3 days
      * tracking: status snapshot if the stock is still tracking at the last bar
    """
    idx = df.index
    closes = df["Close"].values
    opens = df["Open"].values if "Open" in df.columns else closes
    highs = df["High"].values if "High" in df.columns else closes
    lows = df["Low"].values if "Low" in df.columns else closes
    e10 = df["EMA10"].values
    e21 = df["EMA21"].values
    e50 = df["EMA50"].values
    rsi = df["RSI"].values
    n = len(df)

    trades, failed = [], []

    tracking = False
    rsi_pos = None                     # bar where RSI<35 started this cycle
    track_start_pos = None
    cross = {"10_50": None, "21_50": None, "10_21": None}  # first-occurrence (pos, date)

    def reset_cycle():
        nonlocal tracking, rsi_pos, track_start_pos, cross
        tracking = False
        rsi_pos = None
        track_start_pos = None
        cross = {"10_50": None, "21_50": None, "10_21": None}

    for i in range(1, n):
        # Skip bars where required indicators are not yet warm.
        if np.isnan(e50[i]) or np.isnan(e50[i - 1]) or np.isnan(rsi[i]):
            continue

        # --- Step 1: begin tracking on a fresh RSI<35 close (one session/stock) ---
        if not tracking and rsi[i] < RSI_THRESHOLD:
            tracking = True
            rsi_pos = i
            track_start_pos = i
            cross = {"10_50": None, "21_50": None, "10_21": None}
            continue  # crossovers are only monitored on bars AFTER tracking starts

        if not tracking:
            continue

        # --- Step 2: record the FIRST occurrence of each bullish crossover ---
        if cross["10_50"] is None and e10[i - 1] <= e50[i - 1] and e10[i] > e50[i]:
            cross["10_50"] = (i, idx[i])
        if cross["21_50"] is None and e21[i - 1] <= e50[i - 1] and e21[i] > e50[i]:
            cross["21_50"] = (i, idx[i])
        if cross["10_21"] is None and e10[i - 1] <= e21[i - 1] and e10[i] > e21[i]:
            cross["10_21"] = (i, idx[i])

        # --- Step 3/4: all three present -> evaluate the gap on the completion bar ---
        if all(v is not None for v in cross.values()):
            positions = [v[0] for v in cross.values()]
            gap = max(positions) - min(positions)
            rec = {
                "symbol": symbol,
                "rsi_date": idx[rsi_pos],
                "track_start": idx[track_start_pos],
                "c10_50": cross["10_50"][1],
                "c21_50": cross["21_50"][1],
                "c10_21": cross["10_21"][1],
                "gap": int(gap),
            }
            if gap <= MAX_CROSS_GAP:
                buy_pos = i                      # completion bar = final crossover bar
                buy_price = float(closes[buy_pos])
                ex_pos, ex_price, ex_reason = compute_exit(
                    opens, highs, lows, closes, e10, e21, buy_pos, *exit_cfg
                )
                rec.update({
                    "buy_pos": buy_pos,
                    "buy_date": idx[buy_pos],
                    "buy_price": buy_price,
                    "exit_pos": ex_pos,
                    "exit_date": idx[ex_pos],
                    "exit_price": float(ex_price),
                    "exit_reason": ex_reason,
                })
                trades.append(rec)
            else:
                failed.append(rec)
            reset_cycle()  # terminal for this cycle; wait for a fresh RSI<35

    # Snapshot: is the stock still mid-tracking at the last bar?
    tracking_now = None
    if tracking:
        got = {k: (v[1] if v else None) for k, v in cross.items()}
        tracking_now = {
            "symbol": symbol,
            "rsi_date": idx[rsi_pos] if rsi_pos is not None else None,
            "c10_50": got["10_50"],
            "c21_50": got["21_50"],
            "c10_21": got["10_21"],
            "crossovers_found": sum(v is not None for v in cross.values()),
        }

    return {"trades": trades, "failed": failed, "tracking": tracking_now}


# --------------------------------------------------------------------------- #
# Portfolio backtest (event-driven, chronological)
# --------------------------------------------------------------------------- #
def run_backtest(all_trades, price_data, start, end, initial_capital,
                 position_size, max_positions):
    """
    Simulate the portfolio over the selected window. Signals outside [start,end]
    are ignored. Each date: process exits first (freeing cash/slots), then entries.
    Skips a signal when a slot, cash, or whole-share is unavailable, or the symbol
    is already held. Returns (closed_trades, equity_df, metrics, skipped_count).
    """
    signals = [t for t in all_trades if start <= t["buy_date"].date() <= end]
    signals.sort(key=lambda t: (t["buy_date"], t["symbol"]))

    # Master trading calendar across all used symbols, bounded to the window.
    used_syms = {t["symbol"] for t in signals}
    dates_union = sorted({
        d for s in used_syms for d in price_data[s].index
        if start <= d.date() <= end
    })
    if not dates_union:
        return [], pd.DataFrame(columns=["Date", "Equity"]), {}, 0

    # Per-symbol forward-filled close aligned to the master calendar (for MTM).
    close_series = {
        s: price_data[s]["Close"].reindex(dates_union).ffill() for s in used_syms
    }
    signals_by_date: dict[pd.Timestamp, list] = {}
    for t in signals:
        signals_by_date.setdefault(t["buy_date"], []).append(t)

    cash = float(initial_capital)
    open_pos: dict[str, dict] = {}
    closed: list[dict] = []
    equity_rows: list[tuple] = []
    skipped = 0

    for d in dates_union:
        # 1) Exits due today.
        for sym in list(open_pos):
            pos = open_pos[sym]
            if pos["exit_date"] == d:
                proceeds = pos["shares"] * pos["exit_price"]
                cash += proceeds
                pl = proceeds - pos["cost"]
                pos["pl"] = pl
                pos["pl_pct"] = (pl / pos["cost"]) * 100.0 if pos["cost"] else 0.0
                pos["holding_days"] = (pos["exit_date"] - pos["buy_date"]).days
                closed.append(pos)
                del open_pos[sym]

        # 2) Entries signalled today.
        for t in signals_by_date.get(d, []):
            sym = t["symbol"]
            if sym in open_pos:                 # one position per stock
                skipped += 1
                continue
            if len(open_pos) >= max_positions:  # no free slot
                skipped += 1
                continue
            shares = int(position_size // t["buy_price"])
            if shares < 1:                      # price above position size
                skipped += 1
                continue
            cost = shares * t["buy_price"]
            if cost > cash:                     # no capital
                skipped += 1
                continue
            cash -= cost
            open_pos[sym] = {**t, "shares": shares, "cost": cost}

        # 3) Mark-to-market portfolio equity.
        mv = 0.0
        for sym, pos in open_pos.items():
            px = close_series[sym].get(d, np.nan)
            if not np.isnan(px):
                mv += pos["shares"] * px
        equity_rows.append((d, cash + mv))

    # Force-close anything still open at the final bar (safety; usually none).
    last_d = dates_union[-1]
    for sym in list(open_pos):
        pos = open_pos[sym]
        px = float(close_series[sym].get(last_d, pos["buy_price"]))
        proceeds = pos["shares"] * px
        cash += proceeds
        pl = proceeds - pos["cost"]
        pos.update({
            "exit_date": last_d, "exit_price": px, "exit_reason": "Open (End)",
            "pl": pl, "pl_pct": (pl / pos["cost"]) * 100.0 if pos["cost"] else 0.0,
            "holding_days": (last_d - pos["buy_date"]).days,
        })
        closed.append(pos)
        del open_pos[sym]

    equity_df = pd.DataFrame(equity_rows, columns=["Date", "Equity"])
    metrics = compute_metrics(closed, equity_df, initial_capital, start, end)
    return closed, equity_df, metrics, skipped


def compute_metrics(closed, equity_df, initial_capital, start, end):
    """Portfolio performance metrics from closed trades + the equity curve."""
    total = len(closed)
    if total == 0:
        return {
            "total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
            "net_profit": 0.0, "final_capital": float(initial_capital),
            "cagr": None, "max_drawdown": 0.0, "profit_factor": None,
        }

    pls = np.array([t["pl"] for t in closed], dtype=float)
    wins = int((pls > 0).sum())
    losses = int((pls < 0).sum())
    net_profit = float(pls.sum())
    final_capital = float(initial_capital) + net_profit

    gross_profit = float(pls[pls > 0].sum())
    gross_loss = float(-pls[pls < 0].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None

    # CAGR (only if a positive final equity over a positive horizon).
    years = (end - start).days / 365.25
    cagr = None
    if years > 0 and final_capital > 0 and initial_capital > 0:
        cagr = (final_capital / initial_capital) ** (1.0 / years) - 1.0

    # Max drawdown on the mark-to-market equity curve.
    max_dd = 0.0
    if not equity_df.empty:
        eq = equity_df["Equity"].values.astype(float)
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        max_dd = float(dd.min()) if len(dd) else 0.0

    return {
        "total_trades": total, "wins": wins, "losses": losses,
        "win_rate": wins / total * 100.0,
        "net_profit": net_profit, "final_capital": final_capital,
        "cagr": cagr, "max_drawdown": max_dd * 100.0,
        "profit_factor": profit_factor,
    }


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def _d(x):
    return pd.to_datetime(x).strftime("%Y-%m-%d") if x is not None and pd.notna(x) else ""


def build_trade_log(closed: list[dict]) -> pd.DataFrame:
    """Trade log with the exact columns requested by the spec."""
    rows = []
    for t in sorted(closed, key=lambda x: x["buy_date"]):
        rows.append({
            "Symbol": t["symbol"],
            "Tracking Start Date": _d(t.get("track_start")),
            "RSI Below 35 Date": _d(t.get("rsi_date")),
            "EMA10 > EMA50 Date": _d(t.get("c10_50")),
            "EMA21 > EMA50 Date": _d(t.get("c21_50")),
            "EMA10 > EMA21 Date": _d(t.get("c10_21")),
            "Buy Date": _d(t.get("buy_date")),
            "Buy Price": round(float(t["buy_price"]), 2),
            "Exit Date": _d(t.get("exit_date")),
            "Exit Price": round(float(t["exit_price"]), 2),
            "Exit Reason": t.get("exit_reason", ""),
            "Profit/Loss Rs": round(float(t.get("pl", 0.0)), 2),
            "Profit/Loss %": round(float(t.get("pl_pct", 0.0)), 2),
            "Holding Days": int(t.get("holding_days", 0)),
        })
    cols = [
        "Symbol", "Tracking Start Date", "RSI Below 35 Date", "EMA10 > EMA50 Date",
        "EMA21 > EMA50 Date", "EMA10 > EMA21 Date", "Buy Date", "Buy Price",
        "Exit Date", "Exit Price", "Exit Reason", "Profit/Loss Rs",
        "Profit/Loss %", "Holding Days",
    ]
    return pd.DataFrame(rows, columns=cols)


def to_excel_bytes(df: pd.DataFrame, sheet="Sheet1") -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet)
    return buf.getvalue()


def resolve_symbols(source, uploaded, manual_text) -> list[str]:
    """Resolve the symbol universe from the selected source."""
    syms: list[str] = []
    if source == "NIFTY 200 (default)":
        syms = list(NIFTY_200)
    elif source == "Upload CSV/Excel":
        if uploaded is not None:
            try:
                if uploaded.name.lower().endswith((".xlsx", ".xls")):
                    up = pd.read_excel(uploaded, engine="openpyxl")
                else:
                    up = pd.read_csv(uploaded)
                col = next((c for c in up.columns if str(c).strip().lower() == "symbol"), None)
                if col is not None:
                    syms = up[col].dropna().astype(str).tolist()
                else:
                    st.error("Uploaded file must contain a 'Symbol' column.")
            except Exception as e:
                st.error(f"Could not read file: {e}")
    else:  # Manual entry
        raw = manual_text.replace(",", "\n").splitlines()
        syms = [s for s in (r.strip() for r in raw) if s]

    normalized = [_normalize_symbol(s) for s in syms]
    return sorted({s for s in normalized if s})


# --------------------------------------------------------------------------- #
# Scan orchestration (download -> indicators -> per-stock state machine)
# --------------------------------------------------------------------------- #
def run_scan(symbols, start, end, exit_cfg, progress_cb=None):
    """Download, compute indicators, and scan every symbol. Returns aggregated results."""
    fetch_start = (start - timedelta(days=WARMUP_DAYS)).isoformat()
    data = download_data(tuple(symbols), fetch_start, end.isoformat())

    all_trades, all_failed, all_tracking, price_data = [], [], [], {}
    n = len(symbols)
    for k, sym in enumerate(symbols):
        if progress_cb:
            progress_cb((k + 1) / max(n, 1))
        df = data.get(sym)
        if df is None or len(df) < MIN_BARS:
            continue
        df = add_indicators(df.copy())
        price_data[sym] = df

        res = scan_stock(sym, df, exit_cfg)
        # Keep signals/failures whose relevant date falls in the selected window.
        for t in res["trades"]:
            if start <= t["buy_date"].date() <= end:
                all_trades.append(t)
        for f in res["failed"]:
            ref = f["c10_50"] or f["c21_50"] or f["c10_21"]
            if ref is not None and start <= pd.to_datetime(ref).date() <= end:
                all_failed.append(f)
        if res["tracking"] is not None:
            all_tracking.append(res["tracking"])

    return {
        "trades": all_trades, "failed": all_failed,
        "tracking": all_tracking, "price_data": price_data,
        "downloaded": len(data), "requested": n,
    }


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title="NSE Buy-Signal Scanner & Backtester",
                       page_icon="📈", layout="wide")
    st.title("📈 NSE/BSE Buy-Signal Scanner & Backtester")
    st.caption("RSI(14)<35 tracking → three EMA crossovers within 3 trading days → buy on close. "
               "No repainting, no look-ahead.")

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.header("Configuration")

        today = date.today()
        default_start = today - timedelta(days=365 * 3)
        c1, c2 = st.columns(2)
        start_date = c1.date_input("Start date", value=default_start, max_value=today)
        end_date = c2.date_input("End date", value=today, max_value=today)

        st.subheader("Capital & Sizing")
        initial_capital = st.number_input("Initial capital (₹)", min_value=0.0,
                                           value=1_000_000.0, step=50_000.0, format="%.0f")
        position_size = st.number_input("Position size per trade (₹)", min_value=0.0,
                                        value=10_000.0, step=1_000.0, format="%.0f")
        max_positions = st.number_input("Maximum simultaneous positions", min_value=1,
                                        value=10, step=1)

        st.subheader("Symbol Source")
        source = st.radio("Universe", ["NIFTY 200 (default)", "Upload CSV/Excel", "Manual entry"],
                          label_visibility="collapsed")
        uploaded = None
        manual_text = ""
        if source == "Upload CSV/Excel":
            uploaded = st.file_uploader("File with a 'Symbol' column", type=["csv", "xlsx", "xls"])
        elif source == "Manual entry":
            manual_text = st.text_area("Symbols (comma or newline separated)",
                                       value="RELIANCE, TCS, INFY", height=90)

        with st.expander("Exit rules (not in original spec — tune as needed)"):
            st.caption("Long exits are close-based. First rule to trigger wins.")
            target_pct = st.number_input("Target %", min_value=0.0, value=15.0, step=1.0)
            stop_pct = st.number_input("Stop-loss %", min_value=0.0, value=7.0, step=1.0)
            max_hold = st.number_input("Max holding (trading days)", min_value=0, value=30, step=5)
            use_ema_exit = st.checkbox("Exit when EMA10 < EMA21", value=True)

        st.divider()
        run_scanner = st.button("🔍 Run Scanner", use_container_width=True)
        run_bt = st.button("🧪 Run Backtest", use_container_width=True, type="primary")

    exit_cfg = (target_pct, stop_pct, max_hold, use_ema_exit)

    # ---------------- Validation ----------------
    if start_date >= end_date:
        st.warning("Start date must be before end date.")
        return

    if not (run_scanner or run_bt):
        st.info("Configure the sidebar, then run the Scanner or the Backtest.")
        st.stop()

    symbols = resolve_symbols(source, uploaded, manual_text)
    if not symbols:
        st.error("No symbols resolved. Choose a source and provide symbols.")
        return

    # ---------------- Execute scan ----------------
    prog = st.progress(0.0, text=f"Scanning {len(symbols)} symbols…")
    results = run_scan(symbols, start_date, end_date, exit_cfg,
                       progress_cb=lambda p: prog.progress(min(p, 1.0),
                                                           text=f"Scanning… {int(p*100)}%"))
    prog.empty()

    st.caption(f"Data available for {results['downloaded']} of {results['requested']} requested "
               f"symbols (others skipped due to missing/insufficient data).")

    trades = results["trades"]
    failed = results["failed"]
    tracking = results["tracking"]

    # ---------------- Scanner view ----------------
    st.header("Scanner")
    m1, m2, m3 = st.columns(3)
    m1.metric("Buy signals", len(trades))
    m2.metric("Currently tracking", len(tracking))
    m3.metric("Failed (gap > 3d)", len(failed))

    st.subheader("🟢 Buy Signals")
    if trades:
        buy_df = pd.DataFrame([{
            "Symbol": t["symbol"],
            "RSI<35 Date": _d(t["rsi_date"]),
            "EMA10>EMA50": _d(t["c10_50"]),
            "EMA21>EMA50": _d(t["c21_50"]),
            "EMA10>EMA21": _d(t["c10_21"]),
            "Gap (days)": t["gap"],
            "Buy Date": _d(t["buy_date"]),
            "Buy Price": round(t["buy_price"], 2),
        } for t in sorted(trades, key=lambda x: x["buy_date"], reverse=True)])
        st.dataframe(buy_df, use_container_width=True, hide_index=True)
    else:
        st.write("No buy signals in the selected window.")

    cta, ctb = st.columns(2)
    with cta:
        st.subheader("👀 Currently Tracking")
        if tracking:
            trk_df = pd.DataFrame([{
                "Symbol": t["symbol"],
                "RSI<35 Date": _d(t["rsi_date"]),
                "Crossovers found": t["crossovers_found"],
                "EMA10>EMA50": _d(t["c10_50"]),
                "EMA21>EMA50": _d(t["c21_50"]),
                "EMA10>EMA21": _d(t["c10_21"]),
            } for t in tracking])
            st.dataframe(trk_df, use_container_width=True, hide_index=True)
        else:
            st.write("No stocks are mid-tracking.")
    with ctb:
        st.subheader("🔴 Failed Setups (gap > 3 days)")
        if failed:
            fail_df = pd.DataFrame([{
                "Symbol": f["symbol"],
                "RSI<35 Date": _d(f["rsi_date"]),
                "EMA10>EMA50": _d(f["c10_50"]),
                "EMA21>EMA50": _d(f["c21_50"]),
                "EMA10>EMA21": _d(f["c10_21"]),
                "Gap (days)": f["gap"],
            } for f in sorted(failed, key=lambda x: x["gap"], reverse=True)])
            st.dataframe(fail_df, use_container_width=True, hide_index=True)
        else:
            st.write("No failed setups.")

    # ---------------- Backtest view ----------------
    if run_bt:
        st.header("Backtest")
        closed, equity_df, metrics, skipped = run_backtest(
            trades, results["price_data"], start_date, end_date,
            initial_capital, position_size, int(max_positions),
        )

        if metrics.get("total_trades", 0) == 0:
            st.warning("No trades were executed in the backtest window.")
            return

        r1 = st.columns(4)
        r1[0].metric("Total trades", metrics["total_trades"])
        r1[1].metric("Win rate", f"{metrics['win_rate']:.1f}%")
        r1[2].metric("Wins / Losses", f"{metrics['wins']} / {metrics['losses']}")
        r1[3].metric("Skipped signals", skipped)

        r2 = st.columns(4)
        r2[0].metric("Net profit", f"₹{metrics['net_profit']:,.0f}")
        r2[1].metric("Final capital", f"₹{metrics['final_capital']:,.0f}")
        cagr = metrics["cagr"]
        r2[2].metric("CAGR", f"{cagr*100:.2f}%" if cagr is not None else "N/A")
        r2[3].metric("Max drawdown", f"{metrics['max_drawdown']:.2f}%")

        r3 = st.columns(4)
        pf = metrics["profit_factor"]
        r3[0].metric("Profit factor", f"{pf:.2f}" if pf is not None else "∞ / N/A")

        # Equity curve.
        if not equity_df.empty:
            st.subheader("Equity Curve")
            st.line_chart(equity_df.set_index("Date")["Equity"], use_container_width=True)

        # Trade log + downloads.
        st.subheader("Trade Log")
        log_df = build_trade_log(closed)
        st.dataframe(log_df, use_container_width=True, hide_index=True)

        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Download CSV", log_df.to_csv(index=False).encode("utf-8"),
                           file_name="trade_log.csv", mime="text/csv",
                           use_container_width=True)
        d2.download_button("⬇️ Download Excel", to_excel_bytes(log_df, "TradeLog"),
                           file_name="trade_log.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)


if __name__ == "__main__":
    main()
