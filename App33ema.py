"""
NSE Scanner + Backtester
=========================
Strategy = Dynamic Swing Anchored VWAP (Zeiierman) [trend filter]
         + EMA Channel Key K-Lines (akriafly) [trigger]

Both indicators are ported bar-for-bar from their original Pine Script v6
source (as supplied by the user) with NO simplification of the calculations:

  - Dynamic Swing Anchored VWAP (Zeiierman), CC BY-NC-SA 4.0
    swing pivots via ta.highestbars/ta.lowestbars(prd), regime "dir",
    ATR(50)-ratio adaptive APT -> half-life EWMA alpha, full anchored
    VWAP segment recompute on every regime flip (identical loop logic).

  - EMA Channel Key K-Lines (akriafly)
    3x EMA(32) on High/Low/Close, breakout vs channel height, body-ratio
    filter, alternating-signal state machine (no consecutive same-side
    signals) - reproduced exactly incl. Pine's per-bar evaluation order.

Only `dir` (Uptrend=+1 / Downtrend=-1) from the VWAP script feeds the
strategy - the VWAP line itself has no bearing on any condition in the
original indicator, so it is computed for reference/plot purposes only.

Author: built for Ritivk's NSE Scanner Pro suite.
Python 3.11+, Streamlit, Pandas, NumPy, yfinance.
"""

import io
import math
import concurrent.futures as cf
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(page_title="NSE Scanner Pro — VWAP+EMA Channel", layout="wide")

# ============================================================================
# ORIGINAL INDICATOR PARAMETERS (Pine defaults — do not alter without reason)
# ============================================================================
DEFAULTS = dict(
    swing_period=50,       # prd
    base_apt=20.0,         # Adaptive Price Tracking
    use_adapt=False,       # Adapt APT by ATR ratio
    vol_bias=10.0,         # Volatility Bias
    atr_len=50,            # ATR Length (fixed at 50 in the original script)
    ema_length=32,         # EMA Channel EMA Length
    body_ratio=0.66,       # Body Ratio threshold (66%)
)

# ============================================================================
# LOW-LEVEL PINE-EXACT TA PRIMITIVES
# ============================================================================

def pine_ema(src: np.ndarray, length: int) -> np.ndarray:
    """Pine ta.ema: seeds at the very first bar with src[0], then recurses
    with alpha = 2/(length+1). NOT sma-seeded."""
    n = len(src)
    out = np.full(n, np.nan)
    if n == 0:
        return out
    alpha = 2.0 / (length + 1)
    out[0] = src[0]
    for i in range(1, n):
        prev = out[i - 1]
        if np.isnan(prev):
            out[i] = src[i]
        else:
            out[i] = alpha * src[i] + (1 - alpha) * prev
    return out


def pine_rma(src: np.ndarray, length: int) -> np.ndarray:
    """Pine ta.rma: rma = na(rma[1]) ? sma(src,length) : (src-rma[1])*alpha+rma[1]
    alpha = 1/length. rma[1] is na until `length` bars have accumulated."""
    n = len(src)
    out = np.full(n, np.nan)
    if n == 0:
        return out
    alpha = 1.0 / length
    for i in range(n):
        prev = out[i - 1] if i > 0 else np.nan
        if np.isnan(prev):
            if i >= length - 1:
                window = src[i - length + 1:i + 1]
                if np.any(np.isnan(window)):
                    out[i] = np.nan
                else:
                    out[i] = np.mean(window)
            else:
                out[i] = np.nan
        else:
            out[i] = (src[i] - prev) * alpha + prev
    return out


def pine_true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    n = len(high)
    tr = np.empty(n)
    for i in range(n):
        if i == 0 or np.isnan(close[i - 1]):
            tr[i] = high[i] - low[i]
        else:
            tr[i] = max(high[i] - low[i],
                        abs(high[i] - close[i - 1]),
                        abs(low[i] - close[i - 1]))
    return tr


def pine_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int) -> np.ndarray:
    tr = pine_true_range(high, low, close)
    return pine_rma(tr, length)


# ============================================================================
# INDICATOR 1 — Dynamic Swing Anchored VWAP (Zeiierman)
# Exact bar-for-bar port of the supplied Pine v6 source.
# ============================================================================

def compute_dynamic_swing_vwap(df: pd.DataFrame,
                                prd: int = 50,
                                base_apt: float = 20.0,
                                use_adapt: bool = False,
                                vol_bias: float = 10.0,
                                atr_len: int = 50) -> pd.DataFrame:
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)
    volume = df["Volume"].to_numpy(dtype=float)
    n = len(df)
    hlc3 = (high + low + close) / 3.0

    # --- swing pivot detection: ta.highestbars/ta.lowestbars(prd) == 0 -----
    roll_max = pd.Series(high).rolling(prd, min_periods=prd).max().to_numpy()
    roll_min = pd.Series(low).rolling(prd, min_periods=prd).min().to_numpy()
    is_new_high = high == roll_max          # False where roll_max is NaN
    is_new_low = low == roll_min
    is_new_high = np.nan_to_num(is_new_high, nan=0.0).astype(bool) if roll_max.dtype == float else is_new_high
    # (roll_max NaN comparisons already evaluate False in numpy; kept explicit for clarity)

    ph = np.full(n, np.nan)
    pl = np.full(n, np.nan)
    phL = np.zeros(n, dtype=int)
    plL = np.zeros(n, dtype=int)
    cur_ph, cur_pl = np.nan, np.nan
    cur_phL, cur_plL = 0, 0
    for i in range(n):
        if bool(is_new_high[i]):
            cur_ph, cur_phL = high[i], i
        if bool(is_new_low[i]):
            cur_pl, cur_plL = low[i], i
        ph[i], pl[i] = cur_ph, cur_pl
        phL[i], plL[i] = cur_phL, cur_plL

    dir_arr = np.where(phL > plL, 1, -1)

    # --- ATR-adaptive APT --------------------------------------------------
    atr_arr = pine_atr(high, low, close, atr_len)
    atr_avg = pine_rma(atr_arr, atr_len)
    ratio = np.where((~np.isnan(atr_avg)) & (atr_avg > 0), atr_arr / np.where(atr_avg == 0, np.nan, atr_avg), 1.0)
    ratio = np.nan_to_num(ratio, nan=1.0)

    if use_adapt:
        with np.errstate(divide="ignore", invalid="ignore"):
            apt_raw = base_apt / np.power(ratio, vol_bias)
        apt_raw = np.nan_to_num(apt_raw, nan=base_apt, posinf=300.0, neginf=5.0)
    else:
        apt_raw = np.full(n, base_apt)
    apt_clamped = np.clip(apt_raw, 5.0, 300.0)
    apt_series = np.round(apt_clamped).astype(int)

    ln2 = math.log(2.0)

    def alpha_from_apt(apt: int) -> float:
        decay = math.exp(-ln2 / max(1.0, float(apt)))
        return 1.0 - decay

    alpha_arr = np.array([alpha_from_apt(a) for a in apt_series])

    # --- anchored VWAP recursive engine (exact port of the if/else block) --
    vwap_val = np.full(n, np.nan)
    if n > 0:
        p_acc = hlc3[0] * volume[0]
        vol_acc = volume[0]
        for i in range(n):
            d = dir_arr[i]
            flip = (i == 0) or (d != dir_arr[i - 1])
            if flip:
                if d > 0:
                    x, y = plL[i], pl[i]
                else:
                    x, y = phL[i], ph[i]
                if np.isnan(y):
                    # no pivot established yet (warm-up) - hold state
                    vwap_val[i] = vwap_val[i - 1] if i > 0 else np.nan
                    continue
                barsback = i - int(x)
                p_acc = y * volume[int(x)]
                vol_acc = volume[int(x)]
                for k in range(barsback, -1, -1):
                    idx = i - k
                    a = alpha_arr[idx]
                    pxv = hlc3[idx] * volume[idx]
                    p_acc = (1.0 - a) * p_acc + a * pxv
                    vol_acc = (1.0 - a) * vol_acc + a * volume[idx]
                    vwap_val[idx] = p_acc / vol_acc if vol_acc > 0 else np.nan
            else:
                a = alpha_arr[i]
                pxv = hlc3[i] * volume[i]
                p_acc = (1.0 - a) * p_acc + a * pxv
                vol_acc = (1.0 - a) * vol_acc + a * volume[i]
                vwap_val[i] = p_acc / vol_acc if vol_acc > 0 else np.nan

    out = df.copy()
    out["vwap_dir"] = dir_arr           # +1 = Uptrend, -1 = Downtrend
    out["vwap_value"] = vwap_val
    out["vwap_flip"] = np.concatenate(([True], dir_arr[1:] != dir_arr[:-1]))
    return out


# ============================================================================
# INDICATOR 2 — EMA Channel Key K-Lines (akriafly)
# Exact bar-for-bar port incl. sequential buy-then-sell evaluation order.
# ============================================================================

def compute_ema_channel_signals(df: pd.DataFrame,
                                 ema_length: int = 32,
                                 body_ratio_th: float = 0.66) -> pd.DataFrame:
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)
    open_ = df["Open"].to_numpy(dtype=float)
    n = len(df)

    ema_high = pine_ema(high, ema_length)
    ema_low = pine_ema(low, ema_length)
    ema_close = pine_ema(close, ema_length)

    channel_height = ema_high - ema_low
    body_size = np.abs(close - open_)
    total_range = high - low
    with np.errstate(divide="ignore", invalid="ignore"):
        body_ratio = np.where(total_range > 0, body_size / total_range, 0.0)

    key_buy = np.zeros(n, dtype=bool)
    key_sell = np.zeros(n, dtype=bool)

    last_was_buy = False
    last_was_sell = False
    for i in range(n):
        if np.isnan(ema_high[i]) or np.isnan(ema_low[i]):
            continue
        rng_ge_channel = (high[i] - low[i]) >= channel_height[i]

        buy_cond = (close[i] > ema_high[i]) and rng_ge_channel and (body_ratio[i] >= body_ratio_th) and (not last_was_buy)
        if buy_cond:
            key_buy[i] = True
            last_was_buy = True
            last_was_sell = False

        sell_cond = (close[i] < ema_low[i]) and rng_ge_channel and (body_ratio[i] >= body_ratio_th) and (not last_was_sell)
        if sell_cond:
            key_sell[i] = True
            last_was_sell = True
            last_was_buy = False

    out = df.copy()
    out["ema_high"] = ema_high
    out["ema_low"] = ema_low
    out["ema_close"] = ema_close
    out["ema_channel_height"] = channel_height
    out["ema_key_buy"] = key_buy
    out["ema_key_sell"] = key_sell
    return out


# ============================================================================
# STRATEGY — combine both indicators exactly per the specified rules
# ============================================================================

def generate_signals(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Adds vwap_dir/vwap_value, ema channel columns, and final
    'buy_signal' / 'sell_signal' booleans (signal generated ON that bar's
    close, to be executed at the NEXT bar's open by the caller)."""
    d = compute_dynamic_swing_vwap(
        df, prd=params["swing_period"], base_apt=params["base_apt"],
        use_adapt=params["use_adapt"], vol_bias=params["vol_bias"],
        atr_len=params["atr_len"],
    )
    d = compute_ema_channel_signals(
        d, ema_length=params["ema_length"], body_ratio_th=params["body_ratio"]
    )

    vol = d["Volume"].to_numpy(dtype=float)
    prev_vol = np.concatenate(([np.nan], vol[:-1]))
    vol_up = vol > prev_vol

    uptrend = d["vwap_dir"].to_numpy() == 1
    downtrend_now = d["vwap_dir"].to_numpy() == -1
    flip = d["vwap_flip"].to_numpy().copy()
    if len(flip) > 0:
        flip[0] = False  # bar 0 is always a "flip" in Pine (dir != dir[1]=na) — that's the
        # initial VWAP anchor being set, not an actual trend change, so it must not fire a SELL.
    flip_to_down = flip & downtrend_now

    ema_buy = d["ema_key_buy"].to_numpy()
    ema_sell = d["ema_key_sell"].to_numpy()

    buy_signal = uptrend & ema_buy & vol_up
    sell_signal = flip_to_down | ema_sell

    d["buy_signal"] = buy_signal
    d["sell_signal"] = sell_signal
    d["trend"] = np.where(uptrend, "Uptrend", "Downtrend")
    return d


# ============================================================================
# DATA FETCHING (cached)
# ============================================================================

def to_nse_ticker(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if symbol.endswith(".NS") or symbol.endswith(".BO"):
        return symbol
    return f"{symbol}.NS"


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_history(symbol: str, start: str, end: str) -> pd.DataFrame:
    ticker = to_nse_ticker(symbol)
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False, threads=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    return df


def fetch_with_buffer(symbol: str, start_date: date, end_date: date, buffer_days: int = 400) -> pd.DataFrame:
    """Fetch extra history before start_date so the recursive indicators
    (swing pivots / ATR / EMA) have converged by the time the visible
    backtest window begins."""
    buffered_start = start_date - timedelta(days=buffer_days)
    df = fetch_history(symbol, buffered_start.isoformat(), (end_date + timedelta(days=1)).isoformat())
    return df


# ============================================================================
# SCANNER
# ============================================================================

@dataclass
class ScanResult:
    symbol: str
    signal: str
    signal_date: pd.Timestamp | None
    current_price: float | None
    trend: str
    error: str | None = None


def scan_symbol(symbol: str, start_date: date, end_date: date, params: dict) -> ScanResult:
    try:
        raw = fetch_with_buffer(symbol, start_date, end_date)
        if raw.empty or len(raw) < params["swing_period"] + 5:
            return ScanResult(symbol, "-", None, None, "-", error="Insufficient data")
        sig = generate_signals(raw, params)
        sig = sig[(sig.index.date >= start_date) & (sig.index.date <= end_date)]
        if sig.empty:
            return ScanResult(symbol, "-", None, None, "-", error="No data in range")
        last = sig.iloc[-1]
        signal = "BUY" if bool(last["buy_signal"]) else ("SELL" if bool(last["sell_signal"]) else "-")
        return ScanResult(
            symbol=symbol,
            signal=signal,
            signal_date=sig.index[-1],
            current_price=float(last["Close"]),
            trend=str(last["trend"]),
        )
    except Exception as e:  # noqa: BLE001
        return ScanResult(symbol, "-", None, None, "-", error=str(e))


def run_scanner(symbols: list[str], start_date: date, end_date: date, params: dict, max_workers: int = 8) -> pd.DataFrame:
    results = []
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(scan_symbol, s, start_date, end_date, params): s for s in symbols}
        for fut in cf.as_completed(futures):
            results.append(fut.result())
    rows = []
    for r in results:
        rows.append({
            "Symbol": r.symbol,
            "Signal": r.signal,
            "Signal Date": r.signal_date.date() if r.signal_date is not None else None,
            "Current Price": round(r.current_price, 2) if r.current_price is not None else None,
            "Trend": r.trend,
            "Note": r.error or "",
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        order = {"BUY": 0, "SELL": 1, "-": 2}
        out["_ord"] = out["Signal"].map(order).fillna(3)
        out = out.sort_values(["_ord", "Symbol"]).drop(columns="_ord").reset_index(drop=True)
    return out


# ============================================================================
# BACKTEST ENGINE
# Long only, one position per symbol, no pyramiding, no shorts.
# Signal on bar close -> executed at NEXT bar's open. Portfolio-level
# capital / max-open-positions coordination across all symbols.
# ============================================================================

@dataclass
class Trade:
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp | None = None
    exit_price: float | None = None
    qty: float = 0.0


def build_symbol_signal_frames(symbols: list[str], start_date: date, end_date: date, params: dict) -> dict:
    frames = {}
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch_with_buffer, s, start_date, end_date): s for s in symbols}
        raw_data = {}
        for fut in cf.as_completed(futs):
            s = futs[fut]
            try:
                raw_data[s] = fut.result()
            except Exception:
                raw_data[s] = pd.DataFrame()

    for s in symbols:
        raw = raw_data.get(s, pd.DataFrame())
        if raw.empty or len(raw) < params["swing_period"] + 5:
            continue
        sig = generate_signals(raw, params)
        frames[s] = sig
    return frames


def run_backtest(symbols: list[str], start_date: date, end_date: date, params: dict,
                  initial_capital: float, position_size_mode: str, position_size_value: float,
                  brokerage_pct: float, slippage_pct: float, max_open_positions: int) -> dict:

    frames = build_symbol_signal_frames(symbols, start_date, end_date, params)
    if not frames:
        metrics = compute_performance(pd.DataFrame(), pd.DataFrame(), initial_capital, start_date, end_date)
        return {"trades": pd.DataFrame(), "equity_curve": pd.DataFrame(), "metrics": metrics,
                "status": "No usable price data for any symbol in the selected range "
                          "(check symbols / date range / data availability)."}

    # Build execution events: signal on bar i (within [start,end]) -> exec at bar i+1 open.
    # A signal on the LAST bar of a symbol's series has no next bar to execute at, so it
    # is a valid signal but is NOT executable — tracked separately so we can report it
    # explicitly rather than silently returning empty results.
    events = []  # (exec_date, symbol, action, exec_open_price)
    price_panel = {}  # symbol -> date-indexed Close for mark-to-market
    total_signals_in_range = 0
    unexecutable_last_bar_signals = 0
    for s, d in frames.items():
        d = d.sort_index()
        price_panel[s] = d["Close"]
        idx = d.index
        n = len(d)
        for i in range(n):
            bar_date = idx[i].date()
            if bar_date < start_date or bar_date > end_date:
                continue
            is_buy = bool(d["buy_signal"].iloc[i])
            is_sell = bool(d["sell_signal"].iloc[i])
            if not (is_buy or is_sell):
                continue
            total_signals_in_range += 1
            if i == n - 1:
                # no next bar available to execute this signal at
                unexecutable_last_bar_signals += 1
                continue
            exec_date = idx[i + 1]
            exec_open = float(d["Open"].iloc[i + 1])
            if is_buy:
                events.append((exec_date, s, "BUY", exec_open))
            if is_sell:
                events.append((exec_date, s, "SELL", exec_open))

    if not events:
        metrics = compute_performance(pd.DataFrame(), pd.DataFrame(), initial_capital, start_date, end_date)
        if total_signals_in_range == 0:
            status = "No BUY/SELL signals were generated for any symbol in the selected date range."
        elif unexecutable_last_bar_signals == total_signals_in_range:
            status = (f"{total_signals_in_range} signal(s) were generated, but all fell on the last "
                      "available bar of their symbol's data with no next bar to execute at "
                      "(extend End Date or fetch more recent data to make them executable).")
        else:
            status = "Signals were generated but none could be matched to an executable next-bar open."
        return {"trades": pd.DataFrame(), "equity_curve": pd.DataFrame(), "metrics": metrics, "status": status}

    events_df = pd.DataFrame(events, columns=["exec_date", "symbol", "action", "price"])
    events_df = events_df.sort_values(["exec_date", "action", "symbol"])  # SELL < BUY alphabetically -> exits first
    all_dates = sorted(set(events_df["exec_date"]).union(
        *[set(p.index[(p.index.date >= start_date) & (p.index.date <= end_date)]) for p in price_panel.values()]
    ))

    cash = initial_capital
    open_positions: dict[str, Trade] = {}
    closed_trades: list[Trade] = []
    equity_curve = []

    def mark_to_market(as_of):
        mv = 0.0
        for sym, t in open_positions.items():
            s = price_panel.get(sym)
            if s is None:
                continue
            avail = s[s.index <= as_of]
            px = float(avail.iloc[-1]) if len(avail) else t.entry_price
            mv += px * t.qty
        return cash + mv

    grouped = events_df.groupby("exec_date")
    for d_ in all_dates:
        if d_ in grouped.groups:
            day_events = grouped.get_group(d_)

            # 1) process SELL (exits) first — frees capital & slots
            for _, ev in day_events[day_events["action"] == "SELL"].iterrows():
                sym = ev["symbol"]
                if sym not in open_positions:
                    continue
                exit_price_raw = ev["price"]
                if exit_price_raw is None or not np.isfinite(exit_price_raw) or exit_price_raw <= 0:
                    continue  # bad/missing execution price — keep position open rather than mis-price the exit
                t = open_positions.pop(sym)
                exit_price = exit_price_raw * (1 - slippage_pct / 100.0)
                gross = exit_price * t.qty
                fee = gross * (brokerage_pct / 100.0)
                cash += gross - fee
                t.exit_date = d_
                t.exit_price = exit_price
                closed_trades.append(t)

            # 2) process BUY (entries), respecting max open positions & cash
            for _, ev in day_events[day_events["action"] == "BUY"].iterrows():
                sym = ev["symbol"]
                if sym in open_positions:
                    continue  # no pyramiding
                if len(open_positions) >= max_open_positions:
                    continue
                entry_price_raw = ev["price"]
                if entry_price_raw is None or not np.isfinite(entry_price_raw) or entry_price_raw <= 0:
                    continue  # bad/missing execution price (e.g. data gap) — skip this entry
                entry_price = entry_price_raw * (1 + slippage_pct / 100.0)
                if entry_price <= 0:
                    continue

                if position_size_mode == "₹ Fixed":
                    alloc = min(position_size_value, cash)
                else:  # % of current equity
                    equity_now = mark_to_market(d_)
                    alloc = equity_now * (position_size_value / 100.0)
                    alloc = min(alloc, cash)

                fee_buffer = alloc * (brokerage_pct / 100.0)
                usable = alloc - fee_buffer
                if usable <= 0:
                    continue
                qty = usable / entry_price
                if qty <= 0:
                    continue
                cost = qty * entry_price
                fee = cost * (brokerage_pct / 100.0)
                total_cost = cost + fee
                if total_cost > cash:
                    continue
                cash -= total_cost
                open_positions[sym] = Trade(symbol=sym, entry_date=d_, entry_price=entry_price, qty=qty)

        equity_curve.append({"date": d_, "equity": mark_to_market(d_)})

    # Force-close remaining open positions at last available close (for reporting only)
    for sym, t in list(open_positions.items()):
        s = price_panel.get(sym)
        if s is None or s.empty:
            continue
        last_date = s.index[-1]
        last_price = float(s.iloc[-1])
        exit_price = last_price * (1 - slippage_pct / 100.0)
        gross = exit_price * t.qty
        fee = gross * (brokerage_pct / 100.0)
        cash += gross - fee
        t.exit_date = last_date
        t.exit_price = exit_price
        t.qty = t.qty
        closed_trades.append(t)
    open_positions.clear()

    trade_rows = []
    skipped_invalid_entry = 0
    for t in closed_trades:
        entry_price = t.entry_price
        exit_price = t.exit_price
        # Defensive guard: never divide by a zero/None/NaN entry price (e.g. corrupted
        # or corporate-action-gapped feed data slipping through). Such a trade cannot
        # have a meaningful P/L and is excluded from the log rather than crashing.
        if entry_price is None or exit_price is None or not np.isfinite(entry_price) or entry_price <= 0:
            skipped_invalid_entry += 1
            continue
        pnl_pct = (exit_price / entry_price - 1) * 100.0
        pnl_inr = (exit_price - entry_price) * t.qty
        holding_days = (t.exit_date - t.entry_date).days if t.exit_date is not None else None
        trade_rows.append({
            "Symbol": t.symbol,
            "Entry Date": t.entry_date.date() if hasattr(t.entry_date, "date") else t.entry_date,
            "Entry Price": round(entry_price, 2),
            "Exit Date": t.exit_date.date() if hasattr(t.exit_date, "date") else t.exit_date,
            "Exit Price": round(exit_price, 2),
            "Holding Days": holding_days,
            "P/L (%)": round(pnl_pct, 2),
            "P/L (₹)": round(pnl_inr, 2),
        })
    trades_df = pd.DataFrame(trade_rows)
    equity_df = pd.DataFrame(equity_curve).drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)

    metrics = compute_performance(trades_df, equity_df, initial_capital, start_date, end_date)

    status_parts = []
    if unexecutable_last_bar_signals > 0:
        status_parts.append(
            f"{unexecutable_last_bar_signals} signal(s) fell on the last available bar of their "
            "symbol's data and had no next bar to execute at, so were excluded."
        )
    if skipped_invalid_entry > 0:
        status_parts.append(
            f"{skipped_invalid_entry} trade(s) had an invalid (zero/NaN) entry price and were excluded "
            "from the trade log and performance metrics — check the underlying data for that symbol/date."
        )
    status = " ".join(status_parts) if status_parts else None

    return {"trades": trades_df, "equity_curve": equity_df, "metrics": metrics, "status": status}


def compute_performance(trades_df: pd.DataFrame, equity_df: pd.DataFrame, initial_capital: float,
                         start_date: date, end_date: date) -> dict:
    if trades_df.empty:
        final_capital = initial_capital
    else:
        final_capital = initial_capital + trades_df["P/L (₹)"].sum()

    total_trades = len(trades_df)
    wins = trades_df[trades_df["P/L (₹)"] > 0] if not trades_df.empty else trades_df
    losses = trades_df[trades_df["P/L (₹)"] <= 0] if not trades_df.empty else trades_df
    win_rate = (len(wins) / total_trades * 100.0) if total_trades else 0.0
    net_profit = final_capital - initial_capital

    gross_profit = wins["P/L (₹)"].sum() if not wins.empty else 0.0
    gross_loss = -losses["P/L (₹)"].sum() if not losses.empty else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (np.inf if gross_profit > 0 else 0.0)

    max_dd = 0.0
    if not equity_df.empty:
        eq = equity_df["equity"].to_numpy()
        running_max = np.maximum.accumulate(eq)
        dd = np.where(running_max > 0, (eq - running_max) / running_max * 100.0, 0.0)
        max_dd = float(dd.min())

    years = max((end_date - start_date).days / 365.25, 1e-6)
    cagr = ((final_capital / initial_capital) ** (1 / years) - 1) * 100.0 if initial_capital > 0 and final_capital > 0 else 0.0

    return {
        "Total Trades": total_trades,
        "Winning Trades": int(len(wins)),
        "Losing Trades": int(len(losses)),
        "Win Rate (%)": round(win_rate, 2),
        "Net Profit (₹)": round(net_profit, 2),
        "Final Capital (₹)": round(final_capital, 2),
        "Max Drawdown (%)": round(max_dd, 2),
        "Profit Factor": round(profit_factor, 2) if np.isfinite(profit_factor) else "∞",
        "CAGR (%)": round(cagr, 2),
    }


# ============================================================================
# SYMBOL LIST PARSING
# ============================================================================

def parse_symbol_upload(file) -> list[str]:
    name = file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    col = None
    for c in df.columns:
        if str(c).strip().lower() in ("symbol", "symbols", "ticker", "tickers"):
            col = c
            break
    if col is None:
        col = df.columns[0]
    syms = df[col].dropna().astype(str).str.strip().str.upper().tolist()
    return [s for s in syms if s]


def parse_symbol_text(text: str) -> list[str]:
    parts = [p.strip().upper() for chunk in text.split("\n") for p in chunk.split(",")]
    return [p for p in parts if p]


# ============================================================================
# STREAMLIT UI
# ============================================================================

def sidebar_inputs():
    st.sidebar.header("⚙️ Backtest Settings")

    c1, c2 = st.sidebar.columns(2)
    start_date = c1.date_input("Start Date", value=date.today() - timedelta(days=365))
    end_date = c2.date_input("End Date", value=date.today())

    initial_capital = st.sidebar.number_input("Initial Capital (₹)", min_value=1000.0, value=500000.0, step=10000.0)

    position_size_mode = st.sidebar.radio("Position Size Mode", ["₹ Fixed", "% of Equity"], horizontal=True)
    if position_size_mode == "₹ Fixed":
        position_size_value = st.sidebar.number_input("Position Size (₹)", min_value=1000.0, value=50000.0, step=5000.0)
    else:
        position_size_value = st.sidebar.number_input("Position Size (%)", min_value=1.0, max_value=100.0, value=10.0, step=1.0)

    brokerage_pct = st.sidebar.number_input("Brokerage (%) per side", min_value=0.0, value=0.03, step=0.01, format="%.3f")
    slippage_pct = st.sidebar.number_input("Slippage (%) per side", min_value=0.0, value=0.05, step=0.01, format="%.3f")
    max_open_positions = st.sidebar.number_input("Maximum Open Positions", min_value=1, value=10, step=1)

    with st.sidebar.expander("📐 Indicator Settings (originals — edit with care)"):
        swing_period = st.number_input("Swing Period", min_value=2, value=DEFAULTS["swing_period"], step=1)
        base_apt = st.number_input("Adaptive Price Tracking (APT)", min_value=1.0, value=DEFAULTS["base_apt"], step=1.0)
        use_adapt = st.checkbox("Adapt APT by ATR Ratio", value=DEFAULTS["use_adapt"])
        vol_bias = st.number_input("Volatility Bias", min_value=0.1, value=DEFAULTS["vol_bias"], step=0.1)
        atr_len = st.number_input("ATR Length", min_value=1, value=DEFAULTS["atr_len"], step=1)
        ema_length = st.number_input("EMA Channel Length", min_value=1, value=DEFAULTS["ema_length"], step=1)
        body_ratio = st.number_input("Body Ratio Threshold", min_value=0.0, max_value=1.0, value=DEFAULTS["body_ratio"], step=0.01)

    params = dict(
        swing_period=int(swing_period), base_apt=float(base_apt), use_adapt=bool(use_adapt),
        vol_bias=float(vol_bias), atr_len=int(atr_len), ema_length=int(ema_length),
        body_ratio=float(body_ratio),
    )

    st.sidebar.header("📋 Symbol List")
    input_mode = st.sidebar.radio("Input Method", ["Manual Entry", "Upload CSV/Excel"], horizontal=True)
    symbols: list[str] = []
    if input_mode == "Manual Entry":
        text = st.sidebar.text_area("Symbols (comma or newline separated)",
                                     value="RELIANCE\nTCS\nINFY\nHDFCBANK\nICICIBANK")
        symbols = parse_symbol_text(text)
    else:
        up = st.sidebar.file_uploader("Upload CSV/Excel with a 'Symbol' column", type=["csv", "xlsx", "xls"])
        if up is not None:
            try:
                symbols = parse_symbol_upload(up)
            except Exception as e:  # noqa: BLE001
                st.sidebar.error(f"Failed to parse file: {e}")

    return dict(
        start_date=start_date, end_date=end_date, initial_capital=initial_capital,
        position_size_mode=position_size_mode, position_size_value=position_size_value,
        brokerage_pct=brokerage_pct, slippage_pct=slippage_pct,
        max_open_positions=int(max_open_positions), params=params, symbols=symbols,
    )


def render_scanner_tab(cfg):
    st.subheader("📡 Scanner")
    st.caption("BUY = VWAP Uptrend + EMA Channel key buy K-line + rising volume. "
               "SELL = VWAP flips to Downtrend OR EMA Channel key sell K-line. "
               "Signal shown is for the most recent completed bar in range (executes next open).")
    if not cfg["symbols"]:
        st.info("Add symbols in the sidebar to scan.")
        return
    if st.button("🔍 Run Scan", type="primary"):
        with st.spinner(f"Scanning {len(cfg['symbols'])} symbols..."):
            result = run_scanner(cfg["symbols"], cfg["start_date"], cfg["end_date"], cfg["params"])
        st.session_state["scan_result"] = result
    if "scan_result" in st.session_state:
        df = st.session_state["scan_result"]
        if df.empty:
            st.warning("No results.")
        else:
            def highlight(row):
                if row["Signal"] == "BUY":
                    return ["background-color: #103d1f"] * len(row)
                if row["Signal"] == "SELL":
                    return ["background-color: #4a1414"] * len(row)
                return [""] * len(row)
            st.dataframe(df.style.apply(highlight, axis=1), use_container_width=True, hide_index=True)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, "scan_results.csv", "text/csv")


def render_backtest_tab(cfg):
    st.subheader("📊 Backtest")
    if not cfg["symbols"]:
        st.info("Add symbols in the sidebar to backtest.")
        return
    if cfg["end_date"] <= cfg["start_date"]:
        st.error("End Date must be after Start Date.")
        return

    if st.button("🚀 Run Backtest", type="primary"):
        with st.spinner(f"Backtesting {len(cfg['symbols'])} symbols..."):
            result = run_backtest(
                symbols=cfg["symbols"], start_date=cfg["start_date"], end_date=cfg["end_date"],
                params=cfg["params"], initial_capital=cfg["initial_capital"],
                position_size_mode=cfg["position_size_mode"], position_size_value=cfg["position_size_value"],
                brokerage_pct=cfg["brokerage_pct"], slippage_pct=cfg["slippage_pct"],
                max_open_positions=cfg["max_open_positions"],
            )
        st.session_state["bt_result"] = result

    if "bt_result" not in st.session_state:
        return

    result = st.session_state["bt_result"]
    metrics = result["metrics"]
    trades_df = result["trades"]
    equity_df = result["equity_curve"]
    status = result.get("status")

    if status:
        st.info(status)

    if trades_df.empty:
        st.warning("No executable trades for the selected symbols/date range.")
        # metrics still shown below (all zeros) for consistency
        if not metrics:
            return

    st.markdown("### 📈 Performance Report")
    cols = st.columns(4)
    keys = list(metrics.keys())
    for i, k in enumerate(keys):
        cols[i % 4].metric(k, metrics[k])

    if not equity_df.empty:
        st.markdown("### 💰 Equity Curve")
        chart_df = equity_df.set_index("date")[["equity"]]
        st.line_chart(chart_df)

    st.markdown("### 📒 Trade Log")
    if trades_df.empty:
        st.info("No closed trades.")
    else:
        st.dataframe(trades_df, use_container_width=True, hide_index=True)
        csv = trades_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Trade Log CSV", csv, "trade_log.csv", "text/csv")


def main():
    st.title("📈 NSE Scanner Pro — Dynamic Swing VWAP + EMA Channel Key K-Lines")
    st.caption(
        "Long-only strategy combining Zeiierman's Dynamic Swing Anchored VWAP (trend filter) "
        "with the EMA Channel Key K-Lines breakout trigger. Both indicators are ported bar-for-bar "
        "from their original Pine Script v6 source."
    )

    cfg = sidebar_inputs()
    tab1, tab2 = st.tabs(["📡 Scanner", "📊 Backtest"])
    with tab1:
        render_scanner_tab(cfg)
    with tab2:
        render_backtest_tab(cfg)


if __name__ == "__main__":
    main()
