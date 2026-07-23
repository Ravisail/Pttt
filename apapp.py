"""
NSE 62-EMA Channel Tracking Scanner & Backtester
=================================================
Single-file Streamlit application (yfinance-only) implementing a
Tracking -> Buy -> In-Trade state machine on the daily timeframe.

State machine (identical logic for scanner and backtester):
    WAITING -> TRACKING -> BUY_PENDING -> IN_TRADE -> WAITING

    WAITING:
        RSI(14) < 35 AND Close < EMA(High,62) AND Close < EMA(Low,62)
        => start TRACKING (record tracking date/price)

    TRACKING:
        Close > EMA(High,62) AND Close > EMA(Low,62) AND RSI rising
        (RSI[t] > RSI[t-1])
        => BUY signal recorded on this candle; execution deferred to
           the NEXT candle's Open (no look-ahead). State -> BUY_PENDING

    BUY_PENDING (only lasts until the next completed candle):
        Enter trade at that candle's Open.
        Stop Loss = Entry * (1 - sl_pct/100)
        Target    = Entry * (1 + tgt_pct/100)
        State -> IN_TRADE. Exits are checked starting the same candle
        (entry happens at the Open of that candle, so its High/Low are
        still valid information for that day).

    IN_TRADE:
        Exit when Low <= Stop Loss OR High >= Target (whichever the
        candle touches). If both are touched on the same candle, Stop
        Loss is assumed to execute first (conservative). On exit,
        state -> WAITING and a fresh tracking session may begin only
        from a later candle (never the exit candle itself).

Only one tracking session is active at a time per symbol, and only one
open trade per symbol at a time (no pyramiding / no averaging).
"""

import io
import time as time_module
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, time as dtime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

EMA_PERIOD = 62
RSI_PERIOD = 14
WARMUP_BARS = EMA_PERIOD * 4  # bars needed before indicators are considered stable

DEFAULT_WATCHLIST = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR",
    "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK",
    "BAJFINANCE", "ASIANPAINT", "MARUTI", "SUNPHARMA", "TITAN",
    "ULTRACEMCO", "WIPRO", "NESTLEIND", "HCLTECH", "ADANIENT",
    "TATASTEEL", "TATAMOTORS", "POWERGRID", "NTPC", "M&M",
    "BAJAJFINSV", "ONGC", "TECHM", "JSWSTEEL", "GRASIM", "CIPLA",
    "COALINDIA", "DRREDDY", "EICHERMOT", "HEROMOTOCO", "HINDALCO",
    "BRITANNIA", "DIVISLAB", "APOLLOHOSP", "BPCL", "SBILIFE",
    "TATACONSUM", "ADANIPORTS", "INDUSINDBK", "UPL", "SHREECEM",
    "HDFCLIFE", "BAJAJ-AUTO", "VEDL",
]

PERIOD_OPTIONS = {
    "1 Year": "1y",
    "2 Years": "2y",
    "3 Years": "3y",
    "5 Years": "5y",
    "10 Years": "10y",
    "Max": "max",
}

STATE_LABELS = {
    "WAITING": "Waiting",
    "TRACKING": "Tracking",
    "BUY_PENDING": "Buy Signal (Pending Open)",
    "IN_TRADE": "In Trade",
}


# --------------------------------------------------------------------------
# Time / market helpers
# --------------------------------------------------------------------------

def now_ist() -> datetime:
    """Current time in IST without requiring a timezone database."""
    return datetime.utcnow() + timedelta(hours=5, minutes=30)


def drop_incomplete_last_candle(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the last row if it corresponds to today's still-forming candle
    (i.e. today's date and before NSE close at 15:30 IST). Prevents
    repainting / look-ahead off a candle that hasn't finished forming."""
    if df.empty:
        return df
    ist = now_ist()
    last_date = df.index[-1].date()
    if last_date == ist.date() and ist.time() < dtime(15, 30):
        return df.iloc[:-1].copy()
    return df


# --------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------

def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.where(avg_loss != 0.0, 100.0)  # no losses at all -> RSI 100
    return rsi


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adds ema_high, ema_low, rsi columns. Expects Open/High/Low/Close."""
    out = df.copy()
    out["ema_high"] = out["High"].ewm(span=EMA_PERIOD, adjust=False).mean()
    out["ema_low"] = out["Low"].ewm(span=EMA_PERIOD, adjust=False).mean()
    out["rsi"] = compute_rsi(out["Close"], RSI_PERIOD)
    return out


# --------------------------------------------------------------------------
# Core state machine (shared by scanner + backtester -- single source of truth)
# --------------------------------------------------------------------------

def run_symbol_strategy(df: pd.DataFrame, sl_pct: float, tgt_pct: float, symbol: str):
    """
    Walks the dataframe candle-by-candle with no look-ahead.

    Returns:
        trades: list of dicts, one per COMPLETED trade (has an exit)
        current: dict describing the state as of the last processed candle
                 (used by the scanner to show live status, including an
                 open trade or a pending buy signal not yet filled)
    """
    n = len(df)
    trades = []

    current = {
        "Symbol": symbol,
        "state": "WAITING",
        "tracking_date": None,
        "tracking_price": None,
        "buy_signal_date": None,
        "entry_date": None,
        "entry_price": None,
        "stop_loss": None,
        "target": None,
    }

    if n < 2:
        return trades, current

    dates = df.index
    o = df["Open"].to_numpy()
    h = df["High"].to_numpy()
    l = df["Low"].to_numpy()
    c = df["Close"].to_numpy()
    ema_high = df["ema_high"].to_numpy()
    ema_low = df["ema_low"].to_numpy()
    rsi = df["rsi"].to_numpy()

    state = "WAITING"
    tracking_date = None
    tracking_price = None
    buy_signal_date = None
    entry_date = None
    entry_price = None
    stop_loss = None
    target = None

    for i in range(1, n):
        # Skip warm-up rows where indicators aren't valid yet.
        if np.isnan(ema_high[i]) or np.isnan(ema_low[i]) or np.isnan(rsi[i]) or np.isnan(rsi[i - 1]):
            continue

        date = dates[i]

        if state == "BUY_PENDING":
            # Execute at this candle's Open (the candle AFTER the signal candle)
            entry_price = float(o[i])
            entry_date = date
            stop_loss = entry_price * (1.0 - sl_pct / 100.0)
            target = entry_price * (1.0 + tgt_pct / 100.0)
            state = "IN_TRADE"
            # Same-candle exit check is valid since entry happened at this
            # candle's Open; its High/Low are legitimate same-day data.
            hit_sl = l[i] <= stop_loss
            hit_tgt = h[i] >= target
            if hit_sl or hit_tgt:
                exit_reason = "Stop Loss" if hit_sl else "Target"
                exit_price = stop_loss if hit_sl else target
                trades.append(_build_trade(
                    symbol, entry_date, entry_price, stop_loss, target,
                    date, exit_price, exit_reason
                ))
                state = "WAITING"
                entry_date = entry_price = stop_loss = target = None
            continue

        if state == "IN_TRADE":
            hit_sl = l[i] <= stop_loss
            hit_tgt = h[i] >= target
            if hit_sl or hit_tgt:
                exit_reason = "Stop Loss" if hit_sl else "Target"  # SL wins ties
                exit_price = stop_loss if hit_sl else target
                trades.append(_build_trade(
                    symbol, entry_date, entry_price, stop_loss, target,
                    date, exit_price, exit_reason
                ))
                state = "WAITING"
                entry_date = entry_price = stop_loss = target = None
            continue

        if state == "WAITING":
            if rsi[i] < 35.0 and c[i] < ema_high[i] and c[i] < ema_low[i]:
                state = "TRACKING"
                tracking_date = date
                tracking_price = float(c[i])
            continue

        if state == "TRACKING":
            rsi_rising = rsi[i] > rsi[i - 1]
            if c[i] > ema_high[i] and c[i] > ema_low[i] and rsi_rising:
                state = "BUY_PENDING"
                buy_signal_date = date
            continue

    current.update({
        "state": state,
        "tracking_date": tracking_date,
        "tracking_price": tracking_price,
        "buy_signal_date": buy_signal_date,
        "entry_date": entry_date,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "target": target,
    })
    return trades, current


def _build_trade(symbol, entry_date, entry_price, stop_loss, target,
                  exit_date, exit_price, exit_reason):
    holding_days = (exit_date - entry_date).days
    pnl = exit_price - entry_price
    pnl_pct = (pnl / entry_price) * 100.0
    return {
        "Symbol": symbol,
        "Entry Date": entry_date,
        "Entry Price": round(entry_price, 2),
        "Stop Loss": round(stop_loss, 2),
        "Target": round(target, 2),
        "Exit Date": exit_date,
        "Exit Price": round(exit_price, 2),
        "Exit Reason": exit_reason,
        "Holding Days": holding_days,
        "Profit/Loss (Rs)": round(pnl, 2),
        "Profit/Loss (%)": round(pnl_pct, 2),
    }



# --------------------------------------------------------------------------
# Symbol handling
# --------------------------------------------------------------------------

def normalize_symbol(sym: str, suffix: str) -> str:
    sym = str(sym).strip().upper()
    if not sym:
        return sym
    if sym.endswith(".NS") or sym.endswith(".BO"):
        return sym
    return f"{sym}{suffix}"


def parse_symbol_file(uploaded_file, suffix: str):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        raw = pd.read_csv(uploaded_file)
    else:
        raw = pd.read_excel(uploaded_file)
    col = None
    for candidate in ["Symbol", "SYMBOL", "symbol", "Ticker", "TICKER"]:
        if candidate in raw.columns:
            col = candidate
            break
    if col is None:
        col = raw.columns[0]
    symbols = raw[col].dropna().astype(str).tolist()
    return [normalize_symbol(s, suffix) for s in symbols if s.strip()]


# --------------------------------------------------------------------------
# Data fetching (cached + concurrent)
# --------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_one(symbol: str, period: str) -> pd.DataFrame:
    try:
        df = yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=True)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(
            subset=["Open", "High", "Low", "Close"]
        )
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()


def fetch_many(symbols, period: str, progress_label: str, max_workers: int = 10):
    results = {}
    errors = []
    progress = st.progress(0.0, text=progress_label)
    status = st.empty()
    total = len(symbols)
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_one, sym, period): sym for sym in symbols}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                df = fut.result()
                if df.empty:
                    errors.append(sym)
                else:
                    results[sym] = df
            except Exception:
                errors.append(sym)
            done += 1
            progress.progress(done / total, text=f"{progress_label} ({done}/{total})")
            status.caption(f"Fetched {len(results)} / {done} processed, {len(errors)} failed so far")
    progress.empty()
    status.empty()
    return results, errors


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def compute_metrics(trades_df: pd.DataFrame, capital_per_trade: float):
    if trades_df.empty:
        return {}, trades_df

    df = trades_df.copy()
    df["Quantity"] = (capital_per_trade / df["Entry Price"]).apply(np.floor).clip(lower=1)
    df["Position P/L (Rs)"] = df["Quantity"] * (df["Exit Price"] - df["Entry Price"])

    total_trades = len(df)
    wins = df[df["Profit/Loss (%)"] > 0]
    losses = df[df["Profit/Loss (%)"] <= 0]
    win_rate = (len(wins) / total_trades * 100) if total_trades else 0.0

    net_profit = df["Position P/L (Rs)"].sum()
    capital_deployed = capital_per_trade * total_trades
    total_return_pct = (net_profit / capital_deployed * 100) if capital_deployed else 0.0

    avg_profit_pct = wins["Profit/Loss (%)"].mean() if len(wins) else 0.0
    avg_loss_pct = losses["Profit/Loss (%)"].mean() if len(losses) else 0.0

    gross_profit = wins["Position P/L (Rs)"].sum() if len(wins) else 0.0
    gross_loss = abs(losses["Position P/L (Rs)"].sum()) if len(losses) else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.inf

    expectancy = net_profit / total_trades if total_trades else 0.0
    avg_holding_days = df["Holding Days"].mean()

    seq = df.sort_values("Exit Date")
    equity = capital_per_trade + seq["Position P/L (Rs)"].cumsum()
    running_peak = equity.cummax()
    drawdown_pct = ((equity - running_peak) / running_peak * 100)
    max_drawdown_pct = drawdown_pct.min() if len(drawdown_pct) else 0.0

    largest_winner = df["Position P/L (Rs)"].max()
    largest_loser = df["Position P/L (Rs)"].min()

    target_hits = int((df["Exit Reason"] == "Target").sum())
    sl_hits = int((df["Exit Reason"] == "Stop Loss").sum())

    metrics = {
        "Total Trades": total_trades,
        "Winning Trades": len(wins),
        "Losing Trades": len(losses),
        "Win Rate (%)": round(win_rate, 2),
        "Net Profit (Rs)": round(net_profit, 2),
        "Total Return (%)": round(total_return_pct, 2),
        "Average Profit (%)": round(avg_profit_pct, 2),
        "Average Loss (%)": round(avg_loss_pct, 2),
        "Profit Factor": round(profit_factor, 2) if np.isfinite(profit_factor) else float("inf"),
        "Expectancy (Rs)": round(expectancy, 2),
        "Average Holding Days": round(avg_holding_days, 1),
        "Maximum Drawdown (%)": round(max_drawdown_pct, 2),
        "Largest Winner (Rs)": round(largest_winner, 2),
        "Largest Loser (Rs)": round(largest_loser, 2),
        "Target Hits": target_hits,
        "Stop Loss Hits": sl_hits,
    }
    return metrics, df



# --------------------------------------------------------------------------
# Export helpers
# --------------------------------------------------------------------------

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return buf.getvalue()


# --------------------------------------------------------------------------
# Scanner
# --------------------------------------------------------------------------

def build_scanner_row(symbol: str, df: pd.DataFrame, current: dict) -> dict:
    last = df.iloc[-1]
    state = current["state"]
    rsi_trend = "N/A"
    if len(df) >= 2 and not np.isnan(df["rsi"].iloc[-1]) and not np.isnan(df["rsi"].iloc[-2]):
        rsi_trend = "Rising" if df["rsi"].iloc[-1] > df["rsi"].iloc[-2] else "Falling"

    row = {
        "Symbol": symbol.replace(".NS", "").replace(".BO", ""),
        "Current Close": round(float(last["Close"]), 2),
        "EMA High (62)": round(float(last["ema_high"]), 2) if not np.isnan(last["ema_high"]) else None,
        "EMA Low (62)": round(float(last["ema_low"]), 2) if not np.isnan(last["ema_low"]) else None,
        "RSI": round(float(last["rsi"]), 2) if not np.isnan(last["rsi"]) else None,
        "RSI Trend": rsi_trend,
        "Tracking Status": STATE_LABELS.get(state, state),
        "Tracking Date": current["tracking_date"].date() if current["tracking_date"] is not None else None,
        "Buy Signal Date": current["buy_signal_date"].date() if current["buy_signal_date"] is not None else None,
        "Entry Price": round(current["entry_price"], 2) if current["entry_price"] is not None else None,
        "Stop Loss": round(current["stop_loss"], 2) if current["stop_loss"] is not None else None,
        "Target": round(current["target"], 2) if current["target"] is not None else None,
    }
    return row


def run_scanner(symbols, period, sl_pct, tgt_pct):
    data, errors = fetch_many(symbols, period, "Downloading price data")
    rows = []
    prog = st.progress(0.0, text="Running strategy")
    total = len(data)
    for i, (symbol, raw_df) in enumerate(data.items(), start=1):
        clean_df = drop_incomplete_last_candle(raw_df)
        if len(clean_df) < WARMUP_BARS:
            prog.progress(i / total, text="Running strategy")
            continue
        ind_df = compute_indicators(clean_df)
        _, current = run_symbol_strategy(ind_df, sl_pct, tgt_pct, symbol)
        rows.append(build_scanner_row(symbol, ind_df, current))
        prog.progress(i / total, text="Running strategy")
    prog.empty()
    return pd.DataFrame(rows), errors


# --------------------------------------------------------------------------
# Backtester
# --------------------------------------------------------------------------

def run_backtest(symbols, period, sl_pct, tgt_pct):
    data, errors = fetch_many(symbols, period, "Downloading price history")
    all_trades = []
    prog = st.progress(0.0, text="Running backtest")
    total = len(data)
    for i, (symbol, raw_df) in enumerate(data.items(), start=1):
        clean_df = drop_incomplete_last_candle(raw_df)
        if len(clean_df) < WARMUP_BARS:
            prog.progress(i / total, text="Running backtest")
            continue
        ind_df = compute_indicators(clean_df)
        trades, _ = run_symbol_strategy(ind_df, sl_pct, tgt_pct, symbol.replace(".NS", "").replace(".BO", ""))
        all_trades.extend(trades)
        prog.progress(i / total, text="Running backtest")
    prog.empty()
    trades_df = pd.DataFrame(all_trades)
    return trades_df, errors


# --------------------------------------------------------------------------
# Streamlit App
# --------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="62-EMA Channel Scanner & Backtester", layout="wide")
    st.title("NSE 62-EMA Channel Tracking Scanner & Backtester")
    st.caption(
        "Tracking (RSI<35 + close below channel) -> Buy (breakout above channel + rising RSI, "
        "filled next open) -> Exit on Stop Loss or Target. Identical logic drives both the "
        "live scanner and the historical backtester."
    )

    with st.sidebar:
        st.header("Symbols")
        exchange = st.selectbox("Exchange suffix", ["NSE (.NS)", "BSE (.BO)"], index=0)
        suffix = ".NS" if exchange.startswith("NSE") else ".BO"

        source = st.radio("Symbol source", ["Default Watchlist", "Upload CSV/Excel", "Manual Entry"])
        if source == "Upload CSV/Excel":
            uploaded = st.file_uploader("Upload file with a 'Symbol' column", type=["csv", "xlsx", "xls"])
            symbols = parse_symbol_file(uploaded, suffix) if uploaded is not None else []
        elif source == "Manual Entry":
            manual = st.text_area("One symbol per line (or comma-separated)", "RELIANCE\nTCS\nINFY")
            raw_syms = [s for chunk in manual.split("\n") for s in chunk.split(",")]
            symbols = [normalize_symbol(s, suffix) for s in raw_syms if s.strip()]
        else:
            symbols = [normalize_symbol(s, suffix) for s in DEFAULT_WATCHLIST]

        st.caption(f"{len(symbols)} symbol(s) loaded")

        st.header("Strategy Parameters")
        sl_pct = st.number_input("Stop Loss (%)", min_value=0.1, max_value=50.0, value=3.0, step=0.5)
        tgt_pct = st.number_input("Target (%)", min_value=0.1, max_value=100.0, value=9.0, step=0.5)
        capital_per_trade = st.number_input(
            "Capital per Trade (Rs) - for P&L sizing", min_value=1000, value=100000, step=5000
        )

        st.header("Data Period")
        period_label = st.selectbox("History length", list(PERIOD_OPTIONS.keys()), index=2)
        period = PERIOD_OPTIONS[period_label]
        st.caption(f"Warm-up needs >= {WARMUP_BARS} trading days before indicators stabilize.")

    tab_scan, tab_backtest = st.tabs(["Scanner", "Backtester"])

    with tab_scan:
        st.subheader("Live Scanner")
        st.write(
            "Shows each symbol's current indicator values and state "
            "(Waiting / Tracking / Buy Signal Pending / In Trade) as of the last completed candle."
        )
        if st.button("Run Scan", type="primary", key="run_scan"):
            if not symbols:
                st.warning("No symbols loaded. Choose a symbol source in the sidebar.")
            else:
                result_df, errors = run_scanner(symbols, period, sl_pct, tgt_pct)
                st.session_state["scanner_result"] = result_df
                st.session_state["scanner_errors"] = errors

        if "scanner_result" in st.session_state:
            result_df = st.session_state["scanner_result"]
            errors = st.session_state.get("scanner_errors", [])
            if errors:
                st.info(f"{len(errors)} symbol(s) could not be fetched or had insufficient history: "
                        + ", ".join(e.replace('.NS', '').replace('.BO', '') for e in errors[:20])
                        + (" ..." if len(errors) > 20 else ""))

            if result_df.empty:
                st.warning("No results to display.")
            else:
                order = ["In Trade", "Buy Signal (Pending Open)", "Tracking", "Waiting"]
                result_df["_sort"] = result_df["Tracking Status"].apply(
                    lambda s: order.index(s) if s in order else len(order)
                )
                result_df = result_df.sort_values(["_sort", "Symbol"]).drop(columns="_sort")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Scanned", len(result_df))
                c2.metric("In Trade", int((result_df["Tracking Status"] == "In Trade").sum()))
                c3.metric("Buy Pending", int((result_df["Tracking Status"] == "Buy Signal (Pending Open)").sum()))
                c4.metric("Tracking", int((result_df["Tracking Status"] == "Tracking").sum()))

                st.dataframe(result_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download Scanner Results (CSV)",
                    data=to_csv_bytes(result_df),
                    file_name=f"scanner_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                )

    with tab_backtest:
        st.subheader("Backtester")
        st.write("Runs the identical strategy logic across full history and reports every completed trade.")
        if st.button("Run Backtest", type="primary", key="run_backtest"):
            if not symbols:
                st.warning("No symbols loaded. Choose a symbol source in the sidebar.")
            else:
                trades_df, errors = run_backtest(symbols, period, sl_pct, tgt_pct)
                st.session_state["backtest_trades"] = trades_df
                st.session_state["backtest_errors"] = errors

        if "backtest_trades" in st.session_state:
            trades_df = st.session_state["backtest_trades"]
            errors = st.session_state.get("backtest_errors", [])
            if errors:
                st.info(f"{len(errors)} symbol(s) could not be fetched or had insufficient history.")

            if trades_df.empty:
                st.warning("No completed trades found for the selected symbols/period.")
            else:
                metrics, sized_trades = compute_metrics(trades_df, capital_per_trade)

                st.markdown("#### Performance Metrics")
                mcols = st.columns(4)
                metric_items = list(metrics.items())
                for idx, (k, v) in enumerate(metric_items):
                    mcols[idx % 4].metric(k, v)

                st.markdown("#### Trade Log")
                display_trades = sized_trades.copy()
                display_trades["Entry Date"] = pd.to_datetime(display_trades["Entry Date"]).dt.date
                display_trades["Exit Date"] = pd.to_datetime(display_trades["Exit Date"]).dt.date
                display_trades = display_trades.sort_values("Exit Date")
                st.dataframe(display_trades, use_container_width=True, hide_index=True)

                dl1, dl2 = st.columns(2)
                dl1.download_button(
                    "Download Trade Log (CSV)",
                    data=to_csv_bytes(display_trades),
                    file_name=f"trade_log_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                )
                dl2.download_button(
                    "Download Trade Log (Excel)",
                    data=to_excel_bytes(display_trades, "Trade Log"),
                    file_name=f"trade_log_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )


if __name__ == "__main__":
    main()
