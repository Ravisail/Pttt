"""
Signal + Backtesting Only — Single-file Streamlit app.

Strategy
--------
State machine per symbol (no look-ahead, no repainting):

    WAITING
       |  (Close < EMA10) AND (Close < EMA21) AND (RSI14 < 35)  on the same candle
       v
    TRACKING            <- tracking start date/price saved, NO buy signal here
       |  true EMA cross-up:  EMA10[t-1] <= EMA21[t-1]  AND  EMA10[t] > EMA21[t]
       v
    BUY SIGNAL          <- one signal only; entry at NEXT candle open; tracking disabled
       |
       v
    WAITING (restarts only after a fresh tracking condition)

The SAME signal engine feeds both the Scanner and the Backtester, so every
back-tested trade is traceable to a scanner signal.
"""

from __future__ import annotations

import io
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# --------------------------------------------------------------------------- #
# NIFTY 200 universe (static snapshot — edit freely, or use CSV/Excel upload). #
# Symbols are bare NSE tickers; ".NS" is appended automatically at fetch time. #
# --------------------------------------------------------------------------- #
NIFTY_200 = [
    "ABB", "ACC", "APLAPOLLO", "AUBANK", "ADANIENSOL", "ADANIENT", "ADANIGREEN",
    "ADANIPORTS", "ADANIPOWER", "ATGL", "AMBUJACEM", "APOLLOHOSP", "APOLLOTYRE",
    "ASHOKLEY", "ASIANPAINT", "ASTRAL", "AUROPHARMA", "DMART", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BAJAJHLDNG", "BALKRISIND",
    "BANDHANBNK", "BANKBARODA", "BANKINDIA", "BATAINDIA", "BERGEPAINT",
    "BEL", "BHARATFORG", "BHEL", "BPCL", "BHARTIARTL", "BIOCON", "BOSCHLTD",
    "BRITANNIA", "CGPOWER", "CANBK", "CHOLAFIN", "CIPLA", "COALINDIA",
    "COFORGE", "COLPAL", "CONCOR", "COROMANDEL", "CUMMINSIND", "DABUR",
    "DALBHARAT", "DEEPAKNTR", "DELHIVERY", "DIVISLAB", "DIXON", "DLF",
    "DRREDDY", "EICHERMOT", "ESCORTS", "EXIDEIND", "FEDERALBNK", "GAIL",
    "GLAND", "GLENMARK", "GMRINFRA", "GODREJCP", "GODREJPROP", "GRASIM",
    "GUJGASLTD", "HAVELLS", "HCLTECH", "HDFCAMC", "HDFCBANK", "HDFCLIFE",
    "HEROMOTOCO", "HINDALCO", "HINDPETRO", "HINDUNILVR", "HAL", "ICICIBANK",
    "ICICIGI", "ICICIPRULI", "IDBI", "IDFCFIRSTB", "IGL", "INDHOTEL",
    "INDIANB", "INDUSTOWER", "INDUSINDBK", "NAUKRI", "INFY", "INDIGO",
    "IPCALAB", "IRCTC", "IRFC", "ITC", "JINDALSTEL", "JIOFIN", "JSWENERGY",
    "JSWSTEEL", "JUBLFOOD", "KOTAKBANK", "KPITTECH", "LTF", "LTIM", "LT",
    "LAURUSLABS", "LICHSGFIN", "LICI", "LUPIN", "M&M", "M&MFIN", "MANKIND",
    "MARICO", "MARUTI", "MFSL", "MAXHEALTH", "MPHASIS", "MRF", "MUTHOOTFIN",
    "NMDC", "NTPC", "NESTLEIND", "OBEROIRLTY", "OFSS", "OIL", "ONGC",
    "PAGEIND", "PAYTM", "PERSISTENT", "PETRONET", "PIDILITIND", "PIIND",
    "PFC", "POLYCAB", "POONAWALLA", "POWERGRID", "PNB", "PRESTIGE", "RECLTD",
    "RELIANCE", "SBICARD", "SBILIFE", "SBIN", "SHREECEM", "SHRIRAMFIN",
    "SIEMENS", "SOLARINDS", "SONACOMS", "SRF", "SUNPHARMA", "SUNTV",
    "SUPREMEIND", "SYNGENE", "TATACHEM", "TATACOMM", "TATACONSUM",
    "TATAELXSI", "TATAMOTORS", "TATAPOWER", "TATASTEEL", "TATATECH", "TCS",
    "TECHM", "TITAN", "TORNTPHARM", "TORNTPOWER", "TRENT", "TIINDIA",
    "TVSMOTOR", "UBL", "ULTRACEMCO", "UNIONBANK", "UPL", "VBL", "VEDL",
    "VOLTAS", "WIPRO", "YESBANK", "ZOMATO", "ZYDUSLIFE",
]

# --------------------------------------------------------------------------- #
# Symbol / file helpers                                                        #
# --------------------------------------------------------------------------- #
def normalize_symbol(sym: str) -> str:
    """Uppercase, trim, and append '.NS' if no NSE/BSE suffix is present."""
    s = str(sym).strip().upper()
    if not s:
        return s
    if s.endswith(".NS") or s.endswith(".BO"):
        return s
    return s + ".NS"


def load_symbols_from_upload(uploaded) -> list[str]:
    """Read a CSV/Excel file that contains a 'Symbol' column."""
    name = uploaded.name.lower()
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded)
    else:
        df = pd.read_csv(uploaded)
    # Find the symbol column case-insensitively.
    col = next((c for c in df.columns if str(c).strip().lower() == "symbol"), None)
    if col is None:
        raise ValueError("Uploaded file must contain a 'Symbol' column.")
    syms = [normalize_symbol(x) for x in df[col].dropna().astype(str) if str(x).strip()]
    # De-duplicate, preserve order.
    return list(dict.fromkeys(syms))


# --------------------------------------------------------------------------- #
# Data fetch (cached)                                                          #
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_data(symbol: str, start: date, end: date) -> pd.DataFrame:
    """
    Download daily OHLC for `symbol`. A warm-up buffer is fetched BEFORE `start`
    so indicators are already valid at the first in-range candle.
    Returns an empty DataFrame on failure / no data.
    """
    warmup_start = start - timedelta(days=250)
    try:
        df = yf.download(
            symbol,
            start=warmup_start,
            end=end + timedelta(days=1),  # yfinance end is exclusive
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # yfinance can return MultiIndex columns for a single ticker — flatten.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    needed = ["Open", "High", "Low", "Close"]
    if not all(c in df.columns for c in needed):
        return pd.DataFrame()

    df = df[[c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]].copy()
    df = df.dropna(subset=needed)
    df.index = pd.to_datetime(df.index)
    return df


# --------------------------------------------------------------------------- #
# Indicators (vectorized)                                                      #
# --------------------------------------------------------------------------- #
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    out = out.where(avg_loss != 0, 100.0)  # all-gain window -> RSI 100
    return out


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Attach EMA10, EMA21, RSI14 and their 1-bar-lagged EMA values."""
    out = df.copy()
    out["EMA10"] = ema(out["Close"], 10)
    out["EMA21"] = ema(out["Close"], 21)
    out["RSI"] = rsi(out["Close"], 14)
    # Lagged EMAs computed on the FULL series so the first in-range candle
    # still has a valid "previous" value after slicing (no edge artifact).
    out["EMA10_prev"] = out["EMA10"].shift(1)
    out["EMA21_prev"] = out["EMA21"].shift(1)
    return out


# --------------------------------------------------------------------------- #
# Signal engine (event-driven state machine)                                   #
# --------------------------------------------------------------------------- #
def generate_signals(df: pd.DataFrame) -> tuple[list[dict], dict]:
    """
    Walk the candles once and emit buy signals via the tracking state machine.

    Returns
    -------
    signals : list of dicts, each fully self-describing:
        track_start_date, track_start_price, buy_index, buy_signal_date,
        entry_index, entry_date, entry_price  (entry_* is None if the buy
        signal is on the last candle — no "next open" exists yet).
    state   : snapshot of the machine at the LAST candle (for the Scanner).
    """
    n = len(df)
    dates = df.index
    close = df["Close"].to_numpy()
    open_ = df["Open"].to_numpy()
    ema10 = df["EMA10"].to_numpy()
    ema21 = df["EMA21"].to_numpy()
    ema10_p = df["EMA10_prev"].to_numpy()
    ema21_p = df["EMA21_prev"].to_numpy()
    rsi_ = df["RSI"].to_numpy()

    signals: list[dict] = []
    tracking = False
    t_start_idx = None
    t_start_date = None
    t_start_price = None

    for i in range(n):
        if not tracking:
            # --- Step 1: look for a tracking condition (no buy here) ---------
            if (
                not np.isnan(ema10[i])
                and not np.isnan(ema21[i])
                and not np.isnan(rsi_[i])
                and close[i] < ema10[i]
                and close[i] < ema21[i]
                and rsi_[i] < 35
            ):
                tracking = True
                t_start_idx = i
                t_start_date = dates[i]
                t_start_price = float(close[i])
        else:
            # --- Step 2: tracking active -> wait for a TRUE EMA cross-up -----
            if (
                not np.isnan(ema10_p[i])
                and not np.isnan(ema21_p[i])
                and ema10_p[i] <= ema21_p[i]
                and ema10[i] > ema21[i]
            ):
                entry_idx = i + 1 if (i + 1) < n else None
                signals.append(
                    {
                        "track_start_date": t_start_date,
                        "track_start_price": t_start_price,
                        "buy_index": i,
                        "buy_signal_date": dates[i],
                        "entry_index": entry_idx,
                        "entry_date": dates[entry_idx] if entry_idx is not None else None,
                        "entry_price": float(open_[entry_idx]) if entry_idx is not None else None,
                    }
                )
                # One signal per cycle -> back to WAITING.
                tracking = False
                t_start_idx = t_start_date = t_start_price = None

    state = {
        "is_tracking": tracking,
        "track_start_date": t_start_date,
        "tracking_days": (n - 1 - t_start_idx + 1) if (tracking and t_start_idx is not None) else 0,
        "last_date": dates[-1] if n else None,
        "last_close": float(close[-1]) if n else np.nan,
        "last_ema10": float(ema10[-1]) if n else np.nan,
        "last_ema21": float(ema21[-1]) if n else np.nan,
        "last_rsi": float(rsi_[-1]) if n else np.nan,
        "last_index": n - 1,
    }
    return signals, state


# --------------------------------------------------------------------------- #
# Backtest engine                                                              #
# --------------------------------------------------------------------------- #
def _simulate_exit(df: pd.DataFrame, entry_idx: int, entry_price: float, p: dict):
    """
    Determine how a single trade exits, starting from the entry candle.

    Priority on any candle:
      1) Same-day price exits (target / stop) via that candle's High / Low.
         Gaps are honoured (fill at open when the open is already beyond level).
         If both target and stop lie inside a candle, Stop Loss is taken
         (conservative — true intraday order is unknown).
      2) EMA10 crossing BELOW EMA21 is detected on the candle close and exits
         at the NEXT candle's open (never same candle -> no look-ahead).
    Falls back to the last candle's close ("End of Data") if nothing triggers.
    """
    n = len(df)
    open_ = df["Open"].to_numpy()
    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()
    close = df["Close"].to_numpy()
    ema10 = df["EMA10"].to_numpy()
    ema21 = df["EMA21"].to_numpy()
    ema10_p = df["EMA10_prev"].to_numpy()
    ema21_p = df["EMA21_prev"].to_numpy()
    dates = df.index

    target_price = entry_price * (1 + p["target_pct"] / 100.0) if p["use_target"] else None
    stop_price = entry_price * (1 - p["stop_pct"] / 100.0) if p["use_stop"] else None

    j = entry_idx
    while j < n:
        # ---- (1) price-based exits on this candle --------------------------
        if p["use_target"] or p["use_stop"]:
            hit_stop = p["use_stop"] and low[j] <= stop_price
            hit_target = p["use_target"] and high[j] >= target_price

            if hit_stop:  # stop takes priority when both are possible
                px = min(open_[j], stop_price) if open_[j] <= stop_price else stop_price
                return j, dates[j], float(px), "Stop Loss"
            if hit_target:
                px = max(open_[j], target_price) if open_[j] >= target_price else target_price
                return j, dates[j], float(px), "Target"

        # ---- (2) EMA cross-down -> exit at NEXT open -----------------------
        if p["use_ema"]:
            if (
                not np.isnan(ema10_p[j])
                and not np.isnan(ema21_p[j])
                and ema10_p[j] >= ema21_p[j]
                and ema10[j] < ema21[j]
            ):
                if j + 1 < n:
                    return j + 1, dates[j + 1], float(open_[j + 1]), "EMA Cross"
                return j, dates[j], float(close[j]), "EMA Cross (EOD)"

        j += 1

    last = n - 1
    return last, dates[last], float(close[last]), "End of Data"


def run_backtest(symbol: str, df: pd.DataFrame, signals: list[dict], p: dict) -> list[dict]:
    """One trade at a time. New signals during an open trade are ignored."""
    open_ = df["Open"].to_numpy()
    trades: list[dict] = []
    current_exit_index = -1  # last occupied candle; entries must come after it

    for s in signals:
        if s["entry_index"] is None:
            continue  # buy signal on last candle -> no next open to enter on
        if s["entry_index"] <= current_exit_index:
            continue  # already in a trade over this window -> ignore

        entry_idx = s["entry_index"]
        entry_price = float(open_[entry_idx])
        exit_idx, exit_date, exit_price, reason = _simulate_exit(df, entry_idx, entry_price, p)
        current_exit_index = exit_idx

        ret_pct = (exit_price / entry_price - 1.0) * 100.0
        holding_days = int((exit_date.normalize() - s["entry_date"].normalize()).days)

        trades.append(
            {
                "Symbol": symbol,
                "Tracking Start Date": s["track_start_date"].date(),
                "Buy Signal Date": s["buy_signal_date"].date(),
                "Entry Date": s["entry_date"].date(),
                "Entry Price": round(entry_price, 2),
                "Exit Date": exit_date.date(),
                "Exit Price": round(exit_price, 2),
                "Holding Days": holding_days,
                "Return %": round(ret_pct, 2),
                "Profit/Loss": round(exit_price - entry_price, 2),
                "Exit Reason": reason,
            }
        )
    return trades


# --------------------------------------------------------------------------- #
# Summary metrics                                                              #
# --------------------------------------------------------------------------- #
def compute_summary(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {}
    r = trades["Return %"].to_numpy(dtype=float)
    wins = r[r > 0]
    losses = r[r <= 0]

    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.inf

    # Additive equity curve on trade returns -> peak-to-trough drawdown.
    equity = np.cumsum(r)
    peak = np.maximum.accumulate(equity)
    max_dd = float((peak - equity).max()) if len(equity) else 0.0

    return {
        "Total Trades": int(len(r)),
        "Winning Trades": int(len(wins)),
        "Losing Trades": int(len(losses)),
        "Win Rate %": round(len(wins) / len(r) * 100, 2),
        "Average Win %": round(wins.mean(), 2) if len(wins) else 0.0,
        "Average Loss %": round(losses.mean(), 2) if len(losses) else 0.0,
        "Net Return % (sum)": round(r.sum(), 2),
        "Total Profit/Loss (per share)": round(trades["Profit/Loss"].sum(), 2),
        "Profit Factor": round(profit_factor, 2) if np.isfinite(profit_factor) else "∞",
        "Max Drawdown % (return pts)": round(max_dd, 2),
        "Average Holding Days": round(trades["Holding Days"].mean(), 1),
        "Largest Winner %": round(r.max(), 2),
        "Largest Loser %": round(r.min(), 2),
    }


# --------------------------------------------------------------------------- #
# Download helpers                                                             #
# --------------------------------------------------------------------------- #
def to_excel_bytes(df: pd.DataFrame, sheet: str = "Sheet1") -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Core runner shared by Scan + Backtest (single fetch, single signal engine)   #
# --------------------------------------------------------------------------- #
def process_symbol(symbol: str, start: date, end: date):
    """Fetch -> indicators -> slice to range -> signals. Returns (range_df, signals, state) or None."""
    raw = fetch_data(symbol, start, end)
    if raw.empty or len(raw) < 30:
        return None
    ind = add_indicators(raw)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    rng = ind.loc[(ind.index >= start_ts) & (ind.index <= end_ts)].copy()
    if len(rng) < 5:
        return None
    signals, state = generate_signals(rng)
    return rng, signals, state


# --------------------------------------------------------------------------- #
# Streamlit UI                                                                 #
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title="Signal + Backtest", layout="wide")
    st.title("📈 Signal Generation & Backtesting")
    st.caption(
        "EMA10 / EMA21 / RSI14 tracking → true EMA cross-up buy signal. "
        "Event-driven, no look-ahead. Scanner and backtest share one signal engine."
    )

    # ------------------------------- Sidebar -------------------------------- #
    with st.sidebar:
        st.header("Universe")
        source = st.radio("Symbol source", ["NIFTY 200", "Upload CSV / Excel"])

        symbols: list[str] = []
        if source == "NIFTY 200":
            symbols = [normalize_symbol(s) for s in NIFTY_200]
            st.caption(f"{len(symbols)} symbols (static snapshot).")
        else:
            up = st.file_uploader("File with a 'Symbol' column", type=["csv", "xlsx", "xls"])
            if up is not None:
                try:
                    symbols = load_symbols_from_upload(up)
                    st.success(f"Loaded {len(symbols)} symbols.")
                except Exception as e:
                    st.error(str(e))

        st.header("Date range")
        today = date.today()
        c1, c2 = st.columns(2)
        start = c1.date_input("Start", value=today - timedelta(days=365), max_value=today)
        end = c2.date_input("End", value=today, max_value=today)

        st.header("Exit rule")
        exit_choice = st.selectbox(
            "Exit strategy",
            [
                "1) Fixed Target % + Fixed Stop %",
                "2) EMA10 crosses below EMA21",
                "3) Target OR EMA Cross",
                "4) Target OR Stop Loss",
                "5) Target OR Stop OR EMA Cross",
            ],
        )
        target_pct = st.number_input("Target %", 0.1, 500.0, 10.0, 0.5)
        stop_pct = st.number_input("Stop Loss %", 0.1, 100.0, 5.0, 0.5)

        st.header("Run")
        scan_btn = st.button("🔍 Run Scan", use_container_width=True)
        bt_btn = st.button("⚙️ Run Backtest", use_container_width=True)

    # Map the exit choice to feature flags.
    exit_params = {
        "1) Fixed Target % + Fixed Stop %": dict(use_target=True, use_stop=True, use_ema=False),
        "2) EMA10 crosses below EMA21": dict(use_target=False, use_stop=False, use_ema=True),
        "3) Target OR EMA Cross": dict(use_target=True, use_stop=False, use_ema=True),
        "4) Target OR Stop Loss": dict(use_target=True, use_stop=True, use_ema=False),
        "5) Target OR Stop OR EMA Cross": dict(use_target=True, use_stop=True, use_ema=True),
    }[exit_choice]
    exit_params.update(target_pct=target_pct, stop_pct=stop_pct)

    # -------------------------- Guard clauses ------------------------------- #
    if (scan_btn or bt_btn) and not symbols:
        st.warning("No symbols selected. Choose NIFTY 200 or upload a file.")
        return
    if (scan_btn or bt_btn) and start >= end:
        st.warning("Start date must be before end date.")
        return

    # ------------------------------- Scan ----------------------------------- #
    if scan_btn:
        st.subheader("Scanner")
        tracking_rows, signal_rows = [], []
        prog = st.progress(0.0)
        status = st.empty()
        for k, sym in enumerate(symbols, 1):
            status.text(f"Scanning {sym} ({k}/{len(symbols)})")
            prog.progress(k / len(symbols))
            res = process_symbol(sym, start, end)
            if res is None:
                continue
            rng, signals, state = res

            # Currently tracking (state ended in TRACKING, no buy yet).
            if state["is_tracking"]:
                tracking_rows.append(
                    {
                        "Symbol": sym,
                        "Tracking Start Date": state["track_start_date"].date(),
                        "Tracking Days": state["tracking_days"],
                        "Close": round(state["last_close"], 2),
                        "EMA10": round(state["last_ema10"], 2),
                        "EMA21": round(state["last_ema21"], 2),
                        "RSI": round(state["last_rsi"], 2),
                    }
                )
            # Fresh buy signals on the LAST candle -> actionable next open.
            for s in signals:
                if s["buy_index"] == state["last_index"]:
                    signal_rows.append(
                        {
                            "Symbol": sym,
                            "Signal Date": s["buy_signal_date"].date(),
                            "Entry Date": "Next trading day",
                            "Entry Price": "Next open (pending)",
                        }
                    )
        status.text("Scan complete.")

        st.markdown("**Tracking stocks (awaiting EMA cross-up):**")
        if tracking_rows:
            st.dataframe(pd.DataFrame(tracking_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No stocks currently in the tracking state.")

        st.markdown("**Fresh buy signals (buy at next open):**")
        if signal_rows:
            st.dataframe(pd.DataFrame(signal_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No buy signals on the latest candle.")

    # ----------------------------- Backtest --------------------------------- #
    if bt_btn:
        st.subheader("Backtest")
        all_trades = []
        prog = st.progress(0.0)
        status = st.empty()
        for k, sym in enumerate(symbols, 1):
            status.text(f"Backtesting {sym} ({k}/{len(symbols)})")
            prog.progress(k / len(symbols))
            res = process_symbol(sym, start, end)
            if res is None:
                continue
            rng, signals, _ = res
            all_trades.extend(run_backtest(sym, rng, signals, exit_params))
        status.text("Backtest complete.")

        if not all_trades:
            st.info("No trades were generated for the selected symbols / range.")
            return

        trades = pd.DataFrame(all_trades).sort_values(["Entry Date", "Symbol"]).reset_index(drop=True)

        st.markdown("**Summary**")
        summary = compute_summary(trades)
        s_items = list(summary.items())
        cols = st.columns(4)
        for idx, (label, val) in enumerate(s_items):
            cols[idx % 4].metric(label, val)

        st.markdown("**Trade Log**")
        st.dataframe(trades, use_container_width=True, hide_index=True)

        d1, d2 = st.columns(2)
        d1.download_button(
            "⬇️ Download CSV",
            trades.to_csv(index=False).encode("utf-8"),
            "trade_log.csv",
            "text/csv",
            use_container_width=True,
        )
        d2.download_button(
            "⬇️ Download Excel",
            to_excel_bytes(trades, "Trades"),
            "trade_log.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
