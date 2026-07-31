"""
Orderly Pullback Scanner & Backtester — NSE/BSE
Single-file Streamlit app (no charts — tables/data only)

Strategy:
  Stage-2 uptrend (Close>EMA200, EMA10>EMA20>EMA200, EMA200 rising)
  -> recent swing high (lookback window)
  -> 12-20% pullback over 5-15 days
  -> orderly-pullback quality checks (no panic candles / gap-downs / long bearish runs)
  -> price near EMA10/20 support
  -> volume contraction (5d avg < 20d avg)
  -> Mother Candle -> entry on break of Mother Candle High (or next open/close)
  -> stop loss / target per user selection

Run:  streamlit run app.py
Deps: streamlit pandas numpy yfinance openpyxl
"""

import io
import math
import concurrent.futures as cf

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Orderly Pullback Scanner & Backtester", layout="wide")

# ----------------------------------------------------------------------------
# Static NIFTY 200 fallback universe (representative list; upload file to override)
# ----------------------------------------------------------------------------
NIFTY200 = [
    "RELIANCE","TCS","HDFCBANK","ICICIBANK","INFY","HINDUNILVR","ITC","SBIN","BHARTIARTL","KOTAKBANK",
    "LT","AXISBANK","ASIANPAINT","MARUTI","SUNPHARMA","TITAN","BAJFINANCE","NESTLEIND","ULTRACEMCO","WIPRO",
    "ONGC","NTPC","POWERGRID","M&M","TATAMOTORS","TATASTEEL","ADANIENT","ADANIPORTS","COALINDIA","HCLTECH",
    "BAJAJFINSV","GRASIM","TECHM","INDUSINDBK","HINDALCO","JSWSTEEL","DRREDDY","CIPLA","EICHERMOT","BRITANNIA",
    "APOLLOHOSP","DIVISLAB","TATACONSUM","BPCL","HEROMOTOCO","BAJAJ-AUTO","SHRIRAMFIN","LTIM","SBILIFE","HDFCLIFE",
    "PIDILITIND","SIEMENS","DLF","VEDL","AMBUJACEM","GODREJCP","DABUR","HAVELLS","ABB","BANKBARODA",
    "PNB","CANBK","UNIONBANK","IOC","GAIL","HINDPETRO","TATAPOWER","TORNTPHARM","LUPIN","AUROPHARMA",
    "ALKEM","ZYDUSLIFE","BIOCON","GLENMARK","IPCALAB","MANKIND","MAXHEALTH","FORTIS","LALPATHLAB","SYNGENE",
    "PERSISTENT","COFORGE","MPHASIS","OFSS","TATAELXSI","KPITTECH","CYIENT","SONACOMS","BOSCHLTD","MOTHERSON",
    "TVSMOTOR","ASHOKLEY","BHARATFORG","BALKRISIND","MRF","APOLLOTYRE","ESCORTS","EXIDEIND","CUMMINSIND","THERMAX",
    "BHEL","BEL","HAL","BDL","MAZDOCK","COCHINSHIP","RVNL","IRFC","IRCTC","CONCOR",
    "INDIGO","PAYTM","ZOMATO","NYKAA","POLICYBZR","DMART","TRENT","ABFRL","PAGEIND","VBL",
    "UBL","MCDOWELL-N","COLPAL","MARICO","EMAMILTD","JUBLFOOD","DEVYANI","INDHOTEL","CHALET","OBEROIRLTY",
    "GODREJPROP","PRESTIGE","PHOENIXLTD","BRIGADE","LODHA","NCC","KEC","KALPATPOWER","GMRINFRA","IRB",
    "ACC","SHREECEM","JKCEMENT","RAMCOCEM","DALBHARAT","STARCEMENT","SAIL","NMDC","MOIL","JINDALSTEL",
    "NATIONALUM","HINDZINC","RECLTD","PFC","IREDA","SJVN","NHPC","ADANIGREEN","ADANIPOWER","JSWENERGY",
    "TORNTPOWER","CESC","IEX","CDSL","BSE","MCX","ANGELONE","MOTILALOFS","IIFL","CHOLAFIN",
    "BAJAJHLDNG","MUTHOOTFIN","MANAPPURAM","LICHSGFIN","CANFINHOME","PNBHOUSING","AAVAS","APTUS","HOMEFIRST","FEDERALBNK",
    "IDFCFIRSTB","RBLBANK","BANDHANBNK","AUBANK","CUB","KARURVYSYA","JKBANK","YESBANK","IDBI","INDIANB",
    "LICI","GICRE","NIACL","STARHEALTH","ICICIGI","ICICIPRULI","SBICARD","POLYCAB","KEI","FINOLEXIND",
]

# ----------------------------------------------------------------------------
# Data fetch (cached, multi-threaded)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_one(symbol: str, period: str, suffix: str) -> pd.DataFrame | None:
    try:
        df = yf.Ticker(symbol + suffix).history(period=period, auto_adjust=True)
        if df is None or len(df) < 220:
            return None
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception:
        return None


def fetch_all(symbols, period, suffix, progress=None):
    out = {}
    with cf.ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(fetch_one, s, period, suffix): s for s in symbols}
        done = 0
        for fut in cf.as_completed(futs):
            s = futs[fut]
            df = fut.result()
            if df is not None:
                out[s] = df
            done += 1
            if progress:
                progress.progress(done / len(symbols), text=f"Downloaded {done}/{len(symbols)}")
    return out


def add_emas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EMA10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
    df["Vol5"] = df["Volume"].rolling(5).mean()
    df["Vol20"] = df["Volume"].rolling(20).mean()
    return df


# ----------------------------------------------------------------------------
# Setup detection at bar index i (i = last completed bar of the setup / mother candle)
# Returns dict with setup info or None.
# ----------------------------------------------------------------------------
def detect_setup(df, i, p):
    if i < 210:
        return None
    c = df["Close"].iloc[i]
    e10, e20, e200 = df["EMA10"].iloc[i], df["EMA20"].iloc[i], df["EMA200"].iloc[i]

    # 1) Stage-2 trend
    if not (c > e200 and e10 > e20 > e200):
        return None
    if not (e200 > df["EMA200"].iloc[i - p["ema200_rising_days"]]):
        return None

    # 2) Recent swing high within lookback
    lb = df["High"].iloc[max(0, i - p["swing_lookback"]): i + 1]
    swing_high = lb.max()
    swing_idx = lb.idxmax()
    swing_pos = df.index.get_loc(swing_idx)

    # 3) Pullback depth & duration
    low_since = df["Low"].iloc[swing_pos: i + 1].min()
    depth = (swing_high - low_since) / swing_high * 100.0
    dur = i - swing_pos
    if not (p["pb_min"] <= depth <= p["pb_max"]):
        return None
    if not (p["pb_days_min"] <= dur <= p["pb_days_max"]):
        return None

    # 4) Orderly quality checks over the pullback window
    win = df.iloc[swing_pos: i + 1]
    bearish = (win["Close"] < win["Open"]).astype(int)
    # max consecutive bearish candles
    max_run, run = 0, 0
    for b in bearish:
        run = run + 1 if b else 0
        max_run = max(max_run, run)
    if max_run > p["max_consec_bear"]:
        return None
    # large bearish candles (range > large_candle_pct of price)
    rng_pct = (win["High"] - win["Low"]) / win["Close"] * 100.0
    large_bear = ((win["Close"] < win["Open"]) & (rng_pct > p["large_candle_pct"])).sum()
    if large_bear > p["max_large_bear"]:
        return None
    # gap-downs
    prev_low = win["Low"].shift(1)
    gaps = (win["Open"] < prev_low * (1 - p["gap_down_pct"] / 100.0)).sum()
    if gaps > 0:
        return None

    # 5) Near EMA10/20 support
    prox = p["ema_prox_pct"] / 100.0
    near10 = abs(c - e10) / e10 <= prox
    near20 = abs(c - e20) / e20 <= prox
    above = c >= min(e10, e20) * (1 - prox)
    if not ((near10 or near20) and above):
        return None

    # 6) Volume contraction
    if not (df["Vol5"].iloc[i] < df["Vol20"].iloc[i]):
        return None

    # 7) Mother candle = current bar
    return {
        "mc_high": df["High"].iloc[i],
        "mc_low": df["Low"].iloc[i],
        "swing_high": swing_high,
        "swing_low": low_since,
        "pullback_pct": depth,
        "pullback_days": dur,
        "close": c,
        "ema10": e10,
        "ema20": e20,
        "ema200": e200,
    }


# ----------------------------------------------------------------------------
# Backtest a single symbol; returns list of trade dicts
# ----------------------------------------------------------------------------
def backtest_symbol(sym, df, p):
    trades = []
    i = 210
    n = len(df)
    while i < n - 2:
        setup = detect_setup(df, i, p)
        if setup is None:
            i += 1
            continue

        # ---- Entry determination ----
        entry_idx = None
        entry_raw = None
        if p["entry_mode"] == "Break above Mother Candle High":
            # look ahead up to entry_wait days for a break
            for j in range(i + 1, min(i + 1 + p["entry_wait"], n)):
                if df["High"].iloc[j] > setup["mc_high"]:
                    entry_idx = j
                    entry_raw = max(setup["mc_high"], df["Open"].iloc[j])
                    break
        elif p["entry_mode"] == "Next-day Open":
            entry_idx = i + 1
            entry_raw = df["Open"].iloc[entry_idx]
        else:  # Next-day Close
            entry_idx = i + 1
            entry_raw = df["Close"].iloc[entry_idx]

        if entry_idx is None:
            i += 1
            continue

        entry = entry_raw * (1 + p["slippage_pct"] / 100.0)  # slippage-adjusted stored entry

        # ---- Stop loss ----
        if p["sl_mode"] == "Previous-day Low":
            sl = df["Low"].iloc[entry_idx - 1]
        elif p["sl_mode"] == "Mother Candle Low":
            sl = setup["mc_low"]
        elif p["sl_mode"] == "Swing Low":
            sl = setup["swing_low"]
        else:
            sl = entry * (1 - p["sl_fixed_pct"] / 100.0)
        if sl >= entry:
            i += 1
            continue
        risk = entry - sl

        # ---- Target ----
        use_trail = p["target_mode"] == "Trailing Stop"
        if p["target_mode"] == "R:R Multiple":
            target = entry + risk * p["rr_mult"]
        elif p["target_mode"] == "Fixed %":
            target = entry * (1 + p["target_fixed_pct"] / 100.0)
        else:
            target = None

        trail = sl
        exit_idx, exit_price, reason = None, None, None
        for j in range(entry_idx, min(entry_idx + p["max_hold"], n)):
            h, l, cl = df["High"].iloc[j], df["Low"].iloc[j], df["Close"].iloc[j]
            stop_now = trail if use_trail else sl
            # Same-day ambiguity: stop checked first (conservative)
            if l <= stop_now:
                exit_idx, exit_price = j, stop_now
                reason = "Trailing Stop" if (use_trail and stop_now > sl) else "Stop Loss"
                break
            if (target is not None) and h >= target:
                exit_idx, exit_price, reason = j, target, "Target Hit"
                break
            if use_trail:
                trail = max(trail, cl * (1 - p["trail_pct"] / 100.0))
        if exit_idx is None:
            exit_idx = min(entry_idx + p["max_hold"], n) - 1
            exit_price = df["Close"].iloc[exit_idx]
            reason = "Max Holding Period"

        exit_adj = exit_price * (1 - p["slippage_pct"] / 100.0)
        trades.append({
            "Symbol": sym,
            "Setup Date": df.index[i].date(),
            "Entry Date": df.index[entry_idx].date(),
            "Entry": round(entry, 2),
            "Stop": round(sl, 2),
            "Target": round(target, 2) if target else np.nan,
            "Exit Date": df.index[exit_idx].date(),
            "Exit": round(exit_adj, 2),
            "Exit Reason": reason,
            "Holding (trading days)": exit_idx - entry_idx,  # trading days, not calendar
            "Pullback %": round(setup["pullback_pct"], 1),
        })
        i = exit_idx + 1
    return trades


# ----------------------------------------------------------------------------
# Portfolio simulation from raw trade list
# ----------------------------------------------------------------------------
def simulate_portfolio(all_trades, p):
    if not all_trades:
        return pd.DataFrame(), {}, 0
    tl = pd.DataFrame(all_trades).sort_values("Entry Date").reset_index(drop=True)
    capital = p["capital"]
    cash = capital
    open_pos = []  # list of (exit_date, alloc)
    rows, skipped = [], 0
    equity_points = []

    for _, t in tl.iterrows():
        # release closed positions
        open_pos = [op for op in open_pos if op[0] > t["Entry Date"]] if open_pos else []
        alloc = min(p["pos_size"], cash) if p["pos_mode"] == "Fixed ₹" else cash * p["pos_pct"] / 100.0
        if len(open_pos) >= p["max_positions"] or alloc < t["Entry"]:
            skipped += 1
            continue
        qty = int(alloc // t["Entry"])  # whole shares
        if qty <= 0:
            skipped += 1
            continue
        buy_val = qty * t["Entry"]
        sell_val = qty * t["Exit"]
        brokerage = (buy_val + sell_val) * p["brokerage_pct"] / 100.0
        pnl = sell_val - buy_val - brokerage
        cash += pnl
        open_pos.append((t["Exit Date"], buy_val))
        r = t.to_dict()
        r.update({"Qty": qty, "P/L ₹": round(pnl, 2),
                  "P/L %": round((t["Exit"] - t["Entry"]) / t["Entry"] * 100.0, 2),
                  "Equity After": round(cash, 2)})
        rows.append(r)
        equity_points.append(cash)

    log = pd.DataFrame(rows)
    if log.empty:
        return log, {}, skipped

    wins = log[log["P/L ₹"] > 0]
    losses = log[log["P/L ₹"] <= 0]
    gp, gl = wins["P/L ₹"].sum(), abs(losses["P/L ₹"].sum())
    eq = pd.Series(equity_points)
    peak = eq.cummax()
    max_dd = ((eq - peak) / peak).min() * 100.0
    years = max((pd.to_datetime(log["Exit Date"]).max() - pd.to_datetime(log["Entry Date"]).min()).days / 365.25, 0.1)
    cagr = ((cash / capital) ** (1 / years) - 1) * 100.0 if cash > 0 else -100.0

    summary = {
        "Total Trades": len(log),
        "Skipped (slots/cash full)": skipped,
        "Win Rate %": round(len(wins) / len(log) * 100.0, 1),
        "Profit Factor": round(gp / gl, 2) if gl > 0 else float("inf"),
        "Expectancy ₹/trade": round(log["P/L ₹"].mean(), 2),
        "Avg Win %": round(wins["P/L %"].mean(), 2) if len(wins) else 0,
        "Avg Loss %": round(losses["P/L %"].mean(), 2) if len(losses) else 0,
        "Avg Holding (trading days)": round(log["Holding (trading days)"].mean(), 1),
        "Net P/L ₹": round(cash - capital, 2),
        "Final Equity ₹": round(cash, 2),
        "CAGR %": round(cagr, 2),
        "Max Drawdown %": round(max_dd, 2),
    }
    return log, summary, skipped


def to_excel(dfs: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for name, d in dfs.items():
            d.to_excel(w, sheet_name=name[:31], index=False)
    return buf.getvalue()


# ============================================================================
# UI
# ============================================================================
st.title("Orderly Pullback Scanner & Backtester")
st.caption("Stage-2 trend → 12–20% orderly pullback → EMA10/20 support → volume contraction → Mother Candle breakout")

with st.sidebar:
    st.header("Universe")
    exch = st.selectbox("Exchange suffix", [".NS (NSE)", ".BO (BSE)"])
    suffix = ".NS" if exch.startswith(".NS") else ".BO"
    up = st.file_uploader("Upload symbols (CSV/Excel, 1st column)", type=["csv", "xlsx"])
    if up is not None:
        udf = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
        symbols = udf.iloc[:, 0].astype(str).str.strip().str.upper().str.replace(".NS", "", regex=False).tolist()
    else:
        symbols = NIFTY200
    st.write(f"{len(symbols)} symbols")
    period = st.selectbox("History", ["2y", "3y", "5y"], index=1)

    st.header("Setup Parameters")
    p = {}
    p["swing_lookback"] = st.slider("Swing-high lookback (days)", 20, 250, 100)
    p["pb_min"], p["pb_max"] = st.slider("Pullback depth %", 5, 30, (12, 20))
    p["pb_days_min"], p["pb_days_max"] = st.slider("Pullback duration (days)", 3, 30, (5, 15))
    p["ema200_rising_days"] = st.slider("EMA200 rising over N days", 5, 60, 20)
    p["max_consec_bear"] = st.slider("Max consecutive bearish candles", 2, 8, 4)
    p["large_candle_pct"] = st.slider("Large candle range % threshold", 2.0, 10.0, 5.0)
    p["max_large_bear"] = st.slider("Max large bearish candles", 0, 5, 1)
    p["gap_down_pct"] = st.slider("Reject gap-down bigger than %", 0.5, 5.0, 1.0)
    p["ema_prox_pct"] = st.slider("EMA10/20 proximity %", 0.5, 5.0, 2.0)

    st.header("Entry / Exit")
    p["entry_mode"] = st.selectbox("Entry", ["Break above Mother Candle High", "Next-day Open", "Next-day Close"])
    p["entry_wait"] = st.slider("Max days to wait for breakout", 1, 10, 3)
    p["sl_mode"] = st.selectbox("Stop Loss", ["Mother Candle Low", "Previous-day Low", "Swing Low", "Fixed %"])
    p["sl_fixed_pct"] = st.number_input("Fixed SL %", 1.0, 20.0, 5.0)
    p["target_mode"] = st.selectbox("Target", ["R:R Multiple", "Fixed %", "Trailing Stop"])
    p["rr_mult"] = st.slider("R:R multiple", 1.0, 3.0, 2.0, 0.5)
    p["target_fixed_pct"] = st.number_input("Fixed target %", 2.0, 50.0, 10.0)
    p["trail_pct"] = st.number_input("Trailing stop %", 2.0, 20.0, 8.0)
    p["max_hold"] = st.slider("Max holding (trading days)", 10, 120, 40)

    st.header("Portfolio")
    p["capital"] = st.number_input("Initial capital ₹", 50000, 100000000, 500000, step=50000)
    p["pos_mode"] = st.selectbox("Position sizing", ["Fixed ₹", "% of equity"])
    p["pos_size"] = st.number_input("Fixed position ₹", 5000, 10000000, 50000, step=5000)
    p["pos_pct"] = st.number_input("Position % of equity", 1.0, 50.0, 10.0)
    p["max_positions"] = st.slider("Max open positions", 1, 30, 10)
    p["brokerage_pct"] = st.number_input("Brokerage % (round trip on each leg)", 0.0, 1.0, 0.03)
    p["slippage_pct"] = st.number_input("Slippage % per leg", 0.0, 1.0, 0.05)

tab_scan, tab_bt = st.tabs(["🔍 Scanner (Today)", "📊 Backtester"])

with tab_scan:
    if st.button("Run Scanner", type="primary"):
        prog = st.progress(0.0, text="Downloading…")
        data = fetch_all(symbols, period, suffix, prog)
        prog.empty()
        st.info(f"Data OK for {len(data)}/{len(symbols)} symbols")
        rows = []
        for sym, df in data.items():
            df = add_emas(df)
            s = detect_setup(df, len(df) - 1, p)
            if s:
                rows.append({
                    "Symbol": sym, "Close": round(s["close"], 2),
                    "Mother Candle High (Entry Trigger)": round(s["mc_high"], 2),
                    "Mother Candle Low": round(s["mc_low"], 2),
                    "Swing High": round(s["swing_high"], 2),
                    "Pullback %": round(s["pullback_pct"], 1),
                    "Pullback Days": s["pullback_days"],
                    "EMA10": round(s["ema10"], 2), "EMA20": round(s["ema20"], 2), "EMA200": round(s["ema200"], 2),
                })
        if rows:
            res = pd.DataFrame(rows).sort_values("Pullback %")
            st.success(f"{len(res)} setups found")
            st.dataframe(res, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Export Scanner Results (Excel)",
                               to_excel({"Scanner": res}), "scanner_results.xlsx")
        else:
            st.warning("No setups found today.")

with tab_bt:
    if st.button("Run Backtest", type="primary"):
        prog = st.progress(0.0, text="Downloading…")
        data = fetch_all(symbols, period, suffix, prog)
        prog.empty()
        st.info(f"Data OK for {len(data)}/{len(symbols)} symbols")
        all_trades = []
        prog2 = st.progress(0.0, text="Backtesting…")
        for k, (sym, df) in enumerate(data.items()):
            all_trades.extend(backtest_symbol(sym, add_emas(df), p))
            prog2.progress((k + 1) / len(data), text=f"Backtesting {k+1}/{len(data)}")
        prog2.empty()

        log, summary, skipped = simulate_portfolio(all_trades, p)
        if log.empty:
            st.warning("No trades generated with current parameters.")
        else:
            st.subheader("Performance Summary")
            c = st.columns(4)
            for idx, (k2, v) in enumerate(summary.items()):
                c[idx % 4].metric(k2, v)
            st.subheader("Trade Log")
            st.dataframe(log, use_container_width=True, hide_index=True)
            sm = pd.DataFrame([summary])
            st.download_button("⬇️ Export Backtest (Excel)",
                               to_excel({"Trade Log": log, "Summary": sm}),
                               "orderly_pullback_backtest.xlsx")
