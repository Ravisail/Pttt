"""
Ants Pro - MVP Leaders Scanner (NSE)
=====================================
A single-file Streamlit re-implementation of the "Ants Pro - MVP Leaders" concept
(LevelUpTools / David Ryan) for Indian equities.

Ant criteria (all defaults are user-adjustable in the sidebar):
    Momentum : stock closed higher on at least M of the past N days
    Volume   : rolling N-day average volume is X%+ above the rolling 50-day average volume
    Price    : price is up Y%+ over the past N days

Also computes:
    - Distance from 50-SMA (%)
    - ATR Multiple  = (Close - 50SMA) / ATR(20)   -> how "stretched" price is, in ATR units
    - 3-Month & 12-Month % change
    - "Progress toward a new Ant" stats for stocks that haven't qualified yet
    - Full historical scan: finds each Ant *run* in the lookback window (first day of a
      consecutive streak, with the streak length noted), not just the latest bar

Run with:  streamlit run ants_pro_scanner.py
"""

import time
import io
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------------------------------------------------------------------------
# PAGE CONFIG & THEME
# ----------------------------------------------------------------------------------
st.set_page_config(page_title="Ants Pro | MVP Leaders Scanner", page_icon="🐜", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/122.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

FALLBACK_NIFTY50 = [
    "RELIANCE","TCS","HDFCBANK","ICICIBANK","INFY","BHARTIARTL","ITC","SBIN","LT","HINDUNILVR",
    "BAJFINANCE","KOTAKBANK","AXISBANK","ASIANPAINT","MARUTI","TITAN","SUNPHARMA","ULTRACEMCO",
    "NESTLEIND","WIPRO","M&M","NTPC","POWERGRID","ONGC","TATAMOTORS","TATASTEEL","ADANIENT",
    "ADANIPORTS","JSWSTEEL","COALINDIA","HCLTECH","TECHM","BAJAJFINSV","GRASIM","DRREDDY",
    "CIPLA","EICHERMOT","BRITANNIA","DIVISLAB","APOLLOHOSP","HEROMOTOCO","BAJAJ-AUTO","SBILIFE",
    "HDFCLIFE","INDUSINDBK","TATACONSUM","UPL","BPCL","SHREECEM","HINDALCO","LTIM",
]

INDEX_URL_MAP = {
    "NIFTY 50": "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
    "NIFTY 200": "https://archives.nseindia.com/content/indices/ind_nifty200list.csv",
    "NIFTY 500": "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
    "NIFTY MIDCAP 150": "https://archives.nseindia.com/content/indices/ind_niftymidcap150list.csv",
    "NIFTY SMALLCAP 250": "https://archives.nseindia.com/content/indices/ind_niftysmallcap250list.csv",
}

# Keep computed history capped in session_state to bound memory on large universes
# (chart tab only ever displays the most recent 300 bars).
STORE_MAX_ROWS = 400

# ----------------------------------------------------------------------------------
# UNIVERSE LOADING
# ----------------------------------------------------------------------------------
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_index_universe(index_name: str):
    """
    Fetch NSE index constituents from the official archives CSV.

    IMPORTANT: on failure this returns (None, "failed") rather than silently
    substituting a different index's constituent list. The one exception is
    "NIFTY 50" itself, which falls back to a hardcoded snapshot since it's the
    smallest/most stable index and used as the app's baseline default.
    """
    url = INDEX_URL_MAP.get(index_name)
    if url is None:
        return None, "unknown_index"
    try:
        resp = requests.get(url, headers=NSE_HEADERS, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        symbols = df["Symbol"].dropna().astype(str).str.strip().tolist()
        if len(symbols) < 10:
            raise ValueError("Too few symbols parsed")
        return symbols, "live"
    except Exception:
        if index_name == "NIFTY 50":
            return FALLBACK_NIFTY50, "fallback_hardcoded"
        return None, "failed"


def to_yf_ticker(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if symbol.endswith(".NS") or symbol.endswith(".BO"):
        return symbol
    return f"{symbol}.NS"


# ----------------------------------------------------------------------------------
# DATA FETCH (parallel, rate-limit tolerant, with retry support)
# ----------------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_history(ticker: str, period: str = "2y"):
    """Returns (df_or_None, reason). reason is one of:
       'ok', 'no_data', 'insufficient_history', 'error:<msg>'"""
    last_err = "no_data"
    for attempt in range(3):
        try:
            df = yf.download(ticker, period=period, interval="1d",
                              auto_adjust=True, progress=False, threads=False)
            if df is None or df.empty:
                last_err = "no_data"
                time.sleep(1.2 * (attempt + 1))
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.rename(columns=str.title)
            needed = {"Open", "High", "Low", "Close", "Volume"}
            if not needed.issubset(df.columns):
                return None, "no_data"
            df = df.dropna(subset=["Close", "Volume"])
            if len(df) < 60:
                return None, "insufficient_history"
            return df, "ok"
        except Exception as e:
            last_err = f"error:{str(e)[:80]}"
            time.sleep(1.5 * (attempt + 1))
    return None, last_err


def fetch_universe_parallel(tickers, period="2y", max_workers=8, progress_cb=None):
    """Returns (raw_data dict, failures dict[ticker -> reason])."""
    results = {}
    failures = {}
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_history, t, period): t for t in tickers}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                df, reason = fut.result()
            except Exception as e:
                df, reason = None, f"error:{str(e)[:80]}"
            if df is not None:
                results[t] = df
            else:
                failures[t] = reason
            done += 1
            if progress_cb:
                progress_cb(done, len(tickers))
    return results, failures


# ----------------------------------------------------------------------------------
# INDICATOR / ANT LOGIC
# ----------------------------------------------------------------------------------
def wilder_atr(df: pd.DataFrame, period: int = 20) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def compute_ant_series(df: pd.DataFrame, lookback: int, min_up_days: int,
                        vol_pct: float, price_pct: float, sma_period: int, atr_period: int):
    """Vectorized computation of the Ant boolean series across full history."""
    out = df.copy()
    out["Change"] = out["Close"].diff()
    out["UpDay"] = (out["Change"] > 0).astype(int)
    out["UpDayCount"] = out["UpDay"].rolling(lookback).sum()

    out["VolAvg50"] = out["Volume"].rolling(50).mean()
    out["VolAvgN"] = out["Volume"].rolling(lookback).mean()
    out["VolRatioPct"] = np.where(
        out["VolAvg50"] > 0, (out["VolAvgN"] / out["VolAvg50"] - 1.0) * 100.0, np.nan
    )

    prev_close_n = out["Close"].shift(lookback)
    out["PriceChgPct"] = np.where(
        prev_close_n > 0, (out["Close"] / prev_close_n - 1.0) * 100.0, np.nan
    )

    out["SMA"] = out["Close"].rolling(sma_period).mean()
    out["ATR"] = wilder_atr(out, atr_period)
    # Guard against division by zero / NaN denominators (flat or illiquid data)
    out["DistSMAPct"] = np.where(out["SMA"] > 0, (out["Close"] - out["SMA"]) / out["SMA"] * 100.0, np.nan)
    out["ATRMultiple"] = np.where(out["ATR"] > 0, (out["Close"] - out["SMA"]) / out["ATR"], np.nan)

    chg3m_base = out["Close"].shift(63)
    chg12m_base = out["Close"].shift(252)
    out["Chg3M"] = np.where(chg3m_base > 0, (out["Close"] / chg3m_base - 1.0) * 100.0, np.nan)
    out["Chg12M"] = np.where(chg12m_base > 0, (out["Close"] / chg12m_base - 1.0) * 100.0, np.nan)

    out["IsAnt"] = (
        (out["UpDayCount"] >= min_up_days) &
        (out["VolRatioPct"] >= vol_pct) &
        (out["PriceChgPct"] >= price_pct)
    )

    # Group consecutive Ant days into "runs" so a multi-day Ant streak is reported once,
    # with the run length attached, instead of inflating the result count.
    is_ant = out["IsAnt"].fillna(False)
    out["AntRunId"] = (is_ant != is_ant.shift(fill_value=False)).cumsum()
    out["AntRunStart"] = is_ant & (~is_ant.shift(fill_value=False))
    run_lengths = out.loc[is_ant].groupby("AntRunId").size()
    out["AntRunLength"] = out["AntRunId"].map(run_lengths).where(is_ant)

    return out


def progress_to_new_ant(row, min_up_days, vol_pct, price_pct):
    """Return a 0-100 'closeness' score plus the individual gaps for the watch table."""
    up_gap = min_up_days - row["UpDayCount"]
    vol_gap = vol_pct - row["VolRatioPct"]
    price_gap = price_pct - row["PriceChgPct"]

    up_score = np.clip(1 - max(up_gap, 0) / max(min_up_days, 1e-6), 0, 1)
    vol_score = np.clip(1 - max(vol_gap, 0) / max(abs(vol_pct), 1e-6), 0, 1)
    price_score = np.clip(1 - max(price_gap, 0) / max(abs(price_pct), 1e-6), 0, 1)
    closeness = round(float((up_score + vol_score + price_score) / 3 * 100), 1)
    return closeness, up_gap, vol_gap, price_gap


def build_results_for_ticker(ticker, df, params):
    """Compute indicators for one ticker and return (result_rows, watch_row_or_None, trimmed_df)."""
    comp = compute_ant_series(
        df, params["lookback"], params["min_up_days"], params["vol_pct"],
        params["price_pct"], params["sma_period"], params["atr_period"],
    )
    symbol = ticker.replace(".NS", "").replace(".BO", "")

    if params["scan_mode"] == "Latest bar only":
        window = comp.tail(1)
        hit_mask = window["IsAnt"].fillna(False)
    else:
        window = comp.tail(int(params["hist_bars"]))
        hit_mask = window["AntRunStart"] if params["dedup_runs"] else window["IsAnt"].fillna(False)

    result_rows = []
    for dt, row in window[hit_mask].iterrows():
        result_rows.append({
            "Symbol": symbol,
            "Ant Date": dt.strftime("%Y-%m-%d"),
            "Close": round(row["Close"], 2),
            "Up Days": int(row["UpDayCount"]) if pd.notna(row["UpDayCount"]) else np.nan,
            "Vol % vs 50D Avg": round(row["VolRatioPct"], 1) if pd.notna(row["VolRatioPct"]) else np.nan,
            "Price % Chg": round(row["PriceChgPct"], 1) if pd.notna(row["PriceChgPct"]) else np.nan,
            "Dist from SMA %": round(row["DistSMAPct"], 1) if pd.notna(row["DistSMAPct"]) else np.nan,
            "ATR Multiple": round(row["ATRMultiple"], 2) if pd.notna(row["ATRMultiple"]) else np.nan,
            "Run Length (days)": int(row["AntRunLength"]) if pd.notna(row.get("AntRunLength")) else np.nan,
            "3M % Chg": round(row["Chg3M"], 1) if pd.notna(row["Chg3M"]) else np.nan,
            "12M % Chg": round(row["Chg12M"], 1) if pd.notna(row["Chg12M"]) else np.nan,
            "Is Latest Bar": dt == comp.index[-1],
        })

    watch_row = None
    last_row = comp.iloc[-1]
    # Only score stocks whose rolling windows have fully warmed up, and skip stocks that
    # already qualify today (they belong in the results table, not the near-miss watchlist).
    core_ready = pd.notna(last_row["UpDayCount"]) and pd.notna(last_row["VolRatioPct"]) and pd.notna(last_row["PriceChgPct"])
    if core_ready and not bool(last_row["IsAnt"]):
        closeness, up_gap, vol_gap, price_gap = progress_to_new_ant(
            last_row, params["min_up_days"], params["vol_pct"], params["price_pct"]
        )
        watch_row = {
            "Symbol": symbol,
            "Closeness %": closeness,
            "Up Days (have/need)": f"{int(last_row['UpDayCount'])}/{params['min_up_days']}",
            "Vol % (have/need)": f"{last_row['VolRatioPct']:.1f}/{params['vol_pct']}",
            "Price % (have/need)": f"{last_row['PriceChgPct']:.1f}/{params['price_pct']}",
            "Close": round(last_row["Close"], 2),
        }

    trimmed = comp.tail(STORE_MAX_ROWS)[[
        "Open", "High", "Low", "Close", "Volume", "SMA", "ATR",
        "UpDayCount", "VolRatioPct", "PriceChgPct", "DistSMAPct", "ATRMultiple",
        "Chg3M", "Chg12M", "IsAnt",
    ]].copy()

    return result_rows, watch_row, trimmed


def run_universe_scan(tickers, params, progress_container=None):
    """Fetch + compute for a list of tickers. Returns dict with results, watch, computed, failures."""
    if progress_container is not None:
        progress_bar = progress_container.progress(0.0)
        status_text = progress_container.empty()

        def _cb(done, total):
            progress_bar.progress(done / total)
            status_text.text(f"Downloaded {done}/{total} symbols...")
    else:
        _cb = None

    raw_data, failures = fetch_universe_parallel(
        tickers, period=params["data_period"], max_workers=params["max_workers"], progress_cb=_cb
    )

    if progress_container is not None:
        progress_bar.empty()
        status_text.empty()

    result_rows, watch_rows, computed = [], [], {}
    for ticker, df in raw_data.items():
        rows, watch_row, trimmed = build_results_for_ticker(ticker, df, params)
        result_rows.extend(rows)
        if watch_row is not None:
            watch_rows.append(watch_row)
        computed[ticker] = trimmed

    return {
        "result_rows": result_rows,
        "watch_rows": watch_rows,
        "computed": computed,
        "failures": failures,
        "n_fetched": len(raw_data),
    }


# ----------------------------------------------------------------------------------
# SIDEBAR — SETTINGS
# ----------------------------------------------------------------------------------
st.sidebar.title("🐜 Ants Pro Settings")

universe_choice = st.sidebar.selectbox(
    "Universe",
    ["NIFTY 50", "NIFTY 200", "NIFTY 500", "NIFTY MIDCAP 150", "NIFTY SMALLCAP 250", "Custom list"],
    index=0,
)

custom_tickers_raw = ""
if universe_choice == "Custom list":
    custom_tickers_raw = st.sidebar.text_area(
        "Enter symbols (comma or newline separated, NSE symbols without .NS)",
        placeholder="RELIANCE, TCS, INFY, HDFCBANK",
        height=100,
    )

st.sidebar.markdown("---")
st.sidebar.subheader("Ant Requirements (MVP)")
lookback_days = st.sidebar.slider("Lookback window (days)", 5, 30, 15)
min_up_days = st.sidebar.slider("Min up-days within window", 1, lookback_days, min(12, lookback_days))
vol_pct_threshold = st.sidebar.slider("Min volume % above 50-day avg", 0, 100, 20, step=5)
price_pct_threshold = st.sidebar.slider("Min price % change over window", 0, 100, 20, step=5)

st.sidebar.markdown("---")
st.sidebar.subheader("Context Indicators")
sma_period = st.sidebar.number_input("SMA period (for distance/ATR mult.)", 10, 200, 50)
atr_period = st.sidebar.number_input("ATR period", 5, 50, 20)

st.sidebar.markdown("---")
st.sidebar.subheader("Historical Scan")
scan_mode = st.sidebar.radio("Scan mode", ["Latest bar only", "Historical scan"], index=1)
hist_bars = st.sidebar.number_input("Historical bars to search", 20, 500, 250, step=10,
                                     disabled=(scan_mode == "Latest bar only"))
dedup_runs = st.sidebar.checkbox(
    "Collapse consecutive Ant days into one run", value=True,
    disabled=(scan_mode == "Latest bar only"),
    help="When on, a multi-day Ant streak is reported once (first day) with a 'Run "
         "Length' column, instead of one row per day.",
)

st.sidebar.markdown("---")
data_period = st.sidebar.selectbox("Data fetch period", ["1y", "2y", "3y", "5y"], index=1)
max_workers = st.sidebar.slider("Parallel download workers", 2, 16, 8)

run_scan = st.sidebar.button("🚀 Run Scan", type="primary", use_container_width=True)

params = dict(
    lookback=lookback_days, min_up_days=min_up_days, vol_pct=vol_pct_threshold,
    price_pct=price_pct_threshold, sma_period=sma_period, atr_period=atr_period,
    scan_mode=scan_mode, hist_bars=hist_bars, dedup_runs=dedup_runs,
    data_period=data_period, max_workers=max_workers,
)

# ----------------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------------
st.title("🐜 Ants Pro — MVP Leaders Scanner")
st.caption(
    "Momentum · Volume · Price scanner inspired by the Ants / MVP methodology "
    "(David Ryan, LevelUpTools). Ants flag strength, not a buy signal — pair with "
    "pullback-to-support or breakout confirmation before acting."
)

tab_scan, tab_chart, tab_watch, tab_about = st.tabs(
    ["📊 Scanner", "📈 Chart View", "👀 Progress Watch", "ℹ️ About Ants"]
)

# ----------------------------------------------------------------------------------
# STATE
# ----------------------------------------------------------------------------------
for key, default in [
    ("scan_results", None), ("watch_results", pd.DataFrame()), ("computed_data", {}),
    ("failed_tickers", {}), ("scan_params", None), ("attempted_tickers", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ----------------------------------------------------------------------------------
# RUN SCAN
# ----------------------------------------------------------------------------------
if run_scan:
    if universe_choice == "Custom list":
        raw = [s.strip() for s in custom_tickers_raw.replace("\n", ",").split(",") if s.strip()]
        symbols, source_note = raw, "custom"
    else:
        symbols, source_note = load_index_universe(universe_choice)

    if source_note == "failed":
        st.error(
            f"Could not fetch live **{universe_choice}** constituents from the NSE archives "
            f"(request blocked or rate-limited). To avoid scanning the wrong universe, the "
            f"scan was **not** run. Please retry in a moment, or paste the list manually "
            f"via **Custom list** in the sidebar."
        )
        st.session_state.scan_results = None
    elif not symbols:
        st.error("No symbols to scan. Please check your universe selection or custom list.")
    else:
        if source_note == "fallback_hardcoded":
            st.warning(
                "Could not fetch live NIFTY 50 constituents from NSE archives — using a "
                "hardcoded snapshot list instead. Constituents may be slightly stale."
            )

        tickers = [to_yf_ticker(s) for s in symbols]
        st.info(f"Fetching {len(tickers)} symbols from Yahoo Finance ({data_period} history)...")

        progress_container = st.container()
        scan = run_universe_scan(tickers, params, progress_container=progress_container)

        st.session_state.scan_results = pd.DataFrame(scan["result_rows"])
        st.session_state.watch_results = (
            pd.DataFrame(scan["watch_rows"]).sort_values("Closeness %", ascending=False)
            if scan["watch_rows"] else pd.DataFrame()
        )
        st.session_state.computed_data = scan["computed"]
        st.session_state.failed_tickers = scan["failures"]
        st.session_state.scan_params = params
        st.session_state.attempted_tickers = tickers

        st.success(
            f"Scan complete — {scan['n_fetched']}/{len(tickers)} symbols processed, "
            f"{len(scan['result_rows'])} Ant hit(s) found."
        )
        if scan["failures"]:
            st.warning(f"{len(scan['failures'])} symbol(s) failed to download — see the "
                       f"'Failed Symbols' panel below to review or retry them.")

# ----------------------------------------------------------------------------------
# RETRY FAILED DOWNLOADS
# ----------------------------------------------------------------------------------
if st.session_state.failed_tickers:
    with st.expander(f"⚠️ Failed / Skipped Symbols ({len(st.session_state.failed_tickers)})", expanded=False):
        fail_df = pd.DataFrame(
            [{"Ticker": t, "Reason": r} for t, r in st.session_state.failed_tickers.items()]
        ).sort_values("Ticker")
        st.dataframe(fail_df, use_container_width=True, hide_index=True)
        st.caption(
            "Reasons: 'no_data' — Yahoo returned nothing (often a delisted/renamed symbol); "
            "'insufficient_history' — fewer than 60 daily bars available; 'error:...' — "
            "a network/rate-limit error. Rate-limit errors are usually worth retrying."
        )
        if st.button("🔄 Retry Failed Downloads") and st.session_state.scan_params is not None:
            retry_tickers = list(st.session_state.failed_tickers.keys())
            retry_container = st.container()
            retry_scan = run_universe_scan(retry_tickers, st.session_state.scan_params,
                                            progress_container=retry_container)

            if retry_scan["result_rows"]:
                st.session_state.scan_results = pd.concat(
                    [st.session_state.scan_results, pd.DataFrame(retry_scan["result_rows"])],
                    ignore_index=True,
                )
            if retry_scan["watch_rows"]:
                st.session_state.watch_results = pd.concat(
                    [st.session_state.watch_results, pd.DataFrame(retry_scan["watch_rows"])],
                    ignore_index=True,
                ).sort_values("Closeness %", ascending=False)
            st.session_state.computed_data.update(retry_scan["computed"])
            # Symbols that succeeded this time drop out of the failed list; new failures replace old ones
            st.session_state.failed_tickers = retry_scan["failures"]
            st.success(f"Retry complete — recovered {retry_scan['n_fetched']}/{len(retry_tickers)} symbol(s).")
            st.rerun()

# ----------------------------------------------------------------------------------
# TAB: SCANNER RESULTS
# ----------------------------------------------------------------------------------
with tab_scan:
    res = st.session_state.scan_results
    if res is None:
        st.info("Configure your settings in the sidebar and click **Run Scan** to begin.")
    elif res.empty:
        st.warning("No Ants found for the current universe / settings / lookback window.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Ant Hits", len(res))
        c2.metric("Unique Symbols", res["Symbol"].nunique())
        latest_today = res[res["Is Latest Bar"]]["Symbol"].nunique() if "Is Latest Bar" in res else 0
        c3.metric("New Ants (latest bar)", latest_today)
        c4.metric("Avg ATR Multiple", round(res["ATR Multiple"].mean(), 2) if len(res) else 0)

        st.markdown("#### Results")
        only_latest = st.checkbox("Show only latest-bar Ants", value=False)
        display_df = res[res["Is Latest Bar"]] if only_latest else res
        display_df = display_df.sort_values(["Ant Date", "Symbol"], ascending=[False, True])

        st.dataframe(
            display_df.drop(columns=["Is Latest Bar"]),
            use_container_width=True,
            hide_index=True,
        )

        # Excel export — fall back to openpyxl if xlsxwriter isn't installed, and never
        # let a missing optional dependency break the rest of the page.
        export_df = display_df.drop(columns=["Is Latest Bar"])
        excel_bytes, engine_used, export_err = None, None, None
        for engine in ("xlsxwriter", "openpyxl"):
            try:
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine=engine) as writer:
                    export_df.to_excel(writer, sheet_name="Ants", index=False)
                    if engine == "xlsxwriter":
                        workbook = writer.book
                        worksheet = writer.sheets["Ants"]
                        header_fmt = workbook.add_format(
                            {"bold": True, "bg_color": "#16a34a", "font_color": "white"}
                        )
                        for col_num, col_name in enumerate(export_df.columns):
                            worksheet.write(0, col_num, col_name, header_fmt)
                            worksheet.set_column(col_num, col_num, 16)
                excel_bytes, engine_used = buf.getvalue(), engine
                break
            except ImportError:
                export_err = f"'{engine}' not installed"
                continue

        if excel_bytes:
            st.download_button(
                "⬇️ Download Results (Excel)",
                data=excel_bytes,
                file_name=f"ants_pro_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            if engine_used == "openpyxl":
                st.caption("Exported with openpyxl (xlsxwriter not found — `pip install xlsxwriter` for styled headers).")
        else:
            st.warning(
                f"Excel export unavailable ({export_err}). Install one of: "
                f"`pip install xlsxwriter` or `pip install openpyxl`. "
                f"You can still copy data from the table above."
            )
            st.download_button(
                "⬇️ Download Results (CSV)",
                data=export_df.to_csv(index=False).encode("utf-8"),
                file_name=f"ants_pro_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )

# ----------------------------------------------------------------------------------
# TAB: CHART VIEW
# ----------------------------------------------------------------------------------
with tab_chart:
    computed = st.session_state.computed_data
    if not computed:
        st.info("Run a scan first to populate chart data.")
    else:
        symbol_map = {t.replace(".NS", "").replace(".BO", ""): t for t in computed.keys()}
        chosen = st.selectbox("Symbol", sorted(symbol_map.keys()))
        df = computed[symbol_map[chosen]].tail(300)
        sma_label = st.session_state.scan_params["sma_period"] if st.session_state.scan_params else sma_period

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03,
        )
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name="Price", increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA"], name=f"{sma_label}-SMA", line=dict(color="#fbbf24", width=1.5)
        ), row=1, col=1)

        ants = df[df["IsAnt"].fillna(False)]
        fig.add_trace(go.Scatter(
            x=ants.index, y=ants["High"] * 1.02, mode="markers", name="Ant",
            marker=dict(symbol="triangle-up", size=9, color="#111827",
                        line=dict(color="#facc15", width=1.5)),
            hovertext=[
                f"Up days: {r.UpDayCount:.0f}<br>Vol%: {r.VolRatioPct:.1f}<br>"
                f"Price%: {r.PriceChgPct:.1f}<br>ATR mult: "
                f"{r.ATRMultiple:.2f}" if pd.notna(r.ATRMultiple) else "ATR mult: n/a"
                for r in ants.itertuples()
            ],
            hoverinfo="text+x",
        ), row=1, col=1)

        vol_colors = np.where(df["Close"] >= df["Close"].shift(1), "#22c55e", "#ef4444")
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=vol_colors),
                      row=2, col=1)

        fig.update_layout(
            height=650, template="plotly_dark", showlegend=True,
            xaxis_rangeslider_visible=False, margin=dict(t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        last = df.iloc[-1]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Close", f"₹{last['Close']:.2f}")
        c2.metric("Dist from SMA", f"{last['DistSMAPct']:.1f}%" if pd.notna(last['DistSMAPct']) else "—")
        c3.metric("ATR Multiple", f"{last['ATRMultiple']:.2f}" if pd.notna(last['ATRMultiple']) else "—")
        c4.metric("3M % Chg", f"{last['Chg3M']:.1f}%" if pd.notna(last['Chg3M']) else "—")
        c5.metric("12M % Chg", f"{last['Chg12M']:.1f}%" if pd.notna(last['Chg12M']) else "—")

# ----------------------------------------------------------------------------------
# TAB: PROGRESS WATCH
# ----------------------------------------------------------------------------------
with tab_watch:
    watch = st.session_state.watch_results
    if watch is None or watch.empty:
        st.info("Run a scan first to see stocks tracking toward a new Ant. "
                "(Stocks that already qualify as an Ant today are excluded here — "
                "check the Scanner tab for those.)")
    else:
        st.caption(
            "Stocks ranked by how close they are to qualifying for a new Ant on the latest bar. "
            "Only stocks with fully warmed-up rolling windows are scored, and stocks already "
            "qualifying as an Ant today are excluded (see the Scanner tab for those)."
        )
        min_close = st.slider("Min closeness % to display", 0, 100, 60)
        filtered = watch[watch["Closeness %"] >= min_close]
        st.dataframe(filtered, use_container_width=True, hide_index=True)

# ----------------------------------------------------------------------------------
# TAB: ABOUT
# ----------------------------------------------------------------------------------
with tab_about:
    st.markdown("""
### What is an "Ant"?
An Ant flags a stock showing exceptional **M**omentum, **V**olume and **P**rice strength
over a short lookback window — often an early sign of institutional accumulation.

**Default criteria (all adjustable in the sidebar):**
- **Momentum** — closed higher on ≥12 of the past 15 days
- **Volume** — 15-day average volume ≥20% above the 50-day average volume
- **Price** — price up ≥20% over the past 15 days

### Ants are not a buy signal
Chasing an Ant bar is discouraged. The suggested approach:
1. Add the stock to a watchlist when an Ant appears
2. Wait for a **pullback to support** (a moving average or prior demand zone), or
3. Wait for **sideways consolidation followed by a breakout** above the range high

### Ants as a topping signal
The same MVP signature can also show up near a **climax top** — an extended uptrend
followed by a sharp 2–3 week surge on heavy volume. Context distinguishes the two:
- **Distance from the 50-SMA** — how stretched price is above its intermediate trend
- **ATR Multiple** — that distance expressed in units of ATR (20-day by default);
  a high multiple (e.g. 5x) suggests price is very extended and higher risk of a pullback

### Historical scan
Instead of only checking the latest bar, this scanner can look back over N bars
(configurable) and flag Ant occurrences in that window. By default, consecutive Ant
days are collapsed into a single "run" (first day + run length) so a long streak
doesn't inflate the result count — this can be toggled off in the sidebar.

### Best practices
- Relative strength should be trending up
- Look for earnings/sales acceleration as confirmation
- Always define risk before entering — know your exit
- Size positions by volatility/conviction, not emotion
""")
