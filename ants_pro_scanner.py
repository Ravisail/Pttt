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
    - Full historical scan: finds every Ant bar in the lookback window, not just the latest bar

Run with:  streamlit run ants_pro_scanner.py
"""

import time
import io
from datetime import datetime, timedelta
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
    .ant-badge {
        background: linear-gradient(90deg,#16a34a,#22c55e);
        color: white; padding: 2px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600;
    }
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

# ----------------------------------------------------------------------------------
# UNIVERSE LOADING
# ----------------------------------------------------------------------------------
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_index_universe(index_name: str):
    """Fetch NSE index constituents from the official archives CSV. Falls back to a
    hardcoded Nifty50 list if the request fails (NSE blocks non-browser requests often)."""
    url_map = {
        "NIFTY 50": "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
        "NIFTY 200": "https://archives.nseindia.com/content/indices/ind_nifty200list.csv",
        "NIFTY 500": "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
        "NIFTY MIDCAP 150": "https://archives.nseindia.com/content/indices/ind_niftymidcap150list.csv",
        "NIFTY SMALLCAP 250": "https://archives.nseindia.com/content/indices/ind_niftysmallcap250list.csv",
    }
    url = url_map.get(index_name)
    if url is None:
        return FALLBACK_NIFTY50, "fallback"
    try:
        resp = requests.get(url, headers=NSE_HEADERS, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        symbols = df["Symbol"].dropna().astype(str).str.strip().tolist()
        if len(symbols) < 10:
            raise ValueError("Too few symbols parsed")
        return symbols, "live"
    except Exception:
        return FALLBACK_NIFTY50, "fallback"


def to_yf_ticker(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if symbol.endswith(".NS") or symbol.endswith(".BO"):
        return symbol
    return f"{symbol}.NS"


# ----------------------------------------------------------------------------------
# DATA FETCH (parallel, rate-limit tolerant)
# ----------------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_history(ticker: str, period: str = "2y"):
    for attempt in range(3):
        try:
            df = yf.download(ticker, period=period, interval="1d",
                              auto_adjust=True, progress=False, threads=False)
            if df is None or df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.rename(columns=str.title)
            needed = {"Open", "High", "Low", "Close", "Volume"}
            if not needed.issubset(df.columns):
                return None
            df = df.dropna(subset=["Close", "Volume"])
            if len(df) < 60:
                return None
            return df
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return None


def fetch_universe_parallel(tickers, period="2y", max_workers=8, progress_cb=None):
    results = {}
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_history, t, period): t for t in tickers}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                df = fut.result()
            except Exception:
                df = None
            if df is not None:
                results[t] = df
            done += 1
            if progress_cb:
                progress_cb(done, len(tickers))
    return results


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
    out["VolRatioPct"] = (out["VolAvgN"] / out["VolAvg50"] - 1.0) * 100.0

    out["PriceChgPct"] = (out["Close"] / out["Close"].shift(lookback) - 1.0) * 100.0

    out["SMA"] = out["Close"].rolling(sma_period).mean()
    out["ATR"] = wilder_atr(out, atr_period)
    out["DistSMAPct"] = (out["Close"] - out["SMA"]) / out["SMA"] * 100.0
    out["ATRMultiple"] = (out["Close"] - out["SMA"]) / out["ATR"]

    out["Chg3M"] = (out["Close"] / out["Close"].shift(63) - 1.0) * 100.0
    out["Chg12M"] = (out["Close"] / out["Close"].shift(252) - 1.0) * 100.0

    out["IsAnt"] = (
        (out["UpDayCount"] >= min_up_days) &
        (out["VolRatioPct"] >= vol_pct) &
        (out["PriceChgPct"] >= price_pct)
    )
    return out


def progress_to_new_ant(row, lookback, min_up_days, vol_pct, price_pct):
    """Return a 0-100 'closeness' score plus the individual gaps for the watch table."""
    up_gap = min_up_days - row["UpDayCount"]
    vol_gap = vol_pct - row["VolRatioPct"]
    price_gap = price_pct - row["PriceChgPct"]

    up_score = np.clip(1 - max(up_gap, 0) / min_up_days, 0, 1)
    vol_score = np.clip(1 - max(vol_gap, 0) / max(vol_pct, 1e-6), 0, 1)
    price_score = np.clip(1 - max(price_gap, 0) / max(price_pct, 1e-6), 0, 1)
    closeness = round(float((up_score + vol_score + price_score) / 3 * 100), 1)
    return closeness, up_gap, vol_gap, price_gap


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

st.sidebar.markdown("---")
data_period = st.sidebar.selectbox("Data fetch period", ["1y", "2y", "3y", "5y"], index=1)
max_workers = st.sidebar.slider("Parallel download workers", 2, 16, 8)

run_scan = st.sidebar.button("🚀 Run Scan", type="primary", use_container_width=True)

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
if "scan_results" not in st.session_state:
    st.session_state.scan_results = None
if "raw_data" not in st.session_state:
    st.session_state.raw_data = {}
if "computed_data" not in st.session_state:
    st.session_state.computed_data = {}

# ----------------------------------------------------------------------------------
# RUN SCAN
# ----------------------------------------------------------------------------------
if run_scan:
    if universe_choice == "Custom list":
        raw = [s.strip() for s in custom_tickers_raw.replace("\n", ",").split(",") if s.strip()]
        symbols = raw
        source_note = "custom"
    else:
        symbols, source_note = load_index_universe(universe_choice)

    if not symbols:
        st.error("No symbols to scan. Please check your universe selection or custom list.")
    else:
        if source_note == "fallback" and universe_choice != "Custom list":
            st.warning(
                f"Could not fetch live {universe_choice} constituents from NSE archives "
                f"(likely blocked/rate-limited outside a browser session). Using a fallback "
                f"Nifty 50 list instead — you can also paste a custom list in the sidebar."
            )

        tickers = [to_yf_ticker(s) for s in symbols]
        st.info(f"Fetching {len(tickers)} symbols from Yahoo Finance ({data_period} history)...")

        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def _cb(done, total):
            progress_bar.progress(done / total)
            status_text.text(f"Downloaded {done}/{total} symbols...")

        raw_data = fetch_universe_parallel(tickers, period=data_period,
                                            max_workers=max_workers, progress_cb=_cb)
        progress_bar.empty()
        status_text.empty()

        failed = len(tickers) - len(raw_data)
        if failed:
            st.warning(f"{failed} symbol(s) failed to download or had insufficient history and were skipped.")

        st.session_state.raw_data = raw_data

        results_rows = []
        watch_rows = []
        computed = {}

        for ticker, df in raw_data.items():
            comp = compute_ant_series(df, lookback_days, min_up_days, vol_pct_threshold,
                                       price_pct_threshold, sma_period, atr_period)
            computed[ticker] = comp
            symbol = ticker.replace(".NS", "").replace(".BO", "")

            if scan_mode == "Latest bar only":
                window = comp.tail(1)
            else:
                window = comp.tail(int(hist_bars))

            ant_hits = window[window["IsAnt"]]
            for dt, row in ant_hits.iterrows():
                results_rows.append({
                    "Symbol": symbol,
                    "Ant Date": dt.strftime("%Y-%m-%d"),
                    "Close": round(row["Close"], 2),
                    "Up Days": int(row["UpDayCount"]),
                    "Vol % vs 50D Avg": round(row["VolRatioPct"], 1),
                    "Price % Chg": round(row["PriceChgPct"], 1),
                    "Dist from SMA %": round(row["DistSMAPct"], 1),
                    "ATR Multiple": round(row["ATRMultiple"], 2) if pd.notna(row["ATRMultiple"]) else np.nan,
                    "3M % Chg": round(row["Chg3M"], 1) if pd.notna(row["Chg3M"]) else np.nan,
                    "12M % Chg": round(row["Chg12M"], 1) if pd.notna(row["Chg12M"]) else np.nan,
                    "Is Latest Bar": dt == comp.index[-1],
                })

            # progress-to-new-ant, based on latest bar (only if not already an ant today)
            last_row = comp.iloc[-1]
            if pd.notna(last_row["UpDayCount"]) and pd.notna(last_row["VolRatioPct"]) and pd.notna(last_row["PriceChgPct"]):
                closeness, up_gap, vol_gap, price_gap = progress_to_new_ant(
                    last_row, lookback_days, min_up_days, vol_pct_threshold, price_pct_threshold
                )
                watch_rows.append({
                    "Symbol": symbol,
                    "Is Ant Today": bool(last_row["IsAnt"]),
                    "Closeness %": closeness,
                    "Up Days (have/need)": f"{int(last_row['UpDayCount'])}/{min_up_days}",
                    "Vol % (have/need)": f"{last_row['VolRatioPct']:.1f}/{vol_pct_threshold}",
                    "Price % (have/need)": f"{last_row['PriceChgPct']:.1f}/{price_pct_threshold}",
                    "Close": round(last_row["Close"], 2),
                })

        st.session_state.computed_data = computed
        st.session_state.scan_results = pd.DataFrame(results_rows)
        st.session_state.watch_results = pd.DataFrame(watch_rows).sort_values(
            "Closeness %", ascending=False
        ) if watch_rows else pd.DataFrame()

        st.success(f"Scan complete — {len(raw_data)} symbols processed, "
                   f"{len(results_rows)} Ant hit(s) found.")

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

        # Excel export
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            display_df.drop(columns=["Is Latest Bar"]).to_excel(writer, sheet_name="Ants", index=False)
            workbook = writer.book
            worksheet = writer.sheets["Ants"]
            header_fmt = workbook.add_format({"bold": True, "bg_color": "#16a34a", "font_color": "white"})
            for col_num, col_name in enumerate(display_df.drop(columns=["Is Latest Bar"]).columns):
                worksheet.write(0, col_num, col_name, header_fmt)
                worksheet.set_column(col_num, col_num, 16)
        st.download_button(
            "⬇️ Download Results (Excel)",
            data=buf.getvalue(),
            file_name=f"ants_pro_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
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

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03,
        )
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name="Price", increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA"], name=f"{sma_period}-SMA", line=dict(color="#fbbf24", width=1.5)
        ), row=1, col=1)

        ants = df[df["IsAnt"]]
        fig.add_trace(go.Scatter(
            x=ants.index, y=ants["High"] * 1.02, mode="markers", name="Ant",
            marker=dict(symbol="triangle-up", size=9, color="#111827",
                        line=dict(color="#facc15", width=1.5)),
            hovertext=[
                f"Up days: {r.UpDayCount:.0f}<br>Vol%: {r.VolRatioPct:.1f}<br>"
                f"Price%: {r.PriceChgPct:.1f}<br>ATR mult: {r.ATRMultiple:.2f}"
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
        c2.metric("Dist from SMA", f"{last['DistSMAPct']:.1f}%")
        c3.metric("ATR Multiple", f"{last['ATRMultiple']:.2f}")
        c4.metric("3M % Chg", f"{last['Chg3M']:.1f}%" if pd.notna(last['Chg3M']) else "—")
        c5.metric("12M % Chg", f"{last['Chg12M']:.1f}%" if pd.notna(last['Chg12M']) else "—")

# ----------------------------------------------------------------------------------
# TAB: PROGRESS WATCH
# ----------------------------------------------------------------------------------
with tab_watch:
    watch = st.session_state.get("watch_results", pd.DataFrame())
    if watch is None or watch.empty:
        st.info("Run a scan first to see stocks tracking toward a new Ant.")
    else:
        st.caption(
            "Stocks ranked by how close they are to qualifying for a new Ant on the latest bar. "
            "A shallow pullback or continued strength could tip these into Ant status soon."
        )
        min_close = st.slider("Min closeness % to display", 0, 100, 60)
        filtered = watch[(watch["Closeness %"] >= min_close) & (~watch["Is Ant Today"])]
        st.dataframe(filtered.drop(columns=["Is Ant Today"]), use_container_width=True, hide_index=True)

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
(configurable) and flag every historical Ant occurrence — useful for building a
watchlist of past leaders or backtesting how Ants preceded big moves.

### Best practices
- Relative strength should be trending up
- Look for earnings/sales acceleration as confirmation
- Always define risk before entering — know your exit
- Size positions by volatility/conviction, not emotion
""")
