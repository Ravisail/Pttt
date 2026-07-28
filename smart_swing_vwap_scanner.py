"""
Smart Swing VWAP Scanner (NSE) - Streamlit single-file app
============================================================
Original Python re-implementation of the *methodology* published on the
"Smart Swing VWAP (Zeiierman)" TradingView page (multi-scale swing ranking
-> adaptive anchor selection -> decayed volume-weighted average -> structure
tracking -> retest detection). This is NOT a port of the author's Pine
source (which is not reproduced here) - it is an original interpretation of
the described public methodology, built for NSE equities.

Universe   : NIFTY 50 / NIFTY 200 / NIFTY 500 (live from NSE archives,
             cached, with CSV upload fallback for custom lists)
Data       : yfinance (.NS), ThreadPoolExecutor parallel fetch, cached
Scan modes : Latest bar / As-of a specific historical date / Full
             historical range (find every signal bar in a lookback window)

Run: streamlit run smart_swing_vwap_scanner.py
"""

import io
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    st.error("yfinance is not installed. Run: pip install yfinance")
    st.stop()

# ============================================================================
# CONSTANTS / FALLBACK LISTS
# ============================================================================

NSE_INDEX_CSV = {
    "NIFTY 50":  "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv",
    "NIFTY 200": "https://nsearchives.nseindia.com/content/indices/ind_nifty200list.csv",
    "NIFTY 500": "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
}

# Minimal offline fallback (used only if the live NSE fetch fails). Nifty 50
# is stable enough to hardcode as a last resort; 200/500 fall back to asking
# the user to upload a CSV instead of risking a stale/incorrect 500-name list.
FALLBACK_NIFTY50 = [
    "ADANIENT","ADANIPORTS","APOLLOHOSP","ASIANPAINT","AXISBANK","BAJAJ-AUTO",
    "BAJFINANCE","BAJAJFINSV","BEL","BHARTIARTL","CIPLA","COALINDIA","DRREDDY",
    "EICHERMOT","GRASIM","HCLTECH","HDFCBANK","HDFCLIFE","HEROMOTOCO",
    "HINDALCO","HINDUNILVR","ICICIBANK","ITC","INDUSINDBK","INFY","JSWSTEEL",
    "KOTAKBANK","LT","LTIM","M&M","MARUTI","NESTLEIND","NTPC","ONGC",
    "POWERGRID","RELIANCE","SBILIFE","SHRIRAMFIN","SBIN","SUNPHARMA",
    "TCS","TATACONSUM","TATAMOTORS","TATASTEEL","TECHM","TITAN","TRENT",
    "ULTRACEMCO","UPL","WIPRO","BRITANNIA",
]

NSE_REQUEST_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/market-data/live-equity-market",
}

SIGNAL_TYPES = [
    "Structure Break Bull",
    "Structure Break Bear",
    "Retest Support",
    "Retest Resistance",
    "New Anchor",
]

# ============================================================================
# UNIVERSE FETCHING
# ============================================================================

@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def fetch_nse_index_list(index_name: str) -> pd.DataFrame:
    """Fetch live constituent list from NSE archives. Falls back gracefully."""
    url = NSE_INDEX_CSV.get(index_name)
    try:
        with requests.Session() as session:
            session.headers.update(NSE_REQUEST_HEADERS)
            session.get("https://www.nseindia.com", timeout=6)
            resp = session.get(url, timeout=12)
        if resp.status_code == 200 and "Symbol" in resp.text:
            df = pd.read_csv(io.StringIO(resp.text))
            df.columns = [c.strip() for c in df.columns]
            df = df[["Symbol", "Company Name"] if "Company Name" in df.columns
                     else ["Symbol"]]
            df["Symbol"] = df["Symbol"].astype(str).str.strip()
            return df.reset_index(drop=True)
    except Exception:
        pass

    if index_name == "NIFTY 50":
        return pd.DataFrame({"Symbol": FALLBACK_NIFTY50})
    return pd.DataFrame(columns=["Symbol"])  # signal caller to ask for upload


def get_universe(choice: str, uploaded_file) -> list:
    if choice == "Custom (upload CSV)":
        if uploaded_file is None:
            return []
        df = pd.read_csv(uploaded_file)
        col = "Symbol" if "Symbol" in df.columns else df.columns[0]
        return sorted(set(df[col].astype(str).str.strip().str.upper()))

    df = fetch_nse_index_list(choice)
    if df.empty:
        st.warning(
            f"Live fetch for {choice} failed and no offline fallback is bundled "
            "for this list. Upload a CSV with a 'Symbol' column instead, or retry."
        )
        return []
    return sorted(set(df["Symbol"].astype(str).str.strip().str.upper()))


# ============================================================================
# DATA FETCH (yfinance, parallel, cached)
# ============================================================================

@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_ohlcv(symbol: str, start: str, end: str) -> pd.DataFrame:
    ticker = symbol if symbol.endswith((".NS", ".BO")) else f"{symbol}.NS"
    try:
        df = yf.Ticker(ticker).history(start=start, end=end, interval="1d",
                                        auto_adjust=True)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns=str.title)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()


def fetch_universe_data(symbols: list, start: str, end: str, max_workers: int = 8):
    data = {}
    progress = st.progress(0.0, text="Fetching price data...")
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_ohlcv, s, start, end): s for s in symbols}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                df = fut.result()
                if not df.empty and len(df) > 60:
                    data[sym] = df
            except Exception:
                pass
            done += 1
            progress.progress(done / len(symbols),
                               text=f"Fetching price data... ({done}/{len(symbols)})")
    progress.empty()
    return data


# ============================================================================
# SMART SWING VWAP ENGINE
# ============================================================================

@dataclass
class EngineParams:
    scales: tuple = (5, 10, 20, 40, 80)
    active_scales: tuple = (True, True, True, True, True)
    min_anchor_score: float = 60.0
    min_move_strength: float = 1.0        # in ATR multiples
    bars_between_anchors: int = 10
    price_tracking_speed: int = 20        # bars (higher = smoother/slower)
    vol_adjust: bool = True
    vol_adjust_strength: float = 1.0
    retest_touch_zone_atr: float = 0.3
    retest_setup_dist_atr: float = 1.0
    atr_period: int = 14
    score_weights: tuple = (0.25, 0.25, 0.25, 0.25)  # range, volume, volatility, agreement


@dataclass
class PivotEvent:
    bar_idx: int
    confirm_idx: int
    kind: str          # 'high' or 'low'
    price: float
    scale_len: int
    score: float = 0.0
    move_strength: float = 0.0


def _true_range(high, low, close):
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close),
                                            np.abs(low - prev_close)))
    return tr


def _atr(high, low, close, period):
    tr = _true_range(high, low, close)
    atr = pd.Series(tr).rolling(period, min_periods=1).mean().to_numpy()
    return atr


def _find_pivots(high, low, length):
    n = len(high)
    events = []
    for i in range(length, n - length):
        window_h = high[i - length:i + length + 1]
        if high[i] == window_h.max() and np.argmax(window_h) == length:
            events.append((i, i + length, "high", high[i]))
        window_l = low[i - length:i + length + 1]
        if low[i] == window_l.min() and np.argmin(window_l) == length:
            events.append((i, i + length, "low", low[i]))
    return events


def _score_events(all_events, close, volume, atr_arr, active_lengths, weights):
    """Causal scoring: only uses data up to each event's confirm_idx."""
    n = len(close)
    # group by (bar_idx, kind) approx bucket for cross-scale agreement lookup
    by_confirm = sorted(all_events, key=lambda e: e[1])
    scored = []
    avg_vol_series = pd.Series(volume).rolling(20, min_periods=5).mean().to_numpy()
    avg_atr_series = pd.Series(atr_arr).rolling(20, min_periods=5).mean().to_numpy()

    events_full = []
    for e, scale_len in all_events:
        events_full.append((e[0], e[1], e[2], e[3], scale_len))

    for bar_idx, confirm_idx, kind, price, scale_len in events_full:
        if confirm_idx >= n:
            continue
        atr_c = atr_arr[confirm_idx] if atr_arr[confirm_idx] > 0 else np.nan
        if not np.isfinite(atr_c) or atr_c == 0:
            continue

        lookback_start = max(0, bar_idx - scale_len)
        ref_price = close[lookback_start]
        swing_range = abs(price - ref_price)
        range_score = min(swing_range / atr_c, 5.0) / 5.0 * 100

        vseg = volume[bar_idx:confirm_idx + 1]
        v_avg = avg_vol_series[confirm_idx] if avg_vol_series[confirm_idx] > 0 else np.nan
        volume_score = 50.0
        if np.isfinite(v_avg) and v_avg > 0 and len(vseg) > 0:
            volume_score = min(vseg.mean() / v_avg, 3.0) / 3.0 * 100

        a_avg = avg_atr_series[confirm_idx] if avg_atr_series[confirm_idx] > 0 else np.nan
        volatility_score = 50.0
        if np.isfinite(a_avg) and a_avg > 0:
            volatility_score = min(atr_c / a_avg, 3.0) / 3.0 * 100

        # cross-scale agreement: other-scale pivots of same kind near this bar
        tol = max(scale_len, 5)
        agree = 0
        for (b2, c2, k2, p2), l2 in all_events:
            if l2 == scale_len:
                continue
            if k2 == kind and abs(b2 - bar_idx) <= tol and c2 <= confirm_idx + tol:
                agree += 1
        n_other = max(len(active_lengths) - 1, 1)
        agreement_score = min(agree / n_other, 1.0) * 100

        w_r, w_v, w_vol, w_a = weights
        total_score = (w_r * range_score + w_v * volume_score +
                       w_vol * volatility_score + w_a * agreement_score)

        move_strength = abs(close[confirm_idx] - price) / atr_c

        scored.append(PivotEvent(bar_idx, confirm_idx, kind, price, scale_len,
                                  total_score, move_strength))

    scored.sort(key=lambda e: e.confirm_idx)
    return scored


def compute_smart_swing_vwap(df: pd.DataFrame, p: EngineParams) -> dict:
    n = len(df)
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)
    volume = df["Volume"].to_numpy(dtype=float)
    typical = (high + low + close) / 3.0
    atr_arr = _atr(high, low, close, p.atr_period)

    active_lengths = [L for L, on in zip(p.scales, p.active_scales) if on]
    if not active_lengths:
        active_lengths = list(p.scales)

    all_events = []
    for L in active_lengths:
        if n <= 2 * L + 2:
            continue
        for ev in _find_pivots(high, low, L):
            all_events.append((ev, L))

    scored_events = _score_events(all_events, close, volume, atr_arr,
                                   active_lengths, p.score_weights)

    # ---- causal anchor selection ----
    anchors = []  # list of dicts: start_bar, start_confirm, kind, price
    last_anchor_confirm = -10**9
    for ev in scored_events:
        if ev.score < p.min_anchor_score:
            continue
        if ev.move_strength < p.min_move_strength:
            continue
        if ev.confirm_idx - last_anchor_confirm < p.bars_between_anchors:
            continue
        anchors.append(ev)
        last_anchor_confirm = ev.confirm_idx

    vwap = np.full(n, np.nan)
    trend = np.array([""] * n, dtype=object)
    structural_level = np.full(n, np.nan)
    anchor_bar_arr = np.full(n, -1, dtype=int)
    new_anchor_flag = np.zeros(n, dtype=bool)

    avg_atr_series = pd.Series(atr_arr).rolling(20, min_periods=5).mean().to_numpy()

    # merged pivots (all scales), kept with their scale length so structure
    # tracking can require "same-or-larger granularity than the anchor" -
    # using every scale (including the smallest) here causes constant
    # whipsaw since short-scale pivots sit very close to current price.
    all_pivots_sorted = sorted([(e[0][0], e[0][1], e[0][2], e[0][3], e[1]) for e in all_events],
                                key=lambda x: x[1])

    if not anchors:
        segments = [(0, n - 1, None)]
    else:
        segments = []
        for i, a in enumerate(anchors):
            seg_start = a.bar_idx
            seg_end = anchors[i + 1].bar_idx - 1 if i + 1 < len(anchors) else n - 1
            segments.append((seg_start, seg_end, a))
        if anchors[0].bar_idx > 0:
            segments.insert(0, (0, anchors[0].bar_idx - 1, None))

    for seg_start, seg_end, anchor in segments:
        seg_start = max(0, seg_start)
        seg_end = min(n - 1, seg_end)
        if seg_end < seg_start:
            continue

        if anchor is not None:
            anchor_bar_arr[seg_start:seg_end + 1] = anchor.bar_idx
            if anchor.confirm_idx < n:
                new_anchor_flag[anchor.confirm_idx] = True
            state = "bull" if anchor.kind == "low" else "bear"
            # structure is only tracked using pivots at least as large as the
            # anchor's own scale, to avoid whipsaw from smaller-scale noise
            min_scale = anchor.scale_len

            # initial structural level: most extreme opposite pivot before anchor
            opp_kind = "high" if anchor.kind == "low" else "low"
            prior_opp = [pv for pv in all_pivots_sorted
                         if pv[2] == opp_kind and pv[1] <= anchor.confirm_idx
                         and pv[4] >= min_scale]
            cur_struct = np.nan
            if prior_opp:
                prices = [pv[3] for pv in prior_opp[-3:]]
                cur_struct = max(prices) if opp_kind == "high" else min(prices)

            num = 0.0
            den = 0.0
            for t in range(seg_start, seg_end + 1):
                # decay factor, optionally volatility adjusted
                speed = p.price_tracking_speed
                if p.vol_adjust and avg_atr_series[t] > 0 and np.isfinite(avg_atr_series[t]):
                    atr_ratio = atr_arr[t] / avg_atr_series[t] if avg_atr_series[t] else 1.0
                    speed = speed / max(1e-6, (1 + p.vol_adjust_strength * (atr_ratio - 1)))
                    speed = float(np.clip(speed, 2, p.price_tracking_speed * 3))
                decay = max(0.0, 1 - 1.0 / max(speed, 1e-6))

                num = typical[t] * volume[t] + decay * num
                den = volume[t] + decay * den
                vwap[t] = num / den if den > 0 else typical[t]

                # update structural level causally with newly confirmed opposite pivots
                for pv in all_pivots_sorted:
                    if pv[2] == opp_kind and pv[1] == t and pv[4] >= min_scale:
                        if np.isnan(cur_struct):
                            cur_struct = pv[3]
                        elif opp_kind == "high":
                            cur_struct = max(cur_struct, pv[3])
                        else:
                            cur_struct = min(cur_struct, pv[3])

                structural_level[t] = cur_struct
                trend[t] = state

                if np.isfinite(cur_struct):
                    if state == "bull" and close[t] < cur_struct:
                        state = "bear"
                        opp_kind = "low"
                        cur_struct = np.nan
                        prior_lows = [pv[3] for pv in all_pivots_sorted
                                      if pv[2] == "low" and pv[1] <= t and pv[4] >= min_scale]
                        if prior_lows:
                            cur_struct = min(prior_lows[-3:])
                    elif state == "bear" and close[t] > cur_struct:
                        state = "bull"
                        opp_kind = "high"
                        cur_struct = np.nan
                        prior_highs = [pv[3] for pv in all_pivots_sorted
                                       if pv[2] == "high" and pv[1] <= t and pv[4] >= min_scale]
                        if prior_highs:
                            cur_struct = max(prior_highs[-3:])
        else:
            for t in range(seg_start, seg_end + 1):
                vwap[t] = typical[t]
                trend[t] = ""

    # ---- structure break + retest signal extraction (causal) ----
    structure_break = np.array([""] * n, dtype=object)
    retest_signal = np.array([""] * n, dtype=object)
    armed = False
    armed_dir = 0
    for t in range(1, n):
        if trend[t] and trend[t] != trend[t - 1] and trend[t - 1] != "":
            structure_break[t] = "Structure Break Bull" if trend[t] == "bull" else "Structure Break Bear"
            armed = False

        if np.isnan(atr_arr[t]) or atr_arr[t] == 0 or np.isnan(vwap[t]):
            continue
        dist = (close[t] - vwap[t]) / atr_arr[t]
        if not armed and abs(dist) > p.retest_setup_dist_atr:
            armed = True
            armed_dir = 1 if dist > 0 else -1
        elif armed and abs(dist) <= p.retest_touch_zone_atr and np.sign(dist or 1) == armed_dir:
            if trend[t] == "bull":
                retest_signal[t] = "Retest Support"
            elif trend[t] == "bear":
                retest_signal[t] = "Retest Resistance"
            armed = False

    return {
        "vwap": vwap,
        "trend": trend,
        "structural_level": structural_level,
        "structure_break": structure_break,
        "retest_signal": retest_signal,
        "new_anchor": new_anchor_flag,
        "anchor_bar": anchor_bar_arr,
        "atr": atr_arr,
        "anchors": anchors,
    }


# ============================================================================
# SCANNING
# ============================================================================

def scan_symbol(symbol, df, params: EngineParams, mode, as_of_date=None,
                 range_start=None, range_end=None, wanted_signals=None):
    if len(df) < max(params.scales) * 2 + 20:
        return []

    result = compute_smart_swing_vwap(df, params)
    dates = df.index
    close = df["Close"].to_numpy()
    n = len(df)

    rows = []

    def emit(idx, sig_type):
        rows.append({
            "Symbol": symbol,
            "Signal": sig_type,
            "Date": dates[idx].strftime("%Y-%m-%d"),
            "Close": round(float(close[idx]), 2),
            "VWAP": round(float(result["vwap"][idx]), 2) if np.isfinite(result["vwap"][idx]) else None,
            "Dist(ATR)": round(float((close[idx] - result["vwap"][idx]) / result["atr"][idx]), 2)
                          if result["atr"][idx] else None,
            "Trend": result["trend"][idx],
        })

    if mode == "Latest (Today)":
        idx = n - 1
        for sig, arr in (("Structure Break Bull", result["structure_break"]),
                          ("Structure Break Bear", result["structure_break"]),
                          ("Retest Support", result["retest_signal"]),
                          ("Retest Resistance", result["retest_signal"]),
                          ("New Anchor", None)):
            if sig == "New Anchor":
                if result["new_anchor"][idx] and (not wanted_signals or sig in wanted_signals):
                    emit(idx, sig)
            elif arr is not None and arr[idx] == sig and (not wanted_signals or sig in wanted_signals):
                emit(idx, sig)

    elif mode == "As-of Specific Date":
        target = pd.Timestamp(as_of_date)
        valid_idx = np.where(dates <= target)[0]
        if len(valid_idx) == 0:
            return []
        idx = valid_idx[-1]
        for sig, arr in (("Structure Break Bull", result["structure_break"]),
                          ("Structure Break Bear", result["structure_break"]),
                          ("Retest Support", result["retest_signal"]),
                          ("Retest Resistance", result["retest_signal"])):
            if arr[idx] == sig and (not wanted_signals or sig in wanted_signals):
                emit(idx, sig)
        if result["new_anchor"][idx] and (not wanted_signals or "New Anchor" in wanted_signals):
            emit(idx, "New Anchor")

    else:  # Historical Range (all signals)
        rstart = pd.Timestamp(range_start)
        rend = pd.Timestamp(range_end)
        for idx in range(n):
            if not (rstart <= dates[idx] <= rend):
                continue
            sb = result["structure_break"][idx]
            rs = result["retest_signal"][idx]
            if sb and (not wanted_signals or sb in wanted_signals):
                emit(idx, sb)
            if rs and (not wanted_signals or rs in wanted_signals):
                emit(idx, rs)
            if result["new_anchor"][idx] and (not wanted_signals or "New Anchor" in wanted_signals):
                emit(idx, "New Anchor")

    return rows


def run_scan(data: dict, params: EngineParams, mode, as_of_date, range_start,
             range_end, wanted_signals):
    all_rows = []
    progress = st.progress(0.0, text="Scanning...")
    symbols = list(data.keys())
    for i, sym in enumerate(symbols):
        try:
            rows = scan_symbol(sym, data[sym], params, mode, as_of_date,
                                range_start, range_end, wanted_signals)
            all_rows.extend(rows)
        except Exception as e:
            pass
        progress.progress((i + 1) / len(symbols), text=f"Scanning... ({i+1}/{len(symbols)})")
    progress.empty()
    if not all_rows:
        return pd.DataFrame()
    out = pd.DataFrame(all_rows)
    out = out.sort_values(["Date", "Symbol"], ascending=[False, True]).reset_index(drop=True)
    return out


# ============================================================================
# CHARTING
# ============================================================================

def plot_symbol(symbol, df, params: EngineParams):
    result = compute_smart_swing_vwap(df, params)
    dates = df.index

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=dates, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name=symbol, increasing_line_color="#26a69a", decreasing_line_color="#ef5350"
    ))

    trend = result["trend"]
    vwap = result["vwap"]
    seg_start = 0
    for i in range(1, len(dates) + 1):
        if i == len(dates) or trend[i] != trend[seg_start]:
            color = "#26a69a" if trend[seg_start] == "bull" else (
                "#ef5350" if trend[seg_start] == "bear" else "#888888")
            fig.add_trace(go.Scatter(
                x=dates[seg_start:i], y=vwap[seg_start:i], mode="lines",
                line=dict(color=color, width=2), showlegend=False
            ))
            seg_start = i

    anchor_idxs = sorted(set(result["anchor_bar"][result["anchor_bar"] >= 0]))
    if len(anchor_idxs):
        fig.add_trace(go.Scatter(
            x=dates[anchor_idxs], y=df["Low"].to_numpy()[anchor_idxs] * 0.98,
            mode="markers", marker=dict(symbol="triangle-up", size=9, color="gold"),
            name="Anchor"
        ))

    retest_idx = np.where(result["retest_signal"] != "")[0]
    if len(retest_idx):
        fig.add_trace(go.Scatter(
            x=dates[retest_idx], y=df["Close"].to_numpy()[retest_idx],
            mode="markers", marker=dict(symbol="circle", size=7, color="cyan",
                                         line=dict(width=1, color="black")),
            name="Retest"
        ))

    fig.update_layout(
        title=f"{symbol} - Smart Swing VWAP",
        xaxis_rangeslider_visible=False,
        height=560,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ============================================================================
# STREAMLIT UI
# ============================================================================

st.set_page_config(page_title="Smart Swing VWAP Scanner", layout="wide")
st.title("📊 Smart Swing VWAP Scanner — NSE")
st.caption("Multi-scale swing anchoring · decayed volume-weighted VWAP · "
           "structure break & retest detection · historical scanning")

with st.sidebar:
    st.header("Universe")
    universe_choice = st.selectbox(
        "Stock list", ["NIFTY 50", "NIFTY 200", "NIFTY 500", "Custom (upload CSV)"],
        index=0,
    )
    uploaded_file = None
    if universe_choice == "Custom (upload CSV)":
        uploaded_file = st.file_uploader("CSV with a 'Symbol' column", type=["csv"])

    st.header("Data Range")
    lookback_years = st.slider("Years of history to fetch", 1, 5, 2)
    end_date = st.date_input("End date", value=datetime.now().date())
    start_date = end_date - timedelta(days=int(lookback_years * 365.25))

    st.header("Swing Scales")
    scale_defaults = [5, 10, 20, 40, 80]
    active_scales = []
    scale_lengths = []
    cols = st.columns(5)
    for i, (c, default_len) in enumerate(zip(cols, scale_defaults)):
        with c:
            on = st.checkbox(f"S{i+1}", value=True, key=f"scale_on_{i}")
            length = st.number_input("len", min_value=2, max_value=200,
                                      value=default_len, key=f"scale_len_{i}",
                                      label_visibility="collapsed")
            active_scales.append(on)
            scale_lengths.append(int(length))

    st.header("Anchor Rules")
    min_anchor_score = st.slider("Minimum Anchor Score", 0, 100, 60)
    min_move_strength = st.slider("Minimum Move Strength (ATR)", 0.0, 5.0, 1.0, 0.1)
    bars_between_anchors = st.slider("Bars Between New Anchors", 2, 60, 10)

    st.header("Adaptive VWAP")
    price_tracking_speed = st.slider("Price Tracking Speed (bars)", 5, 100, 20)
    vol_adjust = st.checkbox("Adjust Speed With Volatility", value=True)
    vol_adjust_strength = st.slider("Volatility Adjustment Strength", 0.0, 3.0, 1.0, 0.1)

    st.header("Retest")
    retest_touch_zone = st.slider("Retest Touch Zone (ATR)", 0.05, 1.5, 0.3, 0.05)
    retest_setup_dist = st.slider("Retest Setup Distance (ATR)", 0.2, 3.0, 1.0, 0.1)

    st.header("Scan Mode")
    scan_mode = st.radio("Mode", ["Latest (Today)", "As-of Specific Date",
                                   "Historical Range (all signals)"])
    as_of_date = None
    range_start = None
    range_end = None
    if scan_mode == "As-of Specific Date":
        as_of_date = st.date_input("As-of date", value=end_date)
    elif scan_mode == "Historical Range (all signals)":
        range_start = st.date_input("Range start", value=end_date - timedelta(days=180))
        range_end = st.date_input("Range end", value=end_date)

    wanted_signals = st.multiselect("Signal types", SIGNAL_TYPES, default=SIGNAL_TYPES)

    run_button = st.button("🚀 Run Scan", type="primary", use_container_width=True)

params = EngineParams(
    scales=tuple(scale_lengths),
    active_scales=tuple(active_scales),
    min_anchor_score=float(min_anchor_score),
    min_move_strength=float(min_move_strength),
    bars_between_anchors=int(bars_between_anchors),
    price_tracking_speed=int(price_tracking_speed),
    vol_adjust=bool(vol_adjust),
    vol_adjust_strength=float(vol_adjust_strength),
    retest_touch_zone_atr=float(retest_touch_zone),
    retest_setup_dist_atr=float(retest_setup_dist),
)

if "scan_results" not in st.session_state:
    st.session_state.scan_results = None
    st.session_state.scan_data = None

if run_button:
    symbols = get_universe(universe_choice, uploaded_file)
    if not symbols:
        st.error("No symbols to scan. Check the universe selection / upload.")
    else:
        st.info(f"Scanning {len(symbols)} symbols from {universe_choice} "
                f"({start_date} → {end_date})")
        data = fetch_universe_data(symbols, start_date.isoformat(), end_date.isoformat())
        st.session_state.scan_data = data
        if not data:
            st.error("No price data could be fetched for this universe.")
        else:
            results = run_scan(data, params, scan_mode, as_of_date, range_start,
                                range_end, wanted_signals)
            st.session_state.scan_results = results

results = st.session_state.scan_results
data = st.session_state.scan_data

if results is not None:
    if results.empty:
        st.warning("No signals found for the selected criteria.")
    else:
        st.success(f"{len(results)} signal(s) found across {results['Symbol'].nunique()} symbol(s).")

        c1, c2 = st.columns([3, 1])
        with c1:
            sig_filter = st.multiselect("Filter by signal", sorted(results["Signal"].unique()),
                                         default=sorted(results["Signal"].unique()))
        with c2:
            st.write("")
            st.write("")

        filtered = results[results["Signal"].isin(sig_filter)]
        st.dataframe(filtered, use_container_width=True, height=420)

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            filtered.to_excel(writer, index=False, sheet_name="Signals")
        st.download_button("⬇️ Download results (Excel)", data=buf.getvalue(),
                            file_name=f"smart_swing_vwap_scan_{datetime.now():%Y%m%d_%H%M}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        st.subheader("Chart Inspector")
        symbol_pick = st.selectbox("Symbol", sorted(filtered["Symbol"].unique())
                                    if not filtered.empty else sorted(results["Symbol"].unique()))
        if data and symbol_pick in data:
            st.plotly_chart(plot_symbol(symbol_pick, data[symbol_pick], params),
                             use_container_width=True)
else:
    st.info("Configure the universe and rules in the sidebar, then click **Run Scan**.")
