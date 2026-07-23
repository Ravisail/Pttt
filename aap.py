"""
NSE Weekly Swing Signal Scanner & Backtester
=============================================
Single-file Streamlit application.

Strategy
--------
A signal is created when ALL weekly conditions become TRUE on a stock:
  1. Weekly Close > Weekly EMA30
  2. Weekly EMA30 > Previous Week EMA30 (rising)
  3. Weekly EMA10 > Weekly EMA30
  4. Weekly Close > Weekly EMA10
  5. Weekly Close >= Highest Weekly High of previous 52 weeks * 0.95
  6. Weekly Volume > Weekly EMA(Volume, 60) * 1.30
  7. Current Daily Volume > 100,000

No buy occurs on the signal week. The stock is bought only when a later
Daily Close crosses ABOVE the Signal Week High, with entry executed at the
NEXT trading day's Open. The stored Signal Week High is fixed until either
a buy occurs or the signal is invalidated (Weekly Close < Weekly EMA30,
OR Weekly EMA10 < Weekly EMA30, OR the waiting period exceeds 60 trading
days). Only one signal / one trade is tracked per stock at a time.

Exit rule (clarified with user): a trend-break exit -- exit at the next
trading day's Open once a completed weekly bar shows Weekly Close < Weekly
EMA30. This mirrors the entry's trend logic and introduces no look-ahead.

Position sizing (clarified with user): Position Size is a fixed Rupee
amount applied independently to every triggered trade. Capital is used
only to compute Total Return % / drawdown on the realised trade sequence.

The engine is strictly causal: every decision on trading day D only uses
data that would actually be known as of D (weekly bars are only actioned
on the first trading day AFTER the week closes).
"""

import io
import time
import logging
import traceback
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

# =====================================================================
# LOGGING
# =====================================================================
LOG_STREAM = io.StringIO()
logger = logging.getLogger("nse_scanner")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler(LOG_STREAM)
    _handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
    logger.addHandler(_handler)

# =====================================================================
# PAGE CONFIG / STYLE
# =====================================================================
st.set_page_config(
    page_title="NSE Weekly Swing Scanner & Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    div[data-testid="stMetricValue"] {font-size: 1.4rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {
        padding: 8px 18px; border-radius: 8px 8px 0 0; font-weight: 600;
    }
    .signal-caption {opacity: 0.75; font-size: 0.85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================================
# CONSTANTS
# =====================================================================
MAX_DOWNLOAD_WORKERS = 12
MAX_NAME_WORKERS = 8
MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 0.6
CACHE_TTL_SECONDS = 3600
NAME_CACHE_TTL_SECONDS = 86400

STATE_NONE = "NONE"
STATE_SIGNAL_ACTIVE = "SIGNAL_ACTIVE"
STATE_PENDING_ENTRY = "PENDING_ENTRY"
STATE_IN_TRADE = "IN_TRADE"
STATE_PENDING_EXIT = "PENDING_EXIT"

# =====================================================================
# UNIVERSE — built-in curated NSE symbol lists (no ".NS" suffix here;
# appended automatically at download time). Edit these lists to adjust
# the universe. Duplicates are removed automatically.
# =====================================================================

NIFTY_200 = [
    # --- Nifty 50 + large caps ---
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "BHARTIARTL", "ITC", "SBIN",
    "HINDUNILVR", "LT", "HCLTECH", "BAJFINANCE", "KOTAKBANK", "MARUTI", "SUNPHARMA",
    "AXISBANK", "M&M", "ULTRACEMCO", "NTPC", "TITAN", "ADANIENT", "ONGC", "TATAMOTORS",
    "ADANIPORTS", "ASIANPAINT", "BAJAJFINSV", "POWERGRID", "COALINDIA", "NESTLEIND",
    "WIPRO", "JSWSTEEL", "TATASTEEL", "BAJAJ-AUTO", "HINDALCO", "GRASIM", "SBILIFE",
    "TECHM", "HDFCLIFE", "DRREDDY", "CIPLA", "INDUSINDBK", "EICHERMOT", "APOLLOHOSP",
    "DIVISLAB", "BRITANNIA", "BPCL", "TATACONSUM", "HEROMOTOCO", "UPL", "SHRIRAMFIN",
    # --- Nifty Next 50 (approx) ---
    "LTIM", "ADANIGREEN", "ADANIPOWER", "AMBUJACEM", "DLF", "DABUR", "GODREJCP",
    "PIDILITIND", "HAVELLS", "SIEMENS", "VEDL", "BANKBARODA", "PNB", "CANBK", "IOC",
    "GAIL", "INDIGO", "ZOMATO", "NAUKRI", "TVSMOTOR", "BOSCHLTD", "CHOLAFIN",
    "ICICIPRULI", "ICICIGI", "SBICARD", "MARICO", "COLPAL", "BERGEPAINT", "MUTHOOTFIN",
    "LUPIN", "TORNTPHARM", "ALKEM", "AUROPHARMA", "BIOCON", "MOTHERSON", "BEL", "HAL",
    "BHEL", "SAIL", "NMDC", "JINDALSTEL", "RECLTD", "PFC", "IRFC", "CONCOR", "INDHOTEL",
    "TRENT", "PAGEIND", "ABB", "CUMMINSIND",
    # --- Additional Nifty 200 constituents (midcaps) ---
    "POLYCAB", "ASHOKLEY", "MRF", "BALKRISIND", "SRF", "PIIND", "DEEPAKNTR",
    "TATAPOWER", "TATACOMM", "IDFCFIRSTB", "FEDERALBNK", "BANDHANBNK", "AUBANK",
    "YESBANK", "RBLBANK", "PERSISTENT", "MPHASIS", "COFORGE", "LTTS", "OFSS",
    "OBEROIRLTY", "GODREJPROP", "PHOENIXLTD", "PRESTIGE", "LODHA", "IGL", "PETRONET",
    "GUJGASLTD", "ACC", "JKCEMENT", "RAMCOCEM", "DALBHARAT", "ESCORTS", "BHARATFORG",
    "EXIDEIND", "SUPREMEIND", "ASTRAL", "VOLTAS", "WHIRLPOOL", "CROMPTON", "DIXON",
    "AMBER", "KPITTECH", "TATAELXSI", "ZEEL", "SUNTV", "PVRINOX", "JUBLFOOD", "UBL",
    "RADICO", "EMAMILTD", "GILLETTE", "POLICYBZR", "DMART", "METROPOLIS", "LALPATHLAB",
    "FORTIS", "MAXHEALTH", "GLAND", "LAURUSLABS", "GRANULES", "IPCALAB", "NATCOPHARM",
    "ABBOTINDIA", "SANOFI", "PFIZER", "GLAXO", "IDEA", "TATACHEM", "GNFC", "CHAMBLFERT",
    "COROMANDEL", "GSFC", "NATIONALUM", "HINDZINC", "HINDCOPPER", "RATNAMANI",
    "WELCORP", "JSL", "APLAPOLLO", "KEI", "FINCABLES", "CGPOWER", "THERMAX", "HONAUT",
    "KSB", "GRINDWELL", "CARBORUNIV", "SCHAEFFLER", "TIMKEN", "GRAPHITE", "HEG",
    "NAVINFLUOR", "AARTIIND", "VINATIORGA", "ATUL", "FINEORG", "GALAXYSURF",
    "CLEAN", "ALKYLAMINE",
]

NIFTY_500_EXTRA = [
    "CANFINHOME", "LICHSGFIN", "PNBHOUSING", "HUDCO", "IIFL", "MANAPPURAM",
    "CREDITACC", "POONAWALLA", "EQUITASBNK", "UJJIVANSFB", "DCBBANK", "SOUTHBANK",
    "KARURVYSYA", "CUB", "J&KBANK", "BANKINDIA", "CENTRALBK", "UCOBANK", "MAHABANK",
    "PSB", "INDIANB", "KTKBANK", "JBCHEPHARM", "AJANTPHARM", "ERIS", "MANKIND",
    "ZYDUSLIFE", "TORNTPOWER", "JSWENERGY", "NHPC", "SJVN", "RVNL", "IRCON",
    "RAILTEL", "IRCTC", "RITES", "MAZDOCK", "COCHINSHIP", "GRSE", "BDL", "MIDHANI",
    "DATAPATTNS", "PARAS", "ZENTEC", "CYIENT", "KPIT", "SONATSOFTW", "NEWGEN",
    "INTELLECT", "MASTEK", "RATEGAIN", "HAPPSTMNDS", "LATENTVIEW", "TANLA", "ROUTE",
    "SAREGAMA", "NETWORK18", "TV18BRDCST", "DISHTV", "HATHWAY", "GTPL", "NAZARA",
    "DELTACORP", "WESTLIFE", "SAPPHIRE", "DEVYANI", "EASEMYTRIP", "YATRA", "IXIGO",
    "MSTC", "MMTC", "NBCC", "NLCINDIA", "NEYVELI", "PTC", "TATAINVEST", "IFCI",
    "SUNDARMFIN", "MAHINDCIE", "BALRAMCHIN", "DHAMPURSUG", "TRIVENI", "EIDPARRY",
    "RENUKA", "DALMIASUG", "UGARSUGAR", "KRBL", "LTFOODS", "HERITGFOOD", "VADILAL",
    "PARAGMILK", "HATSUN", "AVANTIFEED", "GODFRYPHLP", "VSTIND", "RAJESHEXPO",
    "KALYANKJIL", "PCJEWELLER", "THANGAMAYL", "TBZ", "CAMPUS", "METRO", "RELAXO",
    "BATA", "LIBERTSHOE", "VIPIND", "SAFARI", "WONDERLA", "INOXWIND", "SUZLON",
    "WAAREE", "BOROSIL", "SALASAR", "GENUSPOWER", "VOLTAMP", "TDPOWERSYS", "GMRINFRA",
    "IRB", "ASHOKA", "PNCINFRA", "KNRCON", "HGINFRA", "DBL", "NCC", "GPIL",
    "WELSPUNIND", "WELSPUNLIV", "GARFIBRES", "RUPA", "DOLLAR", "KITEX", "TRIDENT",
    "VARDHMAN", "SUTLEJTEX", "NAHARSPING", "RSWM", "JKPAPER", "WESTCOAST",
    "TNPL", "ANDHRAPAP", "EMAMIPAP", "STARCEMENT", "HEIDELBERG", "PRISMCEM",
    "JKLAKSHMI", "ORIENTCEM", "INDIACEM", "BIRLACORPN", "SAGCEM", "KESORAMIND",
    "VISAKAIND", "SHREECEM", "CENTURYPLY", "GREENPLY", "KAJARIACER", "SOMANYCERA",
    "CERA", "HSIL", "VARUNBEV", "JUBLINGREA", "GOODRICKE", "EIHOTEL", "LEMONTREE",
    "CHALET", "ROHLTD", "TAJGVK", "ADVANIHOTR", "THOMASCOOK", "TCI", "VRL",
    "ALLCARGO", "GATI", "TCIEXP", "SNOWMAN", "MAHLOG", "BLUEDART", "DELHIVERY",
    "SMLISUZU", "FORCEMOT", "ATULAUTO", "OLECTRA", "GREAVESCOT", "SUNDRMFAST",
    "SUNDRMBRAK", "ENDURANCE", "MINDAIND", "SUBROS", "VARROC", "WABAG", "GABRIEL",
    "JAMNAAUTO", "LUMAXTECH", "FIEMIND", "MUNJALSHOW", "RICOAUTO", "TALBROS",
]

NIFTY_500 = sorted(set(NIFTY_200) | set(NIFTY_500_EXTRA))
NIFTY_200 = sorted(set(NIFTY_200))

UNIVERSE_MAP = {
    "NIFTY 200": NIFTY_200,
    "NIFTY 500": NIFTY_500,
}

# =====================================================================
# DATA LAYER — multithreaded, retrying, cached downloads
# =====================================================================


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _download_symbol(symbol_ns: str, start_str: str, end_str: str) -> pd.DataFrame:
    """Download daily OHLCV for one Yahoo-suffixed symbol, with retries."""
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = yf.download(
                symbol_ns,
                start=start_str,
                end=end_str,
                interval="1d",
                progress=False,
                auto_adjust=False,
                threads=False,
            )
            if df is None or df.empty:
                raise ValueError("empty dataframe")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            if df.empty:
                raise ValueError("no valid rows after cleaning")
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SEC * attempt)
    raise RuntimeError(f"{symbol_ns}: failed after {MAX_RETRIES} attempts ({last_err})")


def download_universe(symbols, start_str, end_str, progress_cb=None):
    """Download many symbols in parallel. Returns (data_dict, errors_list)."""
    results, errors = {}, []
    total = len(symbols)
    done = 0
    with ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_WORKERS) as ex:
        futures = {ex.submit(_download_symbol, f"{s}.NS", start_str, end_str): s for s in symbols}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                results[sym] = fut.result()
                logger.info("Downloaded %s (%d rows)", sym, len(results[sym]))
            except Exception as exc:  # noqa: BLE001
                errors.append((sym, str(exc)))
                logger.warning("Failed %s: %s", sym, exc)
            done += 1
            if progress_cb:
                progress_cb(done, total)
    return results, errors


@st.cache_data(ttl=NAME_CACHE_TTL_SECONDS, show_spinner=False)
def _get_company_name(symbol: str) -> str:
    try:
        info = yf.Ticker(f"{symbol}.NS").info
        return info.get("shortName") or info.get("longName") or symbol
    except Exception:  # noqa: BLE001
        return symbol


def fetch_company_names(symbols):
    names = {}
    if not symbols:
        return names
    with ThreadPoolExecutor(max_workers=MAX_NAME_WORKERS) as ex:
        futures = {ex.submit(_get_company_name, s): s for s in symbols}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                names[sym] = fut.result()
            except Exception:  # noqa: BLE001
                names[sym] = sym
    return names


# =====================================================================
# WEEKLY INDICATORS
# =====================================================================


def resample_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    weekly = daily.resample("W-FRI").agg(agg).dropna(subset=["Close"])
    weekly["LastDailyVolume"] = daily["Volume"].resample("W-FRI").last()
    return weekly


def compute_weekly_indicators(weekly, ema10_p, ema30_p, volema_p, lookback_weeks,
                               near_high_frac, vol_mult, min_daily_vol):
    w = weekly.copy()
    w["EMA10"] = w["Close"].ewm(span=ema10_p, adjust=False).mean()
    w["EMA30"] = w["Close"].ewm(span=ema30_p, adjust=False).mean()
    w["EMA30_prev"] = w["EMA30"].shift(1)
    w["VolEMA"] = w["Volume"].ewm(span=volema_p, adjust=False).mean()
    w["Roll52HighPrev"] = w["High"].rolling(lookback_weeks).max().shift(1)
    w["VolRatio"] = w["Volume"] / w["VolEMA"].replace(0, np.nan)

    c1 = w["Close"] > w["EMA30"]
    c2 = w["EMA30"] > w["EMA30_prev"]
    c3 = w["EMA10"] > w["EMA30"]
    c4 = w["Close"] > w["EMA10"]
    c5 = w["Close"] >= (w["Roll52HighPrev"] * near_high_frac)
    c6 = w["Volume"] > (w["VolEMA"] * vol_mult)
    c7 = w["LastDailyVolume"] > min_daily_vol

    w["SignalRaw"] = c1 & c2 & c3 & c4 & c5 & c6 & c7
    w["InvalidCond"] = (w["Close"] < w["EMA30"]) | (w["EMA10"] < w["EMA30"])
    return w


def last_completed_weekly_row(weekly: pd.DataFrame, daily: pd.DataFrame):
    """
    Return the most recent FULLY COMPLETED weekly bar for display purposes.

    resample('W-FRI') labels the current, still-forming week with a Friday
    date even if today is only Tuesday -- that row's EMA/Volume-ratio values
    would keep changing every day (a display-level repaint) even though the
    signal engine itself never *acts* on it (run_engine only actions a
    weekly bar once a daily bar exists after that Friday). This helper keeps
    the on-screen indicator values consistent with what the engine actually
    used, so the Scanner and Backtest never show different numbers for the
    same week.
    """
    if weekly.empty:
        return None
    last_daily_date = daily.index[-1]
    if weekly.index[-1] <= last_daily_date:
        return weekly.iloc[-1]
    if len(weekly) >= 2:
        return weekly.iloc[-2]
    return None


def dynamic_history_days(params: dict, minimum: int) -> int:
    """
    Size the download lookback window to whatever the user's configured
    indicator periods actually need (52-week lookback, EMA30, Volume EMA),
    instead of a fixed constant that could under-warm the indicators if the
    user raises these periods well above their defaults.
    """
    weeks_needed = max(params["lookback"], params["ema30"], params["volema"])
    est_days = int(weeks_needed * 7 * 1.6) + 200  # generous margin for holidays/non-trading days
    return max(est_days, minimum)


# =====================================================================
# STRATEGY ENGINE — causal state machine (no look-ahead)
# =====================================================================


def run_engine(daily: pd.DataFrame, weekly: pd.DataFrame, position_size: float, max_waiting_days: int,
               cost_pct: float = 0.0, slippage_pct: float = 0.0):
    """
    Walk daily bars chronologically. Weekly bars are only actioned on the
    first daily bar strictly AFTER that week has closed (their natural
    'confirmation' date) -- this is what prevents look-ahead / repainting.

    Two checks below are written as EXPLICIT invariants rather than relying
    on the state machine's implicit guarantees, purely for auditability:

    - Breakout is a true crossover: `prev_close <= signal_high AND
      today_close > signal_high`, not just `today_close > signal_high`.
      (Mathematically these are equivalent here, since state only remains
      SIGNAL_ACTIVE while close<=high held on every prior day it was
      checked -- the moment it doesn't, state leaves SIGNAL_ACTIVE. But
      making the crossover explicit removes any reliance on that reasoning
      holding up under future refactors.)
    - A new signal is only opened on a False->True transition of SignalRaw
      relative to the immediately preceding evaluated week (also already
      implied by only checking SignalRaw while state == NONE, since two
      consecutive True weeks while idle would have fired on the first one
      -- again made explicit here rather than implicit.)

    Returns (trades: list[dict], live_state: dict)
    """
    idx = daily.index
    n = len(idx)
    cost_frac = (cost_pct + slippage_pct) / 100.0

    # Map each weekly bar to the daily bar index position on which its
    # data first becomes actionable.
    confirm_map = {}
    for wi, wdate in enumerate(weekly.index):
        pos = idx.searchsorted(wdate, side="right")
        if pos < n:
            confirm_map.setdefault(pos, []).append(wi)

    state = STATE_NONE
    signal = None
    trade = None
    waiting_days = 0
    breakout_date = None
    trades = []
    prev_close = None          # yesterday's daily close, for explicit crossover check
    prev_week_signal_raw = False  # SignalRaw of the last evaluated weekly bar, for explicit False->True check

    for i in range(n):
        d = idx[i]
        row = daily.iloc[i]

        # --- Step 1: action any weekly bar confirmed as of today ---
        for wi in confirm_map.get(i, []):
            wrow = weekly.iloc[wi]
            raw_now = bool(wrow["SignalRaw"])
            if state == STATE_NONE:
                if raw_now and not prev_week_signal_raw:
                    state = STATE_SIGNAL_ACTIVE
                    signal = {
                        "week_date": weekly.index[wi],
                        "high": float(wrow["High"]),
                        "low": float(wrow["Low"]),
                        "close": float(wrow["Close"]),
                        "ema10": float(wrow["EMA10"]),
                        "ema30": float(wrow["EMA30"]),
                    }
                    waiting_days = 0
            elif state == STATE_SIGNAL_ACTIVE:
                if bool(wrow["InvalidCond"]):
                    state = STATE_NONE
                    signal = None
            elif state == STATE_IN_TRADE:
                if wrow["Close"] < wrow["EMA30"]:
                    state = STATE_PENDING_EXIT
            prev_week_signal_raw = raw_now

        # --- Step 2: act on the (possibly just-updated) state ---
        if state == STATE_PENDING_ENTRY:
            entry_price = float(row["Open"]) * (1.0 + cost_frac)
            trade = {
                "Signal Date": signal["week_date"],
                "Breakout Date": breakout_date,
                "Entry Date": d,
                "Entry Price": entry_price,
                "Signal High": signal["high"],
                "entry_i": i,
                "running_high": float(row["High"]),
                "running_low": float(row["Low"]),
                "peak_close": entry_price,
                "max_dd": 0.0,
            }
            state = STATE_IN_TRADE
            signal = None

        elif state == STATE_PENDING_EXIT:
            exit_price = float(row["Open"]) * (1.0 - cost_frac)
            entry_price = trade["Entry Price"]
            profit_pct = (exit_price - entry_price) / entry_price * 100.0
            profit_rs = position_size * profit_pct / 100.0
            holding_days = i - trade["entry_i"]
            mfe = (trade["running_high"] - entry_price) / entry_price * 100.0
            mae = (trade["running_low"] - entry_price) / entry_price * 100.0
            trades.append({
                "Signal Date": trade["Signal Date"],
                "Breakout Date": trade["Breakout Date"],
                "Entry Date": trade["Entry Date"],
                "Entry Price": round(entry_price, 2),
                "Signal High": round(trade["Signal High"], 2),
                "Exit Date": d,
                "Exit Price": round(exit_price, 2),
                "Holding Days": int(holding_days),
                "Profit %": round(profit_pct, 2),
                "Profit ₹": round(profit_rs, 2),
                "Maximum Drawdown": round(trade["max_dd"], 2),
                "Maximum Favourable Excursion": round(mfe, 2),
                "Maximum Adverse Excursion": round(mae, 2),
            })
            state = STATE_NONE
            trade = None

        elif state == STATE_SIGNAL_ACTIVE:
            waiting_days += 1
            if waiting_days > max_waiting_days:
                state = STATE_NONE
                signal = None
            else:
                crossed_above = (prev_close is None or prev_close <= signal["high"]) and row["Close"] > signal["high"]
                if crossed_above:
                    state = STATE_PENDING_ENTRY
                    breakout_date = d

        elif state == STATE_IN_TRADE:
            trade["running_high"] = max(trade["running_high"], float(row["High"]))
            trade["running_low"] = min(trade["running_low"], float(row["Low"]))
            trade["peak_close"] = max(trade["peak_close"], float(row["Close"]))
            dd = (float(row["Close"]) - trade["peak_close"]) / trade["peak_close"] * 100.0
            trade["max_dd"] = min(trade["max_dd"], dd)

        prev_close = float(row["Close"])

    live_state = {"state": state, "signal": signal, "trade": trade,
                  "entry_i": trade["entry_i"] if trade else None, "last_i": n - 1}
    return trades, live_state


def analyze_symbol(symbol, daily_df, params):
    weekly = resample_weekly(daily_df)
    weekly = compute_weekly_indicators(
        weekly,
        params["ema10"], params["ema30"], params["volema"],
        params["lookback"], params["near_high"] / 100.0,
        params["vol_mult"], params["min_vol"],
    )
    trades, live_state = run_engine(
        daily_df, weekly, params["position_size"], params["max_wait"],
        cost_pct=params.get("cost_pct", 0.0), slippage_pct=params.get("slippage_pct", 0.0),
    )
    for t in trades:
        t["Ticker"] = symbol
    return trades, live_state, weekly


# =====================================================================
# SCANNER
# =====================================================================


def run_scanner(universe, params, fetch_names=False, history_days=None):
    history_days = history_days or dynamic_history_days(params, minimum=1460)
    end = datetime.today()
    start = end - timedelta(days=history_days)
    start_str = start.strftime("%Y-%m-%d")
    end_str = (end + timedelta(days=1)).strftime("%Y-%m-%d")

    progress = st.progress(0.0, text="Downloading price history...")

    def _cb(done, total):
        progress.progress(done / total, text=f"Downloading price history... {done}/{total}")

    data, errors = download_universe(universe, start_str, end_str, progress_cb=_cb)
    progress.progress(1.0, text="Evaluating weekly signal conditions...")

    scanner_rows, open_rows = [], []
    for symbol, daily_df in data.items():
        if daily_df is None or len(daily_df) < 80:
            continue
        try:
            _trades, live_state, weekly = analyze_symbol(symbol, daily_df, params)
        except Exception as exc:  # noqa: BLE001
            logger.error("Scan analysis failed for %s: %s\n%s", symbol, exc, traceback.format_exc())
            continue

        st_ = live_state["state"]
        last_close = float(daily_df["Close"].iloc[-1])
        # Use the last FULLY COMPLETED weekly bar for displayed indicator
        # values, never the current still-forming week -- keeps the Scanner
        # from repainting mid-week and keeps it consistent with what the
        # (already causal) signal engine actually used.
        last_w = last_completed_weekly_row(weekly, daily_df)
        if last_w is None:
            continue

        if st_ in (STATE_SIGNAL_ACTIVE, STATE_PENDING_ENTRY):
            sig = live_state["signal"]
            dist = (sig["high"] - last_close) / last_close * 100.0
            if st_ == STATE_PENDING_ENTRY:
                status_label = "\U0001F680 BREAKOUT – Buy Next Open"
                dist = 0.0  # already broken out; no meaningful "distance" remains
            elif dist <= 2:
                status_label = "\U0001F525 Very Close (<2%)"
            elif dist <= 5:
                status_label = "⚠️ Approaching (<5%)"
            else:
                status_label = "\U0001F440 Watching"
            dist = max(0.0, dist)
            scanner_rows.append({
                "Ticker": symbol,
                "Current Price": round(last_close, 2),
                "Signal Date": sig["week_date"].strftime("%Y-%m-%d"),
                "Signal High": round(sig["high"], 2),
                "Current Close": round(last_close, 2),
                "Distance to Breakout %": round(dist, 2),
                "Breakout Status": status_label,
                "Weekly EMA10": round(float(last_w["EMA10"]), 2),
                "Weekly EMA30": round(float(last_w["EMA30"]), 2),
                "Weekly Volume Ratio": round(float(last_w["VolRatio"]), 2) if pd.notna(last_w["VolRatio"]) else None,
            })
        elif st_ in (STATE_IN_TRADE, STATE_PENDING_EXIT):
            tr = live_state["trade"]
            pnl = (last_close - tr["Entry Price"]) / tr["Entry Price"] * 100.0
            open_rows.append({
                "Ticker": symbol,
                "Entry Date": tr["Entry Date"].strftime("%Y-%m-%d"),
                "Entry Price": round(tr["Entry Price"], 2),
                "Current Price": round(last_close, 2),
                "Unrealized P&L %": round(pnl, 2),
                "Status": "Exit pending (next open)" if st_ == STATE_PENDING_EXIT else "Open",
            })

    scanner_df = pd.DataFrame(scanner_rows)
    if not scanner_df.empty:
        scanner_df = scanner_df.sort_values("Distance to Breakout %", ascending=True).reset_index(drop=True)
        if fetch_names:
            names = fetch_company_names(scanner_df["Ticker"].tolist())
            scanner_df.insert(1, "Company", scanner_df["Ticker"].map(names).fillna(scanner_df["Ticker"]))
        else:
            scanner_df.insert(1, "Company", scanner_df["Ticker"])

    open_df = pd.DataFrame(open_rows)
    progress.empty()
    return scanner_df, open_df, errors, len(data)


# =====================================================================
# BACKTEST
# =====================================================================


def run_backtest(universe, params, start_date, end_date):
    warmup_days = dynamic_history_days(params, minimum=900)
    buffer_start = pd.Timestamp(start_date) - pd.Timedelta(days=warmup_days)
    start_str = buffer_start.strftime("%Y-%m-%d")
    end_str = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    progress = st.progress(0.0, text="Downloading price history...")

    def _cb(done, total):
        progress.progress(done / total, text=f"Downloading price history... {done}/{total}")

    data, errors = download_universe(universe, start_str, end_str, progress_cb=_cb)
    progress.progress(1.0, text="Running backtest engine...")

    all_trades = []
    open_position_rows = []
    start_ts, end_ts = pd.Timestamp(start_date), pd.Timestamp(end_date)
    for symbol, daily_df in data.items():
        if daily_df is None or len(daily_df) < 80:
            continue
        try:
            trades, live_state, _weekly = analyze_symbol(symbol, daily_df, params)
        except Exception as exc:  # noqa: BLE001
            logger.error("Backtest analysis failed for %s: %s\n%s", symbol, exc, traceback.format_exc())
            continue
        for t in trades:
            entry_d = pd.Timestamp(t["Entry Date"])
            if start_ts <= entry_d <= end_ts:
                all_trades.append(t)

        # A trade still open on the last available day is NOT force-closed
        # and NOT counted in the trade log / summary stats (that would
        # fabricate an exit price and distort win-rate / P&L). Instead it
        # is reported separately so it never silently disappears.
        st_ = live_state["state"]
        if st_ in (STATE_IN_TRADE, STATE_PENDING_EXIT):
            tr = live_state["trade"]
            entry_d = pd.Timestamp(tr["Entry Date"])
            if entry_d <= end_ts:
                last_close = float(daily_df["Close"].iloc[-1])
                unrealized_pct = (last_close - tr["Entry Price"]) / tr["Entry Price"] * 100.0
                open_position_rows.append({
                    "Ticker": symbol,
                    "Signal Date": tr["Signal Date"],
                    "Entry Date": tr["Entry Date"],
                    "Entry Price": round(tr["Entry Price"], 2),
                    "Last Close": round(last_close, 2),
                    "Holding Days (so far)": int(live_state["last_i"] - live_state["entry_i"]),
                    "Unrealized Profit %": round(unrealized_pct, 2),
                    "Status": "Exit pending (next open)" if st_ == STATE_PENDING_EXIT else "Open",
                })

    trade_log = pd.DataFrame(all_trades)
    if not trade_log.empty:
        cols = ["Ticker", "Signal Date", "Breakout Date", "Entry Date", "Entry Price",
                "Signal High", "Exit Date", "Exit Price", "Holding Days", "Profit %",
                "Profit ₹", "Maximum Drawdown", "Maximum Favourable Excursion",
                "Maximum Adverse Excursion"]
        trade_log = trade_log[cols].sort_values("Entry Date").reset_index(drop=True)

    open_positions_df = pd.DataFrame(open_position_rows)
    if not open_positions_df.empty:
        open_positions_df = open_positions_df.sort_values("Entry Date").reset_index(drop=True)

    progress.empty()
    return trade_log, open_positions_df, errors, len(data)


def compute_summary(trade_log: pd.DataFrame, capital: float):
    if trade_log.empty:
        return None
    wins = trade_log[trade_log["Profit ₹"] > 0]
    losses = trade_log[trade_log["Profit ₹"] <= 0]
    total = len(trade_log)

    net_profit = trade_log["Profit ₹"].sum()
    gross_win = wins["Profit ₹"].sum()
    gross_loss = abs(losses["Profit ₹"].sum())
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else np.inf

    curve = [capital]
    running = capital
    for p in trade_log["Profit ₹"]:
        running += p
        curve.append(running)
    curve = pd.Series(curve, name="Capital")
    peak = curve.cummax()
    dd = (curve - peak) / peak.replace(0, np.nan) * 100.0
    max_dd = dd.min()

    return {
        "Total Trades": total,
        "Winning Trades": len(wins),
        "Losing Trades": len(losses),
        "Win Rate %": round(len(wins) / total * 100.0, 2) if total else 0,
        "Net Profit ₹": round(net_profit, 2),
        "Total Return %": round(net_profit / capital * 100.0, 2) if capital else 0,
        "Average Win %": round(wins["Profit %"].mean(), 2) if len(wins) else 0.0,
        "Average Loss %": round(losses["Profit %"].mean(), 2) if len(losses) else 0.0,
        "Profit Factor": round(profit_factor, 2) if np.isfinite(profit_factor) else float("inf"),
        "Expectancy ₹": round(trade_log["Profit ₹"].mean(), 2),
        "Average Holding Days": round(trade_log["Holding Days"].mean(), 1),
        "Maximum Drawdown %": round(max_dd, 2) if pd.notna(max_dd) else 0.0,
        "Largest Winner ₹": round(trade_log["Profit ₹"].max(), 2),
        "Largest Loser ₹": round(trade_log["Profit ₹"].min(), 2),
        "equity_curve": curve,
    }


# =====================================================================
# EXPORT HELPERS
# =====================================================================


def to_excel_bytes(dfs: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in dfs.items():
            safe_name = name[:31] if name else "Sheet1"
            df.to_excel(writer, sheet_name=safe_name, index=False)
    return buf.getvalue()


def download_buttons(df: pd.DataFrame, base_name: str, key_prefix: str, extra_sheets: dict = None):
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇️ Download CSV", df.to_csv(index=False).encode("utf-8"),
            file_name=f"{base_name}.csv", mime="text/csv", key=f"{key_prefix}_csv",
            use_container_width=True,
        )
    with c2:
        sheets = {base_name: df}
        if extra_sheets:
            sheets.update(extra_sheets)
        st.download_button(
            "⬇️ Download Excel", to_excel_bytes(sheets),
            file_name=f"{base_name}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"{key_prefix}_xlsx", use_container_width=True,
        )


# =====================================================================
# SIDEBAR
# =====================================================================

st.sidebar.header("⚙️ Configuration")

universe_choice = st.sidebar.radio("Universe", ["NIFTY 200", "NIFTY 500"], index=0)
universe_symbols = UNIVERSE_MAP[universe_choice]

custom_file = st.sidebar.file_uploader(
    "Or override with a custom symbol list (CSV/TXT)", type=["csv", "txt"],
    help="One symbol per line, or a CSV with a 'Symbol'/'Ticker' column. "
         "Use this to load the official NSE index constituent list if the "
         "built-in NIFTY 200/500 lists (hand-curated, may drift from the "
         "official index) aren't precise enough for your use case.",
)
if custom_file is not None:
    try:
        raw = custom_file.read().decode("utf-8", errors="ignore")
        custom_symbols = []
        if "," in raw or "\n" in raw and any(h in raw.splitlines()[0].upper() for h in ("SYMBOL", "TICKER")):
            try:
                cdf = pd.read_csv(io.StringIO(raw))
                col = next((c for c in cdf.columns if c.strip().upper() in ("SYMBOL", "TICKER")), cdf.columns[0])
                custom_symbols = cdf[col].astype(str).tolist()
            except Exception:  # noqa: BLE001
                custom_symbols = raw.splitlines()
        else:
            custom_symbols = raw.splitlines()
        custom_symbols = sorted({
            s.strip().upper().replace(".NS", "")
            for s in custom_symbols if s.strip()
        })
        if custom_symbols:
            universe_symbols = custom_symbols
            st.sidebar.success(f"Using {len(universe_symbols)} symbols from uploaded list.")
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"Could not parse uploaded file: {exc}")

st.sidebar.caption(f"{len(universe_symbols)} symbols in active universe")
fetch_names_flag = st.sidebar.checkbox(
    "Fetch company names (slower)", value=False,
    help="Ticker.info lookups are one of Yahoo's slowest calls. Leave off for fast scans; "
         "the Ticker symbol is shown in the Company column instead.",
)

st.sidebar.subheader("Indicator Settings")
ema10_p = st.sidebar.number_input("EMA10 (Weekly)", min_value=2, max_value=50, value=10)
ema30_p = st.sidebar.number_input("EMA30 (Weekly)", min_value=5, max_value=100, value=30)
volema_p = st.sidebar.number_input("Volume EMA Period", min_value=5, max_value=200, value=60)
lookback_weeks = st.sidebar.number_input("52-Week Lookback", min_value=10, max_value=156, value=52)
near_high_pct = st.sidebar.slider("Near High %", min_value=80.0, max_value=100.0, value=95.0, step=0.5)
vol_mult = st.sidebar.number_input("Volume Multiplier", min_value=1.0, max_value=5.0, value=1.30, step=0.05)
min_daily_vol = st.sidebar.number_input("Minimum Daily Volume", min_value=0, max_value=10_000_000, value=100_000, step=10_000)
max_wait_days = st.sidebar.number_input("Maximum Waiting Days", min_value=5, max_value=250, value=60)

st.sidebar.subheader("Capital & Position")
capital = st.sidebar.number_input("Capital (₹)", min_value=10_000, max_value=1_000_000_000, value=1_000_000, step=10_000)
position_size = st.sidebar.number_input("Position Size per Trade (₹)", min_value=1_000, max_value=100_000_000, value=50_000, step=1_000)
st.sidebar.caption("Fixed ₹ amount applied to every triggered trade, independently (unconstrained by capital).")

with st.sidebar.expander("Transaction Costs (optional)", expanded=False):
    cost_pct = st.number_input("Brokerage / Transaction Cost % (per side)", min_value=0.0, max_value=5.0, value=0.0, step=0.05)
    slippage_pct = st.number_input("Slippage % (per side)", min_value=0.0, max_value=5.0, value=0.0, step=0.05)
    st.caption("Applied to both entry and exit fills. Defaults to 0 (frictionless) so existing results are unaffected unless you opt in.")

st.sidebar.subheader("Backtest Period")
default_start = date.today() - timedelta(days=365 * 2)
start_date = st.sidebar.date_input("Start Date", default_start, max_value=date.today())
end_date = st.sidebar.date_input("End Date", date.today(), max_value=date.today())

st.sidebar.divider()
if st.sidebar.button("\U0001F5D1️ Clear Data Cache", use_container_width=True):
    st.cache_data.clear()
    st.sidebar.success("Cache cleared.")

params = dict(
    ema10=ema10_p, ema30=ema30_p, volema=volema_p, lookback=lookback_weeks,
    near_high=near_high_pct, vol_mult=vol_mult, min_vol=min_daily_vol,
    max_wait=max_wait_days, position_size=position_size,
    cost_pct=cost_pct, slippage_pct=slippage_pct,
)

# =====================================================================
# HEADER
# =====================================================================

st.title("\U0001F4C8 NSE Weekly Swing Signal Scanner & Backtester")
st.caption(
    "Weekly-trend signal engine on NSE stocks via yfinance — causal, no repainting, no look-ahead. "
    "Buy only confirmed on a daily close above the fixed Signal Week High, executed at next day's open."
)

tab_scan, tab_bt = st.tabs(["\U0001F4E1 Scanner", "\U0001F501 Backtest"])

# =====================================================================
# SCANNER TAB
# =====================================================================

with tab_scan:
    st.subheader("Current Scanner")
    st.markdown(
        "<span class='signal-caption'>Shows stocks with an active weekly signal awaiting breakout, "
        "sorted by nearest breakout first. Signals and displayed weekly EMA/Volume-ratio values are always "
        "taken from the last fully completed weekly bar (never a still-forming week); only Current Price / "
        "Distance to Breakout use today's live daily close, since the breakout trigger is a daily event.</span>",
        unsafe_allow_html=True,
    )
    run_scan = st.button("▶️ Run Scanner", type="primary", key="run_scan_btn")

    if run_scan:
        with st.spinner("Scanning universe..."):
            scanner_df, open_df, errors, n_downloaded = run_scanner(universe_symbols, params, fetch_names=fetch_names_flag)
        st.session_state["scanner_df"] = scanner_df
        st.session_state["scanner_open_df"] = open_df
        st.session_state["scanner_errors"] = errors
        st.session_state["scanner_n"] = n_downloaded

    if "scanner_df" in st.session_state:
        scanner_df = st.session_state["scanner_df"]
        open_df = st.session_state["scanner_open_df"]
        errors = st.session_state["scanner_errors"]
        n_downloaded = st.session_state["scanner_n"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Symbols Scanned", n_downloaded)
        m2.metric("Active Signals", len(scanner_df))
        m3.metric("Open Positions", len(open_df))
        m4.metric("Failed Downloads", len(errors))

        if scanner_df.empty:
            st.warning("No active pre-breakout signals found for the current settings.")
        else:
            st.success(f"{len(scanner_df)} stock(s) currently on watch for breakout.")
            st.dataframe(scanner_df, use_container_width=True, hide_index=True)
            download_buttons(scanner_df, "nse_scanner_results", "scan")

        if not open_df.empty:
            with st.expander(f"\U0001F4BC Currently Open Positions ({len(open_df)})", expanded=False):
                st.dataframe(open_df, use_container_width=True, hide_index=True)

        if errors:
            with st.expander(f"⚠️ Download Issues ({len(errors)})", expanded=False):
                st.dataframe(pd.DataFrame(errors, columns=["Symbol", "Error"]), use_container_width=True, hide_index=True)

    with st.expander("\U0001F4CB Detailed Logs"):
        st.text(LOG_STREAM.getvalue() or "No logs yet.")

# =====================================================================
# BACKTEST TAB
# =====================================================================

with tab_bt:
    st.subheader("Backtest")
    st.markdown(
        "<span class='signal-caption'>Signals detected only on completed weekly bars; entries/exits execute "
        "at the next trading day's open. Only trades whose Entry Date falls within the selected window are "
        "counted (extra history is downloaded beforehand purely to warm up the indicators).</span>",
        unsafe_allow_html=True,
    )

    if start_date >= end_date:
        st.error("Start Date must be before End Date.")
    else:
        run_bt = st.button("▶️ Generate Trade Log", type="primary", key="run_bt_btn")

        if run_bt:
            with st.spinner("Running backtest..."):
                trade_log, open_positions_df, errors, n_downloaded = run_backtest(universe_symbols, params, start_date, end_date)
            st.session_state["trade_log"] = trade_log
            st.session_state["bt_open_positions"] = open_positions_df
            st.session_state["bt_errors"] = errors
            st.session_state["bt_n"] = n_downloaded

        if "trade_log" in st.session_state:
            trade_log = st.session_state["trade_log"]
            open_positions_df = st.session_state["bt_open_positions"]
            errors = st.session_state["bt_errors"]
            n_downloaded = st.session_state["bt_n"]

            if not open_positions_df.empty:
                with st.expander(f"\U0001F4BC Still Open at End Date ({len(open_positions_df)}) — excluded from stats below", expanded=False):
                    st.caption(
                        "These positions had not exited by End Date, so no exit price exists yet. "
                        "They are shown here for visibility but are NOT force-closed and NOT included "
                        "in the trade log or summary metrics (that would fabricate a result)."
                    )
                    st.dataframe(open_positions_df, use_container_width=True, hide_index=True)

            if trade_log.empty:
                st.warning("No closed trades were generated for this universe / date range / settings.")
                st.caption(f"Symbols downloaded: {n_downloaded} | Failed downloads: {len(errors)}")
            else:
                summary = compute_summary(trade_log, capital)

                st.success(f"Backtest complete — {summary['Total Trades']} trade(s) generated.")

                st.markdown("#### Summary Metrics")
                r1 = st.columns(4)
                r1[0].metric("Total Trades", summary["Total Trades"])
                r1[1].metric("Winning Trades", summary["Winning Trades"])
                r1[2].metric("Losing Trades", summary["Losing Trades"])
                r1[3].metric("Win Rate %", f"{summary['Win Rate %']}%")

                r2 = st.columns(4)
                r2[0].metric("Net Profit ₹", f"{summary['Net Profit ₹']:,.0f}")
                r2[1].metric("Total Return %", f"{summary['Total Return %']}%")
                r2[2].metric("Profit Factor", summary["Profit Factor"])
                r2[3].metric("Expectancy ₹", f"{summary['Expectancy ₹']:,.0f}")

                r3 = st.columns(4)
                r3[0].metric("Average Win %", f"{summary['Average Win %']}%")
                r3[1].metric("Average Loss %", f"{summary['Average Loss %']}%")
                r3[2].metric("Avg Holding Days", summary["Average Holding Days"])
                r3[3].metric("Max Drawdown %", f"{summary['Maximum Drawdown %']}%")

                r4 = st.columns(2)
                r4[0].metric("Largest Winner ₹", f"{summary['Largest Winner ₹']:,.0f}")
                r4[1].metric("Largest Loser ₹", f"{summary['Largest Loser ₹']:,.0f}")

                st.markdown("#### Capital Curve")
                st.line_chart(summary["equity_curve"])

                st.markdown("#### Trade Log")
                st.dataframe(trade_log, use_container_width=True, hide_index=True)

                summary_df = pd.DataFrame(
                    [(k, v) for k, v in summary.items() if k != "equity_curve"],
                    columns=["Metric", "Value"],
                )
                download_buttons(
                    trade_log, "nse_backtest_trade_log", "bt",
                    extra_sheets={"Summary": summary_df},
                )

                if errors:
                    with st.expander(f"⚠️ Download Issues ({len(errors)})", expanded=False):
                        st.dataframe(pd.DataFrame(errors, columns=["Symbol", "Error"]), use_container_width=True, hide_index=True)

    with st.expander("\U0001F4CB Detailed Logs"):
        st.text(LOG_STREAM.getvalue() or "No logs yet.")
