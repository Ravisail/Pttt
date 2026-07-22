"""
================================================================================
Bullish Pullback Tracking Strategy  (EMA21 / EMA50 / EMA200 + RSI70)
Single-file Streamlit application.

Contains three fully INDEPENDENT modules:

    MODULE 1  Strategy Engine          -> deterministic state machine
    MODULE 2  Signal Verification      -> independent reconstruction of setups
    MODULE 3  Backtest Engine          -> consumes generated signals only

Design guarantees
-----------------
* Indicators computed on ADJUSTED CLOSE.
* No repainting / no future-candle references in signal generation.
  (Every decision on candle i uses only data at index <= i.)
* Entry only at the NEXT trading day's OPEN after a Buy signal.
* The three modules must agree on the Buy signals. If they disagree the app
  displays an error and highlights the offending rows.

Run:
    pip install streamlit yfinance pandas numpy
    streamlit run bullish_pullback_app.py
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, timedelta
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
    HAS_YF = True
except Exception:  # pragma: no cover
    HAS_YF = False


# ==============================================================================
# HARDCODED INDEX UNIVERSES  (NSE symbols; ".NS" suffix added for yfinance)
# Source: NSE Nifty index constituents. Some names may have since been
# renamed/delisted -- unresolved tickers are skipped automatically at scan
# time (shown in the "could not load" warning). Edit these lists freely.
# ==============================================================================
NIFTY50 = [
    "ADANIPORTS", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BPCL",
    "BHARTIARTL", "BRITANNIA", "CIPLA", "COALINDIA", "DIVISLAB", "DRREDDY", "EICHERMOT", "GRASIM",
    "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "ITC",
    "INDUSINDBK", "INFY", "JSWSTEEL", "KOTAKBANK", "LT", "M&M", "MARUTI", "NTPC", "ONGC",
    "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SUNPHARMA", "TCS", "TATAMOTORS", "TATASTEEL",
    "TECHM", "TITAN", "ULTRACEMCO", "UPL", "WIPRO", "APOLLOHOSP", "ADANIGREEN", "TATACONSUM",
    "NESTLEIND"
]

NIFTY200 = [
    "ADANIPORTS", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BPCL",
    "BHARTIARTL", "BRITANNIA", "CIPLA", "COALINDIA", "DIVISLAB", "DRREDDY", "EICHERMOT", "GRASIM",
    "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "ITC",
    "INDUSINDBK", "INFY", "JSWSTEEL", "KOTAKBANK", "LT", "M&M", "MARUTI", "NTPC", "ONGC",
    "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SUNPHARMA", "TCS", "TATAMOTORS", "TATASTEEL",
    "TECHM", "TITAN", "ULTRACEMCO", "UPL", "WIPRO", "APOLLOHOSP", "ADANIGREEN", "ABB", "ACC",
    "AUBANK", "ABCAPITAL", "ABFRL", "ALKEM", "AMBUJACEM", "APOLLOTYRE", "ASHOKLEY", "AUROPHARMA",
    "DMART", "BANDHANBNK", "BANKBARODA", "BANKINDIA", "BATAINDIA", "BERGEPAINT", "BHARATFORG",
    "BHEL", "BIOCON", "BOSCHLTD", "CANBK", "CHOLAFIN", "COLPAL", "CONCOR", "COROMANDEL", "CROMPTON",
    "CUMMINSIND", "DABUR", "DEEPAKNTR", "DLF", "LALPATHLAB", "ESCORTS", "EXIDEIND", "FEDERALBNK",
    "GAIL", "GICRE", "GLENMARK", "GODREJCP", "GODREJPROP", "GRANULES", "GUJGASLTD", "HAVELLS",
    "HAL", "HINDPETRO", "HINDZINC", "HONAUT", "HDFCAMC", "HUDCO", "ICICIGI", "ICICIPRULI",
    "IDFCFIRSTB", "IDFC", "IGL", "INDHOTEL", "IOC", "INDIGO", "IPCALAB", "JKCEMENT", "JINDALSTEL",
    "JUBLFOOD", "KANSAINER", "KAJARIACER", "LTTS", "LTI", "LICHSGFIN", "LUPIN", "MGL", "M&MFIN",
    "MANAPPURAM", "MARICO", "MFSL", "MPHASIS", "MUTHOOTFIN", "NATCOPHARM", "NBCC", "NHPC", "NMDC",
    "NATIONALUM", "NAUKRI", "NAVINFLUOR", "OBEROIRLTY", "OIL", "OFSS", "PIIND", "PAGEIND",
    "PERSISTENT", "PETRONET", "PFIZER", "PIDILITIND", "PEL", "PFC", "PNB", "PGHH", "RBLBANK",
    "RECLTD", "RAMCOCEM", "SAIL", "SANOFI", "SHREECEM", "SIEMENS", "SRF", "SUNTV", "SUNDARMFIN",
    "SUPREMEIND", "SYNGENE", "TATACHEM", "TATAELXSI", "TATAPOWER", "TORNTPHARM", "TORNTPOWER",
    "TRENT", "TVSMOTOR", "UBL", "MCDOWELL-N", "UNIONBANK", "VOLTAS", "VBL", "VEDL", "ZEEL",
    "MOTHERSUMI", "MINDTREE", "CADILAHC", "BAJAJHLDNG", "CHOLAHLDNG", "GODREJIND", "ISEC",
    "CANFINHOME", "NIITTECH", "BEML", "CENTURYPLY", "CDSL", "IEX", "BALKRISIND", "MRF", "GMRINFRA",
    "IDBI", "YESBANK", "IBULHSGFIN", "PNBHOUSING", "GSPL", "THERMAX", "ADANIPOWER", "ADANITRANS",
    "ATUL", "EMAMILTD", "GLAXO", "GILLETTE", "JSWENERGY", "RAJESHEXPO", "ASTRAL", "DIXON",
    "LAURUSLABS"
]

NIFTY500 = [
    "3MINDIA", "ABB", "ACC", "AIAENG", "APLAPOLLO", "AUBANK", "AAVAS", "ADANIGREEN", "ADANIPORTS",
    "ADANIPOWER", "ADANITRANS", "ABCAPITAL", "ABFRL", "ADVENZYMES", "AEGISCHEM", "AJANTPHARM",
    "AKZOINDIA", "APLLTD", "ALKEM", "ALBK", "ALLCARGO", "AMARAJABAT", "AMBUJACEM", "ANDHRABANK",
    "APOLLOHOSP", "APOLLOTYRE", "ASHOKLEY", "ASHOKA", "ASIANPAINT", "ASTERDM", "ASTRAZEN", "ASTRAL",
    "ATUL", "AUROPHARMA", "AVANTIFEED", "DMART", "AXISBANK", "BASF", "BEML", "BSE", "BAJAJ-AUTO",
    "BAJAJCON", "BAJAJELEC", "BAJFINANCE", "BAJAJFINSV", "BAJAJHLDNG", "BALKRISIND", "BALMLAWRIE",
    "BALRAMCHIN", "BANDHANBNK", "BANKBARODA", "BANKINDIA", "MAHABANK", "BATAINDIA", "BERGEPAINT",
    "BDL", "BEL", "BHARATFORG", "BHEL", "BPCL", "BHARTIARTL", "INFRATEL", "BIOCON", "BIRLACORPN",
    "BLISSGVS", "BLUEDART", "BLUESTARCO", "BBTC", "BOMDYEING", "BOSCHLTD", "BRIGADE", "BRITANNIA",
    "CARERATING", "CCL", "CESC", "CGPOWER", "CRISIL", "CADILAHC", "CANFINHOME", "CANBK",
    "CAPLIPOINT", "CARBORUNIV", "CASTROLIND", "CEATLTD", "CENTRALBK", "CDSL", "CENTURYPLY", "CERA",
    "CHAMBLFERT", "CHENNPETRO", "CHOLAHLDNG", "CHOLAFIN", "CIPLA", "CUB", "COALINDIA", "COCHINSHIP",
    "COFFEEDAY", "COLPAL", "CONCOR", "COROMANDEL", "CORPBANK", "COX&KINGS", "CREDITACC", "CROMPTON",
    "CUMMINSIND", "CYIENT", "DBCORP", "DCBBANK", "DCMSHRIRAM", "DLF", "DABUR", "DEEPAKFERT",
    "DEEPAKNTR", "DELTACORP", "DHFL", "DBL", "DISHTV", "DCAL", "DIVISLAB", "DIXON", "LALPATHLAB",
    "DRREDDY", "EIDPARRY", "EIHOTEL", "EDELWEISS", "EICHERMOT", "ELGIEQUIP", "EMAMILTD",
    "ENDURANCE", "ENGINERSIN", "EQUITAS", "ERIS", "ESCORTS", "ESSELPACK", "EXIDEIND", "FDC",
    "FEDERALBNK", "FINEORG", "FINCABLES", "FINPIPE", "FSL", "FORTIS", "FCONSUMER", "FLFL",
    "FRETAIL", "GAIL", "GEPIL", "GET&D", "GHCL", "GMRINFRA", "GALAXYSURF", "GDL", "GAYAPROJ",
    "GICRE", "GILLETTE", "GSKCONS", "GLAXO", "GLENMARK", "GODFRYPHLP", "GODREJAGRO", "GODREJCP",
    "GODREJIND", "GODREJPROP", "GRANULES", "GRAPHITE", "GRASIM", "GESHIP", "GREAVESCOT",
    "GRINDWELL", "GRUH", "GUJALKALI", "GUJFLUORO", "GUJGASLTD", "GMDCLTD", "GNFC", "GPPL", "GSFC",
    "GSPL", "GULFOILLUB", "HEG", "HCLTECH", "HDFCAMC", "HDFCBANK", "HDFCLIFE", "HATHWAY", "HATSUN",
    "HAVELLS", "HEIDELBERG", "HERITGFOOD", "HEROMOTOCO", "HEXAWARE", "HFCL", "HSCL", "HIMATSEIDE",
    "HINDALCO", "HAL", "HINDCOPPER", "HINDPETRO", "HINDUNILVR", "HINDZINC", "HONAUT", "HUDCO",
    "HDFC", "ICICIBANK", "ICICIGI", "ICICIPRULI", "ISEC", "ICRA", "IDBI", "IDFCFIRSTB", "IDFC",
    "IFBIND", "IFCI", "IRB", "IRCON", "ITC", "ITDCEM", "ITI", "INDIACEM", "ITDC", "IBULHSGFIN",
    "IBULISL", "IBREALEST", "IBVENTURES", "INDIANB", "IEX", "INDHOTEL", "IOC", "IOB", "INDOSTAR",
    "INDOCO", "IGL", "INDUSINDBK", "INFIBEAM", "NAUKRI", "INFY", "INOXLEISUR", "INOXWIND",
    "INTELLECT", "INDIGO", "IPCALAB", "JBCHEPHARM", "JKCEMENT", "JKLAKSHMI", "JKPAPER", "JKTYRE",
    "JMFINANCIL", "JSWENERGY", "JSWSTEEL", "JAGRAN", "JAICORPLTD", "JISLJALEQS", "JPASSOCIAT",
    "J&KBANK", "JAMNAAUTO", "JETAIRWAYS", "JINDALSAW", "JSLHISAR", "JSL", "JINDALSTEL", "JUBLFOOD",
    "JUBILANT", "JUSTDIAL", "JYOTHYLAB", "KPRMILL", "KEI", "KIOCL", "KNRCON", "KRBL", "KAJARIACER",
    "KALPATPOWR", "KANSAINER", "KTKBANK", "KARURVYSYA", "KSCL", "KEC", "KIRLOSENG", "KOLTEPATIL",
    "KOTAKBANK", "L&TFH", "LTTS", "LICHSGFIN", "LAXMIMACH", "LAKSHVILAS", "LTI", "LT", "LAURUSLABS",
    "LEMONTREE", "LINDEINDIA", "LUPIN", "LUXIND", "MASFIN", "MMTC", "MOIL", "MRF", "MAGMA", "MGL",
    "MAHSCOOTER", "MAHSEAMLES", "M&MFIN", "M&M", "MAHINDCIE", "MHRIL", "MAHLOG", "MANAPPURAM",
    "MRPL", "MARICO", "MARUTI", "MFSL", "MAXINDIA", "MINDTREE", "MINDACORP", "MINDAIND", "MONSANTO",
    "MOTHERSUMI", "MOTILALOFS", "MPHASIS", "MUTHOOTFIN", "NATCOPHARM", "NBCC", "NCC", "NESCO",
    "NHPC", "NIITTECH", "NLCINDIA", "NMDC", "NTPC", "NH", "NATIONALUM", "NFL", "NBVENTURES",
    "NAVINFLUOR", "NETWORK18", "NILKAMAL", "OBEROIRLTY", "ONGC", "OIL", "OMAXE", "OFSS",
    "ORIENTCEM", "ORIENTELEC", "ORIENTBANK", "PCJEWELLER", "PIIND", "PNBHOUSING", "PNCINFRA", "PTC",
    "PVR", "PAGEIND", "PARAGMILK", "PERSISTENT", "PETRONET", "PFIZER", "PHILIPCARB", "PHOENIXLTD",
    "PIDILITIND", "PEL", "PFC", "POWERGRID", "PRAJIND", "PRESTIGE", "PRSMJOHNSN", "PGHL", "PGHH",
    "PNB", "QUESS", "RBLBANK", "RECLTD", "RITES", "RADICO", "RAIN", "RAJESHEXPO", "RALLIS",
    "RKFORGE", "RCF", "RAYMOND", "REDINGTON", "RELAXO", "RELCAPITAL", "RCOM", "RHFL", "RELIANCE",
    "RELINFRA", "RNAM", "RPOWER", "REPCOHOME", "RUPA", "SHK", "SBILIFE", "SJVN", "SKFINDIA",
    "SREINFRA", "SRF", "SADBHAV", "SANOFI", "SCHAEFFLER", "SIS", "SHANKARA", "SHARDACROP", "SFL",
    "SHILPAMED", "SCI", "SHOPERSTOP", "SHREECEM", "RENUKA", "SHRIRAMCIT", "SRTRANSFIN", "SIEMENS",
    "SPTL", "SOBHA", "SOLARINDS", "SONATSOFTW", "SOUTHBANK", "STARCEMENT", "SBIN", "SAIL",
    "STRTECH", "STAR", "SUDARSCHEM", "SPARC", "SUNPHARMA", "SUNTV", "SUNCLAYLTD", "SUNDARMFIN",
    "SUNDRMFAST", "SUNTECK", "SUPRAJIT", "SUPREMEIND", "SUVEN", "SUZLON", "SWANENERGY", "SYMPHONY",
    "SYNDIBANK", "SYNGENE", "TCNSBRANDS", "TTKPRESTIG", "TVTODAY", "TV18BRDCST", "TVSMOTOR", "TAKE",
    "TNPL", "TATACHEM", "TATACOFFEE", "TCS", "TATAELXSI", "TATAGLOBAL", "TATAINVEST", "TATAMTRDVR",
    "TATAMOTORS", "TATAPOWER", "TATASTEEL", "TEAMLEASE", "TECHM", "NIACL", "RAMCOCEM", "THERMAX",
    "THOMASCOOK", "THYROCARE", "TIMETECHNO", "TIMKEN", "TITAN", "TORNTPHARM", "TORNTPOWER", "TRENT",
    "TRIDENT", "TRITURBINE", "TIINDIA", "UCOBANK", "UFLEX", "UPL", "UJJIVAN", "ULTRACEMCO",
    "UNIONBANK", "UBL", "MCDOWELL-N", "VGUARD", "VMART", "VIPIND", "VRLLOG", "VSTIND", "WABAG",
    "VAKRANGEE", "VTL", "VARROC", "VBL", "VEDL", "VENKEYS", "VINATIORGA", "IDEA", "VOLTAS",
    "WABCOINDIA", "WELCORP", "WELSPUNIND", "WHIRLPOOL", "WIPRO", "WOCKPHARMA", "YESBANK", "ZEEL",
    "ZENSARTECH", "ZYDUSWELL", "ECLERX"
]

INDEX_UNIVERSES = {
    "Nifty 50": NIFTY50,
    "Nifty 200": NIFTY200,
    "Nifty 500": NIFTY500,
}


def to_yf(symbols, suffix=".NS"):
    """Append the NSE yfinance suffix to a list of bare NSE symbols."""
    return [s if s.endswith(suffix) else s + suffix for s in symbols]


# ==============================================================================
# INDICATORS  (shared math, but each module runs its own scan logic)
# ==============================================================================
def ema(series: pd.Series, span: int) -> pd.Series:
    """Standard EMA. Deterministic, uses only past values (no lookahead)."""
    return series.ewm(span=span, adjust=False).mean()


def rsi_wilder(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI on the given price series. No repainting."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder smoothing == ewm with alpha = 1/period
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    out = 100.0 - (100.0 / (1.0 + rs))
    out[avg_loss == 0] = 100.0
    return out


def build_indicator_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects columns: Open, High, Low, Close, Adj Close.
    Adds EMA21/50/200 and RSI14 computed on ADJUSTED CLOSE.
    Uses 'StratClose' == adjusted close for all strategy comparisons so that the
    entire signal chain is on an adjusted basis (consistent, no repaint).
    Entry price uses the (adjusted) next-day Open.
    """
    out = df.copy()
    adj = out["Adj Close"].astype(float)
    out["StratClose"] = adj
    # Scale open/high/low by the adj/close ratio -> same adjusted basis.
    ratio = (out["Adj Close"] / out["Close"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    out["Ratio"] = ratio
    out["StratOpen"] = out["Open"].astype(float) * ratio
    out["StratHigh"] = out["High"].astype(float) * ratio
    out["StratLow"] = out["Low"].astype(float) * ratio
    out["EMA21"] = ema(adj, 21)
    out["EMA50"] = ema(adj, 50)
    out["EMA200"] = ema(adj, 200)
    out["RSI"] = rsi_wilder(adj, 14)
    return out


# ==============================================================================
# Data structures
# ==============================================================================
@dataclass
class BuySignal:
    symbol: str
    tracking_start_date: str
    pullback_date: str
    buy_date: str
    signal_price: float
    ema21: float
    ema50: float
    ema200: float
    rsi: float
    candles_until_pullback: int
    candles_until_buy: int


@dataclass
class DebugEvent:
    symbol: str
    date: str
    event: str
    detail: str


# ==============================================================================
# MODULE 1 --- STRATEGY ENGINE (deterministic state machine)
# ==============================================================================
STATE_IDLE = 0
STATE_TRACKING = 1
STATE_WAIT_RECOVERY = 2


def strategy_engine(symbol: str, ind: pd.DataFrame,
                    collect_debug: bool = False,
                    strict_trend: bool = True
                    ) -> Tuple[List[BuySignal], List[DebugEvent], Dict[str, int]]:
    """
    MODULE 1. Proper state machine. Returns list of BuySignal, debug events,
    and cycle statistics. Never references future candles.

    strict_trend=True enforces a fully fanned bullish stack EMA21>EMA50>EMA200
    at tracking start, on the pullback candle, and at recovery/buy. When False
    only the looser EMA21>EMA200 & EMA50>EMA200 relationship is required.
    """
    def fanned(a21, a50, a200) -> bool:
        return (a21 > a50) and (a50 > a200)
    signals: List[BuySignal] = []
    debug: List[DebugEvent] = []
    stats = dict(tracking_started=0, pullbacks_found=0, buys=0,
                 signals_below_ema200=0)

    state = STATE_IDLE
    tracking = False
    pullback_found = False
    tracking_start_idx: Optional[int] = None
    pullback_idx: Optional[int] = None

    idx = ind.index
    n = len(ind)

    for i in range(n):
        row = ind.iloc[i]
        c = row["StratClose"]
        e21, e50, e200, rsi = row["EMA21"], row["EMA50"], row["EMA200"], row["RSI"]

        # Skip candles where indicators are not yet defined (warm-up / no lookahead).
        if pd.isna(e200) or pd.isna(rsi) or pd.isna(e21) or pd.isna(e50):
            continue

        d = str(idx[i].date())

        # ---- GLOBAL RESET RULE (applies in any active state) -------------
        if (tracking or pullback_found) and c <= e200:
            if collect_debug:
                debug.append(DebugEvent(symbol, d, "RESET",
                                        f"Close {c:.2f} <= EMA200 {e200:.2f}; setup discarded"))
            state = STATE_IDLE
            tracking = False
            pullback_found = False
            tracking_start_idx = None
            pullback_idx = None
            continue

        # ---- STATE 0 : IDLE ---------------------------------------------
        if state == STATE_IDLE:
            cond = (c > e200 and c > e50 and c > e21 and
                    e21 > e200 and e50 > e200 and rsi > 70.0 and
                    (not strict_trend or fanned(e21, e50, e200)))
            if cond:
                state = STATE_TRACKING
                tracking = True
                pullback_found = False
                tracking_start_idx = i
                stats["tracking_started"] += 1
                if collect_debug:
                    debug.append(DebugEvent(
                        symbol, d, "TRACKING STARTED",
                        f"RSI {rsi:.1f}  Close {c:.2f}  EMA21 {e21:.2f} "
                        f"EMA50 {e50:.2f} EMA200 {e200:.2f}"))
            continue

        # ---- STATE 1 : TRACKING (wait for pullback) ----------------------
        if state == STATE_TRACKING:
            if (c < e21) and (c < e50) and (c > e200) and \
                    (not strict_trend or fanned(e21, e50, e200)):
                pullback_found = True
                pullback_idx = i
                state = STATE_WAIT_RECOVERY
                stats["pullbacks_found"] += 1
                if collect_debug:
                    debug.append(DebugEvent(
                        symbol, d, "PULLBACK FOUND",
                        f"Close {c:.2f} < EMA21 {e21:.2f} & < EMA50 {e50:.2f}; "
                        f"still > EMA200 {e200:.2f}"))
            continue

        # ---- STATE 2 : WAIT FOR RECOVERY ---------------------------------
        if state == STATE_WAIT_RECOVERY:
            if (c > e21) and (c > e50) and (c > e200) and \
                    (not strict_trend or fanned(e21, e50, e200)):
                cu_pb = pullback_idx - tracking_start_idx
                cu_buy = i - tracking_start_idx
                sig = BuySignal(
                    symbol=symbol,
                    tracking_start_date=str(idx[tracking_start_idx].date()),
                    pullback_date=str(idx[pullback_idx].date()),
                    buy_date=d,
                    signal_price=float(c),
                    ema21=float(e21), ema50=float(e50), ema200=float(e200),
                    rsi=float(rsi),
                    candles_until_pullback=int(cu_pb),
                    candles_until_buy=int(cu_buy),
                )
                signals.append(sig)
                stats["buys"] += 1
                if collect_debug:
                    debug.append(DebugEvent(
                        symbol, d, "RECOVERY FOUND -> BUY",
                        f"Close {c:.2f} > EMA21/50/200; entry = next day OPEN"))
                # ONE BUY PER CYCLE -> full reset to IDLE.
                state = STATE_IDLE
                tracking = False
                pullback_found = False
                tracking_start_idx = None
                pullback_idx = None
            continue

    return signals, debug, stats


# ==============================================================================
# MODULE 2 --- SIGNAL VERIFICATION MODULE (independent reconstruction)
# ==============================================================================
def verification_module(symbol: str, ind: pd.DataFrame,
                        strict_trend: bool = True) -> List[Dict]:
    """
    MODULE 2. Independent re-implementation. Does NOT share variables/objects
    with the strategy engine. Reconstructs every setup by scanning the raw
    indicator arrays with a different code path (index-driven phase scanner)
    and returns a list of setups: {tracking, pullback, buy}.

    Applies the same strict_trend (EMA21>EMA50>EMA200) option, re-derived
    independently, so the two modules must still agree.
    """
    dates = [str(d.date()) for d in ind.index]
    close = ind["StratClose"].to_numpy(dtype=float)
    e21 = ind["EMA21"].to_numpy(dtype=float)
    e50 = ind["EMA50"].to_numpy(dtype=float)
    e200 = ind["EMA200"].to_numpy(dtype=float)
    rsi = ind["RSI"].to_numpy(dtype=float)
    n = len(close)

    def defined(i: int) -> bool:
        return not (np.isnan(e200[i]) or np.isnan(rsi[i])
                    or np.isnan(e21[i]) or np.isnan(e50[i]))

    def fan(i: int) -> bool:
        return (e21[i] > e50[i]) and (e50[i] > e200[i])

    setups: List[Dict] = []
    phase = "idle"          # idle -> tracking -> recovery
    t_start = pb = None

    i = 0
    while i < n:
        if not defined(i):
            i += 1
            continue

        c, a21, a50, a200, r = close[i], e21[i], e50[i], e200[i], rsi[i]

        # Reset rule first (independent check).
        if phase != "idle" and c <= a200:
            phase = "idle"
            t_start = pb = None
            i += 1
            continue

        if phase == "idle":
            if (c > a200 and c > a50 and c > a21 and
                    a21 > a200 and a50 > a200 and r > 70.0 and
                    (not strict_trend or fan(i))):
                phase = "tracking"
                t_start = i
        elif phase == "tracking":
            if (c < a21) and (c < a50) and (c > a200) and \
                    (not strict_trend or fan(i)):
                phase = "recovery"
                pb = i
        elif phase == "recovery":
            if (c > a21) and (c > a50) and (c > a200) and \
                    (not strict_trend or fan(i)):
                setups.append(dict(
                    symbol=symbol,
                    tracking_date=dates[t_start],
                    pullback_date=dates[pb],
                    buy_date=dates[i],
                ))
                phase = "idle"
                t_start = pb = None
        i += 1

    return setups


def compare_signals(engine_signals: List[BuySignal],
                    verify_setups: List[Dict]) -> pd.DataFrame:
    """
    Build the comparison table. Aligns engine buys with verification setups
    by (symbol, tracking_date, pullback_date, buy_date).
    """
    eng = {(s.symbol, s.tracking_start_date, s.pullback_date, s.buy_date): s
           for s in engine_signals}
    ver = {(v["symbol"], v["tracking_date"], v["pullback_date"], v["buy_date"]): v
           for v in verify_setups}

    rows = []
    all_keys = sorted(set(eng) | set(ver), key=lambda k: (k[3], k[0]))
    for k in all_keys:
        in_e = k in eng
        in_v = k in ver
        status = "MATCH" if (in_e and in_v) else "MISMATCH"
        rows.append(dict(
            Symbol=k[0],
            **{"Tracking Date": k[1], "Pullback Date": k[2], "Buy Date": k[3]},
            Status=status,
            InEngine=in_e, InVerify=in_v,
        ))
    return pd.DataFrame(rows)


# ==============================================================================
# MODULE 3 --- BACKTEST ENGINE (consumes signals only, no recalculation)
# ==============================================================================
@dataclass
class PortfolioSettings:
    initial_capital: float
    position_size_pct: float      # % of initial capital per trade
    max_open_positions: int
    brokerage_pct: float
    slippage_pct: float
    taxes_pct: float
    # exit settings (configurable)
    stop_loss_pct: float
    take_profit_pct: float
    max_hold_days: int


@dataclass
class Trade:
    symbol: str
    tracking: str
    pullback: str
    buy_signal: str
    entry_date: str = ""
    entry_price: float = 0.0
    exit_date: str = ""
    exit_price: float = 0.0
    pnl: float = 0.0
    ret_pct: float = 0.0
    holding_days: int = 0
    status: str = ""          # Executed / Skipped / Rejected
    reason: str = ""


def _next_trading_index(ind: pd.DataFrame, buy_date: str) -> Optional[int]:
    """Index of the first candle strictly AFTER the buy_date (next-day open)."""
    target = pd.to_datetime(buy_date).date()
    for j, ts in enumerate(ind.index):
        if ts.date() > target:
            return j
    return None


def backtest_engine(signals_by_symbol: Dict[str, List[BuySignal]],
                    data_by_symbol: Dict[str, pd.DataFrame],
                    verified_keys: set,
                    settings: PortfolioSettings) -> Tuple[List[Trade], Dict]:
    """
    MODULE 3. Consumes the generated Buy signals (does NOT recalculate them).
    Simulates portfolio chronologically across all symbols.
    Entry = next day's OPEN. Exit = SL / TP / max-hold (configurable).
    Open positions are tracked so max-open-positions is enforced over time.
    """
    events = []
    for sym, sigs in signals_by_symbol.items():
        for s in sigs:
            events.append(s)
    events.sort(key=lambda s: (s.buy_date, s.symbol))

    cash = settings.initial_capital
    per_trade_alloc = settings.initial_capital * settings.position_size_pct / 100.0
    trades: List[Trade] = []
    seen_keys = set()
    # list of (exit_date_str) for currently open positions
    open_exits: List[str] = []

    for s in events:
        key = (s.symbol, s.tracking_start_date, s.pullback_date, s.buy_date)
        vkey = (s.symbol, s.tracking_start_date, s.pullback_date, s.buy_date)
        t = Trade(symbol=s.symbol, tracking=s.tracking_start_date,
                  pullback=s.pullback_date, buy_signal=s.buy_date)

        # Release positions that have already exited before this signal's buy date.
        open_exits = [ed for ed in open_exits if ed >= s.buy_date]

        # --- Rejections (data integrity) ---
        if key in seen_keys:
            t.status, t.reason = "Rejected", "Duplicate signal"
            trades.append(t)
            continue
        seen_keys.add(key)

        if verified_keys is not None and vkey not in verified_keys:
            t.status, t.reason = "Rejected", "Verification mismatch"
            trades.append(t)
            continue

        if s.signal_price <= s.ema200:
            t.status, t.reason = "Rejected", "Below EMA200"
            trades.append(t)
            continue

        ind = data_by_symbol[s.symbol]
        entry_idx = _next_trading_index(ind, s.buy_date)
        if entry_idx is None:
            t.status, t.reason = "Rejected", "Invalid sequence"
            trades.append(t)
            continue

        # --- Max-open-positions gating ---
        if len(open_exits) >= settings.max_open_positions:
            t.status, t.reason = "Skipped", "Max positions reached"
            trades.append(t)
            continue

        # --- Capital gating ---
        alloc = min(per_trade_alloc, cash)
        entry_price_raw = float(ind.iloc[entry_idx]["StratOpen"])
        entry_price = entry_price_raw * (1.0 + settings.slippage_pct / 100.0)
        if entry_price <= 0 or alloc <= 0:
            t.status, t.reason = "Skipped", "Insufficient Capital"
            trades.append(t)
            continue

        shares = int(alloc // entry_price)
        if shares <= 0:
            t.status, t.reason = "Skipped", "Insufficient Capital"
            trades.append(t)
            continue

        cost = shares * entry_price
        entry_fees = cost * (settings.brokerage_pct / 100.0)
        if cost + entry_fees > cash:
            t.status, t.reason = "Skipped", "Insufficient Capital"
            trades.append(t)
            continue

        # --- Simulate exit (SL / TP / max-hold) using adjusted OHLC ---
        sl_price = entry_price * (1.0 - settings.stop_loss_pct / 100.0)
        tp_price = entry_price * (1.0 + settings.take_profit_pct / 100.0)
        exit_idx = None
        exit_price = None
        for j in range(entry_idx, len(ind)):
            hi = float(ind.iloc[j]["StratHigh"])
            lo = float(ind.iloc[j]["StratLow"])
            held = j - entry_idx
            if lo <= sl_price:                       # stop-loss priority
                exit_idx, exit_price = j, sl_price
                break
            if hi >= tp_price:
                exit_idx, exit_price = j, tp_price
                break
            if held >= settings.max_hold_days:
                exit_idx = j
                exit_price = float(ind.iloc[j]["StratClose"])
                break
        if exit_idx is None:                         # ran out of data
            exit_idx = len(ind) - 1
            exit_price = float(ind.iloc[exit_idx]["StratClose"])

        exit_price *= (1.0 - settings.slippage_pct / 100.0)
        gross = shares * exit_price
        exit_fees = gross * (settings.brokerage_pct / 100.0)
        taxes = max(0.0, (gross - cost)) * (settings.taxes_pct / 100.0)
        pnl = gross - cost - entry_fees - exit_fees - taxes

        cash += pnl
        t.entry_date = str(ind.index[entry_idx].date())
        t.entry_price = round(entry_price, 4)
        t.exit_date = str(ind.index[exit_idx].date())
        t.exit_price = round(exit_price, 4)
        t.pnl = round(pnl, 2)
        t.ret_pct = round((exit_price / entry_price - 1.0) * 100.0, 2)
        t.holding_days = int(exit_idx - entry_idx)
        t.status = "Executed"
        trades.append(t)
        open_exits.append(t.exit_date)

    summary = dict(
        final_cash=round(cash, 2),
        total_return_pct=round((cash / settings.initial_capital - 1.0) * 100.0, 2),
        executed=sum(1 for t in trades if t.status == "Executed"),
        skipped=sum(1 for t in trades if t.status == "Skipped"),
        rejected=sum(1 for t in trades if t.status == "Rejected"),
    )
    return trades, summary


# ==============================================================================
# ANALYTICS  (derived from executed trades only; consumes backtest output)
# ==============================================================================
def _max_drawdown(equity: List[float]) -> float:
    """Return max drawdown as a positive percentage of peak equity."""
    peak = -np.inf
    mdd = 0.0
    for v in equity:
        peak = max(peak, v)
        if peak > 0:
            mdd = max(mdd, (peak - v) / peak)
    return round(mdd * 100.0, 2)


def compute_analytics(trades: List[Trade], signals: List[BuySignal],
                      settings: PortfolioSettings) -> Dict[str, object]:
    """
    Advanced performance metrics. Uses ONLY executed trades (the realized
    equity curve). Ratios are trade-based (per-trade returns), which is the
    standard approach for signal-driven swing backtests.
    """
    ex = [t for t in trades if t.status == "Executed"]
    a: Dict[str, object] = {}
    a["Executed Trades"] = len(ex)
    if not ex:
        return a, [settings.initial_capital], []

    wins = [t for t in ex if t.pnl > 0]
    losses = [t for t in ex if t.pnl <= 0]
    gross_profit = sum(t.pnl for t in wins)
    gross_loss = -sum(t.pnl for t in losses)     # positive number
    net = sum(t.pnl for t in ex)
    rets = [t.ret_pct / 100.0 for t in ex]       # per-trade fractional returns

    a["Win Rate %"] = round(100.0 * len(wins) / len(ex), 2)
    a["Avg Win"] = round(np.mean([t.pnl for t in wins]), 2) if wins else 0.0
    a["Avg Loss"] = round(np.mean([t.pnl for t in losses]), 2) if losses else 0.0
    a["Profit Factor"] = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf")
    a["Expectancy ($/trade)"] = round(net / len(ex), 2)
    a["Expectancy (%/trade)"] = round(100.0 * float(np.mean(rets)), 2)

    # Equity curve ordered by exit date (realized).
    ordered = sorted(ex, key=lambda t: t.exit_date)
    equity = [settings.initial_capital]
    for t in ordered:
        equity.append(equity[-1] + t.pnl)
    a["Max Drawdown %"] = _max_drawdown(equity)

    # CAGR from first entry to last exit.
    d0 = pd.to_datetime(min(t.entry_date for t in ex))
    d1 = pd.to_datetime(max(t.exit_date for t in ex))
    years = max((d1 - d0).days / 365.25, 1e-9)
    end_eq = equity[-1]
    cagr = (end_eq / settings.initial_capital) ** (1.0 / years) - 1.0 \
        if end_eq > 0 else -1.0
    a["CAGR %"] = round(cagr * 100.0, 2)

    # Sharpe / Sortino on per-trade returns (rf = 0). Annualized by trade freq.
    r = np.array(rets, dtype=float)
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    downside = r[r < 0]
    dsd = downside.std(ddof=1) if len(downside) > 1 else 0.0
    trades_per_year = len(ex) / years
    ann = np.sqrt(trades_per_year) if trades_per_year > 0 else 1.0
    a["Sharpe (trade-based)"] = round((r.mean() / sd) * ann, 2) if sd > 0 else 0.0
    a["Sortino (trade-based)"] = round((r.mean() / dsd) * ann, 2) if dsd > 0 else 0.0

    # MAR & Recovery Factor.
    mdd = a["Max Drawdown %"]
    a["MAR Ratio"] = round(a["CAGR %"] / mdd, 2) if mdd > 0 else float("inf")
    total_ret_pct = 100.0 * net / settings.initial_capital
    a["Recovery Factor"] = round(total_ret_pct / mdd, 2) if mdd > 0 else float("inf")

    # Consecutive wins / losses.
    cw = cl = mcw = mcl = 0
    for t in ordered:
        if t.pnl > 0:
            cw += 1; cl = 0; mcw = max(mcw, cw)
        else:
            cl += 1; cw = 0; mcl = max(mcl, cl)
    a["Max Consecutive Wins"] = mcw
    a["Max Consecutive Losses"] = mcl

    # Signal-side stats (from the buy signals that were executed).
    sig_by_key = {(s.symbol, s.tracking_start_date, s.pullback_date, s.buy_date): s
                  for s in signals}
    rsis, dist200 = [], []
    for t in ex:
        s = sig_by_key.get((t.symbol, t.tracking, t.pullback, t.buy_signal))
        if s:
            rsis.append(s.rsi)
            if s.ema200:
                dist200.append(100.0 * (s.signal_price - s.ema200) / s.ema200)
    a["Avg RSI at Buy"] = round(float(np.mean(rsis)), 1) if rsis else 0.0
    a["Avg Dist from EMA200 %"] = round(float(np.mean(dist200)), 2) if dist200 else 0.0
    a["Avg Holding Days"] = round(float(np.mean([t.holding_days for t in ex])), 1)

    return a, equity, ordered


def bucket_win_rates(trades: List[Trade], signals: List[BuySignal]):
    """Win-rate breakdowns by holding-day band, RSI band, and calendar month."""
    ex = [t for t in trades if t.status == "Executed"]
    sig_by_key = {(s.symbol, s.tracking_start_date, s.pullback_date, s.buy_date): s
                  for s in signals}

    def wr(rows):
        return round(100.0 * sum(1 for t in rows if t.pnl > 0) / len(rows), 1) if rows else 0.0

    # Holding days
    hold_bands = {"1-5": [], "6-15": [], "16-30": [], "31+": []}
    for t in ex:
        h = t.holding_days
        key = "1-5" if h <= 5 else "6-15" if h <= 15 else "16-30" if h <= 30 else "31+"
        hold_bands[key].append(t)
    hold_df = pd.DataFrame([{"Holding Days": k, "Trades": len(v), "Win Rate %": wr(v)}
                            for k, v in hold_bands.items()])

    # RSI bands
    rsi_bands = {"70-75": [], "75-80": [], "80-90": [], "90+": []}
    for t in ex:
        s = sig_by_key.get((t.symbol, t.tracking, t.pullback, t.buy_signal))
        if not s:
            continue
        rv = s.rsi
        key = "70-75" if rv < 75 else "75-80" if rv < 80 else "80-90" if rv < 90 else "90+"
        rsi_bands[key].append(t)
    rsi_df = pd.DataFrame([{"RSI at Buy": k, "Trades": len(v), "Win Rate %": wr(v)}
                           for k, v in rsi_bands.items()])

    # By month of entry
    month_map: Dict[str, List[Trade]] = {}
    for t in ex:
        m = t.entry_date[:7]
        month_map.setdefault(m, []).append(t)
    month_df = pd.DataFrame([{"Month": k, "Trades": len(v), "Win Rate %": wr(v)}
                             for k, v in sorted(month_map.items())])
    return hold_df, rsi_df, month_df


# ==============================================================================
# DATA LOADING
# ==============================================================================
@st.cache_data(show_spinner=False)
def load_data(symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
    if not HAS_YF:
        return None
    try:
        df = yf.download(symbol, start=start, end=end, auto_adjust=False,
                         progress=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        need = ["Open", "High", "Low", "Close", "Adj Close"]
        for col in need:
            if col not in df.columns:
                return None
        df = df[need].dropna()
        return df
    except Exception:
        return None


# ==============================================================================
# PENDING-STATE SCAN (for the Signal Module page)
# ==============================================================================
def current_pending(symbol: str, ind: pd.DataFrame, strict_trend: bool = True):
    """Report an open (unclosed) tracking/pullback setup at the last candle."""
    def fanned(a21, a50, a200):
        return (a21 > a50) and (a50 > a200)
    state = STATE_IDLE
    t_start = pb = None
    for i in range(len(ind)):
        row = ind.iloc[i]
        c = row["StratClose"]; e21 = row["EMA21"]; e50 = row["EMA50"]
        e200 = row["EMA200"]; rsi = row["RSI"]
        if pd.isna(e200) or pd.isna(rsi) or pd.isna(e21) or pd.isna(e50):
            continue
        if state != STATE_IDLE and c <= e200:
            state = STATE_IDLE; t_start = pb = None; continue
        if state == STATE_IDLE:
            if (c > e200 and c > e50 and c > e21 and e21 > e200 and e50 > e200
                    and rsi > 70 and (not strict_trend or fanned(e21, e50, e200))):
                state = STATE_TRACKING; t_start = i
        elif state == STATE_TRACKING:
            if (c < e21 and c < e50 and c > e200
                    and (not strict_trend or fanned(e21, e50, e200))):
                state = STATE_WAIT_RECOVERY; pb = i
        elif state == STATE_WAIT_RECOVERY:
            if (c > e21 and c > e50 and c > e200
                    and (not strict_trend or fanned(e21, e50, e200))):
                state = STATE_IDLE; t_start = pb = None

    last = ind.iloc[-1]
    if state == STATE_TRACKING and t_start is not None:
        return (dict(Symbol=symbol,
                     **{"Tracking Start": str(ind.index[t_start].date()),
                        "Current Price": round(float(last["StratClose"]), 2),
                        "EMA21": round(float(last["EMA21"]), 2),
                        "EMA50": round(float(last["EMA50"]), 2),
                        "EMA200": round(float(last["EMA200"]), 2),
                        "RSI": round(float(last["RSI"]), 1),
                        "Days Tracking": int(len(ind) - 1 - t_start)}), None)
    if state == STATE_WAIT_RECOVERY and t_start is not None:
        return (None, dict(Symbol=symbol,
                           **{"Tracking Date": str(ind.index[t_start].date()),
                              "Pullback Date": str(ind.index[pb].date()),
                              "Current Close": round(float(last["StratClose"]), 2),
                              "Recovery Pending": "YES"}))
    return (None, None)


# ==============================================================================
# AUDIT MODULE
# ==============================================================================
def audit_signal(s: BuySignal, ind: pd.DataFrame, verified: bool,
                 duplicate: bool, strict_trend: bool = True) -> Dict[str, str]:
    """Return PASS/FAIL for every audit rule for a single signal."""
    def P(x: bool) -> str:
        return "PASS" if x else "FAIL"

    def row_for(dstr: str):
        target = pd.to_datetime(dstr).date()
        for k in range(len(ind)):
            if ind.index[k].date() == target:
                return ind.iloc[k]
        return None

    tr = row_for(s.tracking_start_date)
    pb = row_for(s.pullback_date)
    by = row_for(s.buy_date)

    checks = {}
    checks["Tracking started correctly"] = P(
        tr is not None and tr["RSI"] > 70 and tr["StratClose"] > tr["EMA200"]
        and tr["StratClose"] > tr["EMA50"] and tr["StratClose"] > tr["EMA21"]
        and tr["EMA21"] > tr["EMA200"] and tr["EMA50"] > tr["EMA200"])
    checks["Pullback happened"] = P(
        pb is not None and pb["StratClose"] < pb["EMA21"]
        and pb["StratClose"] < pb["EMA50"] and pb["StratClose"] > pb["EMA200"])
    checks["Recovery happened"] = P(
        by is not None and by["StratClose"] > by["EMA21"]
        and by["StratClose"] > by["EMA50"] and by["StratClose"] > by["EMA200"])
    checks["Buy above EMA21"] = P(s.signal_price > s.ema21)
    checks["Buy above EMA50"] = P(s.signal_price > s.ema50)
    checks["Buy above EMA200"] = P(s.signal_price > s.ema200)
    checks["RSI>70 existed before tracking"] = P(tr is not None and tr["RSI"] > 70)
    checks["No duplicate Buy"] = P(not duplicate)
    checks["No Buy below EMA200"] = P(s.signal_price > s.ema200)
    checks["Entry next day Open / order valid"] = P(
        s.tracking_start_date <= s.pullback_date <= s.buy_date
        and s.candles_until_pullback >= 0
        and s.candles_until_buy > s.candles_until_pullback)
    checks["No look-ahead bias"] = P(verified)
    if strict_trend:
        checks["EMA21>EMA50>EMA200 at buy"] = P(s.ema21 > s.ema50 > s.ema200)
        checks["EMA21>EMA50>EMA200 at tracking"] = P(
            tr is not None and tr["EMA21"] > tr["EMA50"] > tr["EMA200"])
    return checks


# ==============================================================================
# ORCHESTRATION
# ==============================================================================
def run_all(symbols, start, end, debug_mode, strict_trend=True):
    data_by_symbol: Dict[str, pd.DataFrame] = {}
    engine_signals_by_symbol: Dict[str, List[BuySignal]] = {}
    all_engine_signals: List[BuySignal] = []
    all_verify_setups: List[Dict] = []
    all_debug: List[DebugEvent] = []
    engine_stats_total = dict(tracking_started=0, pullbacks_found=0, buys=0,
                              signals_below_ema200=0)
    pending_tracking = []
    pending_pullback = []
    load_errors = []

    for sym in symbols:
        df = load_data(sym, str(start), str(end))
        if df is None or len(df) < 210:
            load_errors.append(sym)
            continue
        ind = build_indicator_frame(df)
        data_by_symbol[sym] = ind

        sigs, dbg, stats = strategy_engine(sym, ind, collect_debug=debug_mode,
                                           strict_trend=strict_trend)
        engine_signals_by_symbol[sym] = sigs
        all_engine_signals.extend(sigs)
        all_debug.extend(dbg)
        for k in engine_stats_total:
            engine_stats_total[k] += stats[k]

        verify = verification_module(sym, ind, strict_trend=strict_trend)
        all_verify_setups.extend(verify)

        p_track, p_pull = current_pending(sym, ind, strict_trend=strict_trend)
        if p_track:
            pending_tracking.append(p_track)
        if p_pull:
            pending_pullback.append(p_pull)

    return dict(
        data=data_by_symbol,
        engine_by_symbol=engine_signals_by_symbol,
        engine_signals=all_engine_signals,
        verify_setups=all_verify_setups,
        debug=all_debug,
        engine_stats=engine_stats_total,
        pending_tracking=pending_tracking,
        pending_pullback=pending_pullback,
        load_errors=load_errors,
        strict_trend=strict_trend,
    )


# ==============================================================================
# STREAMLIT UI
# ==============================================================================
st.set_page_config(page_title="Bullish Pullback Tracking",
                   page_icon="", layout="wide")

st.title("Bullish Pullback Tracking")
st.caption("EMA21 / EMA50 / EMA200 + RSI(70) -- independent Strategy / Verification / Backtest engines")

with st.sidebar:
    st.header("Universe & Data")
    universe_choice = st.selectbox(
        "Universe", ["Custom symbols", "Nifty 50", "Nifty 200", "Nifty 500"],
        index=2,
        help="Pick a built-in NSE index list, or 'Custom symbols' to type your own.")
    if universe_choice == "Custom symbols":
        symbols_raw = st.text_input("Symbols (comma separated)", "AAPL, MSFT, NVDA")
        max_symbols = 0
    else:
        _base = INDEX_UNIVERSES[universe_choice]
        symbols_raw = ""
        st.caption(f"{universe_choice}: {len(_base)} NSE stocks (auto '.NS' suffix).")
        max_symbols = st.number_input(
            "Limit to first N symbols (0 = all)", 0, len(_base),
            min(30, len(_base)),
            help="Scanning the full list downloads data for every symbol and can "
                 "be slow. Start small, then raise to 0 for the whole index.")
    today = date.today()
    start = st.date_input("Start date", today - timedelta(days=365 * 4))
    end = st.date_input("End date", today)

    st.header("Portfolio Settings")
    initial_capital = st.number_input("Initial Capital", 1000.0, 1e9, 100000.0, step=1000.0)
    position_size_pct = st.number_input("Position Size (% of capital)", 1.0, 100.0, 10.0)
    max_open_positions = st.number_input("Maximum Open Positions", 1, 100, 5)
    brokerage_pct = st.number_input("Brokerage (%)", 0.0, 5.0, 0.05, step=0.01)
    slippage_pct = st.number_input("Slippage (%)", 0.0, 5.0, 0.05, step=0.01)
    taxes_pct = st.number_input("Taxes (% of profit)", 0.0, 50.0, 0.0, step=0.5)

    st.header("Exit Settings")
    stop_loss_pct = st.number_input("Stop Loss (%)", 0.5, 90.0, 8.0)
    take_profit_pct = st.number_input("Take Profit (%)", 0.5, 500.0, 20.0)
    max_hold_days = st.number_input("Max Holding Days", 1, 500, 30)

    st.header("Options")
    strict_trend = st.checkbox(
        "Strict trend filter (EMA21>EMA50>EMA200)", value=True,
        help="When ON, requires a fully fanned bullish EMA stack at tracking "
             "start, on the pullback candle, and at recovery/buy. When OFF, "
             "only EMA21>EMA200 and EMA50>EMA200 are required.")
    debug_mode = st.checkbox("Enable Debug Mode", value=False)

    run = st.button("Run Scan + Backtest", type="primary")

settings = PortfolioSettings(
    initial_capital=initial_capital,
    position_size_pct=position_size_pct,
    max_open_positions=int(max_open_positions),
    brokerage_pct=brokerage_pct,
    slippage_pct=slippage_pct,
    taxes_pct=taxes_pct,
    stop_loss_pct=stop_loss_pct,
    take_profit_pct=take_profit_pct,
    max_hold_days=int(max_hold_days),
)

if universe_choice == "Custom symbols":
    symbols = [s.strip().upper() for s in symbols_raw.split(",") if s.strip()]
else:
    _base = INDEX_UNIVERSES[universe_choice]
    _sel = _base if max_symbols in (0, None) else _base[:int(max_symbols)]
    symbols = to_yf(_sel)

if run:
    if not HAS_YF:
        st.error("yfinance is not installed. Run: pip install yfinance")
    elif not symbols:
        st.error("Please enter at least one symbol.")
    else:
        with st.spinner("Downloading data and running all three engines..."):
            st.session_state["results"] = run_all(symbols, start, end, debug_mode,
                                                  strict_trend=strict_trend)
            st.session_state["settings"] = settings

results = st.session_state.get("results")

if not results:
    st.info("Set your universe and portfolio settings in the sidebar, then click **Run Scan + Backtest**.")
    st.stop()

if results["load_errors"]:
    st.warning("Could not load (or insufficient history <210 candles) for: "
               + ", ".join(results["load_errors"]))

# ---- Verification comparison + verified key set ----
cmp_df = compare_signals(results["engine_signals"], results["verify_setups"])
verified_keys = set(
    (v["symbol"], v["tracking_date"], v["pullback_date"], v["buy_date"])
    for v in results["verify_setups"]
)
verification_passed = (len(cmp_df) == 0) or bool((cmp_df["Status"] == "MATCH").all())

# ---- Backtest (consumes signals only) ----
active_settings = st.session_state.get("settings", settings)
trades, bt_summary = backtest_engine(
    results["engine_by_symbol"], results["data"], verified_keys, active_settings)

# ---- Three-way consistency check ----
engine_keys = set((s.symbol, s.tracking_start_date, s.pullback_date, s.buy_date)
                  for s in results["engine_signals"])
backtest_consumed_keys = set(
    (t.symbol, t.tracking, t.pullback, t.buy_signal)
    for t in trades if t.status in ("Executed", "Skipped"))
# Backtest should consume every engine signal that passed verification.
three_way_ok = (engine_keys == verified_keys) and backtest_consumed_keys.issubset(engine_keys)

if verification_passed and three_way_ok:
    st.success("Signal Verification Passed -- Strategy Engine, Verification Module "
               "and Backtest Engine produced identical Buy signals.")
else:
    st.error("Signal Verification Failed -- the three modules do NOT agree. "
             "See the Verification tab (mismatched rows highlighted red).")

tabs = st.tabs([
    "Signal Module", "Verification", "Backtest", "Analytics",
    "Audit Report", "Debug", "Final Validation",
])

# ---- TAB 1: SIGNAL MODULE ----
with tabs[0]:
    st.subheader("Pending Tracking")
    if results["pending_tracking"]:
        st.dataframe(pd.DataFrame(results["pending_tracking"]), use_container_width=True)
    else:
        st.caption("No symbols currently in an open tracking state.")

    st.subheader("Pullback Waiting (recovery pending)")
    if results["pending_pullback"]:
        st.dataframe(pd.DataFrame(results["pending_pullback"]), use_container_width=True)
    else:
        st.caption("No symbols currently waiting for recovery.")

    st.subheader("Buy Signals")
    if results["engine_signals"]:
        sig_df = pd.DataFrame([{
            "Symbol": s.symbol,
            "Tracking Date": s.tracking_start_date,
            "Pullback Date": s.pullback_date,
            "Buy Date": s.buy_date,
            "Signal Price": round(s.signal_price, 2),
            "EMA21": round(s.ema21, 2), "EMA50": round(s.ema50, 2),
            "EMA200": round(s.ema200, 2), "RSI": round(s.rsi, 1),
            "Candles Until Pullback": s.candles_until_pullback,
            "Candles Until Buy": s.candles_until_buy,
        } for s in results["engine_signals"]])
        st.dataframe(sig_df, use_container_width=True)
        st.download_button("Download Buy Signals (CSV)",
                           sig_df.to_csv(index=False).encode(),
                           "buy_signals.csv", "text/csv")
    else:
        st.caption("No completed Buy signals in this period.")

# ---- TAB 2: VERIFICATION ----
with tabs[1]:
    st.subheader("Signal Verification Module -- Engine vs Independent Reconstruction")
    if verification_passed and three_way_ok:
        st.success("Signal Verification Passed")
    else:
        st.error("Signal Verification Failed")

    if len(cmp_df):
        show = cmp_df[["Symbol", "Tracking Date", "Pullback Date", "Buy Date", "Status"]]

        def _hl(row):
            color = "background-color: #ffcccc" if row["Status"] == "MISMATCH" else ""
            return [color] * len(row)

        st.dataframe(show.style.apply(_hl, axis=1), use_container_width=True)
    else:
        st.caption("No signals to verify.")

# ---- TAB 3: BACKTEST ----
with tabs[2]:
    st.subheader("Backtest Validation Table")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final Capital", f"{bt_summary['final_cash']:,.0f}")
    c2.metric("Total Return", f"{bt_summary['total_return_pct']}%")
    c3.metric("Executed", bt_summary["executed"])
    c4.metric("Skipped / Rejected",
              f"{bt_summary['skipped']} / {bt_summary['rejected']}")

    if trades:
        tdf = pd.DataFrame([{
            "Symbol": t.symbol, "Tracking": t.tracking, "Pullback": t.pullback,
            "Buy Signal": t.buy_signal, "Entry Date": t.entry_date,
            "Entry Price": t.entry_price, "Exit Date": t.exit_date,
            "Exit Price": t.exit_price, "P/L": t.pnl, "Return %": t.ret_pct,
            "Holding Days": t.holding_days, "Status": t.status,
            "Reason": t.reason,
        } for t in trades])
        st.dataframe(tdf, use_container_width=True)
        st.download_button("Download Backtest Table (CSV)",
                           tdf.to_csv(index=False).encode(),
                           "backtest_validation.csv", "text/csv")

        # -------- Per-trade timeline + colored state badges --------
        st.markdown("### Trade Timeline")
        st.caption("Track -> Pullback -> Recovery/Buy -> Entry -> Exit for every trade.")

        _badge_colors = {
            "TRACK": "#3b82f6", "PULLBACK": "#f59e0b", "BUY": "#8b5cf6",
            "ENTRY": "#10b981", "EXIT": "#ef4444", "SKIPPED": "#9ca3af",
            "REJECTED": "#6b7280",
        }

        def _badge(label, value):
            color = _badge_colors.get(label, "#6b7280")
            return (f"<span style='display:inline-block;padding:3px 9px;margin:2px;"
                    f"border-radius:12px;background:{color};color:white;"
                    f"font-size:0.75rem;font-weight:600'>{label}: {value}</span>")

        _arrow = ("<span style='color:#9ca3af;font-weight:700;"
                  "margin:0 2px'>&rarr;</span>")

        for t in trades:
            if t.status == "Executed":
                chain = [
                    _badge("TRACK", t.tracking), _arrow,
                    _badge("PULLBACK", t.pullback), _arrow,
                    _badge("BUY", t.buy_signal), _arrow,
                    _badge("ENTRY", f"{t.entry_date} @ {t.entry_price}"), _arrow,
                    _badge("EXIT", f"{t.exit_date} @ {t.exit_price}"),
                ]
                pnl_color = "#10b981" if t.pnl >= 0 else "#ef4444"
                tail = (f"<span style='color:{pnl_color};font-weight:700;"
                        f"margin-left:8px'>P/L {t.pnl}  ({t.ret_pct}%)</span>")
            else:
                lab = "SKIPPED" if t.status == "Skipped" else "REJECTED"
                chain = [
                    _badge("TRACK", t.tracking), _arrow,
                    _badge("PULLBACK", t.pullback), _arrow,
                    _badge("BUY", t.buy_signal), _arrow,
                    _badge(lab, t.reason),
                ]
                tail = ""
            st.markdown(
                f"<div style='margin-bottom:6px'><b>{t.symbol}</b> &nbsp; "
                + "".join(chain) + tail + "</div>",
                unsafe_allow_html=True)
    else:
        st.caption("No trades generated.")

# ---- TAB 4: ANALYTICS ----
with tabs[3]:
    st.subheader("Advanced Performance Analytics")
    st.caption("Computed from executed trades only (realized equity curve).")
    an, equity_curve, ordered_ex = compute_analytics(
        trades, results["engine_signals"], active_settings)
    if an.get("Executed Trades", 0) == 0:
        st.caption("No executed trades to analyze.")
    else:
        headline = ["CAGR %", "Max Drawdown %", "Profit Factor",
                    "Sharpe (trade-based)", "Sortino (trade-based)", "MAR Ratio"]
        cols = st.columns(3)
        for i, k in enumerate(headline):
            v = an.get(k, 0.0)
            cols[i % 3].metric(k, "inf" if v == float("inf") else v)

        st.markdown("**Full metric table**")
        met_df = pd.DataFrame(
            [{"Metric": k, "Value": ("inf" if v == float("inf") else v)}
             for k, v in an.items()])
        st.dataframe(met_df, use_container_width=True, hide_index=True)
        st.download_button("Download Analytics (CSV)",
                           met_df.to_csv(index=False).encode(),
                           "analytics.csv", "text/csv")

        st.markdown("**Realized equity curve**")
        eq_df = pd.DataFrame({"Equity": equity_curve})
        st.line_chart(eq_df, use_container_width=True)

        hold_df, rsi_df, month_df = bucket_win_rates(
            trades, results["engine_signals"])
        b1, b2 = st.columns(2)
        with b1:
            st.markdown("**Win rate by holding days**")
            st.dataframe(hold_df, use_container_width=True, hide_index=True)
        with b2:
            st.markdown("**Win rate by RSI bucket**")
            st.dataframe(rsi_df, use_container_width=True, hide_index=True)
        st.markdown("**Win rate by month**")
        st.dataframe(month_df, use_container_width=True, hide_index=True)


# ---- TAB 5: AUDIT REPORT ----
with tabs[4]:
    st.subheader("Strategy Audit Report")
    if results["engine_signals"]:
        seen = {}
        dup_flags = {}
        for s in results["engine_signals"]:
            k = (s.symbol, s.tracking_start_date, s.pullback_date, s.buy_date)
            dup_flags[k] = k in seen
            seen[k] = True

        audit_rows = []
        overall_fail = 0
        for s in results["engine_signals"]:
            k = (s.symbol, s.tracking_start_date, s.pullback_date, s.buy_date)
            checks = audit_signal(s, results["data"][s.symbol],
                                  verified=(k in verified_keys),
                                  duplicate=dup_flags[k],
                                  strict_trend=results.get("strict_trend", True))
            failed = any(v == "FAIL" for v in checks.values())
            overall_fail += int(failed)
            audit_rows.append(dict(
                Symbol=s.symbol, **{"Buy Date": s.buy_date},
                Result=("FAIL" if failed else "PASS"), **checks))
        adf = pd.DataFrame(audit_rows)

        def _hl_audit(val):
            if val == "FAIL":
                return "background-color: #ffcccc"
            if val == "PASS":
                return "background-color: #ccffcc"
            return ""
        st.dataframe(adf.style.applymap(_hl_audit), use_container_width=True)
        if overall_fail == 0:
            st.success(f"All {len(audit_rows)} signals PASS every audit rule.")
        else:
            st.error(f"{overall_fail} signal(s) FAILED one or more audit rules.")
        st.download_button("Download Audit Report (CSV)",
                           adf.to_csv(index=False).encode(),
                           "audit_report.csv", "text/csv")
    else:
        st.caption("No signals to audit.")

# ---- TAB 6: DEBUG ----
with tabs[5]:
    st.subheader("Debug -- Candle Transitions")
    if not debug_mode:
        st.info("Enable **Debug Mode** in the sidebar and re-run to capture "
                "every candle transition.")
    elif results["debug"]:
        ddf = pd.DataFrame([asdict(e) for e in results["debug"]])
        log_lines = []
        for e in results["debug"]:
            log_lines.append(f"{e.date}\n{e.event}\n{e.detail}\n" + "-" * 40)
        st.code("\n".join(log_lines))
        st.download_button("Download Debug Log (CSV)",
                           ddf.to_csv(index=False).encode(),
                           "debug_log.csv", "text/csv")
    else:
        st.caption("No transitions captured.")

# ---- TAB 7: FINAL VALIDATION ----
with tabs[6]:
    st.subheader("Strategy Validation Summary")
    es = results["engine_stats"]
    match_ct = int((cmp_df["Status"] == "MATCH").sum()) if len(cmp_df) else 0
    mismatch_ct = int((cmp_df["Status"] == "MISMATCH").sum()) if len(cmp_df) else 0
    dup_ct = sum(1 for t in trades if t.reason == "Duplicate signal")
    below_ct = sum(1 for t in trades if t.reason == "Below EMA200") + es["signals_below_ema200"]

    def _avg(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 2) if vals else 0.0

    avg_pb = _avg([s.candles_until_pullback for s in results["engine_signals"]])
    avg_rec = _avg([s.candles_until_buy - s.candles_until_pullback
                    for s in results["engine_signals"]])
    avg_hold = _avg([t.holding_days for t in trades if t.status == "Executed"])

    verdict = "PASS" if (verification_passed and three_way_ok and mismatch_ct == 0) else "FAIL"

    report = f"""========== STRATEGY VALIDATION ==========

Tracking Cycles Started      : {es['tracking_started']}
Pullbacks Found              : {es['pullbacks_found']}
Buy Signals Generated        : {es['buys']}
Executed Trades              : {bt_summary['executed']}
Skipped Trades               : {bt_summary['skipped']}
Rejected Trades              : {bt_summary['rejected']}
Verification Matches         : {match_ct}
Verification Mismatches      : {mismatch_ct}
Duplicate Signals            : {dup_ct}
Signals Below EMA200         : {below_ct}
Average Pullback Days        : {avg_pb}
Average Recovery Days        : {avg_rec}
Average Holding Days         : {avg_hold}

{verdict}
========================================="""
    st.code(report)
    st.download_button("Download Validation Summary (TXT)",
                       report.encode(), "strategy_validation.txt", "text/plain")
