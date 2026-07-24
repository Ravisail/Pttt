"""
TradingView EMA Strategy Scanner & Backtester
Replicates exact Pine Script logic in Python/Streamlit.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed


# ===================================================================
# BUILT-IN STOCK UNIVERSES  --  NIFTY 200 / NIFTY 500
# ===================================================================
#
# Stored directly in source so the "Built-in NIFTY 200" / "Built-in
# NIFTY 500" universe options never need an internet lookup to know
# their own constituents (only the per-symbol price history is still
# downloaded from Yahoo Finance).
#
# IMPORTANT CAVEAT: NSE rebalances these indices periodically (adds/
# removes/renames constituents), so any hardcoded list is a point-in-
# time snapshot, not a live feed. NIFTY_500_SYMBOLS below was sourced
# from an older public NSE constituent snapshot and is known to
# contain a handful of since-renamed or since-delisted/merged tickers
# (e.g. ALBK, ANDHRABANK, HDFC, LTI). NIFTY_200_SYMBOLS is a curated
# large/mid-cap approximation, not an official NSE Nifty 200 export.
# For guaranteed current accuracy, use "Custom Upload" with a freshly
# exported NSE index constituent file instead of these built-ins.
# ===================================================================

NIFTY_200_SYMBOLS = [
    'AARTIIND', 'AAVAS', 'ABB', 'ABBOTINDIA', 'ACC', 'ADANIENT', 'ADANIPORTS', 'AJANTPHARM',
    'ALKEM', 'AMBER', 'AMBUJACEM', 'ANGELONE', 'APLAPOLLO', 'APOLLOHOSP', 'APTUS', 'ARVIND',
    'ASIANPAINT', 'ASTRAL', 'AUBANK', 'AUROPHARMA', 'AXISBANK', 'BAJAJ-AUTO', 'BAJAJFINSV', 'BAJFINANCE',
    'BANDHANBNK', 'BANKBARODA', 'BANKINDIA', 'BATAINDIA', 'BAYERCROP', 'BEL', 'BEML', 'BERGEPAINT',
    'BHARTIARTL', 'BHEL', 'BIOCON', 'BOSCHLTD', 'BPCL', 'BRIGADE', 'BRITANNIA', 'BSE',
    'BSOFT', 'CAMS', 'CANBK', 'CANFINHOME', 'CDSL', 'CENTURYPLY', 'CHALET', 'CHAMBLFERT',
    'CHOLAFIN', 'CIPLA', 'COALINDIA', 'COCHINSHIP', 'COFORGE', 'COLPAL', 'CONCOR', 'COROMANDEL',
    'CREDITACC', 'CROMPTON', 'CUMMINSIND', 'CYIENT', 'DABUR', 'DALBHARAT', 'DEEPAKNTR', 'DIVISLAB',
    'DIXON', 'DLF', 'DMART', 'DRREDDY', 'EICHERMOT', 'EIHOTEL', 'EMAMILTD', 'FEDERALBNK',
    'GILLETTE', 'GLAXO', 'GLENMARK', 'GMRINFRA', 'GNFC', 'GODREJCP', 'GODREJPROP', 'GRANULES',
    'GRASIM', 'GREENPLY', 'GRINDWELL', 'GRSE', 'HAL', 'HAVELLS', 'HCLTECH', 'HDFCBANK',
    'HDFCLIFE', 'HEIDELBERG', 'HEROMOTOCO', 'HINDALCO', 'HINDUNILVR', 'HOMEFIRST', 'HUDCO', 'IBREALEST',
    'ICICIBANK', 'IDFCFIRSTB', 'IIFL', 'INDHOTEL', 'INDIAMART', 'INDIANB', 'INDUSINDBK', 'INFY',
    'INTELLECT', 'IOB', 'IOC', 'IPCALAB', 'IRCON', 'IRCTC', 'IRFC', 'ITC',
    'JINDALSTEL', 'JKCEMENT', 'JSWSTEEL', 'JUBLFOOD', 'JUSTDIAL', 'KAJARIACER', 'KOTAKBANK', 'KPITTECH',
    'KRBL', 'LAURUSLABS', 'LICHSGFIN', 'LT', 'LTIM', 'LTTS', 'LUPIN', 'M&M',
    'MANAPPURAM', 'MARICO', 'MARUTI', 'MASFIN', 'MAZDOCK', 'MCDOWELL-N', 'MCX', 'MHRIL',
    'MOTHERSON', 'MPHASIS', 'MUTHOOTFIN', 'NALCO', 'NATCOPHARM', 'NAUKRI', 'NAVINFLUOR', 'NBCC',
    'NESTLEIND', 'NIITTECH', 'NMDC', 'NTPC', 'OBEROIRLTY', 'OFSS', 'ONGC', 'ORIENTCEM',
    'PAGEIND', 'PAYTM', 'PERSISTENT', 'PFC', 'PFIZER', 'PGHH', 'PHOENIXLTD', 'PIDILITIND',
    'PIIND', 'PNB', 'PNBHOUSING', 'POLICYBZR', 'POLYCAB', 'POONAWALLA', 'POWERGRID', 'PRESTIGE',
    'PRISMCEM', 'RADICO', 'RAILTEL', 'RALLIS', 'RAMCOCEM', 'RATNAMANI', 'RAYMOND', 'RBLBANK',
    'RECLTD', 'RELAXO', 'RELIANCE', 'REPCO', 'RVNL', 'SAIL', 'SANOFI', 'SBICARD',
    'SBILIFE', 'SBIN', 'SHOPERSTOP', 'SHREECEM', 'SIEMENS', 'SKFINDIA', 'SOBHA', 'SONATSOFTW',
    'SRF', 'STARCEMENT', 'SUNPHARMA', 'SUNTECK', 'SUPREMEIND', 'SYNGENE', 'TATACHEM', 'TATAELXSI',
    'TATAMOTORS', 'TATASTEEL', 'TCS', 'TECHM', 'THERMAX', 'THOMASCOOK', 'TIMKEN', 'TITAN',
    'TORNTPHARM', 'TRENT', 'UBL', 'ULTRACEMCO', 'UNIONBANK', 'UPL', 'VBL', 'VEDL',
    'VIPIND', 'VMART', 'VOLTAS', 'WESTLIFE', 'WHIRLPOOL', 'WIPRO', 'YESBANK', 'ZENSARTECH',
    'ZOMATO'
]

NIFTY_500_SYMBOLS = [
    '3MINDIA', 'AAVAS', 'ABB', 'ABCAPITAL', 'ABFRL', 'ACC', 'ADANIGREEN', 'ADANIPORTS',
    'ADANIPOWER', 'ADANITRANS', 'ADVENZYMES', 'AEGISCHEM', 'AIAENG', 'AJANTPHARM', 'AKZOINDIA', 'ALBK',
    'ALKEM', 'ALLCARGO', 'AMARAJABAT', 'AMBUJACEM', 'ANDHRABANK', 'APLAPOLLO', 'APLLTD', 'APOLLOHOSP',
    'APOLLOTYRE', 'ASHOKA', 'ASHOKLEY', 'ASIANPAINT', 'ASTERDM', 'ASTRAL', 'ASTRAZEN', 'ATUL',
    'AUBANK', 'AUROPHARMA', 'AVANTIFEED', 'AXISBANK', 'BAJAJ-AUTO', 'BAJAJCON', 'BAJAJELEC', 'BAJAJFINSV',
    'BAJAJHLDNG', 'BAJFINANCE', 'BALKRISIND', 'BALMLAWRIE', 'BALRAMCHIN', 'BANDHANBNK', 'BANKBARODA', 'BANKINDIA',
    'BASF', 'BATAINDIA', 'BBTC', 'BDL', 'BEL', 'BEML', 'BERGEPAINT', 'BHARATFORG',
    'BHARTIARTL', 'BHEL', 'BIOCON', 'BIRLACORPN', 'BLISSGVS', 'BLUEDART', 'BLUESTARCO', 'BOMDYEING',
    'BOSCHLTD', 'BPCL', 'BRIGADE', 'BRITANNIA', 'BSE', 'CADILAHC', 'CANBK', 'CANFINHOME',
    'CAPLIPOINT', 'CARBORUNIV', 'CARERATING', 'CASTROLIND', 'CCL', 'CDSL', 'CEATLTD', 'CENTRALBK',
    'CENTURYPLY', 'CERA', 'CESC', 'CGPOWER', 'CHAMBLFERT', 'CHENNPETRO', 'CHOLAFIN', 'CHOLAHLDNG',
    'CIPLA', 'COALINDIA', 'COCHINSHIP', 'COFFEEDAY', 'COLPAL', 'CONCOR', 'COROMANDEL', 'CORPBANK',
    'COX&KINGS', 'CREDITACC', 'CRISIL', 'CROMPTON', 'CUB', 'CUMMINSIND', 'CYIENT', 'DABUR',
    'DBCORP', 'DBL', 'DCAL', 'DCBBANK', 'DCMSHRIRAM', 'DEEPAKFERT', 'DEEPAKNTR', 'DELTACORP',
    'DHFL', 'DISHTV', 'DIVISLAB', 'DIXON', 'DLF', 'DMART', 'DRREDDY', 'ECLERX',
    'EDELWEISS', 'EICHERMOT', 'EIDPARRY', 'EIHOTEL', 'ELGIEQUIP', 'EMAMILTD', 'ENDURANCE', 'ENGINERSIN',
    'EQUITAS', 'ERIS', 'ESCORTS', 'ESSELPACK', 'EXIDEIND', 'FCONSUMER', 'FDC', 'FEDERALBNK',
    'FINCABLES', 'FINEORG', 'FINPIPE', 'FLFL', 'FORTIS', 'FRETAIL', 'FSL', 'GAIL',
    'GALAXYSURF', 'GAYAPROJ', 'GDL', 'GEPIL', 'GESHIP', 'GET&D', 'GHCL', 'GICRE',
    'GILLETTE', 'GLAXO', 'GLENMARK', 'GMDCLTD', 'GMRINFRA', 'GNFC', 'GODFRYPHLP', 'GODREJAGRO',
    'GODREJCP', 'GODREJIND', 'GODREJPROP', 'GPPL', 'GRANULES', 'GRAPHITE', 'GRASIM', 'GREAVESCOT',
    'GRINDWELL', 'GRUH', 'GSFC', 'GSKCONS', 'GSPL', 'GUJALKALI', 'GUJFLUORO', 'GUJGASLTD',
    'GULFOILLUB', 'HAL', 'HATHWAY', 'HATSUN', 'HAVELLS', 'HCLTECH', 'HDFC', 'HDFCAMC',
    'HDFCBANK', 'HDFCLIFE', 'HEG', 'HEIDELBERG', 'HERITGFOOD', 'HEROMOTOCO', 'HEXAWARE', 'HFCL',
    'HIMATSEIDE', 'HINDALCO', 'HINDCOPPER', 'HINDPETRO', 'HINDUNILVR', 'HINDZINC', 'HONAUT', 'HSCL',
    'HUDCO', 'IBREALEST', 'IBULHSGFIN', 'IBULISL', 'IBVENTURES', 'ICICIBANK', 'ICICIGI', 'ICICIPRULI',
    'ICRA', 'IDBI', 'IDEA', 'IDFC', 'IDFCFIRSTB', 'IEX', 'IFBIND', 'IFCI',
    'IGL', 'INDHOTEL', 'INDIACEM', 'INDIANB', 'INDIGO', 'INDOCO', 'INDOSTAR', 'INDUSINDBK',
    'INFIBEAM', 'INFRATEL', 'INFY', 'INOXLEISUR', 'INOXWIND', 'INTELLECT', 'IOB', 'IOC',
    'IPCALAB', 'IRB', 'IRCON', 'ISEC', 'ITC', 'ITDC', 'ITDCEM', 'ITI',
    'J&KBANK', 'JAGRAN', 'JAICORPLTD', 'JAMNAAUTO', 'JBCHEPHARM', 'JETAIRWAYS', 'JINDALSAW', 'JINDALSTEL',
    'JISLJALEQS', 'JKCEMENT', 'JKLAKSHMI', 'JKPAPER', 'JKTYRE', 'JMFINANCIL', 'JPASSOCIAT', 'JSL',
    'JSLHISAR', 'JSWENERGY', 'JSWSTEEL', 'JUBILANT', 'JUBLFOOD', 'JUSTDIAL', 'JYOTHYLAB', 'KAJARIACER',
    'KALPATPOWR', 'KANSAINER', 'KARURVYSYA', 'KEC', 'KEI', 'KIOCL', 'KIRLOSENG', 'KNRCON',
    'KOLTEPATIL', 'KOTAKBANK', 'KPRMILL', 'KRBL', 'KSCL', 'KTKBANK', 'L&TFH', 'LAKSHVILAS',
    'LALPATHLAB', 'LAURUSLABS', 'LAXMIMACH', 'LEMONTREE', 'LICHSGFIN', 'LINDEINDIA', 'LT', 'LTI',
    'LTTS', 'LUPIN', 'LUXIND', 'M&M', 'M&MFIN', 'MAGMA', 'MAHABANK', 'MAHINDCIE',
    'MAHLOG', 'MAHSCOOTER', 'MAHSEAMLES', 'MANAPPURAM', 'MARICO', 'MARUTI', 'MASFIN', 'MAXINDIA',
    'MCDOWELL-N', 'MFSL', 'MGL', 'MHRIL', 'MINDACORP', 'MINDAIND', 'MINDTREE', 'MMTC',
    'MOIL', 'MONSANTO', 'MOTHERSUMI', 'MOTILALOFS', 'MPHASIS', 'MRF', 'MRPL', 'MUTHOOTFIN',
    'NATCOPHARM', 'NATIONALUM', 'NAUKRI', 'NAVINFLUOR', 'NBCC', 'NBVENTURES', 'NCC', 'NESCO',
    'NETWORK18', 'NFL', 'NH', 'NHPC', 'NIACL', 'NIITTECH', 'NILKAMAL', 'NLCINDIA',
    'NMDC', 'NTPC', 'OBEROIRLTY', 'OFSS', 'OIL', 'OMAXE', 'ONGC', 'ORIENTBANK',
    'ORIENTCEM', 'ORIENTELEC', 'PAGEIND', 'PARAGMILK', 'PCJEWELLER', 'PEL', 'PERSISTENT', 'PETRONET',
    'PFC', 'PFIZER', 'PGHH', 'PGHL', 'PHILIPCARB', 'PHOENIXLTD', 'PIDILITIND', 'PIIND',
    'PNB', 'PNBHOUSING', 'PNCINFRA', 'POWERGRID', 'PRAJIND', 'PRESTIGE', 'PRSMJOHNSN', 'PTC',
    'PVR', 'QUESS', 'RADICO', 'RAIN', 'RAJESHEXPO', 'RALLIS', 'RAMCOCEM', 'RAYMOND',
    'RBLBANK', 'RCF', 'RCOM', 'RECLTD', 'REDINGTON', 'RELAXO', 'RELCAPITAL', 'RELIANCE',
    'RELINFRA', 'RENUKA', 'REPCOHOME', 'RHFL', 'RITES', 'RKFORGE', 'RNAM', 'RPOWER',
    'RUPA', 'SADBHAV', 'SAIL', 'SANOFI', 'SBILIFE', 'SBIN', 'SCHAEFFLER', 'SCI',
    'SFL', 'SHANKARA', 'SHARDACROP', 'SHILPAMED', 'SHK', 'SHOPERSTOP', 'SHREECEM', 'SHRIRAMCIT',
    'SIEMENS', 'SIS', 'SJVN', 'SKFINDIA', 'SOBHA', 'SOLARINDS', 'SONATSOFTW', 'SOUTHBANK',
    'SPARC', 'SPTL', 'SREINFRA', 'SRF', 'SRTRANSFIN', 'STAR', 'STARCEMENT', 'STRTECH',
    'SUDARSCHEM', 'SUNCLAYLTD', 'SUNDARMFIN', 'SUNDRMFAST', 'SUNPHARMA', 'SUNTECK', 'SUNTV', 'SUPRAJIT',
    'SUPREMEIND', 'SUVEN', 'SUZLON', 'SWANENERGY', 'SYMPHONY', 'SYNDIBANK', 'SYNGENE', 'TAKE',
    'TATACHEM', 'TATACOFFEE', 'TATAELXSI', 'TATAGLOBAL', 'TATAINVEST', 'TATAMOTORS', 'TATAMTRDVR', 'TATAPOWER',
    'TATASTEEL', 'TCNSBRANDS', 'TCS', 'TEAMLEASE', 'TECHM', 'THERMAX', 'THOMASCOOK', 'THYROCARE',
    'TIINDIA', 'TIMETECHNO', 'TIMKEN', 'TITAN', 'TNPL', 'TORNTPHARM', 'TORNTPOWER', 'TRENT',
    'TRIDENT', 'TRITURBINE', 'TTKPRESTIG', 'TV18BRDCST', 'TVSMOTOR', 'TVTODAY', 'UBL', 'UCOBANK',
    'UFLEX', 'UJJIVAN', 'ULTRACEMCO', 'UNIONBANK', 'UPL', 'VAKRANGEE', 'VARROC', 'VBL',
    'VEDL', 'VENKEYS', 'VGUARD', 'VINATIORGA', 'VIPIND', 'VMART', 'VOLTAS', 'VRLLOG',
    'VSTIND', 'VTL', 'WABAG', 'WABCOINDIA', 'WELCORP', 'WELSPUNIND', 'WHIRLPOOL', 'WIPRO',
    'WOCKPHARMA', 'YESBANK', 'ZEEL', 'ZENSARTECH', 'ZYDUSWELL'
]


# ===================================================================
# STRATEGY MODULE  --  DO NOT MODIFY
# ===================================================================

def calculate_ema_tv(series: pd.Series, length: int) -> pd.Series:
    """
    Calculate EMA matching TradingView's built-in ta.ema(source, length)
    EXACTLY, per Pine's own documented equivalent:

        alpha = 2 / (length + 1)
        sum := na(sum[1]) ? source : alpha * source + (1 - alpha) * nz(sum[1])

    Key property (verified against Pine's reference implementation): there
    is NO 'length'-bar warm-up and NO SMA seed. The very first bar simply
    seeds the EMA with its own source value (na(sum[1]) is true on bar 0),
    and every bar after that recurses normally. 'length' only sets alpha.

    If a source value is NaN, the output is NaN for that bar AND the seed
    resets (na(sum[1]) becomes true again), so the next valid value reseeds
    with its own source value -- mirroring Pine's na-handling exactly.
    """
    alpha = 2.0 / (length + 1)
    ema = pd.Series(index=series.index, dtype=float)

    prev_ema = np.nan
    for i in range(len(series)):
        val = series.iloc[i]
        if pd.isna(val):
            ema.iloc[i] = np.nan
            prev_ema = np.nan  # sum[1] becomes na -> next valid bar reseeds
            continue

        if pd.isna(prev_ema):
            new_ema = float(val)  # na(sum[1]) ? source : ...
        else:
            new_ema = alpha * val + (1 - alpha) * prev_ema

        ema.iloc[i] = new_ema
        prev_ema = new_ema

    return ema


def generate_signals(df: pd.DataFrame, ema_length: int) -> pd.DataFrame:
    """
    Generate BUY/SELL signals replicating TradingView state logic exactly.
    """
    df = df.copy()

    # Indicators
    df['ema_high'] = calculate_ema_tv(df['High'], ema_length)
    df['ema_low'] = calculate_ema_tv(df['Low'], ema_length)
    df['ema_close'] = calculate_ema_tv(df['Close'], ema_length)

    df['channel_height'] = df['ema_high'] - df['ema_low']
    df['body_size'] = (df['Close'] - df['Open']).abs()
    df['total_range'] = df['High'] - df['Low']

    # Body Ratio: 0 when total_range == 0
    def body_ratio(row):
        tr = row['total_range']
        if tr == 0:
            return 0.0
        return row['body_size'] / tr

    df['body_ratio'] = df.apply(body_ratio, axis=1)

    # Signal generation with TradingView-style state variables
    signals = []
    last_was_buy = False
    last_was_sell = False

    for idx in df.index:
        row = df.loc[idx]

        if pd.isna(row['ema_high']) or pd.isna(row['ema_low']):
            signals.append(None)
            continue

        # BUY conditions
        buy_condition = (
            row['Close'] > row['ema_high']
            and (row['High'] - row['Low']) >= (row['ema_high'] - row['ema_low'])
            and row['body_ratio'] >= 0.66
            and not last_was_buy
        )

        # SELL conditions
        sell_condition = (
            row['Close'] < row['ema_low']
            and (row['High'] - row['Low']) >= (row['ema_high'] - row['ema_low'])
            and row['body_ratio'] >= 0.66
            and not last_was_sell
        )

        if buy_condition:
            signals.append('BUY')
            last_was_buy = True
            last_was_sell = False
        elif sell_condition:
            signals.append('SELL')
            last_was_sell = True
            last_was_buy = False
        else:
            signals.append(None)

    df['signal'] = signals
    return df


# ===================================================================
# ANALYTICS & ENTRY FILTERS MODULE
# ===================================================================
#
# Purely additive, read-only diagnostics computed ONCE per symbol from
# the already-generated 'signal' column above -- NEVER recalculates or
# alters BUY/SELL signals, EMA High/Low/Close, or any STRATEGY MODULE
# value. Entry Filters here only ever decide whether a BUY signal that
# has already fired is actually taken by the BACKTESTING MODULE; they
# can never create, remove, or move a signal, and they never touch
# SELL signals/exits at all.
# ===================================================================

def _rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """Standard Wilder RSI. Pure analytics -- not part of the strategy."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100.0)
    return rsi


def _atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Standard Average True Range (Wilder smoothing). Pure analytics."""
    prev_close = df['Close'].shift(1)
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - prev_close).abs(),
        (df['Low'] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()


def _r2(val):
    """Round to 2 decimals, passing through None/NaN untouched (analytics
    snapshot values can be missing during warm-up)."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return round(float(val), 2)


def _r4(val):
    """Round to 4 decimals, passing through None/NaN untouched."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return round(float(val), 4)


def compute_analytics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add every diagnostic column the Enhanced Trade Log / Entry Filters
    need, computed ONCE per symbol over the full (warm-up-included)
    dataset so nothing is ever recalculated later in the pipeline.
    Read-only with respect to 'signal', 'ema_high', 'ema_low',
    'ema_close' -- those already exist from generate_signals() above
    and are never overwritten here.
    """
    df = df.copy()

    # --- Extra EMAs (analytics only; the strategy's own ema_high/
    # ema_low/ema_close are untouched) ---
    df['ema21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['ema50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['ema200'] = df['Close'].ewm(span=200, adjust=False).mean()

    # --- EMA Channel Width % and distances ---
    df['ema_channel_width_pct'] = ((df['ema_high'] - df['ema_low']) / df['Close']) * 100
    df['dist_close_ema21_pct'] = ((df['Close'] - df['ema21']) / df['ema21']) * 100
    df['dist_close_ema50_pct'] = ((df['Close'] - df['ema50']) / df['ema50']) * 100
    df['dist_close_ema200_pct'] = ((df['Close'] - df['ema200']) / df['ema200']) * 100

    # --- Candle geometry ---
    rng = (df['High'] - df['Low']).replace(0, np.nan)
    df['candle_range_pct'] = (df['High'] - df['Low']) / df['Close'] * 100
    df['candle_body'] = (df['Close'] - df['Open']).abs()
    df['candle_color'] = np.where(df['Close'] > df['Open'], 'Green',
                            np.where(df['Close'] < df['Open'], 'Red', 'Doji'))
    df['upper_wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['lower_wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['upper_wick_ratio'] = (df['upper_wick'] / rng).fillna(0.0)
    df['lower_wick_ratio'] = (df['lower_wick'] / rng).fillna(0.0)
    df['close_position_in_candle_pct'] = ((df['Close'] - df['Low']) / rng * 100).fillna(50.0)
    df['open_position_in_candle_pct'] = ((df['Open'] - df['Low']) / rng * 100).fillna(50.0)

    # --- Gap analysis (vs. previous trading day's High/Low) ---
    prev_high = df['High'].shift(1)
    prev_low = df['Low'].shift(1)
    gap_up_pct = (df['Open'] - prev_high) / prev_high * 100
    gap_down_pct = (prev_low - df['Open']) / prev_low * 100
    # Signed "the day's gap": positive = gap up, negative = gap down
    df['gap_pct'] = np.where(gap_up_pct > 0, gap_up_pct, np.where(gap_down_pct > 0, -gap_down_pct, 0.0))
    df['prev_day_gap_pct'] = df['gap_pct'].shift(1)
    df['gap_2d_ago_pct'] = df['gap_pct'].shift(2)
    df['gap_3d_ago_pct'] = df['gap_pct'].shift(3)
    # Next day's gap (used by the Next Day Entry Gap Filter): gap of the
    # NEXT trading day's Open (or Close) vs. TODAY's Close. Both bases are
    # computed here so the filter can match whichever Entry Type basis is
    # selected -- 'next_day_gap_pct' (the one actually used) is set to the
    # matching one inside evaluate_entry_filters(), based on settings.
    next_open = df['Open'].shift(-1)
    next_close = df['Close'].shift(-1)
    df['next_day_gap_open_pct'] = (next_open - df['Close']) / df['Close'] * 100
    df['next_day_gap_close_pct'] = (next_close - df['Close']) / df['Close'] * 100
    df['next_day_gap_pct'] = df['next_day_gap_open_pct']  # default basis; may be overridden by filter settings

    # --- Volume & momentum ---
    if 'Volume' in df.columns:
        df['avg_volume_20'] = df['Volume'].rolling(20, min_periods=1).mean()
        df['volume_ratio'] = df['Volume'] / df['avg_volume_20'].replace(0, np.nan)
    else:
        df['Volume'] = np.nan
        df['avg_volume_20'] = np.nan
        df['volume_ratio'] = np.nan
    df['rsi'] = _rsi(df['Close'], 14)
    df['atr'] = _atr(df, 14)
    df['atr_pct'] = df['atr'] / df['Close'] * 100
    df['highest_close_30d'] = df['Close'].rolling(30, min_periods=1).max()
    df['lowest_close_30d'] = df['Close'].rolling(30, min_periods=1).min()
    df['dist_from_30d_high_pct'] = (df['Close'] - df['highest_close_30d']) / df['highest_close_30d'] * 100
    df['dist_from_30d_low_pct'] = (df['Close'] - df['lowest_close_30d']) / df['lowest_close_30d'] * 100

    return df


DEFAULT_FILTER_SETTINGS = {
    'ema_width_enabled': False, 'ema_width_min': 1.0, 'ema_width_max': 2.0,
    'body_ratio_enabled': False, 'body_ratio_min': 0.70, 'body_ratio_max': 0.80,
    'candle_range_enabled': False, 'candle_range_max': 4.0,
    'prev_gap_enabled': False, 'prev_gap_lookback': 3, 'prev_gap_up_max': 2.0, 'prev_gap_down_max': 2.0,
    'next_gap_enabled': False, 'next_gap_up_max': 1.0, 'entry_type': 'Next Day Open',
}


def evaluate_entry_filters(df: pd.DataFrame, settings: dict = None) -> pd.DataFrame:
    """
    Evaluate every configured Entry Filter for each BUY signal row, once,
    and store the pass/fail result plus a combined 'final_entry_approved'
    + 'rejection_reason' -- consumed later by build_trigger_events() to
    decide whether a BUY signal is actually taken. Never touches the
    'signal' column itself, and never evaluates filters for SELL rows
    (filters are entry-only, per spec).
    """
    settings = {**DEFAULT_FILTER_SETTINGS, **(settings or {})}
    df = df.copy()
    n = len(df)

    ema_pass = pd.Series(True, index=df.index)
    body_pass = pd.Series(True, index=df.index)
    range_pass = pd.Series(True, index=df.index)
    prev_gap_pass = pd.Series(True, index=df.index)
    next_gap_pass = pd.Series(True, index=df.index)
    df['max_gap_in_lookback_pct'] = 0.0

    if settings['ema_width_enabled']:
        ema_pass = df['ema_channel_width_pct'].between(settings['ema_width_min'], settings['ema_width_max'])

    if settings['body_ratio_enabled']:
        body_pass = df['body_ratio'].between(settings['body_ratio_min'], settings['body_ratio_max'])

    if settings['candle_range_enabled']:
        range_pass = df['candle_range_pct'] <= settings['candle_range_max']

    if settings['prev_gap_enabled']:
        lookback = int(settings['prev_gap_lookback'])
        gap_cols = [df['gap_pct'].shift(k) for k in range(1, lookback + 1)]
        gap_window = pd.concat(gap_cols, axis=1) if gap_cols else pd.DataFrame(index=df.index)
        max_abs_gap = gap_window.abs().max(axis=1).fillna(0.0)
        df['max_gap_in_lookback_pct'] = max_abs_gap
        gap_up_ok = gap_window.where(gap_window > 0).max(axis=1).fillna(0.0) <= settings['prev_gap_up_max']
        gap_down_ok = gap_window.where(gap_window < 0).min(axis=1).fillna(0.0).abs() <= settings['prev_gap_down_max']
        prev_gap_pass = gap_up_ok & gap_down_ok

    if settings['next_gap_enabled']:
        # Match the gap check to whichever price the trade will actually
        # enter at: 'Next Day Close' entry uses the Close-basis gap,
        # otherwise (default) the Open-basis gap is used.
        if settings.get('entry_type') == 'Next Day Close':
            df['next_day_gap_pct'] = df['next_day_gap_close_pct']
        else:
            df['next_day_gap_pct'] = df['next_day_gap_open_pct']
        next_gap_pass = df['next_day_gap_pct'] <= settings['next_gap_up_max']

    df['ema_width_filter_passed'] = ema_pass
    df['body_ratio_filter_passed'] = body_pass
    df['candle_range_filter_passed'] = range_pass
    df['previous_gap_filter_passed'] = prev_gap_pass
    df['next_day_gap_filter_passed'] = next_gap_pass

    final_approved = ema_pass & body_pass & range_pass & prev_gap_pass & next_gap_pass
    df['final_entry_approved'] = final_approved

    reasons = pd.Series('', index=df.index, dtype=object)
    reason_map = [
        (~ema_pass, 'EMA Channel Width'),
        (~body_pass, 'Body Ratio'),
        (~range_pass, 'Candle Range'),
        (~prev_gap_pass, 'Previous Gap'),
        (~next_gap_pass, 'Next Day Gap'),
    ]
    for failed_mask, label in reason_map:
        reasons = np.where(failed_mask & (reasons == ''), label, reasons)
    df['rejection_reason'] = reasons
    return df


def extract_signal_snapshot(symbol: str, signal_date, row: pd.Series) -> dict:
    """
    Build one full analytics snapshot dict for a single signal row (BUY or
    SELL, approved or rejected). Shared by the Scanner Results export and
    the Rejected Signals Log so both carry the same diagnostic detail as
    the Enhanced Trade Log, without requiring the signal to have become an
    executed trade. Computed by simply reading already-computed columns --
    nothing here is recalculated.
    """
    def g(col):
        if col not in row.index:
            return None
        val = row[col]
        if isinstance(val, float) and np.isnan(val):
            return None
        return val

    return {
        'symbol': symbol,
        'signal': g('signal'),
        'signal_date': signal_date,
        'open': _r4(g('Open')), 'high': _r4(g('High')), 'low': _r4(g('Low')), 'close': _r4(g('Close')),
        'candle_color': g('candle_color'),
        'candle_range_pct': _r2(g('candle_range_pct')),
        'candle_body': _r4(g('candle_body')),
        'candle_body_ratio': _r4(g('body_ratio')),
        'upper_wick': _r4(g('upper_wick')), 'upper_wick_ratio': _r4(g('upper_wick_ratio')),
        'lower_wick': _r4(g('lower_wick')), 'lower_wick_ratio': _r4(g('lower_wick_ratio')),
        'close_position_in_candle_pct': _r2(g('close_position_in_candle_pct')),
        'open_position_in_candle_pct': _r2(g('open_position_in_candle_pct')),
        'ema_high': _r4(g('ema_high')), 'ema_low': _r4(g('ema_low')), 'ema_close': _r4(g('ema_close')),
        'ema21': _r4(g('ema21')), 'ema50': _r4(g('ema50')), 'ema200': _r4(g('ema200')),
        'ema_channel_width_pct': _r2(g('ema_channel_width_pct')),
        'dist_close_ema21_pct': _r2(g('dist_close_ema21_pct')),
        'dist_close_ema50_pct': _r2(g('dist_close_ema50_pct')),
        'dist_close_ema200_pct': _r2(g('dist_close_ema200_pct')),
        'previous_day_gap_pct': _r2(g('prev_day_gap_pct')),
        'gap_2d_ago_pct': _r2(g('gap_2d_ago_pct')),
        'gap_3d_ago_pct': _r2(g('gap_3d_ago_pct')),
        'max_gap_in_lookback_pct': _r2(g('max_gap_in_lookback_pct')),
        'next_day_gap_pct': _r2(g('next_day_gap_pct')),
        'next_day_gap_open_pct': _r2(g('next_day_gap_open_pct')),
        'next_day_gap_close_pct': _r2(g('next_day_gap_close_pct')),
        'volume': g('Volume'), 'avg_volume_20': _r2(g('avg_volume_20')), 'volume_ratio': _r2(g('volume_ratio')),
        'rsi': _r2(g('rsi')), 'atr': _r4(g('atr')), 'atr_pct': _r2(g('atr_pct')),
        'highest_close_30d': _r4(g('highest_close_30d')), 'lowest_close_30d': _r4(g('lowest_close_30d')),
        'dist_from_30d_high_pct': _r2(g('dist_from_30d_high_pct')), 'dist_from_30d_low_pct': _r2(g('dist_from_30d_low_pct')),
        'ema_width_filter_passed': g('ema_width_filter_passed'),
        'body_ratio_filter_passed': g('body_ratio_filter_passed'),
        'candle_range_filter_passed': g('candle_range_filter_passed'),
        'previous_gap_filter_passed': g('previous_gap_filter_passed'),
        'next_day_gap_filter_passed': g('next_day_gap_filter_passed'),
        'final_entry_approved': g('final_entry_approved'),
        'rejection_reason': g('rejection_reason'),
    }


def compute_filter_statistics(buy_signal_rows: list) -> dict:
    """
    Aggregate filter-wise pass/fail counts across every BUY signal seen
    during the scan (both approved and rejected), for the Filter
    Statistics dashboard. `buy_signal_rows` is a list of the same dicts
    produced by extract_signal_snapshot() for signal == 'BUY' rows.
    """
    total = len(buy_signal_rows)
    if total == 0:
        return {
            'Total BUY Signals': 0,
            'Rejected by EMA Width Filter': 0,
            'Rejected by Body Ratio Filter': 0,
            'Rejected by Candle Range Filter': 0,
            'Rejected by Previous Gap Filter': 0,
            'Rejected by Next Day Gap Filter': 0,
            'Final Approved': 0,
            'Approval Rate %': 0.0,
        }

    def count_failed(key):
        return sum(1 for r in buy_signal_rows if r.get(key) is False)

    approved = sum(1 for r in buy_signal_rows if r.get('final_entry_approved') is True)
    return {
        'Total BUY Signals': total,
        'Rejected by EMA Width Filter': count_failed('ema_width_filter_passed'),
        'Rejected by Body Ratio Filter': count_failed('body_ratio_filter_passed'),
        'Rejected by Candle Range Filter': count_failed('candle_range_filter_passed'),
        'Rejected by Previous Gap Filter': count_failed('previous_gap_filter_passed'),
        'Rejected by Next Day Gap Filter': count_failed('next_day_gap_filter_passed'),
        'Final Approved': approved,
        'Approval Rate %': round(approved / total * 100, 2),
    }


def build_optimization_bucket_table(trades_df: pd.DataFrame, value_col: str, n_bins: int = 5) -> pd.DataFrame:
    """
    Bucket completed trades by one analytics value (e.g. EMA Channel Width
    % at entry) into up to n_bins quantile ranges and report trade count,
    win rate %, and average return % per bucket -- the "X vs Win Rate"
    optimization report tables. Only executed trades have a win/loss
    outcome, so this is necessarily built from trades_df, not the full
    signal/rejected-signal universe.
    """
    if trades_df.empty or value_col not in trades_df.columns:
        return pd.DataFrame()
    d = trades_df[[value_col, 'pnl_pct', 'win_loss']].dropna(subset=[value_col])
    if d.empty:
        return pd.DataFrame()
    try:
        d = d.assign(bucket=pd.qcut(d[value_col], q=min(n_bins, d[value_col].nunique()), duplicates='drop'))
    except (ValueError, IndexError):
        d = d.assign(bucket=pd.cut(d[value_col], bins=min(n_bins, max(d[value_col].nunique(), 1))))
    grouped = d.groupby('bucket', observed=True).agg(
        Trade_Count=('pnl_pct', 'count'),
        Win_Rate_Pct=('win_loss', lambda s: round((s == 'Win').mean() * 100, 2)),
        Avg_Return_Pct=('pnl_pct', lambda s: round(s.mean(), 2)),
    ).reset_index()
    grouped['bucket'] = grouped['bucket'].astype(str)
    grouped = grouped.rename(columns={
        'bucket': 'Range', 'Trade_Count': 'Trade Count',
        'Win_Rate_Pct': 'Win Rate %', 'Avg_Return_Pct': 'Avg Return %',
    })
    return grouped


# ===================================================================
# BACKTESTING MODULE
# ===================================================================
#
# This module NEVER recalculates or alters BUY/SELL signals. It only
# consumes the 'signal' column already produced by generate_signals()
# (STRATEGY MODULE above), plus the read-only diagnostics from the
# ANALYTICS & ENTRY FILTERS MODULE. All capital management, trade
# execution, per-symbol position tracking, and optional entry-filter /
# stop-loss / target overlays live here.
#
# Architecture: one shared capital pool, multiple independent
# per-symbol open positions (bounded by Maximum Concurrent Positions).
# Signals from every symbol are merged into a single stream of
# executable trigger events and executed in strict chronological order.
# ===================================================================

ANALYTICS_SNAPSHOT_COLUMNS = [
    'ema_high', 'ema_low', 'ema_close', 'ema21', 'ema50', 'ema200',
    'ema_channel_width_pct', 'dist_close_ema21_pct', 'dist_close_ema50_pct', 'dist_close_ema200_pct',
    'Open', 'High', 'Low', 'Close', 'candle_color', 'candle_range_pct', 'candle_body', 'body_ratio',
    'upper_wick', 'upper_wick_ratio', 'lower_wick', 'lower_wick_ratio',
    'close_position_in_candle_pct', 'open_position_in_candle_pct',
    'prev_day_gap_pct', 'gap_2d_ago_pct', 'gap_3d_ago_pct', 'max_gap_in_lookback_pct',
    'next_day_gap_pct', 'next_day_gap_open_pct', 'next_day_gap_close_pct',
    'previous_gap_filter_passed', 'next_day_gap_filter_passed',
    'ema_width_filter_passed', 'body_ratio_filter_passed', 'candle_range_filter_passed',
    'final_entry_approved', 'rejection_reason',
    'Volume', 'avg_volume_20', 'volume_ratio', 'rsi', 'atr', 'atr_pct',
    'highest_close_30d', 'lowest_close_30d', 'dist_from_30d_high_pct', 'dist_from_30d_low_pct',
]


def build_trigger_events(symbol_data: dict, entry_type: str = 'Next Day Open') -> list:
    """
    Convert each symbol's pre-generated 'signal' column into executable
    next-day trigger events. A signal on day i becomes actionable at day
    i+1 (that symbol's own next trading day) -- signals are read exactly
    as produced; nothing here re-derives or changes them.

    Entry Filters (ANALYTICS & ENTRY FILTERS MODULE, evaluated once per
    symbol beforehand) gate BUY events only: a BUY signal whose row has
    'final_entry_approved' == False produces NO event at all -- the
    trade is permanently skipped, exactly as if the signal never fired,
    and (per the Next Day Entry Gap Filter spec) is never retried on a
    later day. SELL signals are never filtered and always produce an
    event.

    `entry_type` controls only the BUY fill price basis: 'Next Day Open'
    (default, matches the original engine) or 'Next Day Close'. SELL
    exits triggered here always use next-day Open, unchanged from the
    original engine.

    A signal on a symbol's LAST available row has no next day to execute
    on within the downloaded data, so it produces no event (consistent
    with only acting on data that actually exists).
    """
    use_close_entry = (entry_type == 'Next Day Close')
    events = []
    for symbol, df in symbol_data.items():
        n = len(df)
        has_analytics = 'final_entry_approved' in df.columns
        for i in range(n - 1):
            sig = df['signal'].iloc[i]
            if pd.isna(sig):
                continue

            if sig == 'BUY' and has_analytics and not bool(df['final_entry_approved'].iloc[i]):
                continue  # entry filter rejected this signal -> skip permanently, no event

            trigger_row = df.iloc[i + 1]
            if sig == 'BUY':
                trigger_price = float(trigger_row['Close']) if use_close_entry else float(trigger_row['Open'])
            else:
                trigger_price = float(trigger_row['Open'])  # SELL exits unchanged: next-day Open

            event = {
                'symbol': symbol,
                'action': sig,  # 'BUY' or 'SELL', exactly as generated
                'signal_date': df.index[i],
                'trigger_date': df.index[i + 1],
                'trigger_open': trigger_price,
            }
            if sig == 'BUY' and has_analytics:
                signal_row = df.iloc[i]
                snapshot = {}
                for col in ANALYTICS_SNAPSHOT_COLUMNS:
                    if col in df.columns:
                        snapshot[col] = signal_row[col]
                event['analytics'] = snapshot
            events.append(event)

    # Strict chronological order. Same-day ties across symbols are broken
    # deterministically by symbol name (daily bars carry no intraday
    # ordering information, so this is the most reproducible tie-break).
    events.sort(key=lambda e: (e['trigger_date'], e['symbol']))
    return events


def run_portfolio_backtest(
    symbol_data: dict,
    initial_capital: float,
    position_size: float,
    enable_stop_loss: bool = False,
    stop_loss_pct: float = 5.0,
    enable_target: bool = False,
    target_pct: float = 10.0,
    max_concurrent_positions: int = 10,
    entry_type: str = 'Next Day Open',
    transaction_cost_pct: float = 0.0,
) -> tuple:
    """
    True multi-position portfolio backtest across every scanned symbol,
    executed in strict chronological order, sharing one capital pool.

    Rules enforced here (capital management / trade execution only --
    no strategy logic; BUY/SELL signals come from the untouched
    STRATEGY MODULE and are never re-derived or altered):
      - Every stock maintains its own independent open position.
      - Every valid BUY signal is executed as long as ALL of the
        following hold: the stock has no open position already, the
        number of currently open positions is below
        max_concurrent_positions, enough cash is available, and the
        position size buys at least one share. Any failing condition
        skips only that one signal -- processing continues normally.
        (Entry Filter rejections happen even earlier, in
        build_trigger_events(), so a filtered-out BUY never reaches
        this stage at all.)
      - Multiple symbols can open AND close positions on the same day.
      - Shares = floor(Position Size / Entry Price); required cash =
        Shares x Entry Price; on exit, invested amount + P&L (net of
        optional transaction cost) returns to available cash
        immediately, reduces the open position count, and can fund a
        new trade (into the freed slot) the same day (checked in
        signal-date order). `entry_type` controls whether the BUY fill
        price is the next day's Open or Close (see build_trigger_events).
      - Optional Stop Loss / Target are backtest-only overlays and never
        touch the scanner's BUY/SELL signal generation:
          * Stop Loss price = Entry Price x (1 - stop_loss_pct/100);
            exits at the Stop Loss price the first day Low <= that price.
          * Target price = Entry Price x (1 + target_pct/100); exits at
            the Target price the first day High >= that price.
          * Exit priority when checked for a given day's candle:
            1) Stop Loss  2) Target  3) original SELL signal  4) End of
            Data. Stop Loss / Target are evaluated once per day for
            positions already open BEFORE that day's signal events are
            processed, so if both a Stop Loss/Target trigger and a SELL
            signal land on the same day, the Stop Loss/Target wins and
            the SELL signal that day is simply ignored (position already
            closed). A newly-opened position's own entry-day Low/High
            is not checked for Stop Loss/Target, and MFE/MAE tracking
            (below) likewise starts the following trading day -- both
            avoid same-candle entry/exit ambiguity.
      - While a position is open, its highest/lowest price reached and
        running peak-to-trough drawdown are tracked daily (from the day
        after entry) to populate MFE/MAE trade-log fields on exit.
      - A position still open when its symbol's data runs out is closed
        at that symbol's last available Close, exit reason "End of Data".

    Returns
    -------
    trades_df : pd.DataFrame       (Enhanced Trade Log -- see
                                     ANALYTICS_SNAPSHOT_COLUMNS /
                                     close_position() below for the
                                     full column set)
    equity_df : pd.DataFrame        (daily portfolio equity across the
                                      union of all symbols' trading dates)
    final_cash : float
    portfolio_stats : dict          (concurrency / capital-utilization /
                                      skip-reason stats used by
                                      compute_backtest_summary)
    """
    cash = float(initial_capital)
    positions = {}  # symbol -> entry snapshot + running MFE/MAE state
    trades = []
    equity_records = []
    concurrent_counts = []
    invested_amounts_daily = []
    skipped_position_limit = 0
    skipped_insufficient_cash = 0
    trade_id_counter = 0

    empty_stats = {
        'max_concurrent_positions_setting': int(max_concurrent_positions),
        'highest_concurrent_positions': 0,
        'avg_concurrent_positions': 0.0,
        'peak_capital_invested': 0.0,
        'avg_capital_invested': 0.0,
        'position_slot_utilization_pct': 0.0,
        'skipped_position_limit': 0,
        'skipped_insufficient_cash': 0,
    }

    if not symbol_data:
        return pd.DataFrame(), pd.DataFrame(columns=['date', 'equity']), float(cash), empty_stats

    events = build_trigger_events(symbol_data, entry_type=entry_type)

    # Master timeline = union of every scanned symbol's trading dates.
    all_dates = sorted(set().union(*[set(df.index) for df in symbol_data.values()]))

    # Forward-filled Close per symbol, used only to mark daily equity
    # while a position is held (never used for signal/entry/exit logic).
    close_lookup = {
        symbol: df['Close'].reindex(all_dates).ffill()
        for symbol, df in symbol_data.items()
    }
    last_index_of = {symbol: df.index[-1] for symbol, df in symbol_data.items()}
    last_close_of = {symbol: float(df['Close'].iloc[-1]) for symbol, df in symbol_data.items()}

    def close_position(symbol, exit_date, exit_price, exit_reason):
        nonlocal cash
        pos = positions.pop(symbol)
        gross_exit_value = pos['shares'] * exit_price
        entry_cost = pos['invested_amount'] * (transaction_cost_pct / 100.0)
        exit_cost = gross_exit_value * (transaction_cost_pct / 100.0)
        transaction_cost = entry_cost + exit_cost
        net_exit_value = gross_exit_value - exit_cost
        cash += net_exit_value
        pnl = net_exit_value - pos['invested_amount'] - entry_cost
        pnl_pct = (pnl / pos['invested_amount']) * 100 if pos['invested_amount'] != 0 else 0.0
        holding_days = (exit_date - pos['entry_date']).days

        highest_price = pos['highest_price']
        lowest_price = pos['lowest_price']
        entry_price = pos['entry_price']
        mfe_pct = (highest_price - entry_price) / entry_price * 100 if entry_price else 0.0
        mae_pct = (entry_price - lowest_price) / entry_price * 100 if entry_price else 0.0

        stop_price = pos['stop_price']
        target_price = pos['target_price']
        risk_reward_ratio = None
        if enable_stop_loss and enable_target and stop_loss_pct:
            risk_reward_ratio = round(target_pct / stop_loss_pct, 2)

        a = pos['analytics']  # entry-day analytics snapshot (may be {})

        trade = {
            # --- Trade Information ---
            'trade_id': pos['trade_id'],
            'symbol': symbol,
            'signal_date': pos['signal_date'],
            'entry_date': pos['entry_date'],
            'entry_price': round(float(entry_price), 4),
            'exit_date': exit_date,
            'exit_price': round(float(exit_price), 4),
            'exit_reason': exit_reason,
            'holding_days': int(holding_days),
            'pnl': round(float(pnl), 2),
            'pnl_pct': round(float(pnl_pct), 2),
            'win_loss': 'Win' if pnl > 0 else 'Loss',

            # --- Position Information ---
            'initial_capital': round(float(initial_capital), 2),
            'available_capital': round(float(cash), 2),
            'position_size': round(float(position_size), 2),
            'shares': int(pos['shares']),
            'invested_amount': round(float(pos['invested_amount']), 2),
            'transaction_cost': round(float(transaction_cost), 2),
            'max_open_positions': int(max_concurrent_positions),
            'capital_utilization_pct': round(float(pos['capital_utilization_pct_at_entry']), 2),

            # --- EMA Data (at signal candle) ---
            'ema_high': _r4(a.get('ema_high')), 'ema_low': _r4(a.get('ema_low')), 'ema_close': _r4(a.get('ema_close')),
            'ema21': _r4(a.get('ema21')), 'ema50': _r4(a.get('ema50')), 'ema200': _r4(a.get('ema200')),
            'ema_channel_width_pct': _r2(a.get('ema_channel_width_pct')),
            'dist_close_ema21_pct': _r2(a.get('dist_close_ema21_pct')),
            'dist_close_ema50_pct': _r2(a.get('dist_close_ema50_pct')),
            'dist_close_ema200_pct': _r2(a.get('dist_close_ema200_pct')),

            # --- Candle Data (signal candle) ---
            'open': _r4(a.get('Open')), 'high': _r4(a.get('High')), 'low': _r4(a.get('Low')), 'close': _r4(a.get('Close')),
            'candle_color': a.get('candle_color'),
            'candle_range_pct': _r2(a.get('candle_range_pct')),
            'candle_body': _r4(a.get('candle_body')),
            'candle_body_ratio': _r4(a.get('body_ratio')),
            'upper_wick': _r4(a.get('upper_wick')), 'upper_wick_ratio': _r4(a.get('upper_wick_ratio')),
            'lower_wick': _r4(a.get('lower_wick')), 'lower_wick_ratio': _r4(a.get('lower_wick_ratio')),
            'close_position_in_candle_pct': _r2(a.get('close_position_in_candle_pct')),
            'open_position_in_candle_pct': _r2(a.get('open_position_in_candle_pct')),

            # --- Gap Analysis ---
            'previous_day_gap_pct': _r2(a.get('prev_day_gap_pct')),
            'gap_2d_ago_pct': _r2(a.get('gap_2d_ago_pct')),
            'gap_3d_ago_pct': _r2(a.get('gap_3d_ago_pct')),
            'max_gap_in_lookback_pct': _r2(a.get('max_gap_in_lookback_pct')),
            'next_day_gap_pct': _r2(a.get('next_day_gap_pct')),
            'next_day_gap_open_pct': _r2(a.get('next_day_gap_open_pct')),
            'next_day_gap_close_pct': _r2(a.get('next_day_gap_close_pct')),

            # --- Volume & Momentum ---
            'volume': a.get('Volume'),
            'avg_volume_20': _r2(a.get('avg_volume_20')),
            'volume_ratio': _r2(a.get('volume_ratio')),
            'rsi': _r2(a.get('rsi')),
            'atr': _r4(a.get('atr')),
            'atr_pct': _r2(a.get('atr_pct')),
            'highest_close_30d': _r4(a.get('highest_close_30d')),
            'lowest_close_30d': _r4(a.get('lowest_close_30d')),
            'dist_from_30d_high_pct': _r2(a.get('dist_from_30d_high_pct')),
            'dist_from_30d_low_pct': _r2(a.get('dist_from_30d_low_pct')),

            # --- Filter Results ---
            'ema_width_filter_passed': a.get('ema_width_filter_passed'),
            'body_ratio_filter_passed': a.get('body_ratio_filter_passed'),
            'candle_range_filter_passed': a.get('candle_range_filter_passed'),
            'previous_gap_filter_passed': a.get('previous_gap_filter_passed'),
            'next_day_gap_filter_passed': a.get('next_day_gap_filter_passed'),
            'final_entry_approved': a.get('final_entry_approved', True),
            'rejection_reason': a.get('rejection_reason', ''),

            # --- Trade Management ---
            'stop_loss_pct': stop_loss_pct if enable_stop_loss else None,
            'stop_price': round(float(stop_price), 4) if stop_price is not None else None,
            'target_pct': target_pct if enable_target else None,
            'target_price': round(float(target_price), 4) if target_price is not None else None,
            'risk_reward_ratio': risk_reward_ratio,
            'highest_price_during_trade': round(float(highest_price), 4),
            'lowest_price_during_trade': round(float(lowest_price), 4),
            'mfe_pct': round(float(mfe_pct), 2),
            'mae_pct': round(float(mae_pct), 2),
            'highest_unrealized_profit_pct': round(float(mfe_pct), 2),
            'max_drawdown_during_trade_pct': round(float(pos['max_drawdown_pct']), 2),

            # kept for backward compatibility with earlier trade-log consumers
            'exit_value': round(float(net_exit_value), 2),
            'trade_duration': int(holding_days),
            'available_cash_after_trade': round(float(cash), 2),
        }
        trades.append(trade)

    event_idx = 0
    n_events = len(events)

    for d in all_dates:
        # --- 1) Stop Loss / Target checks + MFE/MAE tracking for
        #        positions already open (from a prior day), BEFORE
        #        today's signal events, so the exit-priority (Stop Loss
        #        > Target > SELL signal) holds. ---
        for symbol in list(positions.keys()):
            df = symbol_data[symbol]
            if d not in df.index:
                continue
            row = df.loc[d]
            pos = positions[symbol]
            low = float(row['Low'])
            high = float(row['High'])

            pos['highest_price'] = max(pos['highest_price'], high)
            pos['lowest_price'] = min(pos['lowest_price'], low)
            pos['running_peak'] = max(pos['running_peak'], high)
            if pos['running_peak'] > 0:
                drawdown_today = (pos['running_peak'] - low) / pos['running_peak'] * 100
                pos['max_drawdown_pct'] = max(pos['max_drawdown_pct'], drawdown_today)

            hit_stop = enable_stop_loss and pos['stop_price'] is not None and low <= pos['stop_price']
            hit_target = enable_target and pos['target_price'] is not None and high >= pos['target_price']

            if hit_stop:
                close_position(symbol, d, pos['stop_price'], 'Stop Loss')
            elif hit_target:
                close_position(symbol, d, pos['target_price'], 'Target Hit')

        # --- 2) Execute every signal event triggered today, in order ---
        while event_idx < n_events and events[event_idx]['trigger_date'] == d:
            ev = events[event_idx]
            event_idx += 1
            symbol = ev['symbol']
            today_price = ev['trigger_open']

            if ev['action'] == 'BUY':
                if symbol in positions:
                    continue  # this symbol already has an open position -> ignore
                if len(positions) >= max_concurrent_positions:
                    skipped_position_limit += 1
                    continue  # at the concurrent-position cap -> skip only this signal
                calc_shares = int(np.floor(position_size / today_price)) if today_price > 0 else 0
                invested = calc_shares * today_price
                if calc_shares <= 0:
                    continue  # position size can't buy even one share -> skip this entry only
                entry_cost = invested * (transaction_cost_pct / 100.0)
                if cash < invested + entry_cost:
                    skipped_insufficient_cash += 1
                    continue  # insufficient cash -> skip this entry only
                cash -= (invested + entry_cost)
                trade_id_counter += 1
                stop_price = today_price * (1 - stop_loss_pct / 100.0) if enable_stop_loss else None
                target_price = today_price * (1 + target_pct / 100.0) if enable_target else None
                invested_now = sum(p['invested_amount'] for p in positions.values()) + invested
                positions[symbol] = {
                    'trade_id': trade_id_counter,
                    'signal_date': ev['signal_date'],
                    'entry_date': ev['trigger_date'],
                    'entry_price': today_price,
                    'shares': calc_shares,
                    'invested_amount': invested,
                    'stop_price': stop_price,
                    'target_price': target_price,
                    'highest_price': today_price,
                    'lowest_price': today_price,
                    'running_peak': today_price,
                    'max_drawdown_pct': 0.0,
                    'capital_utilization_pct_at_entry': (
                        (invested_now / initial_capital) * 100 if initial_capital else 0.0
                    ),
                    'analytics': ev.get('analytics', {}),
                }

            elif ev['action'] == 'SELL':
                if symbol not in positions:
                    continue  # no open position in this symbol (or already closed by Stop/Target today) -> ignore
                close_position(symbol, d, today_price, 'SELL Signal')

        # --- 3) Force-close any position whose symbol's data ends today ---
        for symbol in list(positions.keys()):
            if d == last_index_of[symbol]:
                close_position(symbol, d, last_close_of[symbol], 'End of Data')

        # --- 4) Mark end-of-day portfolio equity + concurrency stats ---
        invested_value = 0.0
        for symbol, pos in positions.items():
            px = close_lookup[symbol].loc[d]
            if pd.isna(px):
                px = pos['entry_price']  # defensive fallback only
            invested_value += pos['shares'] * float(px)
        equity = cash + invested_value
        equity_records.append({'date': d, 'equity': float(equity)})
        concurrent_counts.append(len(positions))
        invested_amounts_daily.append(invested_value)

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_records)
    highest_concurrent = int(max(concurrent_counts)) if concurrent_counts else 0
    avg_concurrent = float(np.mean(concurrent_counts)) if concurrent_counts else 0.0
    slot_utilization_pct = (
        (avg_concurrent / max_concurrent_positions) * 100 if max_concurrent_positions else 0.0
    )
    portfolio_stats = {
        'max_concurrent_positions_setting': int(max_concurrent_positions),
        'highest_concurrent_positions': highest_concurrent,
        'avg_concurrent_positions': avg_concurrent,
        'peak_capital_invested': float(max(invested_amounts_daily)) if invested_amounts_daily else 0.0,
        'avg_capital_invested': float(np.mean(invested_amounts_daily)) if invested_amounts_daily else 0.0,
        'position_slot_utilization_pct': float(slot_utilization_pct),
        'skipped_position_limit': int(skipped_position_limit),
        'skipped_insufficient_cash': int(skipped_insufficient_cash),
    }
    return trades_df, equity_df, float(cash), portfolio_stats


def compute_backtest_summary(
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    initial_capital: float,
    portfolio_stats: dict = None,
) -> dict:
    """Compute backtest summary statistics from trades and daily equity curve."""
    portfolio_stats = portfolio_stats or {
        'max_concurrent_positions_setting': 0,
        'highest_concurrent_positions': 0,
        'avg_concurrent_positions': 0.0,
        'peak_capital_invested': 0.0,
        'avg_capital_invested': 0.0,
        'position_slot_utilization_pct': 0.0,
        'skipped_position_limit': 0,
        'skipped_insufficient_cash': 0,
    }
    capital_utilization_pct = (
        (portfolio_stats['avg_capital_invested'] / initial_capital) * 100 if initial_capital else 0.0
    )
    concurrency_fields = {
        'Maximum Concurrent Positions (User Setting)': int(portfolio_stats['max_concurrent_positions_setting']),
        'Highest Concurrent Positions Reached': int(portfolio_stats['highest_concurrent_positions']),
        'Average Concurrent Positions': round(float(portfolio_stats['avg_concurrent_positions']), 2),
        'Position Slot Utilization (%)': round(float(portfolio_stats['position_slot_utilization_pct']), 2),
        'Peak Capital Invested': round(float(portfolio_stats['peak_capital_invested']), 2),
        'Average Capital Invested': round(float(portfolio_stats['avg_capital_invested']), 2),
        'Capital Utilization %': round(float(capital_utilization_pct), 2),
        'Trades Skipped Due to Position Limit': int(portfolio_stats['skipped_position_limit']),
        'Trades Skipped Due to Insufficient Cash': int(portfolio_stats['skipped_insufficient_cash']),
    }

    if trades_df.empty:
        final_equity = float(equity_df['equity'].iloc[-1]) if not equity_df.empty else float(initial_capital)
        return {
            'Initial Capital': round(float(initial_capital), 2),
            'Final Capital': round(final_equity, 2),
            'Available Cash': round(final_equity, 2),
            'Net Profit': 0.0,
            'Net Return %': 0.0,
            'Total Trades': 0,
            'Winning Trades': 0,
            'Losing Trades': 0,
            'Win Rate %': 0.0,
            'Average Win %': 0.0,
            'Average Loss %': 0.0,
            'Largest Winner': 0.0,
            'Largest Loser': 0.0,
            'Gross Profit': 0.0,
            'Gross Loss': 0.0,
            'Profit Factor': 0.0,
            'Maximum Drawdown': 0.0,
            'Average Holding Days': 0.0,
            **concurrency_fields,
            'Target Hits': 0,
            'Stop Loss Hits': 0,
            'Signal Exits': 0,
            'End-of-Data Exits': 0,
        }

    total_trades = len(trades_df)
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]

    win_count = len(wins)
    loss_count = len(losses)
    win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0.0

    avg_win_pct = wins['pnl_pct'].mean() if not wins.empty else 0.0
    avg_loss_pct = losses['pnl_pct'].mean() if not losses.empty else 0.0

    largest_winner = wins['pnl'].max() if not wins.empty else 0.0
    largest_loser = losses['pnl'].min() if not losses.empty else 0.0

    gross_profit = float(wins['pnl'].sum()) if not wins.empty else 0.0
    gross_loss = float(abs(losses['pnl'].sum())) if not losses.empty else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')

    # Maximum drawdown from daily portfolio equity curve
    equity_values = equity_df['equity'].values
    peak = np.maximum.accumulate(equity_values)
    drawdown = (peak - equity_values) / peak
    max_drawdown = float(np.max(drawdown) * 100)

    net_profit = float(trades_df['pnl'].sum())
    final_equity = float(equity_df['equity'].iloc[-1]) if not equity_df.empty else float(initial_capital) + net_profit
    net_return_pct = (net_profit / initial_capital) * 100 if initial_capital != 0 else 0.0
    avg_holding = float(trades_df['holding_days'].mean())

    exit_reasons = trades_df['exit_reason'] if 'exit_reason' in trades_df.columns else pd.Series(dtype=str)
    target_hits = int((exit_reasons == 'Target Hit').sum())
    stop_loss_hits = int((exit_reasons == 'Stop Loss').sum())
    signal_exits = int((exit_reasons == 'SELL Signal').sum())
    eod_exits = int((exit_reasons == 'End of Data').sum())

    return {
        'Initial Capital': round(float(initial_capital), 2),
        'Final Capital': round(final_equity, 2),
        'Available Cash': round(final_equity, 2),
        'Net Profit': round(float(net_profit), 2),
        'Net Return %': round(float(net_return_pct), 2),
        'Total Trades': int(total_trades),
        'Winning Trades': int(win_count),
        'Losing Trades': int(loss_count),
        'Win Rate %': round(float(win_rate), 2),
        'Average Win %': round(float(avg_win_pct), 2),
        'Average Loss %': round(float(avg_loss_pct), 2),
        'Largest Winner': round(float(largest_winner), 2),
        'Largest Loser': round(float(largest_loser), 2),
        'Gross Profit': round(float(gross_profit), 2),
        'Gross Loss': round(float(gross_loss), 2),
        'Profit Factor': round(float(profit_factor), 2) if profit_factor != float('inf') else float('inf'),
        'Maximum Drawdown': round(float(max_drawdown), 2),
        'Average Holding Days': round(float(avg_holding), 2),
        **concurrency_fields,
        'Target Hits': target_hits,
        'Stop Loss Hits': stop_loss_hits,
        'Signal Exits': signal_exits,
        'End-of-Data Exits': eod_exits,
    }


def plot_equity_curve(equity_df: pd.DataFrame, title: str = 'Portfolio Equity Curve'):
    """Plot daily portfolio equity curve using Plotly."""
    if equity_df.empty:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_df['date'],
        y=equity_df['equity'],
        mode='lines',
        name='Equity',
        line=dict(color='royalblue', width=2),
    ))
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Capital (₹)',
        template='plotly_white',
        height=500,
    )
    return fig


# ===================================================================
# DATE-RANGE / WARM-UP HANDLING
# ===================================================================
#
# NOT strategy logic and NOT backtest execution logic -- this is purely
# data-window orchestration. It exists so EMA calculations never start
# cold at the user's selected Start Date (which would make early-window
# EMA values diverge from what TradingView shows, since TradingView's
# indicator has been running on the full chart history). Instead we:
#   1. Download extra history before Start Date (>= WARMUP_MIN_BARS,
#      targeting WARMUP_TARGET_BARS) so all EMAs are fully converged.
#   2. Run the untouched STRATEGY MODULE over that FULL extended dataset
#      (never truncated), so ema_high/ema_low/signal/lastWasBuy/
#      lastWasSell are all computed exactly as TradingView would.
#   3. Only AFTER that computation is done, slice the result down to the
#      user's [Start Date, End Date] window for display and trading.
#      Everything before Start Date is discarded at this point -- it has
#      already done its job of warming up the indicators and state.
# ===================================================================

WARMUP_MIN_BARS = 300
WARMUP_TARGET_BARS = 500


def compute_warmup_download_start(start_date, warmup_bars: int = WARMUP_TARGET_BARS):
    """
    Estimate a calendar date far enough before `start_date` that
    downloading from it forward will very likely yield at least
    `warmup_bars` trading days of history before `start_date`.

    ~252 trading days/year -> ~1.45 calendar days per trading day.
    A 1.7x multiplier plus a flat 30-day cushion comfortably covers
    weekends, exchange holidays, and short gaps without under-shooting.
    """
    calendar_days = int(warmup_bars * 1.7) + 30
    return start_date - timedelta(days=calendar_days)


def restrict_to_date_range(df: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    """
    Slice a fully-computed (warm-up + selected range) dataframe down to
    only the bars within [start_date, end_date]. Called AFTER
    generate_signals() has already run over the complete extended
    history, so every retained row's ema_high/ema_low/signal value
    already reflects the correct, uninterrupted TradingView-style state
    (lastWasBuy/lastWasSell) built up from real history before
    start_date. Rows before start_date or after end_date are simply
    dropped here -- they've already done their warm-up job.
    """
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    return df[(df.index >= start_ts) & (df.index <= end_ts)].copy()


# ===================================================================
# SYMBOL INPUT HANDLING
# ===================================================================
#
# Not strategy logic, not backtest logic -- purely cleaning/normalizing
# whatever the user typed or uploaded before it's used as a Yahoo
# Finance ticker.
# ===================================================================

def normalize_symbol(raw) -> str:
    """
    Normalize one raw ticker for NSE/BSE + Yahoo Finance convention:
      - uppercase
      - strip leading/trailing whitespace
      - leave an existing .NS or .BO suffix untouched
      - otherwise append .NS
    Returns '' for blank/unusable input.
    """
    sym = str(raw).strip().upper()
    if not sym or sym == 'NAN':
        return ''
    if sym.endswith('.NS') or sym.endswith('.BO'):
        return sym
    return f'{sym}.NS'


def normalize_symbols(raw_symbols: list) -> tuple:
    """
    Normalize a list of raw ticker strings: uppercase, strip, auto-suffix,
    drop blanks, de-duplicate while preserving first-seen order.

    Returns (normalized_list, duplicate_count).
    """
    seen = set()
    normalized = []
    duplicate_count = 0
    for raw in raw_symbols:
        sym = normalize_symbol(raw)
        if not sym:
            continue
        if sym in seen:
            duplicate_count += 1
            continue
        seen.add(sym)
        normalized.append(sym)
    return normalized, duplicate_count


def detect_symbol_column(df: pd.DataFrame) -> pd.Series:
    """
    Return the first column in df that actually contains a non-blank
    value, so a file whose first column is blank or a header/label-only
    column doesn't silently produce zero symbols.
    """
    for col in df.columns:
        series = df[col]
        non_blank = series.dropna().astype(str).str.strip()
        non_blank = non_blank[non_blank != '']
        if not non_blank.empty:
            return series
    return df.iloc[:, 0]  # nothing usable found; fall back to first column


# ===================================================================
# BATCH / PARALLEL DOWNLOAD
# ===================================================================
#
# Replaces one-symbol-at-a-time yf.download() calls with true batch
# downloading: 50-100 symbols per yfinance call (one HTTP round trip
# covering the whole batch), several batches fetched concurrently via
# ThreadPoolExecutor, automatic per-batch retry, and a session-level
# cache so re-running the same symbols/date-range doesn't re-download
# unchanged data. This is purely a data-fetching optimization -- it
# does not touch signal generation or backtest execution.
# ===================================================================

BATCH_SIZE = 75
MAX_BATCH_WORKERS = 4
MAX_BATCH_RETRIES = 2


def _download_one_batch(batch_symbols: list, download_start, end_date):
    """
    Download OHLC history for a batch of symbols in a single yfinance
    call. Retries the whole batch a couple of times on failure before
    giving up (never raises -- returns (None, error) on total failure).
    """
    last_err = None
    for _attempt in range(MAX_BATCH_RETRIES + 1):
        try:
            data = yf.download(
                tickers=batch_symbols,
                start=download_start,
                end=end_date + timedelta(days=1),
                group_by='ticker',
                progress=False,
                auto_adjust=False,
                threads=True,
            )
            return data, None
        except Exception as e:
            last_err = e
    return None, last_err


def _split_batch_result(data, batch_symbols: list) -> dict:
    """Slice a (possibly multi-ticker) yfinance batch result into one raw
    OHLC DataFrame per symbol. Returns {symbol: df_or_None}."""
    out = {}
    if data is None or data.empty:
        for s in batch_symbols:
            out[s] = None
        return out

    for s in batch_symbols:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                top_level = data.columns.get_level_values(0)
                if len(batch_symbols) == 1:
                    df = data.copy()
                    df.columns = df.columns.get_level_values(0)
                elif s in set(top_level):
                    df = data[s].copy()
                else:
                    out[s] = None
                    continue
            else:
                # Single-ticker batch sometimes comes back as a flat frame
                df = data.copy()
            df = df.dropna(how='all')
            out[s] = df if not df.empty else None
        except Exception:
            out[s] = None
    return out


def download_all_symbols(symbols: list, download_start, end_date, cache: dict,
                          progress_callback=None) -> dict:
    """
    Batch-download every symbol's OHLC history, processing several
    50-100 symbol batches in parallel. `cache` is a dict (typically
    st.session_state-backed) keyed by symbol, storing the last
    downloaded DataFrame plus the (download_start, end_date) it was
    fetched for -- reused whenever a re-run asks for the same window,
    so repeated scans of the same universe don't re-download unchanged
    history.

    `progress_callback(done_batches, total_batches)` is invoked after
    every completed batch so the caller can drive a progress bar.

    Returns {symbol: raw_ohlc_df_or_None}.
    """
    cache_key = f'{download_start}|{end_date}'
    to_fetch = [s for s in symbols if cache.get(s, {}).get('key') != cache_key]
    batches = [to_fetch[i:i + BATCH_SIZE] for i in range(0, len(to_fetch), BATCH_SIZE)]

    fetched = {}
    total_batches = len(batches)
    done_batches = 0

    if batches:
        with ThreadPoolExecutor(max_workers=min(MAX_BATCH_WORKERS, total_batches)) as executor:
            future_to_batch = {
                executor.submit(_download_one_batch, batch, download_start, end_date): batch
                for batch in batches
            }
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                data, _err = future.result()
                fetched.update(_split_batch_result(data, batch))
                done_batches += 1
                if progress_callback:
                    progress_callback(done_batches, total_batches)

    all_data = {}
    for s in symbols:
        if s in fetched:
            cache[s] = {'key': cache_key, 'df': fetched[s]}
        all_data[s] = cache.get(s, {}).get('df')
    return all_data


# ===================================================================
# SCAN ORCHESTRATION
# ===================================================================
#
# Glue between data download and the untouched STRATEGY/BACKTESTING
# modules. Every symbol is processed independently and failures are
# always captured, never raised, so one bad symbol never stops the scan.
# ===================================================================

def process_symbol_df(symbol: str, raw_df, ema_length: int, start_date, end_date,
                       filter_settings: dict = None) -> dict:
    """
    Generate signals for one symbol from an already-downloaded raw OHLC
    DataFrame (fetched via download_all_symbols) and summarize its
    current state. Never raises: any failure is captured in
    result['status'] so the caller can always move on to the next symbol.

    result['status'] is one of:
      'Success', 'Download Error: <msg>', 'No Yahoo Finance data',
      'Missing column(s): <cols>', 'No data in selected range'
    """
    result = {
        'symbol': symbol,
        'status': 'Success',
        'in_range_df': None,
        'bars_before_start': 0,
        'last_close': None,
        'ema_high': None,
        'ema_low': None,
        'current_signal': 'No Signal',
        'last_signal_date': None,
    }

    if raw_df is None or raw_df.empty:
        result['status'] = 'No Yahoo Finance data'
        return result

    try:
        df = raw_df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        missing_cols = [c for c in ['Open', 'High', 'Low', 'Close'] if c not in df.columns]
        if missing_cols:
            result['status'] = f'Missing column(s): {", ".join(missing_cols)}'
            return result

        # Generate signals over the COMPLETE downloaded dataset, warm-up
        # bars included (STRATEGY MODULE -- untouched, never truncated).
        df = generate_signals(df, ema_length)

        # Additive, read-only analytics + entry-filter evaluation, also
        # computed ONCE over the complete dataset (so rolling windows /
        # gap-lookbacks spanning the Start Date boundary are correct, and
        # nothing is ever recalculated later in the pipeline).
        df = compute_analytics(df)
        df = evaluate_entry_filters(df, filter_settings)

        start_ts = pd.Timestamp(start_date)
        result['bars_before_start'] = int((df.index < start_ts).sum())

        # Only NOW -- after all calculations are finished -- filter down
        # to the selected [Start Date, End Date] window.
        in_range_df = restrict_to_date_range(df, start_date, end_date)
        if in_range_df.empty:
            result['status'] = 'No data in selected range'
            return result

        last_row = in_range_df.iloc[-1]
        result['last_close'] = float(last_row['Close'])
        result['ema_high'] = float(last_row['ema_high']) if pd.notna(last_row['ema_high']) else None
        result['ema_low'] = float(last_row['ema_low']) if pd.notna(last_row['ema_low']) else None

        signal_df = in_range_df[in_range_df['signal'].notna()]
        if not signal_df.empty:
            result['current_signal'] = signal_df.iloc[-1]['signal']
            result['last_signal_date'] = signal_df.index[-1]

        result['in_range_df'] = in_range_df

    except Exception as e:
        result['status'] = f'Download Error: {e}'
        result['in_range_df'] = None

    return result


# ===================================================================
# STREAMLIT UI
# ===================================================================

def main():
    st.set_page_config(page_title='TV EMA Strategy Scanner & Backtester', layout='wide')
    st.title('📈 TradingView EMA Strategy Scanner & Backtester')
    st.markdown('Replicates exact TradingView Pine Script logic. No extra filters.')

    with st.sidebar:
        st.header('Settings')

        ema_length = st.number_input('EMA Length', min_value=1, max_value=500, value=32, step=1)
        initial_capital = st.number_input('Initial Capital (₹)', min_value=1000.0, value=100000.0, step=1000.0)
        position_size = st.number_input('Position Size (₹ per trade)', min_value=1000.0, value=50000.0, step=1000.0)

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input('Start Date', value=datetime(2020, 1, 1))
        with col2:
            end_date = st.date_input('End Date', value=datetime.today())

        warmup_bars = st.number_input(
            'Warm-up Bars (history before Start Date)',
            min_value=WARMUP_MIN_BARS, value=WARMUP_TARGET_BARS, step=50,
            help=(
                'Extra historical bars downloaded before Start Date so all '
                'EMAs are fully converged before your selected period begins '
                '(matches TradingView, which computes indicators over full '
                'chart history, not just the visible range). This warm-up '
                'data is used only to keep EMA values and BUY/SELL state '
                'correct -- it is never shown or traded.'
            ),
        )

        st.subheader('Risk Management (Backtest Only)')
        st.caption('Stop Loss / Target affect only the backtest engine -- they never change scanner BUY/SELL signals.')
        enable_stop_loss = st.checkbox('Enable Stop Loss', value=False)
        stop_loss_pct = st.slider('Stop Loss %', min_value=1.0, max_value=20.0, value=5.0, step=0.5,
                                   disabled=not enable_stop_loss)
        enable_target = st.checkbox('Enable Target', value=False)
        target_pct = st.slider('Target %', min_value=1.0, max_value=50.0, value=10.0, step=0.5,
                                disabled=not enable_target)

        st.subheader('Entry Filter Settings (Backtest Only)')
        st.caption('Every filter only decides whether an already-fired BUY signal is taken. Scanner BUY/SELL signals are never changed.')

        enable_ema_width_filter = st.checkbox('Enable EMA Channel Width Filter', value=False)
        ema_width_min, ema_width_max = st.slider(
            'EMA Channel Width % (min, max)', min_value=0.50, max_value=5.00,
            value=(1.0, 2.0), step=0.05, disabled=not enable_ema_width_filter,
        )

        enable_body_ratio_filter = st.checkbox('Enable Body Ratio Filter', value=False)
        body_ratio_min, body_ratio_max = st.slider(
            'Body Ratio (min, max)', min_value=0.50, max_value=1.00,
            value=(0.70, 0.80), step=0.01, disabled=not enable_body_ratio_filter,
        )

        enable_candle_range_filter = st.checkbox('Enable Candle Range Filter', value=False)
        candle_range_max = st.slider(
            'Maximum Candle Range %', min_value=2.0, max_value=8.0, value=4.0, step=0.1,
            disabled=not enable_candle_range_filter,
        )

        enable_prev_gap_filter = st.checkbox('Enable Previous Gap Filter', value=False)
        prev_gap_lookback = st.slider(
            'Lookback Days', min_value=1, max_value=10, value=3, step=1,
            disabled=not enable_prev_gap_filter,
        )
        prev_gap_up_max = st.slider(
            'Maximum Gap Up %', min_value=0.5, max_value=5.0, value=2.0, step=0.1,
            disabled=not enable_prev_gap_filter,
        )
        prev_gap_down_max = st.slider(
            'Maximum Gap Down %', min_value=0.5, max_value=5.0, value=2.0, step=0.1,
            disabled=not enable_prev_gap_filter,
        )

        enable_next_gap_filter = st.checkbox('Enable Next Day Gap Filter', value=False)
        next_gap_up_max = st.slider(
            'Maximum Next Day Gap Up %', min_value=0.0, max_value=3.0, value=1.0, step=0.1,
            disabled=not enable_next_gap_filter,
        )
        entry_type = st.selectbox(
            'Entry Type', ['Next Day Open', 'Next Day Close'], index=0,
            help=(
                'Fill price basis for every BUY entry (independent of whether '
                'the Next Day Gap Filter is enabled). When the Next Day Gap '
                'Filter is on, its gap % check automatically uses this same '
                'basis -- Next Day Open vs. signal Close, or Next Day Close '
                'vs. signal Close.'
            ),
        )

        filter_settings = {
            'ema_width_enabled': enable_ema_width_filter, 'ema_width_min': ema_width_min, 'ema_width_max': ema_width_max,
            'body_ratio_enabled': enable_body_ratio_filter, 'body_ratio_min': body_ratio_min, 'body_ratio_max': body_ratio_max,
            'candle_range_enabled': enable_candle_range_filter, 'candle_range_max': candle_range_max,
            'prev_gap_enabled': enable_prev_gap_filter, 'prev_gap_lookback': prev_gap_lookback,
            'prev_gap_up_max': prev_gap_up_max, 'prev_gap_down_max': prev_gap_down_max,
            'next_gap_enabled': enable_next_gap_filter, 'next_gap_up_max': next_gap_up_max,
            'entry_type': entry_type,
        }

        st.subheader('Portfolio Sizing')
        max_concurrent_positions = st.number_input(
            'Maximum Concurrent Positions',
            min_value=1, max_value=500, value=10, step=1,
            help='The backtester will never hold more than this many open trades at once, across all symbols.',
        )
        transaction_cost_pct = st.number_input(
            'Transaction Cost % (per side)', min_value=0.0, max_value=5.0, value=0.0, step=0.01,
            help='Applied to both entry and exit trade value; reduces available cash and net P&L accordingly.',
        )

        st.subheader('Stock Universe')
        universe_choice = st.radio(
            'Select universe',
            ['Custom (Upload/Manual)', 'Built-in NIFTY 200', 'Built-in NIFTY 500'],
            help=(
                'Built-in universes are stored directly in the app (no internet '
                'lookup needed) but are point-in-time snapshots -- NSE rebalances '
                'these indices periodically. Use Custom Upload with a freshly '
                'exported constituent file for guaranteed current accuracy.'
            ),
        )

        uploaded_file = None
        manual_symbols = ''
        if universe_choice == 'Custom (Upload/Manual)':
            st.subheader('Symbols')
            uploaded_file = st.file_uploader('Upload CSV/Excel with symbols', type=['csv', 'xlsx', 'xls'])
            manual_symbols = st.text_area('Or enter symbols (comma or newline separated)', placeholder='RELIANCE.NS, TCS.NS, INFY.NS')
        else:
            built_in_list = NIFTY_200_SYMBOLS if universe_choice == 'Built-in NIFTY 200' else NIFTY_500_SYMBOLS
            st.caption(f'Using built-in list of {len(built_in_list)} symbols.')

        run_btn = st.button('▶️ Run Scanner & Backtest')

    # Parse raw symbols (blank rows ignored; no cleaning/suffixing yet)
    raw_symbols = []
    if universe_choice != 'Custom (Upload/Manual)':
        raw_symbols.extend(NIFTY_200_SYMBOLS if universe_choice == 'Built-in NIFTY 200' else NIFTY_500_SYMBOLS)
    else:
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    uploaded_df = pd.read_csv(uploaded_file)
                else:
                    uploaded_df = pd.read_excel(uploaded_file)
                col = detect_symbol_column(uploaded_df)
                file_symbols = [s for s in col.dropna().astype(str).tolist() if s.strip()]
                raw_symbols.extend(file_symbols)
                st.sidebar.success(f'Loaded {len(file_symbols)} symbol(s) from file.')
            except Exception as e:
                st.sidebar.error(f'Error reading file: {e}')

        if manual_symbols.strip():
            manual_list = [s for s in manual_symbols.replace('\n', ',').split(',') if s.strip()]
            raw_symbols.extend(manual_list)

    uploaded_count = len(raw_symbols)
    symbols, duplicate_count = normalize_symbols(raw_symbols)
    cleaned_count = len(symbols)

    if not symbols:
        st.info('👈 Please upload a symbol file or enter symbols in the sidebar, then click **Run Scanner & Backtest**.')
        return

    if not run_btn:
        st.info(f'Ready to process {len(symbols)} symbol(s) (from {uploaded_count} uploaded, {duplicate_count} duplicate(s) removed). Click **Run Scanner & Backtest** to begin.')
        return

    # Progress info: bar + live text
    progress_bar = st.progress(0)
    progress_text = st.empty()

    all_scanner_rows = []
    rejected_signal_rows = []  # BUY signals that fired but were rejected by an Entry Filter
    buy_signal_snapshots = []  # every BUY signal's snapshot (approved + rejected), for Filter Statistics
    symbol_data = {}  # symbol -> in-range df with signals, fed into the portfolio engine
    failed_symbols = []
    warmup_notes = []  # symbols where fewer than WARMUP_MIN_BARS were available before Start Date
    symbol_overview_rows = []  # one row per symbol: current status, for the Scanner Output table

    total = len(symbols)
    download_start = compute_warmup_download_start(start_date, warmup_bars)

    success_count = 0
    failed_count = 0

    # --- Phase 1: batch-download every symbol's OHLC history, several
    # batches of BATCH_SIZE symbols fetched in parallel, with a session
    # cache so re-running the same symbols/date-range is instant. ---
    if 'yf_cache' not in st.session_state:
        st.session_state['yf_cache'] = {}

    def _on_batch_done(done, total_batches):
        progress_bar.progress(min(done / max(total_batches, 1) * 0.5, 0.5))
        progress_text.text(f'Downloading data... batch {done}/{total_batches}')

    progress_text.text(f'Downloading data for {total} symbol(s)...')
    raw_data = download_all_symbols(
        symbols, download_start, end_date, st.session_state['yf_cache'],
        progress_callback=_on_batch_done,
    )

    # --- Phase 2: generate signals per symbol (CPU-bound, fast) from the
    # already-downloaded data -- untouched STRATEGY MODULE, run once per
    # stock, never recalculated. ---
    for i, symbol in enumerate(symbols):
        remaining = total - (i + 1)
        progress_bar.progress(min(0.5 + (i + 1) / total * 0.5, 1.0))
        progress_text.text(
            f'Processing [{i+1}/{total}] {symbol}  |  '
            f'Success: {success_count}   Failed: {failed_count}   Remaining: {remaining}'
        )

        # Signal generation for this one symbol from its pre-downloaded
        # data. Never raises -- any failure (bad ticker, no data, ...) is
        # captured in r['status'] so the scan always continues.
        r = process_symbol_df(symbol, raw_data.get(symbol), ema_length, start_date, end_date, filter_settings=filter_settings)

        if r['in_range_df'] is not None:
            success_count += 1
            symbol_data[symbol] = r['in_range_df']

            if r['bars_before_start'] < WARMUP_MIN_BARS:
                warmup_notes.append((symbol, r['bars_before_start']))

            # Scanner rows -- same in-range signals the backtester will use,
            # now carrying the full analytics snapshot (same detail as the
            # Enhanced Trade Log) so signals can be analyzed even when they
            # never become an executed trade.
            signal_df = r['in_range_df'][r['in_range_df']['signal'].notna()]
            for idx, row in signal_df.iterrows():
                snapshot = extract_signal_snapshot(symbol, idx, row)
                all_scanner_rows.append(snapshot)
                if snapshot['signal'] == 'BUY':
                    buy_signal_snapshots.append(snapshot)
                    if snapshot.get('final_entry_approved') is False:
                        rejected_signal_rows.append(snapshot)
        else:
            failed_count += 1
            failed_symbols.append((symbol, r['status']))

        symbol_overview_rows.append({
            'Symbol': symbol,
            'Last Close': round(r['last_close'], 4) if r['last_close'] is not None else None,
            'EMA High': round(r['ema_high'], 4) if r['ema_high'] is not None else None,
            'EMA Low': round(r['ema_low'], 4) if r['ema_low'] is not None else None,
            'Current Signal': r['current_signal'],
            'Last Signal Date': r['last_signal_date'],
            'Download Status': r['status'],
        })

    progress_bar.empty()
    progress_text.empty()

    # ------------------------------------------------------------------
    # Run the single shared-capital, single-position backtest across
    # every scanned symbol (BACKTESTING MODULE – consumes signals only)
    # ------------------------------------------------------------------
    trades_df, combined_equity, _, portfolio_stats = run_portfolio_backtest(
        symbol_data, initial_capital, position_size,
        enable_stop_loss=enable_stop_loss, stop_loss_pct=stop_loss_pct,
        enable_target=enable_target, target_pct=target_pct,
        max_concurrent_positions=max_concurrent_positions,
        entry_type=entry_type, transaction_cost_pct=transaction_cost_pct,
    )
    if combined_equity.empty:
        combined_equity = pd.DataFrame({'date': [pd.Timestamp(start_date)], 'equity': [float(initial_capital)]})

    # ------------------------------------------------------------------
    # Display Results
    # ------------------------------------------------------------------

    overview_df = pd.DataFrame(symbol_overview_rows)
    buy_count = int((overview_df['Current Signal'] == 'BUY').sum()) if not overview_df.empty else 0
    sell_count = int((overview_df['Current Signal'] == 'SELL').sum()) if not overview_df.empty else 0
    no_signal_count = int((overview_df['Current Signal'] == 'No Signal').sum()) if not overview_df.empty else 0

    st.subheader('🔎 Scan Summary')
    scan_summary = {
        'Symbols Uploaded': uploaded_count,
        'Symbols After Cleaning': cleaned_count,
        'Duplicate Symbols Removed': duplicate_count,
        'Successfully Downloaded': success_count,
        'Failed Downloads': failed_count,
        'Symbols With BUY Signal': buy_count,
        'Symbols With SELL Signal': sell_count,
        'Symbols Without Signal': no_signal_count,
    }
    st.dataframe(pd.DataFrame([scan_summary]), use_container_width=True, hide_index=True)

    st.divider()

    st.subheader('🔍 Scanner Output')
    show_only_signals = st.checkbox('Show Only BUY/SELL Signals', value=True)
    display_overview = overview_df
    if show_only_signals and not overview_df.empty:
        display_overview = overview_df[overview_df['Current Signal'].isin(['BUY', 'SELL'])]
    if not display_overview.empty:
        st.dataframe(display_overview, use_container_width=True, hide_index=True)

        csv_overview = display_overview.to_csv(index=False).encode('utf-8')
        st.download_button(
            label='📥 Download Scanner Output (CSV)',
            data=csv_overview,
            file_name='scanner_output.csv',
            mime='text/csv',
        )
    else:
        st.warning('No symbols match the current filter.')

    st.divider()

    scanner_df = pd.DataFrame(all_scanner_rows)

    st.subheader('📊 Scanner Results (Signal History)')
    if not scanner_df.empty:
        st.dataframe(scanner_df, use_container_width=True, hide_index=True)

        csv_scanner = scanner_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label='📥 Download Scanner Results (CSV)',
            data=csv_scanner,
            file_name='scanner_results.csv',
            mime='text/csv',
        )
    else:
        st.warning('No signals generated for the given symbols and date range.')

    st.divider()

    st.subheader('🧮 Filter Statistics')
    filter_stats = compute_filter_statistics(buy_signal_snapshots)
    st.dataframe(pd.DataFrame([filter_stats]), use_container_width=True, hide_index=True)

    st.divider()

    st.subheader('🚫 Rejected Signals Log')
    st.caption('Every BUY signal that fired but was rejected by an Entry Filter, with its full analytics snapshot -- lets you compare accepted vs. rejected signals for optimization.')
    if rejected_signal_rows:
        rejected_df = pd.DataFrame(rejected_signal_rows)
        st.dataframe(rejected_df, use_container_width=True, hide_index=True)

        csv_rejected = rejected_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label='📥 Download Rejected Signals Log (CSV)',
            data=csv_rejected,
            file_name='rejected_signals_log.csv',
            mime='text/csv',
        )
    else:
        st.info('No BUY signals were rejected by an Entry Filter (or no Entry Filters are enabled).')

    st.divider()

    st.subheader('📋 Backtest Trade Log')
    if not trades_df.empty:
        # Explicit friendly labels for the original columns (kept stable for
        # anyone already relying on these names); every other column (the
        # new Enhanced Trade Log fields) gets a generic Title Case label so
        # nothing is ever silently dropped from the export.
        known_labels = {
            'trade_id': 'Trade ID',
            'symbol': 'Symbol',
            'signal_date': 'Signal Date',
            'entry_date': 'Entry Date',
            'entry_price': 'Entry Price',
            'exit_date': 'Exit Date',
            'exit_price': 'Exit Price',
            'shares': 'Shares',
            'invested_amount': 'Invested Amount',
            'exit_value': 'Exit Value',
            'pnl': 'Profit/Loss (₹)',
            'pnl_pct': 'Profit/Loss (%)',
            'holding_days': 'Holding Days',
            'exit_reason': 'Exit Reason',
            'stop_price': 'Stop Price',
            'target_price': 'Target Price',
            'trade_duration': 'Trade Duration',
            'available_cash_after_trade': 'Available Cash After Trade',
        }
        display_labels = {
            col: known_labels.get(col, col.replace('_', ' ').replace('pct', '%').strip().title())
            for col in trades_df.columns
        }
        display_trades = trades_df.rename(columns=display_labels)
        st.dataframe(display_trades, use_container_width=True, hide_index=True)

        col_csv, col_xlsx = st.columns(2)
        with col_csv:
            csv_trades = display_trades.to_csv(index=False).encode('utf-8')
            st.download_button(
                label='📥 Download Trade Log (CSV)',
                data=csv_trades,
                file_name='trade_log.csv',
                mime='text/csv',
            )
        with col_xlsx:
            xlsx_buffer = io.BytesIO()
            with pd.ExcelWriter(xlsx_buffer, engine='openpyxl') as writer:
                display_trades.to_excel(writer, index=False, sheet_name='Trade Log')
            st.download_button(
                label='📥 Download Trade Log (Excel)',
                data=xlsx_buffer.getvalue(),
                file_name='trade_log.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            )
    else:
        st.warning('No trades executed.')

    st.divider()

    st.subheader('📈 Backtest Summary')
    summary = compute_backtest_summary(trades_df, combined_equity, initial_capital, portfolio_stats)
    summary_df = pd.DataFrame([summary])
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    csv_summary = summary_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label='📥 Download Backtest Summary (CSV)',
        data=csv_summary,
        file_name='backtest_summary.csv',
        mime='text/csv',
    )

    # Optional equity curve
    fig = plot_equity_curve(combined_equity)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader('🧪 Optimization Report')
    st.caption('Executed-trade outcomes bucketed by each parameter, to help identify the best filter/setting ranges. Built only from completed trades (win/loss requires an exit).')
    if not trades_df.empty:
        opt_tables = [
            ('EMA Width vs Win Rate', 'ema_channel_width_pct'),
            ('Body Ratio vs Win Rate', 'candle_body_ratio'),
            ('Candle Range vs Win Rate', 'candle_range_pct'),
            ('Gap % vs Win Rate', 'previous_day_gap_pct'),
            ('Holding Days vs Return', 'holding_days'),
        ]
        for title, col in opt_tables:
            st.markdown(f'**{title}**')
            table = build_optimization_bucket_table(trades_df, col)
            if not table.empty:
                st.dataframe(table, use_container_width=True, hide_index=True)
            else:
                st.caption('Not enough data to bucket (missing values or too few trades).')
    else:
        st.info('No completed trades to build an optimization report from.')

    if failed_symbols:
        with st.expander(f'⚠️ Failed Symbols ({len(failed_symbols)})'):
            for sym, reason in failed_symbols:
                st.text(f'{sym}: {reason}')

    if warmup_notes:
        with st.expander(f'ℹ️ Limited Warm-up Data ({len(warmup_notes)})'):
            st.caption(
                f'These symbols had fewer than {WARMUP_MIN_BARS} trading days of '
                'history available before Start Date (e.g. recent listings). '
                'Their EMA values near the start of the selected period may not '
                'fully match TradingView, since less real history was available '
                'to converge on.'
            )
            for sym, bars in warmup_notes:
                st.text(f'{sym}: {bars} bar(s) available before Start Date')


if __name__ == '__main__':
    main()
