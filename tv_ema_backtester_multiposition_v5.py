"""
TradingView EMA Strategy Scanner & Backtester
Replicates exact Pine Script logic in Python/Streamlit.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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
# BACKTESTING MODULE
# ===================================================================
#
# This module NEVER recalculates or alters BUY/SELL signals. It only
# consumes the 'signal' column already produced by generate_signals()
# (STRATEGY MODULE above). All capital management, trade execution,
# and single-position-at-a-time rules live here.
#
# Architecture: one shared capital pool and at most ONE open position
# across the ENTIRE scanned universe at any point in time. Signals
# from every symbol are merged into a single stream of next-day
# trigger events and executed in strict chronological order.
# ===================================================================

def build_trigger_events(symbol_data: dict) -> list:
    """
    Convert each symbol's pre-generated 'signal' column into executable
    next-day trigger events. A signal on day i becomes actionable at day
    i+1's Open (that symbol's own next trading day) -- signals are read
    exactly as produced; nothing here re-derives or changes them.

    A signal on a symbol's LAST available row has no next day to execute
    on within the downloaded data, so it produces no event (consistent
    with only acting on data that actually exists).
    """
    events = []
    for symbol, df in symbol_data.items():
        n = len(df)
        for i in range(n - 1):
            sig = df['signal'].iloc[i]
            if pd.notna(sig):
                events.append({
                    'symbol': symbol,
                    'action': sig,  # 'BUY' or 'SELL', exactly as generated
                    'signal_date': df.index[i],
                    'trigger_date': df.index[i + 1],
                    'trigger_open': float(df['Open'].iloc[i + 1]),
                })

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
      - Multiple symbols can open AND close positions on the same day.
      - Shares = floor(Position Size / Entry Price); required cash =
        Shares x Entry Price; on exit, invested amount + P&L returns to
        available cash immediately, reduces the open position count,
        and can fund a new trade (into the freed slot) the same day
        (checked in signal-date order).
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
            is not checked for Stop Loss/Target -- monitoring starts the
            next trading day, avoiding same-candle entry/exit ambiguity.
      - A position still open when its symbol's data runs out is closed
        at that symbol's last available Close, exit reason "End of Data".

    Returns
    -------
    trades_df : pd.DataFrame
    equity_df : pd.DataFrame        (daily portfolio equity across the
                                      union of all symbols' trading dates)
    final_cash : float
    portfolio_stats : dict          (concurrency / capital-utilization /
                                      skip-reason stats used by
                                      compute_backtest_summary)
    """
    cash = float(initial_capital)
    positions = {}  # symbol -> {signal_date, entry_date, entry_price, shares, invested_amount, stop_price, target_price}
    trades = []
    equity_records = []
    concurrent_counts = []
    invested_amounts_daily = []
    skipped_position_limit = 0
    skipped_insufficient_cash = 0

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

    events = build_trigger_events(symbol_data)

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
        exit_value = pos['shares'] * exit_price
        cash += exit_value
        pnl = exit_value - pos['invested_amount']
        pnl_pct = (pnl / pos['invested_amount']) * 100 if pos['invested_amount'] != 0 else 0.0
        holding_days = (exit_date - pos['entry_date']).days

        trades.append({
            'symbol': symbol,
            'signal_date': pos['signal_date'],
            'entry_date': pos['entry_date'],
            'entry_price': round(float(pos['entry_price']), 4),
            'exit_date': exit_date,
            'exit_price': round(float(exit_price), 4),
            'shares': int(pos['shares']),
            'invested_amount': round(float(pos['invested_amount']), 2),
            'exit_value': round(float(exit_value), 2),
            'pnl': round(float(pnl), 2),
            'pnl_pct': round(float(pnl_pct), 2),
            'holding_days': int(holding_days),
            'exit_reason': exit_reason,
            'stop_price': round(float(pos['stop_price']), 4) if pos['stop_price'] is not None else None,
            'target_price': round(float(pos['target_price']), 4) if pos['target_price'] is not None else None,
            'trade_duration': int(holding_days),
            'available_cash_after_trade': round(float(cash), 2),
        })

    event_idx = 0
    n_events = len(events)

    for d in all_dates:
        # --- 1) Stop Loss / Target checks for positions already open
        #        (from a prior day), BEFORE today's signal events, so the
        #        exit-priority (Stop Loss > Target > SELL signal) holds. ---
        for symbol in list(positions.keys()):
            df = symbol_data[symbol]
            if d not in df.index:
                continue
            row = df.loc[d]
            pos = positions[symbol]
            low = float(row['Low'])
            high = float(row['High'])

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
            today_open = ev['trigger_open']

            if ev['action'] == 'BUY':
                if symbol in positions:
                    continue  # this symbol already has an open position -> ignore
                if len(positions) >= max_concurrent_positions:
                    skipped_position_limit += 1
                    continue  # at the concurrent-position cap -> skip only this signal
                calc_shares = int(np.floor(position_size / today_open)) if today_open > 0 else 0
                invested = calc_shares * today_open
                if calc_shares <= 0:
                    continue  # position size can't buy even one share -> skip this entry only
                if cash < invested:
                    skipped_insufficient_cash += 1
                    continue  # insufficient cash -> skip this entry only
                cash -= invested
                stop_price = today_open * (1 - stop_loss_pct / 100.0) if enable_stop_loss else None
                target_price = today_open * (1 + target_pct / 100.0) if enable_target else None
                positions[symbol] = {
                    'signal_date': ev['signal_date'],
                    'entry_date': ev['trigger_date'],
                    'entry_price': today_open,
                    'shares': calc_shares,
                    'invested_amount': invested,
                    'stop_price': stop_price,
                    'target_price': target_price,
                }

            elif ev['action'] == 'SELL':
                if symbol not in positions:
                    continue  # no open position in this symbol (or already closed by Stop/Target today) -> ignore
                close_position(symbol, d, today_open, 'SELL Signal')

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

def process_symbol_df(symbol: str, raw_df, ema_length: int, start_date, end_date) -> dict:
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

        st.subheader('Portfolio Sizing')
        max_concurrent_positions = st.number_input(
            'Maximum Concurrent Positions',
            min_value=1, max_value=500, value=10, step=1,
            help='The backtester will never hold more than this many open trades at once, across all symbols.',
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
        r = process_symbol_df(symbol, raw_data.get(symbol), ema_length, start_date, end_date)

        if r['in_range_df'] is not None:
            success_count += 1
            symbol_data[symbol] = r['in_range_df']

            if r['bars_before_start'] < WARMUP_MIN_BARS:
                warmup_notes.append((symbol, r['bars_before_start']))

            # Scanner rows -- same in-range signals the backtester will use
            signal_df = r['in_range_df'][r['in_range_df']['signal'].notna()]
            for idx, row in signal_df.iterrows():
                all_scanner_rows.append({
                    'symbol': symbol,
                    'signal': row['signal'],
                    'signal_date': idx,
                    'close': round(row['Close'], 4),
                    'ema_high': round(row['ema_high'], 4),
                    'ema_low': round(row['ema_low'], 4),
                    'channel_height': round(row['channel_height'], 4),
                    'candle_range': round(row['total_range'], 4),
                    'body_size': round(row['body_size'], 4),
                    'body_ratio': round(row['body_ratio'], 4),
                })
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

    st.subheader('📋 Backtest Trade Log')
    if not trades_df.empty:
        # Display-friendly column labels (internal engine keys unchanged above)
        display_trades = trades_df.rename(columns={
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
        })
        st.dataframe(display_trades, use_container_width=True, hide_index=True)

        csv_trades = display_trades.to_csv(index=False).encode('utf-8')
        st.download_button(
            label='📥 Download Trade Log (CSV)',
            data=csv_trades,
            file_name='trade_log.csv',
            mime='text/csv',
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
