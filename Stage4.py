import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import io
import os

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Weinstein 4-Stage Analysis Scanner",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1f77b4; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #666; margin-bottom: 2rem; }
    .stage-1 { color: #e74c3c; }
    .stage-2 { color: #27ae60; }
    .stage-3 { color: #f39c12; }
    .stage-4 { color: #8e44ad; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# NSE Sector Index Configuration (Item 1)
# yfinance ticker symbols for NSE sector indices. These are editable in the
# sidebar because Yahoo's coverage of NSE sector indices is inconsistent —
# if one returns no data, override it there.
# ---------------------------------------------------------------------------
DEFAULT_SECTOR_INDICES = {
    "NIFTY IT": "^CNXIT",
    "NIFTY BANK": "^NSEBANK",
    "NIFTY FIN SERVICE": "NIFTY_FIN_SERVICE.NS",
    "NIFTY FMCG": "^CNXFMCG",
    "NIFTY PHARMA": "^CNXPHARMA",
    "NIFTY AUTO": "^CNXAUTO",
    "NIFTY METAL": "^CNXMETAL",
    "NIFTY ENERGY": "^CNXENERGY",
    "NIFTY REALTY": "^CNXREALTY",
    "NIFTY MEDIA": "^CNXMEDIA",
    "NIFTY PSU BANK": "^CNXPSUBANK",
    "NIFTY INFRA": "^CNXINFRA",
    "NIFTY CONSUMPTION": "^CNXCONSUM",
}

# Stock -> Sector mapping (NSE symbols without suffix)
STOCK_SECTOR_MAP = {
    # IT
    "TCS": "NIFTY IT", "INFY": "NIFTY IT", "WIPRO": "NIFTY IT", "HCLTECH": "NIFTY IT",
    "TECHM": "NIFTY IT", "LTIM": "NIFTY IT", "COFORGE": "NIFTY IT", "PERSISTENT": "NIFTY IT",
    "MPHASIS": "NIFTY IT", "LTTS": "NIFTY IT", "OFSS": "NIFTY IT", "TATAELXSI": "NIFTY IT",
    "KPITTECH": "NIFTY IT",
    # Banks
    "HDFCBANK": "NIFTY BANK", "ICICIBANK": "NIFTY BANK", "KOTAKBANK": "NIFTY BANK",
    "AXISBANK": "NIFTY BANK", "SBIN": "NIFTY BANK", "INDUSINDBK": "NIFTY BANK",
    "FEDERALBNK": "NIFTY BANK", "IDFCFIRSTB": "NIFTY BANK", "BANDHANBNK": "NIFTY BANK",
    "AUBANK": "NIFTY BANK",
    # PSU Banks
    "BANKBARODA": "NIFTY PSU BANK", "PNB": "NIFTY PSU BANK", "CANBK": "NIFTY PSU BANK",
    "UNIONBANK": "NIFTY PSU BANK", "IOB": "NIFTY PSU BANK", "IDBI": "NIFTY PSU BANK",
    # Financial Services
    "BAJFINANCE": "NIFTY FIN SERVICE", "BAJAJFINSV": "NIFTY FIN SERVICE",
    "HDFCLIFE": "NIFTY FIN SERVICE", "SBILIFE": "NIFTY FIN SERVICE",
    "SHRIRAMFIN": "NIFTY FIN SERVICE", "CHOLAFIN": "NIFTY FIN SERVICE",
    "ICICIGI": "NIFTY FIN SERVICE", "ICICIPRULI": "NIFTY FIN SERVICE",
    "MUTHOOTFIN": "NIFTY FIN SERVICE", "LICI": "NIFTY FIN SERVICE",
    "PFC": "NIFTY FIN SERVICE", "RECLTD": "NIFTY FIN SERVICE", "IRFC": "NIFTY FIN SERVICE",
    "JIOFIN": "NIFTY FIN SERVICE", "HDFCAMC": "NIFTY FIN SERVICE",
    "LICHSGFIN": "NIFTY FIN SERVICE", "M&MFIN": "NIFTY FIN SERVICE",
    "SUNDARMFIN": "NIFTY FIN SERVICE", "MFSL": "NIFTY FIN SERVICE",
    "BAJAJHLDNG": "NIFTY FIN SERVICE", "STARHEALTH": "NIFTY FIN SERVICE",
    # FMCG
    "HINDUNILVR": "NIFTY FMCG", "ITC": "NIFTY FMCG", "NESTLEIND": "NIFTY FMCG",
    "BRITANNIA": "NIFTY FMCG", "DABUR": "NIFTY FMCG", "GODREJCP": "NIFTY FMCG",
    "MARICO": "NIFTY FMCG", "COLPAL": "NIFTY FMCG", "TATACONSUM": "NIFTY FMCG",
    "UNITDSPR": "NIFTY FMCG", "VBL": "NIFTY FMCG", "PATANJALI": "NIFTY FMCG",
    # Pharma
    "SUNPHARMA": "NIFTY PHARMA", "DRREDDY": "NIFTY PHARMA", "CIPLA": "NIFTY PHARMA",
    "DIVISLAB": "NIFTY PHARMA", "LUPIN": "NIFTY PHARMA", "AUROPHARMA": "NIFTY PHARMA",
    "TORNTPHARM": "NIFTY PHARMA", "ZYDUSLIFE": "NIFTY PHARMA", "ALKEM": "NIFTY PHARMA",
    "BIOCON": "NIFTY PHARMA", "IPCALAB": "NIFTY PHARMA", "LAURUSLABS": "NIFTY PHARMA",
    "ABBOTINDIA": "NIFTY PHARMA", "MANKIND": "NIFTY PHARMA", "APOLLOHOSP": "NIFTY PHARMA",
    "MAXHEALTH": "NIFTY PHARMA", "FORTIS": "NIFTY PHARMA",
    # Auto
    "MARUTI": "NIFTY AUTO", "M&M": "NIFTY AUTO", "TATAMOTORS": "NIFTY AUTO",
    "BAJAJ-AUTO": "NIFTY AUTO", "HEROMOTOCO": "NIFTY AUTO", "EICHERMOT": "NIFTY AUTO",
    "TVSMOTOR": "NIFTY AUTO", "ASHOKLEY": "NIFTY AUTO", "BALKRISIND": "NIFTY AUTO",
    "MOTHERSON": "NIFTY AUTO", "BHARATFORG": "NIFTY AUTO", "MRF": "NIFTY AUTO",
    "EXIDEIND": "NIFTY AUTO", "SONACOMS": "NIFTY AUTO", "TIINDIA": "NIFTY AUTO",
    "ESCORTS": "NIFTY AUTO", "BOSCHLTD": "NIFTY AUTO",
    # Metal
    "TATASTEEL": "NIFTY METAL", "JSWSTEEL": "NIFTY METAL", "HINDALCO": "NIFTY METAL",
    "VEDL": "NIFTY METAL", "JINDALSTEL": "NIFTY METAL", "SAIL": "NIFTY METAL",
    "NATIONALUM": "NIFTY METAL", "NMDC": "NIFTY METAL", "APLAPOLLO": "NIFTY METAL",
    "COALINDIA": "NIFTY METAL",
    # Energy
    "RELIANCE": "NIFTY ENERGY", "ONGC": "NIFTY ENERGY", "NTPC": "NIFTY ENERGY",
    "POWERGRID": "NIFTY ENERGY", "BPCL": "NIFTY ENERGY", "IOC": "NIFTY ENERGY",
    "GAIL": "NIFTY ENERGY", "HINDPETRO": "NIFTY ENERGY", "TATAPOWER": "NIFTY ENERGY",
    "ADANIGREEN": "NIFTY ENERGY", "ADANIPOWER": "NIFTY ENERGY", "JSWENERGY": "NIFTY ENERGY",
    "NHPC": "NIFTY ENERGY", "SJVN": "NIFTY ENERGY", "TORNTPOWER": "NIFTY ENERGY",
    "OIL": "NIFTY ENERGY", "PETRONET": "NIFTY ENERGY", "IGL": "NIFTY ENERGY",
    "GUJGASLTD": "NIFTY ENERGY", "ATGL": "NIFTY ENERGY", "SUZLON": "NIFTY ENERGY",
    "ADANIENSOL": "NIFTY ENERGY",
    # Realty
    "DLF": "NIFTY REALTY", "GODREJPROP": "NIFTY REALTY", "OBEROIRLTY": "NIFTY REALTY",
    "PRESTIGE": "NIFTY REALTY", "LODHA": "NIFTY REALTY",
    # Media
    "ZEEL": "NIFTY MEDIA", "PVRINOX": "NIFTY MEDIA", "SUNTV": "NIFTY MEDIA",
    "NAUKRI": "NIFTY MEDIA",
    # Infra
    "LT": "NIFTY INFRA", "ADANIPORTS": "NIFTY INFRA", "ULTRACEMCO": "NIFTY INFRA",
    "GRASIM": "NIFTY INFRA", "SHREECEM": "NIFTY INFRA", "AMBUJACEM": "NIFTY INFRA",
    "ACC": "NIFTY INFRA", "BHARTIARTL": "NIFTY INFRA", "SIEMENS": "NIFTY INFRA",
    "ABB": "NIFTY INFRA", "BEL": "NIFTY INFRA", "HAL": "NIFTY INFRA",
    "BHEL": "NIFTY INFRA", "CGPOWER": "NIFTY INFRA", "CUMMINSIND": "NIFTY INFRA",
    "POLYCAB": "NIFTY INFRA", "HAVELLS": "NIFTY INFRA", "GMRAIRPORT": "NIFTY INFRA",
    "CONCOR": "NIFTY INFRA", "RVNL": "NIFTY INFRA", "INDUSTOWER": "NIFTY INFRA",
    "TATACOMM": "NIFTY INFRA", "SOLARINDS": "NIFTY INFRA", "ASTRAL": "NIFTY INFRA",
    "SUPREMEIND": "NIFTY INFRA", "DIXON": "NIFTY INFRA",
    # Consumption
    "TITAN": "NIFTY CONSUMPTION", "ASIANPAINT": "NIFTY CONSUMPTION",
    "TRENT": "NIFTY CONSUMPTION", "BERGEPAINT": "NIFTY CONSUMPTION",
    "PIDILITIND": "NIFTY CONSUMPTION", "VOLTAS": "NIFTY CONSUMPTION",
    "INDHOTEL": "NIFTY CONSUMPTION", "JUBLFOOD": "NIFTY CONSUMPTION",
    "PAGEIND": "NIFTY CONSUMPTION", "BATAINDIA": "NIFTY CONSUMPTION",
    "ZOMATO": "NIFTY CONSUMPTION", "INDIGO": "NIFTY CONSUMPTION",
    "DELHIVERY": "NIFTY CONSUMPTION", "PAYTM": "NIFTY CONSUMPTION",
    "BSE": "NIFTY CONSUMPTION", "IEX": "NIFTY CONSUMPTION",
    # Chemicals / others -> Infra bucket fallback
    "UPL": "NIFTY INFRA", "SRF": "NIFTY INFRA", "PIIND": "NIFTY INFRA",
    "DEEPAKNTR": "NIFTY INFRA", "AARTIIND": "NIFTY INFRA", "COROMANDEL": "NIFTY INFRA",
    "ADANIENT": "NIFTY INFRA",
}

# Sub-stage -> Recommended Action (Item 3)
SUBSTAGE_ACTION_MAP = {
    "1A": "Avoid (Dead Money)",
    "1B": "Watchlist / Prepare",
    "2A": "Primary BUY Zone",
    "2B": "Hold / Tighten Stops",
    "3A": "Trim 50% Position",
    "3B": "Sell Remaining on Rallies",
    "4A": "Exit Longs",
    "4B": "Cash / Stay Out",
}


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def normalize_symbol(sym):
    """Auto-append .NS for NSE symbols if no exchange suffix is present."""
    sym = str(sym).strip().upper()
    if not sym:
        return sym
    if sym.startswith('^') or '.' in sym:
        return sym
    return sym + '.NS'


def base_symbol(sym):
    """Strip exchange suffix to look up sector mapping."""
    return str(sym).upper().replace('.NS', '').replace('.BO', '').strip()


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_stock_data(symbol, start, end):
    """Fetch OHLCV data from yfinance. Cached (Item 21)."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, auto_adjust=True)
        if df.empty or len(df) < 50:
            return None, f"Insufficient data for {symbol}"
        df.reset_index(inplace=True)
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]
        if 'date' not in df.columns and 'datetime' in df.columns:
            df.rename(columns={'datetime': 'date'}, inplace=True)
        if 'adj_close' in df.columns and 'close' not in df.columns:
            df.rename(columns={'adj_close': 'close'}, inplace=True)
        needed = ['date', 'open', 'high', 'low', 'close', 'volume']
        for col in needed:
            if col not in df.columns:
                return None, f"Missing column {col} for {symbol}"
        df = df[needed].copy()
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        df.sort_values('date', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df, None
    except Exception as e:
        return None, f"Error fetching {symbol}: {str(e)}"


def calculate_slope(series, lookback):
    """Percentage slope: (today - lookback) / lookback * 100"""
    return (series - series.shift(lookback)) / series.shift(lookback) * 100


def calculate_indicators(df, params):
    """Calculate all technical indicators exactly per Pine Script logic."""
    df = df.copy()

    df['sma50'] = df['close'].rolling(window=params['sma50_period']).mean()
    df['sma200'] = df['close'].rolling(window=params['sma200_period']).mean()
    df['sma200_slope_pct'] = calculate_slope(df['sma200'], params['slope_lookback'])
    df['rising_ma'] = df['sma200_slope_pct'] > params['rising_threshold']
    df['falling_ma'] = df['sma200_slope_pct'] < -params['falling_threshold']
    df['above_ma'] = (df['close'] > df['sma200']) & (df['sma50'] > df['sma200'])
    df['below_ma'] = (df['close'] < df['sma200']) & (df['sma50'] < df['sma200'])

    df['avg_volume'] = df['volume'].rolling(window=params['volume_avg_length']).mean()
    df['volume_expansion'] = df['volume'] > (df['avg_volume'] * params['volume_expansion_mult'])
    df['volume_contraction'] = df['volume'] < (df['avg_volume'] / params['volume_expansion_mult'])

    df['volume_state'] = np.where(
        df['avg_volume'].isna(), 'Neutral',
        np.where(df['volume_expansion'], 'Expansion',
                 np.where(df['volume_contraction'], 'Contraction', 'Neutral'))
    )

    df['extension_pct'] = (df['close'] - df['sma50']) / df['sma50'] * 100
    df['support'] = df['low'].shift(1).rolling(window=params['support_lookback']).min()
    df['break_support'] = df['close'] < df['support']
    df['stock_return'] = df['close'] / df['close'].shift(params['rs_lookback']) - 1

    return df


def calculate_benchmark_rs(benchmark_df, params):
    """Benchmark return series for RS comparison."""
    b = benchmark_df.copy()
    b['benchmark_return'] = b['close'] / b['close'].shift(params['rs_lookback']) - 1
    return b[['date', 'benchmark_return']]


def calculate_sector_state(sector_df, params):
    """Sector Close > 50 SMA = Strong, else Weak (Item 1 / Item 2)."""
    s = sector_df.copy()
    s['sector_sma50'] = s['close'].rolling(window=params['sma50_period']).mean()
    s['sector_state'] = np.where(
        s['sector_sma50'].isna(), 'N/A',
        np.where(s['close'] > s['sector_sma50'], 'Strong', 'Weak')
    )
    return s[['date', 'sector_state']]


def classify_stages(df, params):
    """Pine Script stage state machine applied day-by-day."""
    stages = []
    days_in_stage = []
    prev_stage = 1

    rising_arr = df['rising_ma'].values
    falling_arr = df['falling_ma'].values
    above_arr = df['above_ma'].values
    below_arr = df['below_ma'].values

    for i in range(len(df)):
        rising, falling = rising_arr[i], falling_arr[i]
        above, below = above_arr[i], below_arr[i]

        if rising and above:
            curr_stage = 2
        elif falling and below:
            curr_stage = 4
        elif prev_stage == 2:
            curr_stage = 3
        elif prev_stage == 4:
            curr_stage = 1
        else:
            curr_stage = prev_stage

        if curr_stage == prev_stage and i > 0:
            days = days_in_stage[-1] + 1
        else:
            days = 1

        stages.append(curr_stage)
        days_in_stage.append(days)
        prev_stage = curr_stage

    df['stage'] = stages
    df['days_in_stage'] = days_in_stage
    df['weeks_in_stage'] = df['days_in_stage'] / 5.0
    df['prev_stage'] = df['stage'].shift(1).fillna(1).astype(int)

    df['buy_signal'] = (df['stage'] == 2) & (df['prev_stage'] != 2)
    df['sell_signal'] = (df['stage'] == 3) & (df['prev_stage'] == 2)
    df['stage4_confirmed'] = (df['stage'] == 4) & (df['prev_stage'] != 4)

    return df


def classify_substage(row, params):
    """Sub-stage classification exactly per Pine Script logic."""
    stage = row['stage']
    slope = row['sma200_slope_pct']
    weeks = row['weeks_in_stage']
    extension = row['extension_pct']
    break_support = row['break_support']
    falling_ma = row['falling_ma']
    volume_state = row['volume_state']

    if stage == 1:
        return '1A' if slope < -params['flat_threshold'] else '1B'
    elif stage == 2:
        if weeks <= params['late_stage_2_weeks'] and extension <= params['extended_pct']:
            return '2A'
        elif weeks > params['late_stage_2_weeks'] or extension > params['extended_pct']:
            return '2B'
        return '2A'
    elif stage == 3:
        if (not break_support) and (not falling_ma):
            return '3A'
        elif break_support or falling_ma:
            return '3B'
        return '3A'
    elif stage == 4:
        steep_decline = slope < -params['steep_decline_pct']
        if weeks <= params['late_stage_4_weeks'] and not steep_decline:
            return '4A'
        elif weeks > params['late_stage_4_weeks'] or (volume_state == 'Contraction' and steep_decline):
            return '4B'
        return '4A'
    return ''


def add_alert_windows(df, alert_window):
    """
    Item 6/7/8/9/10: rolling alert window + last buy/sell dates.
    A signal stays 'in window' for `alert_window` trading days after it fires.
    """
    w = int(alert_window)
    df['buy_window'] = df['buy_signal'].rolling(window=w, min_periods=1).max().astype(bool)
    df['sell_window'] = df['sell_signal'].rolling(window=w, min_periods=1).max().astype(bool)

    buy_dates = df['date'].where(df['buy_signal']).ffill()
    sell_dates = df['date'].where(df['sell_signal']).ffill()
    df['last_buy_date'] = buy_dates
    df['last_sell_date'] = sell_dates
    return df


def fmt_date(val):
    if pd.isna(val):
        return ''
    return pd.Timestamp(val).strftime('%d-%b-%Y')


def process_symbol(symbol, params, mode, start_date, end_date,
                   benchmark_rs, sector_state_map, sector_name_map,
                   sector_ticker_map, alert_window):
    """Main processing pipeline for a single symbol."""
    max_lookback = max(200, params['slope_lookback'], params['rs_lookback'], params['volume_avg_length'])

    if mode == 'Current':
        fetch_start = (datetime.now() - timedelta(days=max_lookback + 400)).strftime('%Y-%m-%d')
        fetch_end = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        fetch_start = (start_date - timedelta(days=max_lookback + 400)).strftime('%Y-%m-%d')
        fetch_end = (end_date + timedelta(days=1)).strftime('%Y-%m-%d')

    df, error = fetch_stock_data(symbol, fetch_start, fetch_end)
    if error:
        return None, error

    min_required = max_lookback + 10
    if len(df) < min_required:
        return None, f"Insufficient history for {symbol} ({len(df)} bars, need {min_required})"

    df = calculate_indicators(df, params)
    df = classify_stages(df, params)

    # Relative Strength vs benchmark
    if benchmark_rs is not None:
        df = df.merge(benchmark_rs, on='date', how='left')
        df['rs_state'] = np.where(
            df['benchmark_return'].isna(), 'N/A',
            np.where(df['stock_return'] > df['benchmark_return'], 'Outperform', 'Underperform')
        )
    else:
        df['rs_state'] = 'N/A'

    # Sector strength (Item 2)
    sector_name = sector_name_map.get(base_symbol(symbol), 'Unknown')
    sector_index = sector_ticker_map.get(sector_name, '')
    sector_df = sector_state_map.get(sector_name)
    if sector_df is not None:
        df = df.merge(sector_df, on='date', how='left')
        df['sector_state'] = df['sector_state'].fillna('N/A')
    else:
        df['sector_state'] = 'N/A'

    df['sub_stage'] = df.apply(lambda row: classify_substage(row, params), axis=1)
    df['recommended_action'] = df['sub_stage'].map(SUBSTAGE_ACTION_MAP).fillna('')

    # Alert windows computed on full history BEFORE trimming (Item 6)
    df = add_alert_windows(df, alert_window)

    if mode == 'Historical':
        df = df[(df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= pd.Timestamp(end_date))].copy()
        if df.empty:
            return None, f"No data in selected range for {symbol}"
    else:
        df = df.iloc[[-1]].copy()

    results = []
    for _, row in df.iterrows():
        results.append({
            'Symbol': symbol,
            'Date': row['date'].strftime('%Y-%m-%d'),
            'Close': round(row['close'], 2),
            'Current Stage': int(row['stage']),
            'Sub Stage': row['sub_stage'],
            'Recommended Action': row['recommended_action'],
            'Sector': sector_name,
            'Sector Index': sector_index,
            'Sector State': row['sector_state'],
            'Days in Stage': int(row['days_in_stage']),
            'Weeks in Stage': round(row['weeks_in_stage'], 1),
            '50 SMA': round(row['sma50'], 2) if pd.notna(row['sma50']) else None,
            '200 SMA': round(row['sma200'], 2) if pd.notna(row['sma200']) else None,
            '200 SMA Slope %': round(row['sma200_slope_pct'], 2) if pd.notna(row['sma200_slope_pct']) else None,
            'Rising MA': bool(row['rising_ma']),
            'Falling MA': bool(row['falling_ma']),
            'Extension %': round(row['extension_pct'], 2) if pd.notna(row['extension_pct']) else None,
            'Volume State': row['volume_state'],
            'Relative Strength State': row['rs_state'],
            'Buy Signal': bool(row['buy_signal']),
            'Sell Signal': bool(row['sell_signal']),
            'Stage 4 Confirmed': bool(row['stage4_confirmed']),
            'Buy Window': 'YES' if bool(row['buy_window']) else 'NO',
            'Sell Window': 'YES' if bool(row['sell_window']) else 'NO',
            'Last Buy': fmt_date(row['last_buy_date']),
            'Last Sell': fmt_date(row['last_sell_date']),
        })

    return pd.DataFrame(results), None


# ---------------------------------------------------------------------------
# Sidebar UI
# ---------------------------------------------------------------------------
st.sidebar.markdown("## ⚙️ Scanner Settings")

scanner_mode = st.sidebar.radio(
    "Scanner Mode",
    options=["Current Scanner", "Historical Scanner"],
    index=0,
    help="Current: latest trading day only. Historical: full period evaluation."
)
mode = "Current" if scanner_mode == "Current Scanner" else "Historical"

if mode == "Historical":
    st.sidebar.markdown("### 📅 Date Range")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=90))
    with c2:
        end_date = st.date_input("End Date", value=datetime.now())
    if start_date > end_date:
        st.sidebar.error("Start date must be before end date")
        st.stop()
else:
    start_date = None
    end_date = None

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Symbol Selection")
symbol_option = st.sidebar.radio("Input Method", options=["Upload Excel", "Manual Symbols"], index=1)

symbols = []
if symbol_option == "Upload Excel":
    uploaded_file = st.sidebar.file_uploader("Upload Excel (Column: Symbol)", type=["xlsx", "xls"])
    if uploaded_file:
        try:
            xl = pd.read_excel(uploaded_file)
            if 'Symbol' not in xl.columns:
                st.sidebar.error("Excel must contain a 'Symbol' column")
            else:
                symbols = [normalize_symbol(s) for s in xl['Symbol'].dropna().astype(str).tolist()]
                st.sidebar.success(f"Loaded {len(symbols)} symbols")
        except Exception as e:
            st.sidebar.error(f"Error reading Excel: {e}")
else:
    manual_input = st.sidebar.text_area("Enter symbols (one per line)",
                                        value="RELIANCE\nTCS\nINFY\nSBIN", height=150)
    if manual_input:
        symbols = [normalize_symbol(s) for s in manual_input.split('\n') if s.strip()]

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Parameters")

with st.sidebar.expander("Stage Classification", expanded=True):
    sma200_period = st.number_input("200 SMA Period", value=200, min_value=1, disabled=True)
    sma50_period = st.number_input("50 SMA Period", value=50, min_value=1, disabled=True)
    slope_lookback = st.number_input("Slope Lookback", value=21, min_value=1)
    rising_threshold = st.number_input("Rising Threshold %", value=0.5, step=0.1, format="%.2f")
    falling_threshold = st.number_input("Falling Threshold %", value=0.5, step=0.1, format="%.2f")

with st.sidebar.expander("Sub Stage Inputs", expanded=False):
    flat_threshold = st.number_input("Flat Threshold %", value=0.15, step=0.05, format="%.2f")
    late_stage_2_weeks = st.number_input("Late Stage 2 Weeks", value=35, min_value=1)
    extended_pct = st.number_input("Extended Above 50 SMA %", value=25.0, step=1.0, format="%.1f")
    late_stage_4_weeks = st.number_input("Late Stage 4 Weeks", value=35, min_value=1)
    steep_decline_pct = st.number_input("Steep Decline %", value=1.5, step=0.1, format="%.1f")
    support_lookback = st.number_input("Support Break Lookback", value=10, min_value=1)

with st.sidebar.expander("Volume & RS", expanded=False):
    volume_avg_length = st.number_input("Average Length", value=50, min_value=1)
    volume_expansion_mult = st.number_input("Expansion Multiplier", value=1.2, step=0.1, format="%.1f")
    benchmark_symbol = st.text_input("Benchmark", value="^NSEI")
    rs_lookback = st.number_input("RS Lookback", value=21, min_value=1)

# Item 6: Alert Window
with st.sidebar.expander("Alert Window", expanded=False):
    alert_window = st.slider("Alert Window (days)", min_value=1, max_value=20, value=3,
                             help="A Buy/Sell signal keeps showing as 'in window' for this many trading days after it fires.")

# Item 1: editable sector index tickers
with st.sidebar.expander("Sector Indices (editable)", expanded=False):
    st.caption("Yahoo's NSE sector index coverage is inconsistent. If a sector shows N/A, correct its ticker here.")
    sector_ticker_map = {}
    for name, default_ticker in DEFAULT_SECTOR_INDICES.items():
        sector_ticker_map[name] = st.text_input(name, value=default_ticker, key=f"sect_{name}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Scanner Filters")

with st.sidebar.expander("Stage / Signal Filters", expanded=True):
    # Item 18: Current scanner quick checkboxes
    if mode == "Current":
        quick_view = st.radio("Quick View",
                              ["All Stages", "Buy Only", "Sell Only", "Stage 4 Only"],
                              index=0)
    else:
        # Item 4: historical row export mode
        quick_view = st.radio("Historical Rows",
                              ["All Days", "Signal Days Only", "Buy Signals Only",
                               "Sell Signals Only", "Stage 4 Only"],
                              index=0)

    stage_filter = st.multiselect("Stage", [1, 2, 3, 4], default=[1, 2, 3, 4])
    substage_filter = st.multiselect("Sub Stage",
                                     ["1A", "1B", "2A", "2B", "3A", "3B", "4A", "4B"],
                                     default=["1A", "1B", "2A", "2B", "3A", "3B", "4A", "4B"])

with st.sidebar.expander("State Filters", expanded=False):
    volume_filter = st.multiselect("Volume State", ["Expansion", "Neutral", "Contraction"],
                                   default=["Expansion", "Neutral", "Contraction"])
    rs_filter = st.multiselect("RS State", ["Outperform", "Underperform", "N/A"],
                               default=["Outperform", "Underperform", "N/A"])
    sector_filter = st.multiselect("Sector State", ["Strong", "Weak", "N/A"],
                                   default=["Strong", "Weak", "N/A"])

with st.sidebar.expander("Duration & Extension Filters", expanded=False):
    days_min, days_max = st.slider("Days in Stage", 0, 1000, (0, 1000))
    weeks_min, weeks_max = st.slider("Weeks in Stage", 0.0, 200.0, (0.0, 200.0))
    ext_min, ext_max = st.slider("Extension %", -100.0, 200.0, (-100.0, 200.0))

params = {
    'sma200_period': int(sma200_period),
    'sma50_period': int(sma50_period),
    'slope_lookback': int(slope_lookback),
    'rising_threshold': float(rising_threshold),
    'falling_threshold': float(falling_threshold),
    'flat_threshold': float(flat_threshold),
    'late_stage_2_weeks': float(late_stage_2_weeks),
    'extended_pct': float(extended_pct),
    'late_stage_4_weeks': float(late_stage_4_weeks),
    'steep_decline_pct': float(steep_decline_pct),
    'support_lookback': int(support_lookback),
    'volume_avg_length': int(volume_avg_length),
    'volume_expansion_mult': float(volume_expansion_mult),
    'rs_lookback': int(rs_lookback)
}

# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">📊 Weinstein 4-Stage Analysis Scanner</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">NSE edition — stage engine, sector strength, relative strength, alert windows</div>', unsafe_allow_html=True)

if not symbols:
    st.info("👈 Please configure symbols in the sidebar to begin scanning.")
    st.stop()

run_scan = st.button("🚀 Run Scanner", type="primary", use_container_width=True)

if run_scan:
    scan_started = datetime.now()

    # ----- Item 5 / 22: benchmark + sector use the SAME extended range as stocks -----
    max_lookback = max(200, params['slope_lookback'], params['rs_lookback'], params['volume_avg_length'])
    if mode == 'Current':
        idx_start = (datetime.now() - timedelta(days=max_lookback + 400)).strftime('%Y-%m-%d')
        idx_end = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        idx_start = (start_date - timedelta(days=max_lookback + 400)).strftime('%Y-%m-%d')
        idx_end = (end_date + timedelta(days=1)).strftime('%Y-%m-%d')

    # Benchmark (fetched once per scan, cached separately - Item 21)
    benchmark_rs = None
    with st.spinner("Fetching benchmark..."):
        if benchmark_symbol:
            b_df, b_err = fetch_stock_data(benchmark_symbol.strip(), idx_start, idx_end)
            if b_err is None and b_df is not None:
                benchmark_rs = calculate_benchmark_rs(b_df, params)
            else:
                st.warning(f"Benchmark {benchmark_symbol} unavailable — RS State will show N/A. ({b_err})")

    # Sector indices: fetch ONLY the sectors actually needed by this symbol list
    sector_name_map = STOCK_SECTOR_MAP
    needed_sectors = sorted({sector_name_map.get(base_symbol(s), 'Unknown') for s in symbols})
    needed_sectors = [s for s in needed_sectors if s != 'Unknown' and sector_ticker_map.get(s)]

    sector_state_map = {}
    sector_errors = []
    if needed_sectors:
        with st.spinner(f"Fetching {len(needed_sectors)} sector indices..."):
            for sect in needed_sectors:
                s_df, s_err = fetch_stock_data(sector_ticker_map[sect], idx_start, idx_end)
                if s_err is None and s_df is not None:
                    sector_state_map[sect] = calculate_sector_state(s_df, params)
                else:
                    sector_errors.append(f"{sect} ({sector_ticker_map[sect]}): {s_err}")

    if sector_errors:
        with st.expander(f"⚠️ Sector indices unavailable ({len(sector_errors)}) — those stocks show Sector State = N/A"):
            for e in sector_errors:
                st.text(e)

    # ----- Item 21: worker count scaled to CPU, capped -----
    cpu = os.cpu_count() or 4
    max_workers = max(4, min(16, cpu * 2))

    progress_text = st.empty()
    progress_bar = st.progress(0)
    all_results, errors = [], []
    total = len(symbols)
    progress_text.text(f"Scanning {total} symbols with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_symbol, sym, params, mode, start_date, end_date,
                            benchmark_rs, sector_state_map, sector_name_map,
                            sector_ticker_map, alert_window): sym
            for sym in symbols
        }
        completed = 0
        for future in as_completed(futures):
            sym = futures[future]
            completed += 1
            progress_bar.progress(min(int((completed / total) * 100), 100))
            progress_text.text(f"Scanning... {completed}/{total} — {sym}")
            try:
                result, error = future.result()
                if error:
                    errors.append(f"{sym}: {error}")
                elif result is not None:
                    all_results.append(result)
            except Exception as e:
                errors.append(f"{sym}: {str(e)}")

    progress_bar.empty()
    progress_text.empty()

    if not all_results:
        st.error("No valid results. Check errors below:")
        for err in errors[:20]:
            st.error(err)
        st.stop()

    final_df = pd.concat(all_results, ignore_index=True)
    unfiltered_count = len(final_df)

    # ----- Apply filters (Items 4, 11-18) -----
    fdf = final_df.copy()

    if quick_view == "Buy Only" or quick_view == "Buy Signals Only":
        fdf = fdf[fdf['Buy Signal']]
    elif quick_view == "Sell Only" or quick_view == "Sell Signals Only":
        fdf = fdf[fdf['Sell Signal']]
    elif quick_view == "Stage 4 Only":
        fdf = fdf[fdf['Current Stage'] == 4]
    elif quick_view == "Signal Days Only":
        fdf = fdf[fdf['Buy Signal'] | fdf['Sell Signal'] | fdf['Stage 4 Confirmed']]

    fdf = fdf[fdf['Current Stage'].isin(stage_filter)]
    fdf = fdf[fdf['Sub Stage'].isin(substage_filter)]
    fdf = fdf[fdf['Volume State'].isin(volume_filter)]
    fdf = fdf[fdf['Relative Strength State'].isin(rs_filter)]
    fdf = fdf[fdf['Sector State'].isin(sector_filter)]
    fdf = fdf[(fdf['Days in Stage'] >= days_min) & (fdf['Days in Stage'] <= days_max)]
    fdf = fdf[(fdf['Weeks in Stage'] >= weeks_min) & (fdf['Weeks in Stage'] <= weeks_max)]
    fdf = fdf[fdf['Extension %'].isna() | ((fdf['Extension %'] >= ext_min) & (fdf['Extension %'] <= ext_max))]

    if mode == "Historical":
        fdf = fdf.sort_values(['Date', 'Symbol']).reset_index(drop=True)
    else:
        fdf = fdf.sort_values(['Current Stage', 'Symbol']).reset_index(drop=True)

    elapsed = (datetime.now() - scan_started).total_seconds()
    st.caption(f"Scan completed in {elapsed:.1f}s · {unfiltered_count} rows produced · {len(fdf)} rows after filters")

    # ----- Item 19: Statistics -----
    st.markdown("### 📈 Statistics")
    r1 = st.columns(6)
    r1[0].metric("Total Stocks", fdf['Symbol'].nunique())
    r1[1].metric("Stage 1", int((fdf['Current Stage'] == 1).sum()))
    r1[2].metric("Stage 2", int((fdf['Current Stage'] == 2).sum()))
    r1[3].metric("Stage 3", int((fdf['Current Stage'] == 3).sum()))
    r1[4].metric("Stage 4", int((fdf['Current Stage'] == 4).sum()))
    r1[5].metric("Buy Signals", int(fdf['Buy Signal'].sum()))

    r2 = st.columns(6)
    r2[0].metric("Sell Signals", int(fdf['Sell Signal'].sum()))
    r2[1].metric("Stage 4 Confirmed", int(fdf['Stage 4 Confirmed'].sum()))
    r2[2].metric("Outperform", int((fdf['Relative Strength State'] == 'Outperform').sum()))
    r2[3].metric("Underperform", int((fdf['Relative Strength State'] == 'Underperform').sum()))
    r2[4].metric("Strong Sector", int((fdf['Sector State'] == 'Strong').sum()))
    r2[5].metric("Weak Sector", int((fdf['Sector State'] == 'Weak').sum()))

    # ----- Results table -----
    st.markdown("### 📋 Scanner Results")
    if fdf.empty:
        st.warning("No rows match the current filters. Loosen the filters in the sidebar.")
    else:
        def color_stage(val):
            colors = {1: '#ffcccc', 2: '#ccffcc', 3: '#ffffcc', 4: '#e6ccff'}
            return f'background-color: {colors.get(val, "")}'

        try:
            styled = fdf.style.map(color_stage, subset=['Current Stage'])
        except AttributeError:
            styled = fdf.style.applymap(color_stage, subset=['Current Stage'])

        st.dataframe(styled, use_container_width=True,
                     height=min(650, max(300, len(fdf) * 35)), hide_index=True)

    # ----- Item 20: multi-sheet Excel export -----
    st.markdown("---")
    st.markdown("### 💾 Download Results")

    buy_sheet = fdf[fdf['Buy Signal']].copy()
    sell_sheet = fdf[fdf['Sell Signal']].copy()
    stage4_sheet = fdf[fdf['Current Stage'] == 4].copy()

    stage_summary = (
        fdf.groupby(['Current Stage', 'Sub Stage'])
        .agg(Count=('Symbol', 'count'),
             Symbols=('Symbol', lambda s: ', '.join(sorted(set(s))[:40])))
        .reset_index()
        .sort_values(['Current Stage', 'Sub Stage'])
    ) if not fdf.empty else pd.DataFrame(columns=['Current Stage', 'Sub Stage', 'Count', 'Symbols'])

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        (fdf if not fdf.empty else final_df).to_excel(writer, sheet_name='All Results', index=False)
        (buy_sheet if not buy_sheet.empty else pd.DataFrame(columns=fdf.columns)).to_excel(
            writer, sheet_name='Buy Signals', index=False)
        (sell_sheet if not sell_sheet.empty else pd.DataFrame(columns=fdf.columns)).to_excel(
            writer, sheet_name='Sell Signals', index=False)
        (stage4_sheet if not stage4_sheet.empty else pd.DataFrame(columns=fdf.columns)).to_excel(
            writer, sheet_name='Stage 4', index=False)
        stage_summary.to_excel(writer, sheet_name='Stage Summary', index=False)
    excel_buffer.seek(0)

    file_label = "current" if mode == "Current" else "historical"
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            label="📥 Download Excel (5 sheets)",
            data=excel_buffer,
            file_name=f"weinstein_scanner_{file_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    with d2:
        st.download_button(
            label="📥 Download CSV",
            data=fdf.to_csv(index=False).encode('utf-8'),
            file_name=f"weinstein_scanner_{file_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    if errors:
        with st.expander(f"⚠️ Errors ({len(errors)} symbols failed)"):
            for err in errors:
                st.text(err)

st.markdown("---")
st.caption("Weinstein stage engine unchanged | Streamlit + yfinance | NSE benchmark & sector adaptation")
