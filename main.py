import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import numpy as np
import time

# ==========================================
# 1. PAGE CONFIGURATION & CSS (TradingView Look)
# ==========================================
st.set_page_config(
    page_title="Pro Crypto Scalper",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for TradingView Vibes
st.markdown("""
    <style>
        /* Background & General Text */
        .stApp { background-color: #131722; color: #d1d4dc; }
        
        /* Remove Top Padding */
        .block-container { padding-top: 0rem; padding-bottom: 0rem; max-width: 100% !important; }
        
        /* Hide Streamlit Default Elements */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Top Information Bar */
        .top-bar {
            background-color: #1e222d;
            padding: 10px 20px;
            border-bottom: 1px solid #363c4e;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Fira Sans", "Droid Sans", "Helvetica Neue", sans-serif;
            margin-bottom: 10px;
        }
        
        .coin-info { font-size: 20px; font-weight: bold; color: #d1d4dc; }
        .price-info { font-size: 20px; font-weight: bold; }
        
        /* Sidebar Styling */
        section[data-testid="stSidebar"] { background-color: #1e222d; border-right: 1px solid #363c4e; }
        div[data-testid="stSidebarUserContent"] { color: #d1d4dc; }
        
        /* Signal Badges */
        .badge-buy { background-color: #00E676; color: black; padding: 5px 10px; border-radius: 4px; font-weight: bold; }
        .badge-sell { background-color: #FF1744; color: white; padding: 5px 10px; border-radius: 4px; font-weight: bold; }
        .badge-wait { background-color: #78808A; color: white; padding: 5px 10px; border-radius: 4px; font-weight: bold; }

        /* Animation */
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
        .pulse { animation: pulse 2s infinite; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SIDEBAR CONTROLS
# ==========================================
st.sidebar.title("⚙️ Market Settings")

# Coin Selection
symbol_input = st.sidebar.text_input("Symbol (e.g. BTC/USDT)", "BTC/USDT").upper()

# Timeframe Selection
timeframe_options = ["1m", "5m", "15m", "1h", "4h", "1d"]
timeframe = st.sidebar.selectbox("Timeframe", timeframe_options, index=1) # Default 5m

# Auto Refresh Control
refresh_rate = st.sidebar.slider("Refresh Rate (Seconds)", 2, 60, 5)

st.sidebar.markdown("---")
st.sidebar.info("Designed like TradingView\nUses MEXC Data")

# ==========================================
# 3. DATA & LOGIC ENGINE
# ==========================================
@st.cache_data(ttl=5) # Cache data for 5 seconds to prevent spamming API on interactions
def get_market_data(sym, tf):
    try:
        # Connect to MEXC (No API key needed for public data)
        exchange = ccxt.mexc({
            'enableRateLimit': True, 
            'options': {'defaultType': 'swap'} # Futures data
        })
        
        # Fetch OHLCV
        limit = 300
        ohlcv = exchange.fetch_ohlcv(sym, tf, limit=limit)
        
        if not ohlcv:
            return pd.DataFrame()
            
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # --- STRATEGY CALCULATION (Your Pine Script Logic) ---
        # 1. EMA & Baseline
        df['ema200'] = ta.ema(df['close'], length=200)
        df['baseline'] = ta.ema(df['close'], length=80)
        
        # 2. ATR & Deviation
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=15)
        min_tick = 0.0000001
        multiplier = 1.8 # Deviation Multiplier
        
        # Avoid division by zero
        divisor = (df['atr'] * multiplier).clip(lower=min_tick)
        df['rawDeviation'] = (df['close'] - df['baseline']) / divisor
        
        # 3. Smoothing Signal
        df['signal_line'] = ta.ema(df['rawDeviation'], length=8)
        
        # 4. Hyperbolic Tangent (Tanh) Clamping
        # Custom Tanh implementation to match Pine Script behavior safely
        def tanh_clamped(x):
            clipped = np.clip(x, -20.0, 20.0)
            return np.tanh(clipped) # Using numpy's native tanh
            
        df['clamped'] = tanh_clamped(df['signal_line'])
        
        # 5. Determine State (Color Logic)
        # 1 = Bullish (Green), -1 = Bearish (Red), 0 = Neutral (Gray)
        threshold = 0.08
        conditions = [
            (df['clamped'] > threshold),
            (df['clamped'] < -threshold)
        ]
        choices = [1, -1]
        df['state'] = np.select(conditions, choices, default=0)
        
        # 6. Signal Generation
        # Logic: State changes + Confirmation with EMA200
        df['Buy_Signal'] = False
        df['Sell_Signal'] = False
        
        last_sig = 0 # 0=None, 1=Buy, -1=Sell
        
        # Loop for signals (Loops are slower in Python but necessary for state-dependent logic)
        # Optimizing by converting to numpy arrays
        states = df['state'].values
        closes = df['close'].values
        ema200s = df['ema200'].values
        buys = np.zeros(len(df), dtype=bool)
        sells = np.zeros(len(df), dtype=bool)
        
        for i in range(1, len(df)):
            current_state = states[i]
            prev_state = states[i-1]
            
            # Raw crossover detection
            cross_up = (current_state == 1) and (prev_state != 1)
            cross_down = (current_state == -1) and (prev_state != -1)
            
            # Filter Logic
            if cross_up and (last_sig != 1):
                last_sig = 1
                if closes[i] > ema200s[i]: # Trend Filter
                    buys[i] = True
            
            elif cross_down and (last_sig != -1):
                last_sig = -1
                if closes[i] < ema200s[i]: # Trend Filter
                    sells[i] = True
                    
        df['Buy_Signal'] = buys
        df['Sell_Signal'] = sells
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# ==========================================
# 4. CHART RENDERING
# ==========================================
df = get_market_data(symbol_input, timeframe)

if not df.empty:
    # --- Top Bar Data ---
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    price = latest['close']
    change = price - prev['close']
    change_pct = (change / prev['close']) * 100
    color_hex = "#00E676" if change >= 0 else "#FF1744"
    
    # Check for active signals (Current or Previous Candle)
    signal_status = "<span class='badge-wait'>WAITING</span>"
    if latest['Buy_Signal'] or prev['Buy_Signal']:
        signal_status = "<span class='badge-buy pulse'>BUY SIGNAL</span>"
    elif latest['Sell_Signal'] or prev['Sell_Signal']:
        signal_status = "<span class='badge-sell pulse'>SELL SIGNAL</span>"

    # Display Top Bar
    st.markdown(f"""
        <div class="top-bar">
            <div class="coin-info">{symbol_input} <span style="font-size:14px; color:#78808A">({timeframe})</span></div>
            <div class="price-info" style="color: {color_hex}">
                {price} 
                <span style="font-size:16px">({change:+.2f} / {change_pct:+.2f}%)</span>
            </div>
            <div>{signal_status}</div>
        </div>
    """, unsafe_allow_html=True)

    # --- Plotly Chart ---
    fig = go.Figure()

    # Colors
    c_bull = '#00E676' # Bright Green
    c_bear = '#FF1744' # Bright Red
    c_neut = '#78808A' # Gray

    # Add Candles by State (To color whole candles correctly)
    # State 1: Bullish Candles
    bull_df = df[df['state'] == 1]
    if not bull_df.empty:
        fig.add_trace(go.Candlestick(
            x=bull_df['timestamp'], open=bull_df['open'], high=bull_df['high'], low=bull_df['low'], close=bull_df['close'],
            increasing_line_color=c_bull, decreasing_line_color=c_bull, name="Bullish Zone"
        ))
    
    # State -1: Bearish Candles
    bear_df = df[df['state'] == -1]
    if not bear_df.empty:
        fig.add_trace(go.Candlestick(
            x=bear_df['timestamp'], open=bear_df['open'], high=bear_df['high'], low=bear_df['low'], close=bear_df['close'],
            increasing_line_color=c_bear, decreasing_line_color=c_bear, name="Bearish Zone"
        ))

    # State 0: Neutral Candles
    neut_df = df[df['state'] == 0]
    if not neut_df.empty:
        fig.add_trace(go.Candlestick(
            x=neut_df['timestamp'], open=neut_df['open'], high=neut_df['high'], low=neut_df['low'], close=neut_df['close'],
            increasing_line_color=c_neut, decreasing_line_color=c_neut, name="Neutral Zone"
        ))

    # Add EMA 200
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['ema200'], 
        line=dict(color='#FF9800', width=2), name="EMA 200"
    ))
    
    # Add Baseline (Points style as per original logic visual)
    base_colors = [c_bull if s==1 else (c_bear if s==-1 else c_neut) for s in df['state']]
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['baseline'],
        mode='markers', marker=dict(color=base_colors, size=2),
        name="Baseline", hoverinfo='skip'
    ))

    # Add Buy/Sell Markers
    buys = df[df['Buy_Signal']]
    if not buys.empty:
        fig.add_trace(go.Scatter(
            x=buys['timestamp'], y=buys['low'] * 0.998,
            mode='markers+text', marker=dict(symbol='triangle-up', size=14, color='#00E676'),
            text="BUY", textposition="bottom center", textfont=dict(color='#00E676', size=12),
            name="Buy Signal"
        ))

    sells = df[df['Sell_Signal']]
    if not sells.empty:
        fig.add_trace(go.Scatter(
            x=sells['timestamp'], y=sells['high'] * 1.002,
            mode='markers+text', marker=dict(symbol='triangle-down', size=14, color='#FF1744'),
            text="SELL", textposition="top center", textfont=dict(color='#FF1744', size=12),
            name="Sell Signal"
        ))

    # Chart Layout (Crucial for TradingView Feel)
    fig.update_layout(
        height=750, # Large Height
        template='plotly_dark',
        paper_bgcolor='#131722',
        plot_bgcolor='#131722',
        margin=dict(l=0, r=60, t=10, b=0), # Maximize space
        showlegend=False,
        # AXIS SETTINGS
        xaxis=dict(
            type='date',
            rangeslider=dict(visible=False), # Hide bottom slider
            showgrid=True, gridcolor='#2a2e39',
            tickformat='%H:%M',
        ),
        yaxis=dict(
            side='right', # Price scale on right
            showgrid=True, gridcolor='#2a2e39',
            tickprefix="$",
            fixedrange=False 
        ),
        hovermode='x unified', # Crosshair
        dragmode='pan', # Default to Panning
        
        # ZOOM LOCK: This prevents reset on refresh
        uirevision=f"{symbol_input}_{timeframe}" 
    )

    # Render Chart
    st.plotly_chart(
        fig, 
        use_container_width=True, 
        config={
            'scrollZoom': True, 
            'displayModeBar': False,
            'doubleClick': 'reset',
        }
    )

else:
    st.warning(f"Could not load data for {symbol_input}. Please check the symbol name.")

# ==========================================
# 5. AUTO REFRESH LOOP
# ==========================================
# This keeps the script running and refreshing without manual clicks
time.sleep(refresh_rate)
st.rerun()