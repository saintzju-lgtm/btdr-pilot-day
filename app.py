import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import requests
import altair as alt
from datetime import datetime, timedelta
import pytz
from scipy.stats import norm

# --- 1. 页面配置 & 样式 ---
st.set_page_config(page_title="BTDR Pilot v12.8 Trader", layout="centered")

CUSTOM_CSS = """
<style>
    html { overflow-y: scroll; }
    .stApp > header { display: none; }
    .stApp { margin-top: -30px; background-color: #ffffff; }
    div[data-testid="stStatusWidget"] { visibility: hidden; }
    
    h1, h2, h3, div, p, span { 
        color: #212529 !important; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important; 
    }
    
    div[data-testid="stAltairChart"] {
        overflow: hidden !important; border: 1px solid #f8f9fa; border-radius: 8px;
    }
    
    /* Trade Plan Card */
    .plan-card {
        background: #fff; border: 1px solid #e9ecef; border-radius: 12px;
        padding: 15px; margin-bottom: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    .plan-header { font-size: 1rem; font-weight: 800; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; }
    .plan-row { display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 0.9rem; align-items: center; }
    .plan-label { color: #868e96; font-weight: 500; }
    .plan-val { font-weight: 700; font-family: monospace; font-size: 1.1rem; }
    
    /* R/R Visual Bar */
    .rr-bar-container { display: flex; align-items: center; height: 12px; width: 100%; margin-top: 8px; border-radius: 6px; overflow: hidden; background: #f1f3f5; }
    .rr-risk { background-color: #fa5252; height: 100%; transition: width 0.5s; }
    .rr-reward { background-color: #40c057; height: 100%; transition: width 0.5s; }
    .rr-mid { width: 2px; background: #212529; height: 100%; z-index: 10; }
    
    /* Metric Card */
    .metric-card {
        background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 12px;
        height: 85px; padding: 0 16px; display: flex; flex-direction: column; justify-content: center;
    }
    .metric-label { font-size: 0.75rem; color: #888; margin-bottom: 2px; }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #212529; line-height: 1.2; }
    .metric-delta { font-size: 0.85rem; font-weight: 600; margin-top: 2px; }
    
    /* Chart Legend */
    .chart-legend {
        display: flex; flex-wrap: wrap; gap: 10px; font-size: 0.75rem; color: #555;
        background: #f8f9fa; padding: 6px 10px; border-radius: 6px; margin-bottom: 5px;
        border: 1px solid #eee; align-items: center;
    }
    .legend-item { display: flex; align-items: center; gap: 4px; }
    .legend-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
    
    .color-up { color: #e03131; } .color-down { color: #0ca678; }
    .tag-buy { background: #e6fcf5; color: #099268; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: bold; border: 1px solid #099268; }
    .tag-sell { background: #fff5f5; color: #e03131; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: bold; border: 1px solid #e03131; }
    .tag-hold { background: #f8f9fa; color: #495057; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: bold; border: 1px solid #adb5bd; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- 2. 基础配置 ---
MINER_SHARES = {"MARA": 300, "RIOT": 330, "CLSK": 220, "CORZ": 190, "IREN": 180, "WULF": 410, "CIFR": 300, "HUT": 100}
MINER_POOL = list(MINER_SHARES.keys())

# --- 3. 辅助函数 ---
def card_html(label, value_str, delta_str=None, delta_val=0):
    delta_html = ""
    if delta_str:
        color_class = "color-up" if delta_val >= 0 else "color-down"
        delta_html = f"<div class='metric-delta {color_class}'>{delta_str}</div>"
    return f"""<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value_str}</div>{delta_html}</div>"""

def trade_plan_html(type, entry, stop, target, probability, rr_ratio):
    # Calculate widths for the R/R bar
    total_range = (entry - stop) + (target - entry)
    if total_range <= 0: total_range = 1
    risk_width = ((entry - stop) / total_range) * 100
    reward_width = ((target - entry) / total_range) * 100
    
    # Cap visualizations to avoid broken UI
    if risk_width < 5: risk_width = 5; reward_width = 95
    if reward_width < 5: reward_width = 5; risk_width = 95
    
    header_color = "#0ca678" if type == "BUY" else "#e03131"
    tag_class = "tag-buy" if type == "BUY" else "tag-sell"
    
    return f"""
    <div class="plan-card" style="border-left: 5px solid {header_color};">
        <div class="plan-header">
            <span style="color:{header_color}">{type} SETUP</span>
            <span class="{tag_class}">Win Rate: {probability}%</span>
        </div>
        <div class="plan-row">
            <span class="plan-label">🎯 目标 (Target)</span>
            <span class="plan-val" style="color:#1c7ed6">${target:.2f}</span>
        </div>
        <div class="plan-row">
            <span class="plan-label">⚡ 入场 (Entry)</span>
            <span class="plan-val" style="color:#212529">${entry:.2f}</span>
        </div>
        <div class="plan-row">
            <span class="plan-label">🛑 止损 (Stop)</span>
            <span class="plan-val" style="color:#fa5252">${stop:.2f}</span>
        </div>
        <div style="margin-top:10px; font-size:0.7rem; color:#888; display:flex; justify-content:space-between;">
            <span>RISK (${(entry-stop):.2f})</span>
            <span>R/R Ratio 1:{rr_ratio:.1f}</span>
            <span>REWARD (${(target-entry):.2f})</span>
        </div>
        <div class="rr-bar-container">
            <div class="rr-risk" style="width:{risk_width}%"></div>
            <div class="rr-mid"></div>
            <div class="rr-reward" style="width:{reward_width}%"></div>
        </div>
    </div>
    """

# --- 4. 核心计算 ---
def run_kalman_filter(y, x):
    try:
        n = len(y); beta = np.zeros(n); P = np.zeros(n); beta[0]=1.0; P[0]=1.0; R=0.002; Q=1e-4
        for t in range(1, n):
            beta_pred = beta[t-1]; P_pred = P[t-1] + Q
            if x[t] == 0: x[t] = 1e-6
            residual = y[t] - beta_pred * x[t]; S = P_pred * x[t]**2 + R; K = P_pred * x[t] / S
            beta[t] = beta_pred + K * residual; P[t] = (1 - K * x[t]) * P_pred
        return beta[-1]
    except: return 1.0

def calculate_hurst(series):
    try:
        lags = range(2, 20); tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1); return poly[0] * 2.0
    except: return 0.5

@st.cache_data(ttl=300)
def run_grandmaster_analytics(live_price=None):
    default_factors = {"vwap": 0, "adx": 20, "regime": "Neutral", "rsi": 50, "vol_base": 0.05, "atr": 0.5, "macd": 0, "boll_u": 0, "boll_l": 0, "boll_m": 0}
    try:
        # 1. Fetch History (Clean)
        btdr = yf.Ticker("BTDR").history(period="6mo", interval="1d")
        btdr.index = btdr.index.tz_localize(None) # Remove TZ
        
        if btdr.empty: return default_factors, "No Data"

        # 2. Inject Live Data (If valid)
        if live_price and live_price > 0:
            last_date = btdr.index[-1].date()
            today = datetime.now().date()
            
            # Construct new row
            last_row = btdr.iloc[-1].copy()
            last_row['Close'] = live_price
            last_row['High'] = max(last_row['High'], live_price)
            last_row['Low'] = min(last_row['Low'], live_price)
            
            if last_date == today:
                btdr.iloc[-1] = last_row
            else:
                new_idx = btdr.index[-1] + timedelta(days=1)
                new_df = pd.DataFrame([last_row], index=[new_idx])
                btdr = pd.concat([btdr, new_df])
        
        # 3. Clean Data (Fill NaNs)
        btdr = btdr.ffill().bfill()
        close = btdr['Close']
        
        # 4. Indicators
        # VWAP
        pv = (close * btdr['Volume']); vol_sum = btdr['Volume'].tail(30).sum()
        vwap = pv.tail(30).sum() / vol_sum if vol_sum > 0 else close.mean()
        
        # BOLL
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        boll_u = sma20 + 2*std20
        boll_l = sma20 - 2*std20
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # MACD
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9, adjust=False).mean()
        
        # ATR
        tr = np.maximum(btdr['High'] - btdr['Low'], np.abs(btdr['High'] - close.shift(1)))
        atr = tr.rolling(14).mean()
        
        # Hurst
        hurst = calculate_hurst(close.values[-50:])
        
        # 5. Pack Results (Last Values)
        # Use simple fallback if calc fails (NaN)
        def get_val(series, fallback):
            val = series.iloc[-1]
            return val if not np.isnan(val) else fallback

        factors = {
            "vwap": vwap,
            "rsi": get_val(rsi, 50),
            "macd": get_val(macd, 0),
            "macd_sig": get_val(signal, 0),
            "boll_u": get_val(boll_u, close.iloc[-1]*1.1),
            "boll_l": get_val(boll_l, close.iloc[-1]*0.9),
            "boll_m": get_val(sma20, close.iloc[-1]),
            "atr": get_val(atr, close.iloc[-1]*0.05),
            "hurst": hurst,
            "vol_base": close.pct_change().std(),
            "regime": "Trend" if get_val(rsi, 50) > 60 or get_val(rsi, 50) < 40 else "Range"
        }
        return factors, "v12.8 Trader"
        
    except Exception as e:
        print(e)
        return default_factors, "Offline"

# --- 5. 绘图函数 ---
def draw_kline_chart(df, live_price, factors):
    if df.empty: return alt.Chart(pd.DataFrame()).mark_text().encode(text=alt.value("No Data"))
    
    # Inject Live Data for Chart
    df = df.copy()
    if live_price > 0:
        last_idx = df.index[-1]
        today = datetime.now().date()
        last_row = df.iloc[-1].to_dict()
        last_row['Close'] = live_price
        last_row['High'] = max(last_row['High'], live_price)
        last_row['Low'] = min(last_row['Low'], live_price)
        if last_idx.date() == today: df.iloc[-1] = pd.Series(last_row)
        else:
            new_idx = last_idx + timedelta(days=1)
            df = pd.concat([df, pd.DataFrame([last_row], index=[new_idx])])
    
    # Recalculate indicators for chart lines
    df['MA5'] = df['Close'].rolling(5).mean()
    df['BOLL_MID'] = df['Close'].rolling(20).mean()
    std = df['Close'].rolling(20).std()
    df['BOLL_U'] = df['BOLL_MID'] + 2*std
    df['BOLL_L'] = df['BOLL_MID'] - 2*std
    
    # Legend
    last = df.iloc[-1]
    legend_html = f"""
    <div class="chart-legend">
        <div class="legend-item"><span class="legend-dot" style="background:#228be6;"></span><span style="color:#228be6;">MA5: {last['MA5']:.2f}</span></div>
        <div class="legend-item"><span class="legend-dot" style="background:#f59f00;"></span><span style="color:#f59f00;">BOLL(Mid): {last['BOLL_MID']:.2f}</span></div>
        <div class="legend-item"><span class="legend-dot" style="background:#868e96;"></span><span style="color:#868e96;">Up: {last['BOLL_U']:.2f}</span></div>
        <div class="legend-item"><span class="legend-dot" style="background:#868e96;"></span><span style="color:#868e96;">Low: {last['BOLL_L']:.2f}</span></div>
    </div>
    """
    
    # Chart
    df = df.tail(60).reset_index()
    df.columns = ['T'] + list(df.columns[1:])
    
    base = alt.Chart(df).encode(x=alt.X('T:T', axis=alt.Axis(format='%m/%d', title=None)))
    rule = base.mark_rule().encode(y=alt.Y('Low:Q', scale=alt.Scale(zero=False)), y2='High:Q')
    bar = base.mark_bar(width=6).encode(
        y='Open:Q', y2='Close:Q',
        color=alt.condition("datum.Close >= datum.Open", alt.value("#e03131"), alt.value("#0ca678"))
    )
    line_5 = base.mark_line(color='#228be6', size=1).encode(y='MA5:Q')
    line_m = base.mark_line(color='#f59f00', size=1).encode(y='BOLL_MID:Q')
    line_u = base.mark_line(color='#adb5bd', strokeDash=[4,2]).encode(y='BOLL_U:Q')
    line_l = base.mark_line(color='#adb5bd', strokeDash=[4,2]).encode(y='BOLL_L:Q')
    
    return (rule + bar + line_5 + line_m + line_u + line_l).properties(height=300).interactive(), legend_html

# --- 6. 核心逻辑：生成交易计划 ---
def generate_trade_plan(price, factors):
    atr = factors['atr']
    b_low = factors['boll_l']
    b_up = factors['boll_u']
    
    # Logic: If price is low (near BOLL_L or RSI < 40), suggest BUY
    if price < b_low * 1.02 or factors['rsi'] < 40:
        action = "BUY"
        # Aggressive entry: Current price. Conservative: BOLL_L
        entry = price 
        # Stop: Recent low - ATR
        stop = price - (atr * 1.5)
        # Target: BOLL_MID or BOLL_UP
        target = factors['boll_m']
        if target <= entry: target = entry + (atr * 2) # Min reward
        
        prob = 65 + (50 - factors['rsi']) # Higher prob if lower RSI
        
    # Logic: If price is high (near BOLL_U or RSI > 60), suggest SELL
    elif price > b_up * 0.98 or factors['rsi'] > 60:
        action = "SELL"
        entry = price
        stop = price + (atr * 1.5)
        target = factors['boll_m']
        if target >= entry: target = entry - (atr * 2)
        
        prob = 65 + (factors['rsi'] - 50)
        
    else:
        # Hold/Range
        action = "WAIT"
        entry = b_low
        stop = b_low - atr
        target = b_up
        prob = 50
        
    # R/R Calculation
    risk = abs(entry - stop)
    reward = abs(target - entry)
    rr = reward / risk if risk > 0 else 1.0
    
    return action, entry, stop, target, min(int(prob), 95), rr

# --- 7. 主仪表盘 ---
@st.fragment(run_every=15)
def show_live_dashboard():
    # 1. Init Variables FIRST (Prevents NameError)
    tz_ny = pytz.timezone('America/New_York')
    now_ny = datetime.now(tz_ny).strftime('%H:%M:%S')
    
    # 2. Get Data
    try:
        t = yf.Ticker("BTDR")
        # Try fast info
        try:
            live_price = t.fast_info['last_price']
            prev_close = t.fast_info['previous_close']
        except:
            # Fallback
            h = t.history(period='2d')
            live_price = h['Close'].iloc[-1]
            prev_close = h['Close'].iloc[-2]
            
        pct_chg = ((live_price - prev_close)/prev_close)*100
        
        # Get Analysis
        factors, status = run_grandmaster_analytics(live_price)
        
    except Exception as e:
        st.error(f"Data Feed Error: {e}")
        return

    # 3. Generate Plan
    action, entry, stop, target, prob, rr = generate_trade_plan(live_price, factors)

    # 4. Render UI
    st.markdown(f"<div class='time-bar'>美东 {now_ny} &nbsp;|&nbsp; 引擎: {status} &nbsp;|&nbsp; ATR: {factors['atr']:.2f}</div>", unsafe_allow_html=True)
    
    # Top Metrics
    c1, c2, c3 = st.columns(3)
    c1.markdown(card_html("BTDR 现价", f"${live_price:.2f}", f"{pct_chg:+.2f}%", pct_chg), unsafe_allow_html=True)
    c2.markdown(card_html("RSI (14)", f"{factors['rsi']:.0f}", "强" if factors['rsi']>50 else "弱", factors['rsi']-50), unsafe_allow_html=True)
    
    # VWAP Distance
    vwap_dist = (live_price - factors['vwap'])/factors['vwap']*100
    c3.markdown(card_html("VWAP 乖离", f"{vwap_dist:+.1f}%", None, 0), unsafe_allow_html=True)
    
    st.markdown("<div style='margin-bottom:15px'></div>", unsafe_allow_html=True)

    # Trade Plan Card (Visual R/R)
    if action == "WAIT":
        st.info(f"⏳ 震荡观望区间。下方支撑 ${entry:.2f}，上方阻力 ${target:.2f}。")
    else:
        st.markdown(trade_plan_html(action, entry, stop, target, prob, rr), unsafe_allow_html=True)

    # Chart
    # Get history for chart
    btdr_hist = yf.Ticker("BTDR").history(period="6mo", interval="1d")
    btdr_hist.index = btdr_hist.index.tz_localize(None)
    
    chart, legend = draw_kline_chart(btdr_hist, live_price, factors)
    st.markdown("### 📈 实时 K 线 (Live Chart)")
    st.markdown(legend, unsafe_allow_html=True)
    st.altair_chart(chart, use_container_width=True)
    
    # Data Matrix
    st.markdown("### 📊 核心数据矩阵")
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(factor_html("MACD", f"{factors['macd']:.3f}", "Bull" if factors['macd']>factors['macd_sig'] else "Bear", 0), unsafe_allow_html=True)
    m2.markdown(factor_html("BOLL Width", f"{(factors['boll_u']-factors['boll_l']):.2f}", "Vol", 0), unsafe_allow_html=True)
    m3.markdown(factor_html("Support", f"${factors['boll_l']:.2f}", "Level", 0), unsafe_allow_html=True)
    m4.markdown(factor_html("Resist", f"${factors['boll_u']:.2f}", "Level", 0), unsafe_allow_html=True)

st.markdown("### ⚡ BTDR Pilot v12.8 Trader")
show_live_dashboard()
