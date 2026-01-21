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
st.set_page_config(page_title="BTDR Pilot v13.3 Scenario", layout="centered")

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
    
    /* Metric Card */
    .metric-card {
        background-color: #f8f9fa; border: 1px solid #e9ecef;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02); height: 95px; padding: 0 16px;
        display: flex; flex-direction: column; justify-content: center;
        position: relative; transition: all 0.2s;
    }
    .metric-card.has-tooltip { cursor: help; }
    .metric-card.has-tooltip:hover { border-color: #ced4da; }
    
    .metric-label { font-size: 0.75rem; color: #888; margin-bottom: 2px; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #212529; line-height: 1.2; }
    .metric-delta { font-size: 0.9rem; font-weight: 600; margin-top: 2px; }
    
    /* Miner Card */
    .miner-card {
        background-color: #fff; border: 1px solid #e9ecef;
        border-radius: 10px; padding: 8px 10px;
        text-align: center; height: 100px;
        display: flex; flex-direction: column; justify-content: space-between;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .miner-sym { font-size: 0.75rem; color: #888; font-weight: 600; margin-bottom: 2px; }
    .miner-price { font-size: 1.1rem; font-weight: 700; color: #212529; }
    .miner-sub { font-size: 0.7rem; display: flex; justify-content: space-between; margin-top: 4px; }
    .miner-pct { font-weight: 600; }
    .miner-turn { color: #868e96; }
    
    /* Factor Box */
    .factor-box {
        background: #fff;
        border: 1px solid #eee; border-radius: 8px; padding: 6px; text-align: center;
        height: 75px; display: flex; flex-direction: column; justify-content: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02); position: relative; cursor: help; transition: transform 0.1s;
    }
    .factor-box:hover { border-color: #ced4da; transform: translateY(-1px); }
    .factor-title { font-size: 0.65rem; color: #999; text-transform: uppercase; letter-spacing: 0.5px; }
    .factor-val { font-size: 1.1rem; font-weight: bold; color: #495057; margin: 2px 0; }
    .factor-sub { font-size: 0.7rem; font-weight: 600; }
    
    /* Action Banner */
    .action-banner {
        padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px;
        display: flex; align-items: center; justify-content: space-between;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }
    .act-buy { background: linear-gradient(135deg, #e6fcf5 0%, #099268 100%); color: white; }
    .act-sell { background: linear-gradient(135deg, #fff5f5 0%, #c92a2a 100%); color: white; }
    .act-hold { background: linear-gradient(135deg, #f8f9fa 0%, #868e96 100%); color: #343a40; }
    .act-title { font-size: 0.9rem; font-weight: 600; text-transform: uppercase; opacity: 0.9; }
    .act-main { font-size: 2rem; font-weight: 800; letter-spacing: 1px; }
    .act-sub { font-size: 0.8rem; font-weight: 500; opacity: 0.95; }

    /* Chart Legend */
    .chart-legend {
        display: flex; flex-wrap: wrap; gap: 10px; font-size: 0.75rem; color: #555;
        background: #f8f9fa; padding: 6px 10px; border-radius: 6px; margin-bottom: 5px;
        border: 1px solid #eee; align-items: center;
    }
    .legend-item { display: flex; align-items: center; gap: 4px; }
    .legend-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
    .legend-val { font-weight: 700; color: #212529; margin-left: 2px; }

    /* Tooltip Core */
    .tooltip-text {
        visibility: hidden;
        width: 180px; background-color: rgba(33, 37, 41, 0.95);
        color: #fff !important; text-align: center; border-radius: 6px; padding: 8px;
        position: absolute; z-index: 999;
        bottom: 110%; left: 50%; margin-left: -90px;
        opacity: 0; transition: opacity 0.3s; font-size: 0.7rem !important;
        font-weight: normal; line-height: 1.4; pointer-events: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .tooltip-text::after {
        content: "";
        position: absolute; top: 100%; left: 50%; margin-left: -5px;
        border-width: 5px; border-style: solid;
        border-color: rgba(33, 37, 41, 0.95) transparent transparent transparent;
    }
    
    .factor-box:hover .tooltip-text { visibility: visible; opacity: 1; }
    .metric-card:hover .tooltip-text { visibility: visible; opacity: 1; }
    
    .color-up { color: #e03131; } .color-down { color: #0ca678; } .color-neutral { color: #adb5bd; }
    
    .status-dot { height: 6px; width: 6px; border-radius: 50%; display: inline-block; margin-left: 6px; margin-bottom: 2px; }
    .dot-pre { background-color: #f59f00; box-shadow: 0 0 4px #f59f00; }
    .dot-reg { background-color: #0ca678; box-shadow: 0 0 4px #0ca678; }
    .dot-post { background-color: #1c7ed6; box-shadow: 0 0 4px #1c7ed6; }
    .dot-night { background-color: #7048e8; box-shadow: 0 0 4px #7048e8; }
    .dot-closed { background-color: #adb5bd; }
    
    /* NEW Scenario Box */
    .scenario-container {
        display: flex; gap: 10px; margin-top: 10px; margin-bottom: 20px;
    }
    .scenario-box {
        flex: 1; border-radius: 8px; padding: 12px 10px; text-align: center;
        border: 1px solid #eee; display: flex; flex-direction: column; justify-content: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03); transition: transform 0.1s;
    }
    .scenario-box:hover { transform: translateY(-2px); }
    
    .scen-bear { background: #e6fcf5; border-color: #b2f2bb; border-top: 4px solid #0ca678; }
    .scen-base { background: #f8f9fa; border-color: #e9ecef; border-top: 4px solid #adb5bd; }
    .scen-bull { background: #fff5f5; border-color: #ffc9c9; border-top: 4px solid #e03131; }
    
    .scen-title { font-size: 0.75rem; font-weight: bold; text-transform: uppercase; margin-bottom: 4px; opacity: 0.8;}
    .scen-val { font-size: 1.3rem; font-weight: 800; color: #333; margin-bottom: 2px; }
    .scen-sub { font-size: 0.7rem; font-weight: 600; }
    
    .time-bar { font-size: 0.75rem; color: #999; text-align: center; margin-bottom: 20px; padding: 6px; background: #fafafa; border-radius: 6px; }
    .badge-ai { background: linear-gradient(90deg, #6366f1, #a855f7); color:white; padding:1px 6px; border-radius:3px; font-size:0.6rem; font-weight:bold;}
    
    /* 强制图表宽度适配 */
    canvas { width: 100% !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- 2. 基础配置 ---
MINER_SHARES = {"MARA": 300, "RIOT": 330, "CLSK": 220, "CORZ": 190, "IREN": 180, "WULF": 410, "CIFR": 300, "HUT": 100}
MINER_POOL = list(MINER_SHARES.keys())

# --- 3. 辅助函数 ---
def card_html(label, value_str, delta_str=None, delta_val=0, extra_tag="", tooltip_text=None):
    delta_html = ""
    if delta_str:
        color_class = "color-up" if delta_val >= 0 else "color-down"
        delta_html = f"<div class='metric-delta {color_class}'>{delta_str}</div>"
    
    tooltip_html = f"<div class='tooltip-text'>{tooltip_text}</div>" if tooltip_text else ""
    card_class = "metric-card has-tooltip" if tooltip_text else "metric-card"
    return f"""<div class="{card_class}">{tooltip_html}<div class="metric-label">{label} {extra_tag}</div><div class="metric-value">{value_str}</div>{delta_html}</div>"""

def factor_html(title, val, delta_str, delta_val, tooltip_text, reverse_color=False):
    is_positive = delta_val >= 0
    if reverse_color: is_positive = not is_positive
    color_class = "color-up" if is_positive else "color-down"
    return f"""<div class="factor-box"><div class="tooltip-text">{tooltip_text}</div><div class="factor-title">{title}</div><div class="factor-val">{val}</div><div class="factor-sub {color_class}">{delta_str}</div></div>"""

def miner_card_html(sym, price, pct, turnover):
    color_class = "color-up" if pct >= 0 else "color-down"
    return f"""<div class="miner-card"><div class="miner-sym">{sym}</div><div class="miner-price ${color_class}">${price:.2f}</div><div class="miner-sub"><span class="miner-pct {color_class}">{pct:+.1f}%</span><span class="miner-turn">换 {turnover:.1f}%</span></div></div>"""

def action_banner_html(action, reason, sub_text):
    if action in ["STRONG BUY", "ACCUMULATE", "BUY"]: css_class = "act-buy"; icon = "🚀"
    elif action in ["STRONG SELL", "REDUCE", "SELL"]: css_class = "act-sell"; icon = "⚠️"
    else: css_class = "act-hold"; icon = "🛡️"
        
    return f"""
    <div class="action-banner {css_class}">
        <div style="text-align:left;">
            <div class="act-title">AI TACTICAL ADVICE</div>
            <div class="act-main">{icon} {action}</div>
        </div>
        <div style="text-align:right; max-width: 60%;">
            <div style="font-size:0.9rem; font-weight:bold;">{reason}</div>
            <div class="act-sub">{sub_text}</div>
        </div>
    </div>
    """

def scenario_html(open_p, atr):
    # Calculations
    bear_low = open_p - (atr * 1.5)
    bear_high = open_p - (atr * 0.5)
    
    base_low = open_p - (atr * 0.6)
    base_high = open_p + (atr * 0.6)
    
    bull_low = open_p + (atr * 0.5)
    bull_high = open_p + (atr * 1.5)
    
    return f"""
    <div style="margin-bottom:5px; font-weight:bold; color:#555;">🎲 今日股价情景推演 (Scenario Forecast)</div>
    <div class="scenario-container">
        <div class="scenario-box scen-bear">
            <div class="scen-title" style="color:#0ca678;">🐻 悲观 (Bear)</div>
            <div class="scen-val" style="color:#0ca678;">${bear_low:.2f}</div>
            <div class="scen-sub">下探至 ${bear_high:.2f}</div>
        </div>
        <div class="scenario-box scen-base">
            <div class="scen-title" style="color:#495057;">⚖️ 中性 (Base)</div>
            <div class="scen-val" style="color:#495057;">${base_low:.2f}</div>
            <div class="scen-sub">震荡至 ${base_high:.2f}</div>
        </div>
        <div class="scenario-box scen-bull">
            <div class="scen-title" style="color:#d6336c;">🚀 乐观 (Bull)</div>
            <div class="scen-val" style="color:#d6336c;">${bull_high:.2f}</div>
            <div class="scen-sub">上攻至 ${bull_low:.2f}</div>
        </div>
    </div>
    <div style="text-align:center; font-size:0.7rem; color:#999; margin-top:-15px; margin-bottom:20px;">基于今日开盘价 ${open_p:.2f} 与 ATR(${atr:.2f}) 波动率推演</div>
    """

# --- 4. 核心计算 ---
def run_kalman_filter(y, x, delta=1e-4):
    try:
        n = len(y)
        if n < 2: return 1.0
        beta = np.zeros(n); P = np.zeros(n); beta[0]=1.0; P[0]=1.0; R=0.002; Q=delta/(1-delta)
        for t in range(1, n):
            beta_pred = beta[t-1]; P_pred = P[t-1] + Q
            if x[t] == 0: x[t] = 1e-6
            residual = y[t] - beta_pred * x[t]; S = P_pred * x[t]**2 + R; K = P_pred * x[t] / S
            beta[t] = beta_pred + K * residual; P[t] = (1 - K * x[t]) * P_pred
        return beta[-1]
    except: return 1.0

def calculate_hurst(series):
    try:
        if len(series) < 20: return 0.5
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    except: return 0.5

@st.cache_data(ttl=300)
def run_grandmaster_analytics(live_price=None):
    default_factors = {"vwap": 0, "adx": 20, "regime": "Neutral", "beta_btc": 1.5, "beta_qqq": 1.2, "rsi": 50, "vol_base": 0.05, "atr_ratio": 0.05, "hurst": 0.5, "macd": 0, "macd_sig": 0, "boll_u": 0, "boll_l": 0, "boll_m": 0, "atr": 0.5}

    try:
        tickers_str = "BTDR BTC-USD QQQ " + " ".join(MINER_POOL)
        data = yf.download(tickers_str, period="6mo", interval="1d", group_by='ticker', threads=True, progress=False)
        if data.empty: return {}, default_factors, "No Data"

        btdr = data['BTDR'].dropna(); btc = data['BTC-USD'].dropna(); qqq = data['QQQ'].dropna()
        idx = btdr.index.intersection(btc.index).intersection(qqq.index)
        btdr, btc, qqq = btdr.loc[idx], btc.loc[idx], qqq.loc[idx]
        
        # 清洗时区
        btdr.index = btdr.index.tz_localize(None)
        
        # 注入实时
        if live_price and live_price > 0:
            last_date = btdr.index[-1].date()
            today = datetime.now().date()
            last_row = btdr.iloc[-1].copy()
            last_row['Close'] = live_price
            last_row['High'] = max(last_row['High'], live_price)
            last_row['Low'] = min(last_row['Low'], live_price)
            
            if last_date == today: btdr.iloc[-1] = last_row
            else:
                new_idx = btdr.index[-1] + timedelta(days=1)
                new_df = pd.DataFrame([last_row], index=[new_idx])
                btdr = pd.concat([btdr, new_df])

        if len(btdr) < 30: return {}, default_factors, "Insufficient Data"

        # Calc
        close = btdr['Close']
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        boll_u = sma20 + 2 * std20
        boll_l = sma20 - 2 * std20
        
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        exp12 = close.ewm(span=12).mean(); exp26 = close.ewm(span=26).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9).mean()
        
        tr = np.maximum(btdr['High'] - btdr['Low'], np.abs(btdr['High'] - close.shift(1)))
        atr = tr.rolling(14).mean()
        
        pv = (close * btdr['Volume']); vwap = pv.tail(30).sum() / btdr['Volume'].tail(30).sum()

        factors = {
            "beta_btc": 1.5, "beta_qqq": 1.2, # Placeholder
            "vwap": vwap, 
            "adx": 25, "regime": "Trend", 
            "rsi": rsi.iloc[-1], 
            "vol_base": 0.05, 
            "atr_ratio": (atr.iloc[-1]/close.iloc[-1]), 
            "hurst": 0.5,
            "macd": macd.iloc[-1], "macd_sig": signal.iloc[-1], 
            "boll_u": boll_u.iloc[-1], "boll_l": boll_l.iloc[-1], "boll_m": sma20.iloc[-1],
            "atr": atr.iloc[-1]
        }
        return {}, factors, "v13.3 Scenario"
    except Exception as e:
        return {}, default_factors, "Offline"

# --- 5. 辅助逻辑 ---
def determine_market_state(now_ny):
    weekday = now_ny.weekday(); curr_min = now_ny.hour * 60 + now_ny.minute
    if weekday >= 5: return "Weekend", "dot-closed"
    if 240 <= curr_min < 570: return "Pre-Mkt", "dot-pre"
    if 570 <= curr_min < 960: return "Mkt Open", "dot-reg"
    return "Closed", "dot-night"

def get_signal_recommendation(curr_price, factors):
    score = 0; reasons = []
    
    rsi = factors['rsi']
    if rsi < 30: score += 2; reasons.append("RSI超卖")
    elif rsi > 70: score -= 2; reasons.append("RSI超买")
    
    range_boll = factors['boll_u'] - factors['boll_l']
    if range_boll <= 0: range_boll = 0.01
    bp = (curr_price - factors['boll_l']) / range_boll
    
    if bp < 0: score += 3; reasons.append("跌破下轨")
    elif bp > 1: score -= 3; reasons.append("突破上轨")
    elif bp < 0.2: score += 1; reasons.append("近下轨")
    elif bp > 0.8: score -= 1; reasons.append("近上轨")

    action = "HOLD"; sub_text = "多空均衡"
    if score >= 4: action = "STRONG BUY"; sub_text = "技术共振，建议买入"
    elif score >= 1: action = "ACCUMULATE"; sub_text = "趋势偏多，分批建仓"
    elif score <= -4: action = "STRONG SELL"; sub_text = "风险极高，建议清仓"
    elif score <= -1: action = "REDUCE"; sub_text = "阻力较大，逢高减仓"
        
    return action, " | ".join(reasons[:2]), sub_text, score

def get_realtime_data():
    try:
        t = yf.Ticker("BTDR")
        try: 
            price = t.fast_info['last_price']
            prev = t.fast_info['previous_close']
            vol = 0
            open_p = t.fast_info['open']
        except:
            h = t.history(period="2d")
            price = h['Close'].iloc[-1]
            prev = h['Close'].iloc[-2]
            vol = h['Volume'].iloc[-1]
            open_p = h['Open'].iloc[-1]
            
        pct = ((price - prev) / prev) * 100 if prev > 0 else 0
        
        # Hist for Chart
        hist = t.history(period="6mo")
        hist.index = hist.index.tz_localize(None)
        
        return {"price": price, "pct": pct, "prev": prev, "open": open_p, "vol": vol}, 50, 0.05, hist
    except: return {"price": 0}, 50, 0.05, pd.DataFrame()

# --- 6. 绘图 ---
def draw_kline_chart(df, live_price):
    if df.empty: return alt.Chart(pd.DataFrame()).mark_text().encode(text=alt.value("No Data")), ""
    
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

    df['MA5'] = df['Close'].rolling(5).mean()
    df['BOLL_MID'] = df['Close'].rolling(20).mean()
    std = df['Close'].rolling(20).std()
    df['BOLL_U'] = df['BOLL_MID'] + 2*std
    df['BOLL_L'] = df['BOLL_MID'] - 2*std
    
    last = df.iloc[-1]
    # Check NaN
    m5 = last['MA5'] if not np.isnan(last['MA5']) else live_price
    b_mid = last['BOLL_MID'] if not np.isnan(last['BOLL_MID']) else live_price
    b_up = last['BOLL_U'] if not np.isnan(last['BOLL_U']) else live_price * 1.1
    b_low = last['BOLL_L'] if not np.isnan(last['BOLL_L']) else live_price * 0.9
    
    legend_html = f"""
    <div class="chart-legend">
        <div class="legend-item"><span class="legend-dot" style="background:#228be6;"></span><span style="color:#228be6;">MA5: {m5:.2f}</span></div>
        <div class="legend-item"><span class="legend-dot" style="background:#f59f00;"></span><span style="color:#f59f00;">BOLL(Mid): {b_mid:.2f}</span></div>
        <div class="legend-item"><span class="legend-dot" style="background:#868e96;"></span><span style="color:#868e96;">BOLL(Up): {b_up:.2f}</span></div>
        <div class="legend-item"><span class="legend-dot" style="background:#868e96;"></span><span style="color:#868e96;">BOLL(Low): {b_low:.2f}</span></div>
    </div>
    """
    
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
    
    # CSS width fallback
    return (rule + bar + line_5 + line_m + line_u + line_l).properties(height=300).interactive(), legend_html

# --- 7. Main ---
@st.fragment(run_every=15)
def show_live_dashboard():
    # Init vars
    tz_ny = pytz.timezone('America/New_York')
    now_ny = datetime.now(tz_ny).strftime('%H:%M:%S')
    badge_class = "badge-ai"
    ai_status = "Init"
    act, reason, sub = "WAIT", "Init...", ""
    
    # Data
    quotes, fng, vol, btdr_hist = get_realtime_data()
    live_price = quotes.get('price', 0)
    
    if live_price <= 0:
        st.warning("📡 Connecting..."); time.sleep(2); st.rerun(); return

    # Analytics
    _, factors, ai_status = run_grandmaster_analytics(live_price)
    
    act, reason, sub, score = get_signal_recommendation(live_price, factors)
    
    # UI
    st.markdown(f"<div class='time-bar'>美东 {now_ny} &nbsp;|&nbsp; 引擎: <b>{ai_status}</b></div>", unsafe_allow_html=True)
    st.markdown(action_banner_html(act, reason, sub), unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1: st.markdown(card_html("BTDR 现价", f"${live_price:.2f}", f"{quotes['pct']:.2f}%", quotes['pct']), unsafe_allow_html=True)
    with c2: st.markdown(card_html("恐慌指数", f"{fng}", None, 0, tooltip_text="0-25: Extreme Fear"), unsafe_allow_html=True)
    
    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    
    # --- NEW: Scenario Analysis (Replaces Tickets) ---
    open_p = quotes['open'] if quotes['open'] > 0 else quotes['prev']
    atr_val = factors['atr'] if not np.isnan(factors['atr']) else live_price * 0.05
    st.markdown(scenario_html(open_p, atr_val), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Chart
    st.markdown("### 📈 实时 K 线 (Live Chart)")
    if not btdr_hist.empty:
        c, l = draw_kline_chart(btdr_hist, live_price)
        st.markdown(l, unsafe_allow_html=True)
        st.altair_chart(c, use_container_width=True)
        
    # Matrix
    st.markdown("### 📊 核心指标")
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.markdown(factor_html("RSI", f"{factors['rsi']:.0f}", "Neu", 0, "Relative Strength"), unsafe_allow_html=True)
    with m2: st.markdown(factor_html("MACD", f"{factors['macd']:.3f}", "Cross", 0, "Trend"), unsafe_allow_html=True)
    
    # Range calc
    rng = factors['boll_u'] - factors['boll_l']
    if rng <= 0: rng = 0.01
    pos = (live_price - factors['boll_l']) / rng * 100
    
    with m3: st.markdown(factor_html("BOLL Pos", f"{pos:.0f}%", "Band", 0, "Position"), unsafe_allow_html=True)
    with m4: st.markdown(factor_html("ATR", f"{atr_val:.2f}", "Vol", 0, "Volatility"), unsafe_allow_html=True)

    # Probability Chart at bottom (Optional, kept for completeness)
    # st.markdown("### ☁️ 概率推演")
    # render_probability_chart(live_price, factors['vol_base'])

st.markdown("### ⚡ BTDR 领航员 v13.3 Scenario")
show_live_dashboard()
