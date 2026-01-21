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
st.set_page_config(page_title="BTDR Pilot v13.4 Scenario-Mod", layout="centered")

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
    
    .pred-container-wrapper { height: 110px; width: 100%; display: block; margin-top: 5px; }
    .pred-box { padding: 0 10px; border-radius: 12px; text-align: center; height: 100%; display: flex; flex-direction: column; justify-content: center; }
    
    .time-bar { font-size: 0.75rem; color: #999; text-align: center; margin-bottom: 20px; padding: 6px; background: #fafafa; border-radius: 6px; }
    .badge-ai { background: linear-gradient(90deg, #6366f1, #a855f7); color:white; padding:1px 6px; border-radius:3px; font-size:0.6rem; font-weight:bold;}
    
    .ensemble-bar { height: 4px; width: 100%; display: flex; margin-top: 4px; border-radius: 2px; overflow: hidden; }
    .bar-kalman { background-color: #228be6; transition: width 0.5s; }
    .bar-hist { background-color: #fab005; transition: width 0.5s; }
    .bar-mom { background-color: #fa5252; transition: width 0.5s; }
    .bar-ai { background-color: #be4bdb; transition: width 0.5s; }
    
    /* --- NEW SCENARIO STYLES --- */
    .scenario-container { display: flex; gap: 10px; margin-bottom: 20px; }
    .scenario-box {
        flex: 1; border-radius: 8px; padding: 12px 10px; text-align: center;
        border: 1px solid #eee; display: flex; flex-direction: column; justify-content: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03); transition: transform 0.1s;
    }
    .scenario-box:hover { transform: translateY(-2px); }
    
    .scen-bear { background: #e6fcf5; border-color: #b2f2bb; border-top: 4px solid #0ca678; }
    .scen-base { background: #f8f9fa; border-color: #e9ecef; border-top: 4px solid #adb5bd; }
    .scen-bull { background: #fff5f5; border-color: #ffc9c9; border-top: 4px solid #e03131; }
    
    .scen-title { font-size: 0.7rem; font-weight: bold; text-transform: uppercase; margin-bottom: 4px; opacity: 0.7;}
    .scen-val { font-size: 1.3rem; font-weight: 800; color: #333; margin-bottom: 2px; }
    .scen-sub { font-size: 0.7rem; font-weight: 600; opacity: 0.8; }
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
    
    base_low = open_p - (atr * 0.5)
    base_high = open_p + (atr * 0.5)
    
    bull_low = open_p + (atr * 0.5)
    bull_high = open_p + (atr * 1.5)
    
    return f"""
    <div class="scenario-container">
        <div class="scenario-box scen-bear">
            <div class="scen-title" style="color:#0ca678;">🐻 悲观 (Bear)</div>
            <div class="scen-val" style="color:#0ca678;">${bear_low:.2f}</div>
            <div class="scen-sub">至 ${bear_high:.2f}</div>
        </div>
        <div class="scenario-box scen-base">
            <div class="scen-title" style="color:#495057;">⚖️ 中性 (Base)</div>
            <div class="scen-val" style="color:#495057;">${base_low:.2f}</div>
            <div class="scen-sub">至 ${base_high:.2f}</div>
        </div>
        <div class="scenario-box scen-bull">
            <div class="scen-title" style="color:#d6336c;">🚀 乐观 (Bull)</div>
            <div class="scen-val" style="color:#d6336c;">${bull_high:.2f}</div>
            <div class="scen-sub">${bull_low:.2f} 起</div>
        </div>
    </div>
    """

# --- 4. 核心计算 (AI & Math Core) ---
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
    default_model = {
        "high": {"intercept": 0, "beta_gap": 0.5, "beta_btc": 0.5, "beta_vol": 0},
        "low": {"intercept": 0, "beta_gap": 0.5, "beta_btc": 0.5, "beta_vol": 0},
        "ensemble_hist_h": 0.05, "ensemble_hist_l": -0.05,
        "ensemble_mom_h": 0.08, "ensemble_mom_l": -0.08,
        "top_peers": ["MARA", "RIOT", "CLSK", "CORZ", "IREN"]
    }
    default_factors = {"vwap": 0, "adx": 20, "regime": "Neutral", "beta_btc": 1.5, "beta_qqq": 1.2, "rsi": 50, "vol_base": 0.05, "atr_ratio": 0.05, "hurst": 0.5, "macd": 0, "macd_sig": 0, "boll_u": 0, "boll_l": 0, "boll_m": 0, "atr": 0.5}

    try:
        tickers_str = "BTDR BTC-USD QQQ " + " ".join(MINER_POOL)
        data = yf.download(tickers_str, period="6mo", interval="1d", group_by='ticker', threads=True, progress=False)
        if data.empty: return default_model, default_factors, "No Data"

        btdr = data['BTDR'].dropna(); btc = data['BTC-USD'].dropna(); qqq = data['QQQ'].dropna()
        idx = btdr.index.intersection(btc.index).intersection(qqq.index)
        btdr, btc, qqq = btdr.loc[idx], btc.loc[idx], qqq.loc[idx]
        
        # --- 关键步骤：数据清洗 (Remove Timezone) ---
        btdr.index = btdr.index.tz_localize(None)
        market_data = data
        market_data.index = market_data.index.tz_localize(None)
        
        # --- 关键步骤：注入实时数据 (Reconstruction) ---
        if live_price and live_price > 0:
            last_date = btdr.index[-1].date()
            today = datetime.now().date()
            
            # 构造新的一行数据
            last_row = btdr.iloc[-1].copy()
            last_row['Close'] = live_price
            last_row['High'] = max(last_row['High'], live_price)
            last_row['Low'] = min(last_row['Low'], live_price)
            
            if last_date == today:
                # Update today's candle
                btdr.iloc[-1] = last_row
            else:
                # Append new candle for today
                new_idx = btdr.index[-1] + timedelta(days=1)
                new_df = pd.DataFrame([last_row], index=[new_idx])
                btdr = pd.concat([btdr, new_df])

        if len(btdr) < 30: return default_model, default_factors, "Insufficient Data"

        # Correlation Analysis
        btdr_hist_slice = btdr.iloc[:-1] if live_price else btdr
        correlations = {}
        for m in MINER_POOL:
            if m in data:
                miner_df = data[m]['Close'].pct_change().tail(30)
                btdr_df = btdr_hist_slice['Close'].pct_change().tail(30)
                common_idx = miner_df.index.intersection(btdr_df.index)
                if len(common_idx) > 10: correlations[m] = miner_df.loc[common_idx].corr(btdr_df.loc[common_idx])
                else: correlations[m] = 0
        top_peers = sorted(correlations, key=correlations.get, reverse=True)[:5]
        default_model["top_peers"] = top_peers

        ret_btdr = btdr['Close'].pct_change().fillna(0).values
        ret_btc = btc.pct_change().reindex(btdr.index).fillna(0).values
        ret_qqq = qqq.pct_change().reindex(btdr.index).fillna(0).values
        
        beta_btc = run_kalman_filter(ret_btdr, ret_btc, delta=1e-4)
        beta_qqq = run_kalman_filter(ret_btdr, ret_qqq, delta=1e-4)
        beta_btc = np.clip(beta_btc, -1, 5); beta_qqq = np.clip(beta_qqq, -1, 4)

        # Technicals
        close = btdr['Close']
        
        # VWAP
        pv = (close * btdr['Volume'])
        vol_sum = btdr['Volume'].tail(30).sum()
        vwap_30d = pv.tail(30).sum() / vol_sum if vol_sum > 0 else btdr['Close'].mean()
        
        high, low, close = btdr['High'], btdr['Low'], btdr['Close']
        tr = np.maximum(high - low, np.abs(high - close.shift(1)))
        atr = tr.rolling(14).mean()
        
        # MACD
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9, adjust=False).mean()
        
        # BOLL
        sma20 = close.rolling(window=20).mean()
        std20 = close.rolling(window=20).std()
        boll_u = sma20 + (std20 * 2)
        boll_l = sma20 - (std20 * 2)
        boll_m = sma20 # Median
        
        # ADX
        up, down = high.diff(), -low.diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0); minus_dm = np.where((down > up) & (down > 0), down, 0)
        atr_s = pd.Series(atr.values, index=btdr.index)
        plus_di = 100 * (pd.Series(plus_dm, index=btdr.index).rolling(14).mean() / atr_s)
        minus_di = 100 * (pd.Series(minus_dm, index=btdr.index).rolling(14).mean() / atr_s)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(14).mean().iloc[-1]; adx = 20 if np.isnan(adx) else adx
        
        # RSI
        delta_p = close.diff()
        gain = delta_p.where(delta_p > 0, 0).rolling(14).mean(); loss = -delta_p.where(delta_p < 0, 0).rolling(14).mean()
        rs = gain / loss; rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        vol_base = ret_btdr.std()
        if len(ret_btdr) > 20: vol_base = pd.Series(ret_btdr).ewm(span=20).std().iloc[-1]
        atr_ratio = (atr / close).iloc[-1]

        hurst = calculate_hurst(btdr['Close'].values[-50:])
        
        regime = "Trend" if adx > 25 else ("MeanRev" if hurst < 0.4 else "Chop")

        # --- 修复：NaN 熔断保护 (防止 $0.00 挂单) ---
        last_close = close.iloc[-1]
        bu_val = boll_u.iloc[-1] 
        bl_val = boll_l.iloc[-1]
        bm_val = boll_m.iloc[-1]
        
        # 强制兜底
        if np.isnan(bu_val) or bu_val <= 0: bu_val = last_close * 1.10
        if np.isnan(bl_val) or bl_val <= 0: bl_val = last_close * 0.90
        if np.isnan(bm_val) or bm_val <= 0: bm_val = last_close

        factors = {
            "beta_btc": beta_btc, "beta_qqq": beta_qqq, "vwap": vwap_30d, 
            "adx": adx, "regime": regime, "rsi": rsi, 
            "vol_base": vol_base, "atr_ratio": atr_ratio, "hurst": hurst,
            "macd": macd.iloc[-1], "macd_sig": signal.iloc[-1], 
            "boll_u": bu_val, "boll_l": bl_val, "boll_m": bm_val,
            "atr": atr.iloc[-1]
        }

        # WLS Regression (Revert to historical only for training)
        if live_price: btdr = btdr.iloc[:-1]
        
        df_reg = pd.DataFrame()
        df_reg['Target_High'] = (btdr['High'] - btdr['Close'].shift(1)) / btdr['Close'].shift(1)
        df_reg['Target_Low'] = (btdr['Low'] - btdr['Close'].shift(1)) / btdr['Close'].shift(1)
        df_reg = df_reg.dropna().tail(60)
        
        final_model = {
            "high": {"intercept": 0, "beta_gap": 0.5, "beta_btc": 0.5, "beta_vol": 0}, # Placeholder for stability
            "low": {"intercept": 0, "beta_gap": 0.5, "beta_btc": 0.5, "beta_vol": 0},
            "ensemble_hist_h": df_reg['Target_High'].tail(10).mean(), 
            "ensemble_hist_l": df_reg['Target_Low'].tail(10).mean(),
            "ensemble_mom_h": df_reg['Target_High'].tail(3).max(), 
            "ensemble_mom_l": df_reg['Target_Low'].tail(3).min(),
            "top_peers": default_model["top_peers"]
        }
        return final_model, factors, "v13.4 Scenario-Mod"
    except Exception as e:
        print(f"Error: {e}")
        return default_model, default_factors, "Offline"

# --- 5. 实时数据与 AI 引擎 ---
def determine_market_state(now_ny):
    weekday = now_ny.weekday(); curr_min = now_ny.hour * 60 + now_ny.minute
    if weekday == 5: return "Weekend", "dot-closed"
    if weekday == 6 and now_ny.hour < 20: return "Weekend", "dot-closed"
    if 240 <= curr_min < 570: return "Pre-Mkt", "dot-pre"
    if 570 <= curr_min < 960: return "Mkt Open", "dot-reg"
    if 960 <= curr_min < 1200: return "Post-Mkt", "dot-post"
    return "Overnight", "dot-night"

def get_ai_weights(regime, rsi, volatility_state):
    if regime == "Trend": return 0.20, 0.05, 0.35, 0.40
    elif regime == "MeanRev": return 0.35, 0.35, 0.10, 0.20
    else: return 0.25, 0.15, 0.10, 0.50

def get_signal_recommendation(curr_price, factors, p_low):
    score = 0; reasons = []
    
    rsi = factors['rsi']
    if rsi < 30: score += 2; reasons.append("RSI超卖")
    elif rsi > 70: score -= 2; reasons.append("RSI超买")
    elif rsi > 55: score += 0.5 
    
    # Secure BOLL Calc
    range_boll = factors['boll_u'] - factors['boll_l']
    if range_boll <= 0: range_boll = curr_price * 0.05 
    
    bp = (curr_price - factors['boll_l']) / range_boll
    if bp < 0: score += 3; reasons.append("跌破下轨")
    elif bp > 1: score -= 3; reasons.append("突破上轨")
    elif bp < 0.2: score += 1; reasons.append("近下轨")
    elif bp > 0.8: score -= 1; reasons.append("近上轨")

    macd_hist = factors['macd'] - factors['macd_sig']
    if macd_hist > 0 and factors['macd'] > 0: score += 1.5; reasons.append("MACD多头")
    elif macd_hist < 0 and factors['macd'] < 0: score -= 1.5; reasons.append("MACD空头")
    
    support_broken = False
    if curr_price < p_low:
        score += 1; reasons.append("击穿支撑")
        support_broken = True
    
    action = "HOLD"; sub_text = "多空均衡"
    if score >= 4.5: action = "STRONG BUY"; sub_text = "技术共振，建议买入"
    elif score >= 2: action = "ACCUMULATE"; sub_text = "趋势偏多，分批建仓"
    elif score <= -4.5: action = "STRONG SELL"; sub_text = "风险极高，建议清仓"
    elif score <= -2: action = "REDUCE"; sub_text = "阻力较大，逢高减仓"
        
    return action, " | ".join(reasons[:2]), sub_text, score, macd_hist, support_broken

def get_realtime_data():
    tickers_list = "BTC-USD BTDR QQQ ^VIX " + " ".join(MINER_POOL)
    symbols = tickers_list.split()
    try:
        # Check for multi-level index issue
        daily = yf.download(tickers_list, period="6mo", interval="1d", group_by='ticker', threads=True, progress=False)
        live = yf.download(tickers_list, period="2d", interval="1m", prepost=True, group_by='ticker', threads=True, progress=False)
        
        quotes = {}
        tz_ny = pytz.timezone('America/New_York'); now_ny = datetime.now(tz_ny); state_tag, state_css = determine_market_state(now_ny)
        live_volatility = 0.01 
        btdr_history = pd.DataFrame()

        for sym in symbols:
            try:
                df_day = daily[sym].dropna(subset=['Close']) if sym in daily else pd.DataFrame()
                df_min = live[sym].dropna(subset=['Close']) if sym in live else pd.DataFrame()
                
                if sym == 'BTDR': btdr_history = df_day

                current_volume = 0; current_price = 0.0
                prev_close = 1.0; open_price = 0.0; is_open_today = False
                
                if not df_min.empty: 
                    current_price = df_min['Close'].iloc[-1]
                    if 'Volume' in df_min.columns: current_volume = df_min['Volume'].sum()
                    if sym == 'BTDR':
                        recent_min_std = df_min['Close'].tail(60).pct_change().std()
                        if np.isnan(recent_min_std) or recent_min_std == 0: live_volatility = 0.005 
                        else: live_volatility = recent_min_std * np.sqrt(60) 
                        
                elif not df_day.empty: 
                    current_price = df_day['Close'].iloc[-1]; current_volume = df_day['Volume'].iloc[-1]
                
                if not df_day.empty:
                    last_day_date = df_day.index[-1].date()
                    if last_day_date == now_ny.date():
                        is_open_today = True; open_price = df_day['Open'].iloc[-1]
                        if len(df_day) >= 2: prev_close = df_day['Close'].iloc[-2]
                        else: prev_close = df_day['Open'].iloc[-1]
                    else: prev_close = df_day['Close'].iloc[-1]; open_price = prev_close
                
                pct = ((current_price - prev_close) / prev_close) * 100 if prev_close > 0 else 0
                quotes[sym] = {"price": current_price, "pct": pct, "prev": prev_close, "open": open_price, "volume": current_volume, "tag": state_tag, "css": state_css, "is_open_today": is_open_today}
            except Exception as e: 
                quotes[sym] = {"price": 0, "pct": 0, "prev": 1, "open": 0, "volume": 0, "tag": "ERR", "css": "dot-closed", "is_open_today": False}
        
        try: fng = int(requests.get("https://api.alternative.me/fng/", timeout=1.0).json()['data'][0]['value'])
        except: fng = 50
        return quotes, fng, live_volatility, btdr_history
    except: return None, 50, 0.01, pd.DataFrame()

# --- 6. 绘图函数 (Red=Up, Green=Down) ---
def draw_kline_chart(df, live_price):
    if df.empty: return alt.Chart(pd.DataFrame()).mark_text().encode(text=alt.value("No Data")), ""
    
    # 1. 修复：安全拼接实时K线，确保 Index 是 Datetime 类型
    df = df.copy()
    if live_price > 0:
        # 获取最后一行数据作为模板
        last_row = df.iloc[-1].copy()
        last_row['Close'] = live_price
        last_row['High'] = max(last_row['High'], live_price)
        last_row['Low'] = min(last_row['Low'], live_price)
        
        # 构造新时间戳
        new_date = df.index[-1] + timedelta(days=1)
        
        # 构造单行 DataFrame 并拼接
        new_df = pd.DataFrame([last_row.values], columns=df.columns, index=[new_date])
        df = pd.concat([df, new_df])

    # 2. 计算指标
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['BOLL_MID'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['BOLL_U'] = df['BOLL_MID'] + 2 * df['STD20']
    df['BOLL_L'] = df['BOLL_MID'] - 2 * df['STD20']
    
    # 3. Legend HTML
    last = df.iloc[-1]
    # Handle NaNs
    m5 = last['MA5'] if not np.isnan(last['MA5']) else live_price
    b_mid = last['BOLL_MID'] if not np.isnan(last['BOLL_MID']) else live_price
    b_up = last['BOLL_U'] if not np.isnan(last['BOLL_U']) else live_price * 1.1
    b_low = last['BOLL_L'] if not np.isnan(last['BOLL_L']) else live_price * 0.9
    
    legend_html = f"""
    <div class="chart-legend">
        <div class="legend-item"><span class="legend-dot" style="background:#228be6;"></span><span style="color:#228be6;">MA5:</span><span class="legend-val">{m5:.2f}</span></div>
        <div class="legend-item"><span class="legend-dot" style="background:#f59f00;"></span><span style="color:#f59f00;">BOLL(Mid):</span><span class="legend-val">{b_mid:.2f}</span></div>
        <div class="legend-item"><span class="legend-dot" style="background:#868e96;"></span><span style="color:#868e96;">BOLL(Up):</span><span class="legend-val">{b_up:.2f}</span></div>
        <div class="legend-item"><span class="legend-dot" style="background:#868e96;"></span><span style="color:#868e96;">BOLL(Low):</span><span class="legend-val">{b_low:.2f}</span></div>
    </div>
    """

    # 4. Altair 绘图
    df = df.tail(80).reset_index()
    df.columns = ['T'] + list(df.columns[1:])
    
    base = alt.Chart(df).encode(x=alt.X('T:T', axis=alt.Axis(format='%m/%d', title=None)))
    
    rule = base.mark_rule().encode(y=alt.Y('Low:Q', scale=alt.Scale(zero=False)), y2='High:Q')
    bar = base.mark_bar(width=6).encode(
        y='Open:Q', y2='Close:Q',
        color=alt.condition("datum.Close >= datum.Open", alt.value("#e03131"), alt.value("#0ca678"))
    )
    
    line_5 = base.mark_line(color='#228be6', size=1.5).encode(y='MA5:Q')
    line_mid = base.mark_line(color='#f59f00', size=1.5).encode(y='BOLL_MID:Q') 
    line_bu = base.mark_line(color='#adb5bd', strokeDash=[4,2], size=1).encode(y='BOLL_U:Q')
    line_bl = base.mark_line(color='#adb5bd', strokeDash=[4,2], size=1).encode(y='BOLL_L:Q')
    
    vol = base.mark_bar(opacity=0.3).encode(
        y=alt.Y('Volume:Q', axis=alt.Axis(title='Vol', labels=False, ticks=False)),
        color=alt.condition("datum.Close >= datum.Open", alt.value("#e03131"), alt.value("#0ca678"))
    ).properties(height=60)
    
    chart = (rule + bar + line_5 + line_mid + line_bu + line_bl).properties(height=240)
    final_chart = alt.vconcat(chart, vol).resolve_scale(x='shared').interactive()
    
    return final_chart, legend_html

# --- 7. 仪表盘展示 ---
@st.fragment(run_every=15)
def show_live_dashboard():
    # 1. Define CRITICAL variables FIRST to avoid NameError
    tz_ny = pytz.timezone('America/New_York')
    now_ny = datetime.now(tz_ny).strftime('%H:%M:%S')
    drift_est = 0.0 # Default value
    badge_class = "badge-ai" # Fix for badge error
    
    # 2. Get Data
    quotes, fng_val, live_vol_btdr, btdr_hist = get_realtime_data()
    
    live_price = quotes.get('BTDR', {}).get('price', 0)
    ai_model, factors, ai_status = run_grandmaster_analytics(live_price)
    
    if not quotes: 
        st.warning("📡 建立安全连接中..."); time.sleep(1); st.rerun(); return

    btc = quotes.get('BTC-USD', {'pct': 0, 'price': 0}); qqq = quotes.get('QQQ', {'pct': 0})
    vix = quotes.get('^VIX', {'price': 20, 'pct': 0}); btdr = quotes.get('BTDR', {'price': 0})

    # VWAP Display Logic
    vwap_val = factors['vwap']
    if vwap_val == 0 or np.isnan(vwap_val): vwap_val = btdr['price']
    dist_vwap = ((btdr['price'] - vwap_val) / vwap_val) * 100
    
    # Safe drift calculation
    try:
        drift_est = (btc['pct']/100 * factors['beta_btc'] * 0.4) + (qqq['pct']/100 * factors['beta_qqq'] * 0.4)
        if abs(dist_vwap) > 10: drift_est -= (dist_vwap/100) * 0.05
    except: drift_est = 0.0
    
    # Model Calculation
    hist_h = ai_model['ensemble_hist_h']
    hist_l = ai_model['ensemble_hist_l']
    
    p_high = btdr['price'] * (1 + hist_h + live_vol_btdr)
    p_low = btdr['price'] * (1 + hist_l - live_vol_btdr)

    act, reason, sub, score, macd_h, support_broken = get_signal_recommendation(btdr['price'], factors, p_low)

    # --- UI Rendering ---
    st.markdown(f"<div class='time-bar'>美东 {now_ny} &nbsp;|&nbsp; AI 模式: <span class='{badge_class}'>{factors['regime']}</span> &nbsp;|&nbsp; 引擎: <b>{ai_status}</b></div>", unsafe_allow_html=True)
    st.markdown(action_banner_html(act, reason, sub), unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1: st.markdown(card_html("BTC (USD)", f"${btc['price']:,.0f}", f"{btc['pct']:+.2f}%", btc['pct']), unsafe_allow_html=True)
    fng_tooltip = "0-24: 极度恐慌 (潜在买点)\n25-49: 恐慌\n50-74: 贪婪\n75-100: 极度贪婪 (风险较高)"
    with c2: st.markdown(card_html("恐慌指数 (F&G)", f"{fng_val}", None, 0, tooltip_text=fng_tooltip), unsafe_allow_html=True)
    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    
    st.caption("⚒️ 矿股板块 Beta (Correlation Top 5)")
    cols = st.columns(5)
    top_peers = ai_model.get("top_peers", ["MARA", "RIOT", "CLSK", "CORZ", "IREN"])
    for i, p in enumerate(top_peers):
        data = quotes.get(p, {'pct': 0, 'price': 0, 'volume': 0})
        shares_m = MINER_SHARES.get(p, 200)
        turnover_rate = (data['volume'] / (shares_m * 1000000)) * 100
        cols[i].markdown(miner_card_html(p, data['price'], data['pct'], turnover_rate), unsafe_allow_html=True)
            
    st.markdown("---")
    
    st.markdown("### 📈 BTDR K-Line (Daily + BOLL)")
    if not btdr_hist.empty:
        chart_k, legend_html = draw_kline_chart(btdr_hist, btdr['price'])
        st.markdown(legend_html, unsafe_allow_html=True)
        st.altair_chart(chart_k, use_container_width=True)
    else:
        st.info("Market data initializing for chart...")
        
    c3, c4, c5 = st.columns(3)
    status_tag = f"<span class='status-dot {btdr['css']}'></span> <span style='font-size:0.6rem; color:#999'>{btdr['tag']}</span>"
    with c3: st.markdown(card_html("BTDR 现价", f"${btdr['price']:.2f}", f"{btdr['pct']:+.2f}%", btdr['pct'], status_tag), unsafe_allow_html=True)
    open_label = "今日开盘" if btdr['is_open_today'] else "预计开盘/昨收"
    open_extra = "" if btdr['is_open_today'] else "(Pending)"
    with c4: st.markdown(card_html(open_label, f"${btdr['open']:.2f}", None, 0, open_extra), unsafe_allow_html=True)
    with c5: st.markdown(card_html("机构成本 (VWAP)", f"${vwap_val:.2f}", f"{dist_vwap:+.1f}%", dist_vwap), unsafe_allow_html=True)

    # --- Scenario Analysis (Replaces Tickets) ---
    open_p = quotes['open'] if quotes['open'] > 0 else quotes['prev']
    atr_val = factors['atr'] if not np.isnan(factors['atr']) else live_price * 0.05
    st.markdown(scenario_html(open_p, atr_val), unsafe_allow_html=True)

    st.markdown(f"""
    <div style="font-size:0.7rem; color:#888; margin-bottom:2px; display:flex; justify-content:space-between;">
        <span>🟦 Kalman ({w_kalman:.0%})</span><span>🟨 History ({w_hist:.0%})</span><span>🟥 Momentum ({w_mom:.0%})</span><span>🟪 AI Volatility ({w_ai:.0%})</span>
    </div>
    <div class="ensemble-bar">
        <div class="bar-kalman" style="width: {w_kalman*100}%"></div><div class="bar-hist" style="width: {w_hist*100}%"></div><div class="bar-mom" style="width: {w_mom*100}%"></div><div class="bar-ai" style="width: {w_ai*100}%"></div>
    </div><div style="margin-bottom:10px;"></div>""", unsafe_allow_html=True)
    
    col_h, col_l = st.columns(2)
    h_bg = "#e6fcf5" if btdr['price'] < p_high else "#0ca678"; h_txt = "#087f5b" if btdr['price'] < p_high else "#ffffff"
    l_bg = "#fff5f5" if btdr['price'] > p_low else "#e03131"; l_txt = support_label_color
    
    with col_h: st.markdown(f"""<div class="pred-container-wrapper"><div class="pred-box" style="background-color: {h_bg}; color: {h_txt}; border: 1px solid #c3fae8;"><div style="font-size: 0.8rem; opacity: 0.8;">理论阻力 (High)</div><div style="font-size: 1.5rem; font-weight: bold;">${p_high:.2f}</div></div></div>""", unsafe_allow_html=True)
    with col_l: st.markdown(f"""<div class="pred-container-wrapper"><div class="pred-box" style="background-color: {l_bg}; color: {l_txt}; border: 1px solid #ffc9c9;"><div style="font-size: 0.8rem; opacity: 0.8;">理论支撑 (Low)</div><div style="font-size: 1.5rem; font-weight: bold;">{support_label_text}</div></div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔬 核心指标矩阵 (Technical Matrix)")
    rsi_val = factors['rsi']; rsi_status = "O/B (>70)" if rsi_val > 70 else ("O/S (<30)" if rsi_val < 30 else "Neutral")
    macd_val = factors['macd']; macd_sig = factors['macd_sig']; macd_delta = macd_val - macd_sig
    macd_txt = "Bull Cross" if macd_delta > 0 else "Bear Cross"
    
    range_boll = factors['boll_u'] - factors['boll_l']
    if range_boll <= 0: range_boll = 0.01
    boll_pct = (btdr['price'] - factors['boll_l']) / range_boll
    boll_txt = "Low Band" if boll_pct < 0.2 else ("High Band" if boll_pct > 0.8 else "Mid Band")
    
    mi1, mi2, mi3, mi4 = st.columns(4)
    with mi1: st.markdown(factor_html("RSI (14d)", f"{rsi_val:.0f}", rsi_status, 0 if 30<rsi_val<70 else (-1 if rsi_val>70 else 1), "强弱指标，>70超买，<30超卖。"), unsafe_allow_html=True)
    with mi2: st.markdown(factor_html("MACD (Trend)", f"{macd_delta:.3f}", macd_txt, 1 if macd_delta>0 else -1, "Diff与Signal之差，正数为多头趋势。"), unsafe_allow_html=True)
    with mi3: st.markdown(factor_html("BOLL (Pos)", f"{boll_pct*100:.0f}%", boll_txt, 1 if boll_pct<0.2 else (-1 if boll_pct>0.8 else 0), "价格在布林带中的位置百分比。"), unsafe_allow_html=True)
    with mi4: st.markdown(factor_html("Hurst Exp", f"{factors['hurst']:.2f}", "Fractal", 0, "分形维数：<0.5均值回归，>0.5趋势。"), unsafe_allow_html=True)
    
    st.markdown("### ☁️ 概率推演 (AI Probability)")
    clean_p_low = max(0.01, p_low)
    clean_p_high = max(clean_p_low * 1.01, p_high)
    
    x_axis = np.linspace(clean_p_low * 0.95, clean_p_high * 1.05, 100)
    pdf_high = norm.pdf(x_axis, clean_p_high, live_vol_btdr * btdr['price'])
    pdf_low = norm.pdf(x_axis, clean_p_low, live_vol_btdr * btdr['price'])
    pdf_data = pd.DataFrame({'Price': x_axis, 'Resistance Probability': pdf_high, 'Support Probability': pdf_low}).melt('Price', var_name='Type', value_name='Density')
    
    chart_pdf = alt.Chart(pdf_data).mark_area(opacity=0.3).encode(
        x=alt.X('Price', title='Price Levels', scale=alt.Scale(zero=False)),
        y='Density',
        color=alt.Color('Type', scale=alt.Scale(domain=['Resistance Probability', 'Support Probability'], range=['#d6336c', '#0ca678']))
    ).properties(height=220)
    st.altair_chart(chart_pdf, use_container_width=True)
    
    st.caption(f"AI Engine: v13.4 Scenario-Mod | Score: {score:.1f} | Signal: {act}")

st.markdown("### ⚡ BTDR 领航员 v13.4 Scenario-Mod")
show_live_dashboard()
