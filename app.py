import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import pytz

# --- 1. 页面配置 ---
st.set_page_config(page_title="BTDR Pilot v7.7 (Stable)", layout="centered")

# 注意：移除了 st_autorefresh，改用原生的 @st.fragment 实现无感刷新

# CSS: 保持样式不变
st.markdown("""
    <style>
    /* 基础重置 */
    html { overflow-y: scroll; }
    .stApp > header { display: none; }
    .stApp { margin-top: -30px; background-color: #ffffff; }
    div[data-testid="stStatusWidget"] { visibility: hidden; }
    
    h1, h2, h3, div, p, span { 
        color: #212529 !important; 
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important; 
    }
    
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        height: 95px;
        padding: 0 16px;
        display: flex; flex-direction: column; justify-content: center;
        overflow: hidden; transition: transform 0.2s;
    }
    .metric-label { font-size: 0.75rem; color: #888; display: flex; align-items: center; margin-bottom: 2px; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #212529; line-height: 1.2; letter-spacing: -0.5px; }
    .metric-delta { font-size: 0.9rem; font-weight: 600; margin-top: 2px; }
    .color-up { color: #0ca678; } .color-down { color: #d6336c; } .color-neutral { color: #adb5bd; }
    
    .pred-container-wrapper { height: 110px; width: 100%; display: block; }
    .pred-box { padding: 0 10px; border-radius: 12px; text-align: center; height: 100%; display: flex; flex-direction: column; justify-content: center; }
    
    .status-dot { height: 6px; width: 6px; border-radius: 50%; display: inline-block; margin-left: 6px; margin-bottom: 2px;}
    .dot-pre { background-color: #f59f00; box-shadow: 0 0 4px #f59f00; }
    .dot-reg { background-color: #0ca678; box-shadow: 0 0 4px #0ca678; }
    .dot-post { background-color: #1c7ed6; box-shadow: 0 0 4px #1c7ed6; }
    .dot-closed { background-color: #adb5bd; }
    
    .time-bar { font-size: 0.75rem; color: #999; text-align: center; margin-bottom: 20px; padding: 6px; background: #fafafa; border-radius: 6px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 辅助函数 ---
def card_html(label, value_str, delta_str=None, delta_val=0, extra_tag=""):
    delta_html = ""
    if delta_str:
        color_class = "color-up" if delta_val >= 0 else "color-down"
        delta_html = f"<div class='metric-delta {color_class}'>{delta_str}</div>"
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label} {extra_tag}</div>
        <div class="metric-value">{value_str}</div>
        {delta_html}
    </div>
    """

# --- 3. 数据与模型逻辑 (缓存层) ---

@st.cache_resource(ttl=None) 
def get_ai_model_version(date_key):
    """AI 每天自动训练一次"""
    default_model = {
        "high": {"intercept": 4.29, "beta_open": 0.67, "beta_btc": 0.52},
        "low":  {"intercept": -3.22, "beta_open": 0.88, "beta_btc": 0.42},
        "beta_sector": 0.25
    }
    log_msg = f"Last Training: {datetime.now().strftime('%H:%M:%S')}"
    
    try:
        df = yf.download("BTDR", period="65d", interval="1d", progress=False)
        if len(df) < 20: return default_model, log_msg + " (Low Data)"
        if isinstance(df.columns, pd.MultiIndex): df = df.xs('BTDR', axis=1, level=1)
        df = df.dropna()
        df['PrevClose'] = df['Close'].shift(1)
        df = df.dropna()
        
        x = ((df['Open'] - df['PrevClose']) / df['PrevClose'] * 100).values
        y_high = ((df['High'] - df['PrevClose']) / df['PrevClose'] * 100).values
        y_low = ((df['Low'] - df['PrevClose']) / df['PrevClose'] * 100).values
        
        length = len(x)
        weights = np.exp(np.linspace(-1, 0, length)) 
        
        def weighted_stats(x_in, y_in, w_in):
            w_mean_x = np.average(x_in, weights=w_in)
            w_mean_y = np.average(y_in, weights=w_in)
            num = np.sum(w_in * (x_in - w_mean_x) * (y_in - w_mean_y))
            den = np.sum(w_in * (x_in - w_mean_x) ** 2)
            slope = num / den if den != 0 else 0
            intercept = w_mean_y - slope * w_mean_x
            return slope, intercept

        beta_h, int_h = weighted_stats(x, y_high, weights)
        beta_l, int_l = weighted_stats(x, y_low, weights)
        
        volatility = np.std(y_high[-5:]) 
        dampener = 0.9 if volatility > 5 else 1.0
        
        beta_h = np.clip(beta_h, 0.2, 1.4) * dampener
        beta_l = np.clip(beta_l, 0.3, 1.8) * dampener
        
        final_model = {
            "high": {"intercept": 0.5*4.29 + 0.5*int_h, "beta_open": 0.5*0.67 + 0.5*beta_h, "beta_btc": 0.52},
            "low": {"intercept": 0.5*-3.22 + 0.5*int_l, "beta_open": 0.5*0.88 + 0.5*beta_l, "beta_btc": 0.42},
            "beta_sector": 0.25
        }
        return final_model, log_msg + " (Success)"
    except Exception as e:
        return default_model, log_msg + f" (Err: {str(e)[:5]})"

@st.cache_data(ttl=5) # 短缓存，防止频繁请求卡顿
def get_data_v77():
    tickers_list = "BTC-USD BTDR MARA RIOT CORZ CLSK IREN"
    try:
        daily = yf.download(tickers_list, period="5d", interval="1d", group_by='ticker', threads=True, progress=False)
        live = yf.download(tickers_list, period="1d", interval="1m", prepost=True, group_by='ticker', threads=True, progress=False)
        quotes = {}
        symbols = tickers_list.split()
        today_ny = datetime.now(pytz.timezone('America/New_York')).date()
        for sym in symbols:
            try:
                df_day = daily[sym] if sym in daily else pd.DataFrame()
                if not df_day.empty: df_day = df_day.dropna(subset=['Close'])
                df_min = live[sym] if sym in live else pd.DataFrame()
                if not df_min.empty: df_min = df_min.dropna(subset=['Close'])
                
                state = "REG" if not df_min.empty else "CLOSED"
                current_price = df_min['Close'].iloc[-1] if not df_min.empty else (df_day['Close'].iloc[-1] if not df_day.empty else 0)
                
                prev_close = 1.0
                if not df_day.empty:
                    last_date = df_day.index[-1].date()
                    if last_date == today_ny:
                        if len(df_day) >= 2: prev_close = df_day['Close'].iloc[-2]
                        elif not df_day.empty: prev_close = df_day['Open'].iloc[-1]
                    else: prev_close = df_day['Close'].iloc[-1]
                
                pct = ((current_price - prev_close) / prev_close) * 100 if prev_close > 0 else 0
                open_price = df_day['Open'].iloc[-1] if not df_day.empty and df_day.index[-1].date() == today_ny else current_price
                quotes[sym] = {"price": current_price, "pct": pct, "prev": prev_close, "open": open_price, "tag": state}
            except: quotes[sym] = {"price": 0, "pct": 0, "prev": 0, "open": 0, "tag": "ERR"}
        return quotes
    except: return None

@st.cache_data(ttl=3600)
def get_fng():
    try: return int(requests.get("https://api.alternative.me/fng/", timeout=1).json()['data'][0]['value'])
    except: return 50

# --- 4. 核心：无感刷新仪表盘 ---
# 使用 @st.fragment 装饰器，实现局部刷新，杜绝全页跳动
@st.fragment(run_every=5)
def dashboard_fragment():
    
    # 1. 准备数据
    ny_now = datetime.now(pytz.timezone('America/New_York'))
    # 如果是美东下午4点后，视为新的一天，触发重新训练
    training_key = ny_now.date() if ny_now.hour >= 4 else ny_now.date() - timedelta(days=1)
    
    ai_model, ai_status = get_ai_model_version(str(training_key))
    quotes = get_data_v77()
    fng_val = get_fng()

    if not quotes:
        st.warning("📡 正在获取实时数据...")
        return

    # 2. 解包数据
    btc_chg = quotes['BTC-USD']['pct']
    btc_price = quotes['BTC-USD']['price']
    btdr = quotes['BTDR']
    
    # 时间栏
    tz_bj = pytz.timezone('Asia/Shanghai')
    now_bj = datetime.now(tz_bj).strftime('%H:%M:%S')
    now_ny = ny_now.strftime('%H:%M:%S')
    st.markdown(f"<div class='time-bar'>北京 {now_bj} | 美东 {now_ny} | 🧠 {ai_status}</div>", unsafe_allow_html=True)

    # 3. 核心指标区 (不再需要 st.empty，直接布局)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(card_html("BTC (全时段)", f"${btc_price:,.0f}", f"{btc_chg:+.2f}%", btc_chg), unsafe_allow_html=True)
    with c2:
        st.markdown(card_html("恐慌指数", f"{fng_val}", None, 0), unsafe_allow_html=True)

    # 4. 板块区
    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    st.caption("⚒️ 矿股板块 Beta")
    cols = st.columns(5)
    peers = ["MARA", "RIOT", "CORZ", "CLSK", "IREN"]
    valid_peers_pct = []
    
    for i, p in enumerate(peers):
        if p in quotes:
            val = quotes[p]['pct']
            cols[i].markdown(card_html(p, f"{val:+.1f}%", f"{val:+.1f}%", val), unsafe_allow_html=True)
            if quotes[p]['price'] > 0: valid_peers_pct.append(val)

    # 分割线
    st.markdown("---")

    # 5. BTDR 实时区
    c3, c4 = st.columns(2)
    btdr_open_pct = 0
    if btdr['price'] > 0:
        btdr_open_pct = ((btdr['open'] - btdr['prev']) / btdr['prev']) * 100
        
    state_map = {"PRE": "dot-pre", "REG": "dot-reg", "POST": "dot-post", "CLOSED": "dot-closed"}
    dot_class = state_map.get(btdr.get('tag', 'CLOSED'), 'dot-closed')
    status_tag = f"<span class='status-dot {dot_class}'></span> <span style='margin-left:2px; font-size:0.7rem;'>{btdr.get('tag', 'CLOSED')}</span>"

    with c3:
        st.markdown(card_html("BTDR 实时", f"${btdr['price']:.2f}", f"{btdr['pct']:+.2f}%", btdr['pct'], status_tag), unsafe_allow_html=True)
    with c4:
        st.markdown(card_html("计算用开盘", f"${btdr['open']:.2f}", f"{btdr_open_pct:+.2f}%", btdr_open_pct), unsafe_allow_html=True)

    # 6. 预测区
    st.markdown("### 🎯 AI 托管预测 (每日校准)")
    
    # 计算预测逻辑
    sector_avg = sum(valid_peers_pct)/len(valid_peers_pct) if valid_peers_pct else 0
    sector_alpha = sector_avg - btc_chg
    sentiment_adj = (fng_val - 50) * 0.03
    
    M = ai_model
    pred_h_pct = M['high']['intercept'] + (M['high']['beta_open']*btdr_open_pct) + (M['high']['beta_btc']*btc_chg) + (M['beta_sector']*sector_alpha) + sentiment_adj
    pred_l_pct = M['low']['intercept'] + (M['low']['beta_open']*btdr_open_pct) + (M['low']['beta_btc']*btc_chg) + (M['beta_sector']*sector_alpha) + sentiment_adj
    
    p_h = btdr['prev'] * (1 + pred_h_pct/100)
    p_l = btdr['prev'] * (1 + pred_l_pct/100)
    
    col_h, col_l = st.columns(2)
    
    h_bg = "#e6fcf5" if btdr['price'] < p_h else "#0ca678"; h_tx = "#087f5b" if btdr['price'] < p_h else "#fff"
    l_bg = "#fff5f5" if btdr['price'] > p_l else "#e03131"; l_tx = "#c92a2a" if btdr['price'] > p_l else "#fff"

    with col_h:
        st.markdown(f"""<div class='pred-container-wrapper'><div class='pred-box' style='background:{h_bg};color:{h_tx};border:1px solid #c3fae8'>
            <div style='font-size:0.8rem;opacity:0.8'>阻力位 (High)</div><div style='font-size:1.5rem;font-weight:bold'>${p_h:.2f}</div>
            <div style='font-size:0.75rem;opacity:0.9'>预期: {pred_h_pct:+.2f}%</div></div></div>""", unsafe_allow_html=True)
    
    with col_l:
        st.markdown(f"""<div class='pred-container-wrapper'><div class='pred-box' style='background:{l_bg};color:{l_tx};border:1px solid #ffc9c9'>
            <div style='font-size:0.8rem;opacity:0.8'>支撑位 (Low)</div><div style='font-size:1.5rem;font-weight:bold'>${p_l:.2f}</div>
            <div style='font-size:0.75rem;opacity:0.9'>预期: {pred_l_pct:+.2f}%</div></div></div>""", unsafe_allow_html=True)

    st.caption(f"Update: {now_ny} ET | Auto-Tuned by AI (Smart-Train)")

# --- 5. 主程序入口 ---
st.markdown("### ⚡ BTDR Pilot v7.7 (Stable)")

# 调用局部刷新函数
# 这部分代码每 5 秒自动只运行 dashboard_fragment 内部的内容
# 页面其他部分完全静止，不会跳动
dashboard_fragment()
