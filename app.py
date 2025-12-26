import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import pytz

# --- 1. 页面配置 ---
st.set_page_config(page_title="BTDR Pilot v7.9", layout="centered")

# CSS: 重点修复高度塌陷
st.markdown("""
    <style>
    .stApp > header { display: none; }
    .stApp { margin-top: -30px; background-color: #ffffff; }
    div[data-testid="stStatusWidget"] { visibility: hidden; }
    
    h1, h2, h3, div, p, span { 
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important; 
        color: #212529 !important;
    }
    
    /* 核心卡片样式 */
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        height: 95px !important;
        min-height: 95px !important;
        display: flex; flex-direction: column; justify-content: center;
        padding: 0 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        overflow: hidden;
    }
    .metric-label { font-size: 0.75rem; color: #888; margin-bottom: 2px; height: 16px; overflow: hidden; white-space: nowrap;}
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #212529; line-height: 1.2; height: 35px; overflow: hidden;}
    .metric-delta { font-size: 0.9rem; font-weight: 600; margin-top: 2px; height: 18px; overflow: hidden;}
    .color-up { color: #0ca678; } .color-down { color: #d6336c; }
    
    /* 预测框 */
    .pred-container-wrapper { height: 110px; width: 100%; display: block; }
    .pred-box { 
        padding: 0 10px; border-radius: 12px; text-align: center; 
        height: 110px !important; 
        display: flex; flex-direction: column; justify-content: center; 
    }
    
    .status-dot { height: 6px; width: 6px; border-radius: 50%; display: inline-block; margin-left: 6px; margin-bottom: 2px;}
    .dot-pre { background-color: #f59f00; } .dot-reg { background-color: #0ca678; } 
    .dot-post { background-color: #1c7ed6; } .dot-closed { background-color: #adb5bd; }
    
    /* --- 修复抖动的关键 CSS --- */
    /* 我们定义一个高度锁死的容器，里面放时间条 */
    .fixed-height-container {
        height: 32px;            /* 强制高度 */
        min-height: 32px;        /* 双重保险 */
        line-height: 32px;
        overflow: hidden;        /* 防止溢出 */
        margin-bottom: 20px;
        background: #fafafa;
        border-radius: 6px;
        text-align: center;
        width: 100%;
        display: flex;           /* Flex布局保证垂直居中 */
        align-items: center;
        justify-content: center;
    }
    
    .time-text {
        font-size: 0.75rem; 
        color: #999;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 状态初始化 ---
if 'last_quotes' not in st.session_state: st.session_state['last_quotes'] = None
if 'last_fng' not in st.session_state: st.session_state['last_fng'] = 50

# --- 3. 辅助函数 ---
def card_html(label, value_str, delta_str=None, delta_val=0, extra_tag=""):
    delta_html = "&nbsp;" 
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

def get_time_html(msg):
    """
    生成高度锁死的时间条 HTML
    注意外层的 div style='height:32px'，这是防止抖动的核心
    """
    return f"""
    <div class="fixed-height-container">
        <span class="time-text">{msg}</span>
    </div>
    """

# --- 4. 核心逻辑 ---
@st.cache_resource
def get_ai_model():
    default_model = {
        "high": {"intercept": 4.29, "beta_open": 0.67, "beta_btc": 0.52},
        "low":  {"intercept": -3.22, "beta_open": 0.88, "beta_btc": 0.42},
        "beta_sector": 0.25
    }
    try:
        df = yf.download("BTDR", period="65d", interval="1d", progress=False)
        if len(df) < 15: return default_model, "Low Data"
        if isinstance(df.columns, pd.MultiIndex): df = df.xs('BTDR', axis=1, level=1)
        df = df.dropna()
        df['PrevClose'] = df['Close'].shift(1)
        df = df.dropna()
        x = ((df['Open'] - df['PrevClose']) / df['PrevClose'] * 100).values
        y_high = ((df['High'] - df['PrevClose']) / df['PrevClose'] * 100).values
        y_low = ((df['Low'] - df['PrevClose']) / df['PrevClose'] * 100).values
        weights = np.exp(np.linspace(-1, 0, len(x))) 
        def w_stats(X, Y, W):
            mx, my = np.average(X, weights=W), np.average(Y, weights=W)
            num = np.sum(W * (X - mx) * (Y - my))
            den = np.sum(W * (X - mx) ** 2)
            slope = num / den if den != 0 else 0
            return slope, my - slope * mx
        bh, ih = w_stats(x, y_high, weights)
        bl, il = w_stats(x, y_low, weights)
        return {
            "high": {"intercept": 0.5*4.29 + 0.5*ih, "beta_open": 0.5*0.67 + 0.5*bh, "beta_btc": 0.52},
            "low": {"intercept": 0.5*-3.22 + 0.5*il, "beta_open": 0.5*0.88 + 0.5*bl, "beta_btc": 0.42},
            "beta_sector": 0.25
        }, "AI-Ready"
    except: return default_model, "Fallback"

def fetch_data_silent():
    tickers = "BTC-USD BTDR MARA RIOT CORZ CLSK IREN"
    try:
        daily = yf.download(tickers, period="5d", interval="1d", group_by='ticker', threads=True, progress=False)
        live = yf.download(tickers, period="1d", interval="1m", prepost=True, group_by='ticker', threads=True, progress=False)
        q = {}
        symbols = tickers.split()
        today_ny = datetime.now(pytz.timezone('America/New_York')).date()
        for sym in symbols:
            try:
                d_day = daily[sym].dropna(subset=['Close']) if sym in daily else pd.DataFrame()
                d_min = live[sym].dropna(subset=['Close']) if sym in live else pd.DataFrame()
                curr = d_min['Close'].iloc[-1] if not d_min.empty else (d_day['Close'].iloc[-1] if not d_day.empty else 0)
                prev = 1.0
                if not d_day.empty:
                    last_date = d_day.index[-1].date()
                    prev = d_day['Close'].iloc[-2] if (last_date == today_ny and len(d_day) >= 2) else d_day['Close'].iloc[-1]
                pct = ((curr - prev)/prev)*100 if prev > 0 else 0
                open_p = d_day['Open'].iloc[-1] if (not d_day.empty and d_day.index[-1].date() == today_ny) else curr
                tag = "REG" if not d_min.empty else "CLOSED"
                q[sym] = {"price": curr, "pct": pct, "prev": prev, "open": open_p, "tag": tag}
            except: q[sym] = {"price": 0, "pct": 0, "prev": 0, "open": 0, "tag": "ERR"}
        return q
    except: return None

# --- 5. 布局骨架 ---
st.markdown("### ⚡ BTDR Pilot v7.9")

# 【重点修改】这里先给一个有高度的占位符，防止第一次加载时塌陷
ph_header = st.empty()
ph_header.markdown(get_time_html("Loading..."), unsafe_allow_html=True)

c1, c2 = st.columns(2)
ph_btc = c1.empty()
ph_fng = c2.empty()

st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
st.caption("⚒️ 矿股板块 Beta")
peer_cols = st.columns(5)
ph_peers = [col.empty() for col in peer_cols]

st.markdown("---")
c3, c4 = st.columns(2)
ph_btdr_p = c3.empty()
ph_btdr_o = c4.empty()

st.markdown("### 🎯 AI 托管预测")
c5, c6 = st.columns(2)
ph_pred_h = c5.empty()
ph_pred_l = c6.empty()
ph_footer = st.empty()

# --- 6. 局部刷新逻辑 ---
@st.fragment(run_every=5)
def update_dashboard():
    # 1. 尝试静默获取数据
    new_quotes = fetch_data_silent()
    
    # 2. 状态保持：如果获取失败，使用旧数据
    if new_quotes:
        st.session_state['last_quotes'] = new_quotes
        if np.random.rand() > 0.9: 
            try: st.session_state['last_fng'] = int(requests.get("https://api.alternative.me/fng/", timeout=1).json()['data'][0]['value'])
            except: pass
            
    quotes = st.session_state['last_quotes']
    if not quotes:
        # 即使没有数据，也要显示时间条，保持高度
        tz_bj = pytz.timezone('Asia/Shanghai')
        t_bj = datetime.now(tz_bj).strftime('%H:%M:%S')
        ph_header.markdown(get_time_html(f"北京 {t_bj} | ⏳ 初始化数据流..."), unsafe_allow_html=True)
        return

    # 3. 准备展示内容
    ai_model, ai_msg = get_ai_model()
    fng = st.session_state['last_fng']
    
    # 时间计算
    tz_bj = pytz.timezone('Asia/Shanghai')
    tz_ny = pytz.timezone('America/New_York')
    t_bj = datetime.now(tz_bj).strftime('%H:%M:%S')
    t_ny = datetime.now(tz_ny).strftime('%H:%M:%S')
    
    # 【修复重点】调用 get_time_html 生成带有固定高度外壳的 HTML
    ph_header.markdown(get_time_html(f"北京 {t_bj} | 美东 {t_ny} | {ai_msg}"), unsafe_allow_html=True)

    # 渲染其他卡片 (保持不变)
    btc = quotes['BTC-USD']
    ph_btc.markdown(card_html("BTC (全时段)", f"${btc['price']:,.0f}", f"{btc['pct']:+.2f}%", btc['pct']), unsafe_allow_html=True)
    ph_fng.markdown(card_html("恐慌指数", f"{fng}", None, 0), unsafe_allow_html=True)

    peers_list = ["MARA", "RIOT", "CORZ", "CLSK", "IREN"]
    valid_pcts = []
    for i, p in enumerate(peers_list):
        if p in quotes:
            v = quotes[p]['pct']
            ph_peers[i].markdown(card_html(p, f"{v:+.1f}%", f"{v:+.1f}%", v), unsafe_allow_html=True)
            if quotes[p]['price'] > 0: valid_pcts.append(v)

    btdr = quotes['BTDR']
    b_open_pct = 0
    if btdr['price'] > 0:
        b_open_pct = ((btdr['open'] - btdr['prev'])/btdr['prev'])*100
        
    tag_html = f"<span class='status-dot dot-{btdr.get('tag','CLOSED').lower()}'></span>"
    ph_btdr_p.markdown(card_html("BTDR 实时", f"${btdr['price']:.2f}", f"{btdr['pct']:+.2f}%", btdr['pct'], tag_html), unsafe_allow_html=True)
    ph_btdr_o.markdown(card_html("计算用开盘", f"${btdr['open']:.2f}", f"{b_open_pct:+.2f}%", b_open_pct), unsafe_allow_html=True)

    sec_avg = sum(valid_pcts)/len(valid_pcts) if valid_pcts else 0
    alpha = sec_avg - btc['pct']
    sent = (fng - 50) * 0.03
    
    M = ai_model
    p_h_pct = M['high']['intercept'] + (M['high']['beta_open']*b_open_pct) + (M['high']['beta_btc']*btc['pct']) + (M['beta_sector']*alpha) + sent
    p_l_pct = M['low']['intercept'] + (M['low']['beta_open']*b_open_pct) + (M['low']['beta_btc']*btc['pct']) + (M['beta_sector']*alpha) + sent
    
    p_h = btdr['prev'] * (1 + p_h_pct/100)
    p_l = btdr['prev'] * (1 + p_l_pct/100)
    
    h_bg = "#e6fcf5" if btdr['price'] < p_h else "#0ca678"; h_c = "#087f5b" if btdr['price'] < p_h else "#fff"
    l_bg = "#fff5f5" if btdr['price'] > p_l else "#e03131"; l_c = "#c92a2a" if btdr['price'] > p_l else "#fff"
    
    ph_pred_h.markdown(f"<div class='pred-container-wrapper'><div class='pred-box' style='background:{h_bg};color:{h_c};border:1px solid #c3fae8'><div style='font-size:0.8rem;opacity:0.8'>阻力位 (High)</div><div style='font-size:1.5rem;font-weight:bold'>${p_h:.2f}</div><div style='font-size:0.75rem;opacity:0.9'>预期: {p_h_pct:+.2f}%</div></div></div>", unsafe_allow_html=True)
    ph_pred_l.markdown(f"<div class='pred-container-wrapper'><div class='pred-box' style='background:{l_bg};color:{l_c};border:1px solid #ffc9c9'><div style='font-size:0.8rem;opacity:0.8'>支撑位 (Low)</div><div style='font-size:1.5rem;font-weight:bold'>${p_l:.2f}</div><div style='font-size:0.75rem;opacity:0.9'>预期: {p_l_pct:+.2f}%</div></div></div>", unsafe_allow_html=True)

    ph_footer.caption(f"Last Upd: {t_ny} ET (Locked Layout)")

# --- 7. 启动 ---
if st.session_state.get('last_quotes'):
    update_dashboard()
update_dashboard()
