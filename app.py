import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import pytz

# --- 1. 页面配置 ---
st.set_page_config(page_title="BTDR Pilot v7.8", layout="centered")

# CSS: 强制锁定高度，防止塌陷抖动
st.markdown("""
    <style>
    /* 隐藏顶部 */
    .stApp > header { display: none; }
    .stApp { margin-top: -30px; background-color: #ffffff; }
    div[data-testid="stStatusWidget"] { visibility: hidden; }
    
    /* 字体优化 */
    h1, h2, h3, div, p, span { 
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important; 
        color: #212529 !important;
    }
    
    /* 核心卡片样式 - 强制固定高度与宽度，防止重绘时跳动 */
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        height: 95px !important;       /* 强制高度 */
        min-height: 95px !important;   /* 最小高度 */
        display: flex; 
        flex-direction: column; 
        justify-content: center;
        padding: 0 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        overflow: hidden;
    }
    
    .metric-label { font-size: 0.75rem; color: #888; margin-bottom: 2px; height: 16px; overflow: hidden; white-space: nowrap;}
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #212529; line-height: 1.2; height: 35px; overflow: hidden;}
    .metric-delta { font-size: 0.9rem; font-weight: 600; margin-top: 2px; height: 18px; overflow: hidden;}
    
    .color-up { color: #0ca678; } .color-down { color: #d6336c; }
    
    /* 预测框样式 */
    .pred-container-wrapper { height: 110px; width: 100%; display: block; }
    .pred-box { 
        padding: 0 10px; border-radius: 12px; text-align: center; 
        height: 110px !important; /* 强制高度 */
        display: flex; flex-direction: column; justify-content: center; 
    }
    
    /* 状态点 */
    .status-dot { height: 6px; width: 6px; border-radius: 50%; display: inline-block; margin-left: 6px; margin-bottom: 2px;}
    .dot-pre { background-color: #f59f00; } .dot-reg { background-color: #0ca678; } 
    .dot-post { background-color: #1c7ed6; } .dot-closed { background-color: #adb5bd; }
    
    .time-bar { 
        font-size: 0.75rem; color: #999; text-align: center; 
        margin-bottom: 20px; padding: 6px; background: #fafafa; border-radius: 6px;
        height: 30px; line-height: 18px; /* 固定高度 */
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 状态初始化 (防止首次加载闪烁) ---
if 'last_quotes' not in st.session_state:
    st.session_state['last_quotes'] = None
if 'last_model' not in st.session_state:
    st.session_state['last_model'] = None
if 'last_fng' not in st.session_state:
    st.session_state['last_fng'] = 50

# --- 3. 辅助函数 ---
def card_html(label, value_str, delta_str=None, delta_val=0, extra_tag=""):
    delta_html = "&nbsp;" # 默认占位符，防止高度塌陷
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

# --- 4. 核心逻辑 (分离计算) ---
# 使用 ttl=0 但配合 session_state 实现无缝切换
@st.cache_resource
def get_ai_model():
    """模型计算逻辑"""
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

def fetch_data_silent
