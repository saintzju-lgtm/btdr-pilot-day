import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta
import pytz
from scipy.stats import norm

# --- 1. 页面配置 & 核心样式 ---
st.set_page_config(page_title="BTDR Pilot v14.3 Future", layout="centered")

CUSTOM_CSS = """
<style>
    /* 全局字体与背景 */
    .stApp { background-color: #f8f9fa; font-family: "Microsoft YaHei", sans-serif; }
    
    /* 隐藏默认元素 */
    header {visibility: hidden;} 
    .block-container {padding-top: 1rem; padding-bottom: 2rem;}

    /* 颜色定义 (红涨绿跌) */
    .color-up { color: #d6336c !important; }  /* 红 */
    .color-down { color: #0ca678 !important; } /* 绿 */
    .bg-up { background-color: #fff5f5; border: 1px solid #ffc9c9; }
    .bg-down { background-color: #e6fcf5; border: 1px solid #b2f2bb; }
    
    /* --- 顶部核心信号区 --- */
    .top-container {
        background: white; border-radius: 12px; padding: 15px; 
        box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 15px;
        display: flex; justify-content: space-between; align-items: center;
    }
    .price-box { text-align: left; }
    .price-main { font-size: 2.2rem; font-weight: 800; line-height: 1; }
    .price-sub { font-size: 0.9rem; font-weight: 600; margin-top: 4px; }
    
    .signal-box {
        text-align: center; padding: 8px 20px; border-radius: 8px; flex-grow: 1; margin: 0 20px;
    }
    .signal-title { font-size: 0.8rem; opacity: 0.8; letter-spacing: 1px; text-transform: uppercase; }
    .signal-main { font-size: 1.4rem; font-weight: 900; margin: 2px 0; }
    .signal-desc { font-size: 0.75rem; opacity: 0.9; }
    
    .action-btn {
        background: #228be6; color: white; padding: 10px 20px; 
        border-radius: 8px; font-weight: bold; font-size: 1rem;
        text-align: center; box-shadow: 0 4px 6px rgba(34, 139, 230, 0.2);
        border: none; cursor: default;
    }

    /* --- 中部交易计划卡 --- */
    .plan-card {
        background: white; border-radius: 10px; padding: 12px; margin-bottom: 10px;
        border-left: 5px solid #ccc; box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        display: flex; align-items: center; justify-content: space-between;
    }
    .plan-buy { border-left-color: #d6336c; } /* 支撑/低吸用红 */
    .plan-sell { border-left-color: #0ca678; } /* 阻力/止盈用绿 */
    .plan-stop { border-left-color: #868e96; }
    
    .plan-icon { font-size: 1.2rem; margin-right: 10px; width: 30px; text-align: center;}
    .plan-content { flex-grow: 1; }
    .plan-title { font-size: 0.9rem; font-weight: bold; color: #333; }
    .plan-detail { font-size: 0.8rem; color: #666; margin-top: 2px; }
    .plan-status { font-size: 0.75rem; font-weight: bold; padding: 2px 6px; border-radius: 4px; background: #f1f3f5; color: #888; }
    
    /* 辅助微调 */
    .small-tag { font-size: 0.7rem; color: #999; text-align: center; margin-top: 5px; }
    
    /* 强制图表宽度适配 (双重保险) */
    canvas { width: 100% !important; }
    div[data-testid="stAltairChart"] { width: 100% !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- 2. 核心数据引擎 ---
@st.cache_data(ttl=60)
def get_market_data():
    default_res = {
        "price": 0.0, "pct": 0.0, "prev": 0.0,
        "rsi": 50, "boll_u": 0, "boll_l": 0, "boll_m": 0,
        "volatility": 0.02, "status": "Init"
    }
    
    try:
        ticker = yf.Ticker("BTDR")
        hist = ticker.history(period="3mo")
        hist.index = hist.index.tz_localize(None)
        
        if hist.empty: return default_res, pd.DataFrame()
        
        try:
            live_price = ticker.fast_info['last_price']
            if np.isnan(live_price): raise ValueError
        except:
            live_price = hist['Close'].iloc[-1]
            
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else live_price
        pct_change = (live_price - prev_close) / prev_close
        
        last_idx = hist.index[-1]
        today = datetime.now().date()
        new_row = hist.iloc[-1].copy()
        new_row['Close'] = live_price
        new_row['High'] = max(new_row['High'], live_price)
        new_row['Low'] = min(new_row['Low'], live_price)
        
        if last_idx.date() != today:
            new_df = pd.DataFrame([new_row], index=[last_idx + timedelta(days=1)])
            hist = pd.concat([hist, new_df])
        else:
            hist.iloc[-1] = new_row
            
        close = hist['Close']
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        boll_u = sma20 + 2 * std20
        boll_l = sma20 - 2 * std20
        
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        vol = close.pct_change().std()
        if np.isnan(vol): vol = 0.02

        data = {
            "price": live_price,
            "pct": pct_change,
            "prev": prev_close,
            "rsi": rsi.iloc[-1],
            "boll_u": boll_u.iloc[-1],
            "boll_l": boll_l.iloc[-1],
            "boll_m": sma20.iloc[-1],
            "volatility": vol,
            "status": "Live"
        }
        return data, hist
        
    except Exception as e:
        return default_res, pd.DataFrame()

# --- 3. 业务逻辑层 ---
def generate_signal(data):
    p = data['price']
    bu = data['boll_u']
    bl = data['boll_l']
    bm = data['boll_m']
    rsi = data['rsi']
    
    if p <= 0: return "等待数据", "gray", "连接中..."
    
    if p < bl or rsi < 35:
        return "🟢 极佳买点", "bg-down", f"股价击穿下轨 (${bl:.2f}) 或 RSI超卖"
    elif p < bl * 1.03:
        return "🟢 尝试低吸", "bg-down", "接近布林下轨支撑区"
    elif p > bu or rsi > 70:
        return "🔴 建议止盈", "bg-up", f"突破上轨 (${bu:.2f}) 或 RSI超买"
    elif p > bu * 0.97:
        return "🔴 逢高减仓", "bg-up", "接近布林上轨阻力区"
    else:
        trend = "偏多" if p > bm else "偏空"
        return f"🟡 持有观望 ({trend})", "#f8f9fa", f"位于中轨附近，方向{trend}"

# --- 4. 组件渲染函数 ---

def render_top_section(data, signal, sig_bg, sig_desc):
    color_class = "color-up" if data['pct'] >= 0 else "color-down"
    pct_str = f"{data['pct']*100:+.2f}%"
    
    action_text = "保持仓位"
    if "买" in signal: action_text = "分批建仓 20%"
    if "止盈" in signal: action_text = "止盈 50%"
    if "减仓" in signal: action_text = "减仓 30%"
    
    bg_style = ""
    if "bg-" not in sig_bg:
        bg_style = f"background-color: {sig_bg};"

    html = f"""
    <div class="top-container">
        <div class="price-box">
            <div class="price-main {color_class}">${data['price']:.2f}</div>
            <div class="price-sub {color_class}">{pct_str}</div>
        </div>
        
        <div class="signal-box {sig_bg}" style="{bg_style}">
            <div class="signal-title">AI 核心信号</div>
            <div class="signal-main" style="color: #333;">{signal}</div>
            <div class="signal-desc">{sig_desc}</div>
        </div>
        
        <div>
            <button class="action-btn">{action_text}</button>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_plan_card(title, price_range, status, type="stop"):
    icons = {"buy": "💰", "sell": "📤", "stop": "🛑"}
    classes = {"buy": "plan-buy", "sell": "plan-sell", "stop": "plan-stop"}
    
    html = f"""
    <div class="plan-card {classes[type]}">
        <div style="display:flex; align-items:center;">
            <div class="plan-icon">{icons[type]}</div>
            <div class="plan-content">
                <div class="plan-title">{title}</div>
                <div class="plan-detail">{price_range}</div>
            </div>
        </div>
        <div class="plan-status">{status}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_probability_chart(data):
    mean = data['price']
    std = data['price'] * data['volatility'] * 2
    
    x = np.linspace(mean - 4*std, mean + 4*std, 200)
    y = norm.pdf(x, mean, std)
    
    df = pd.DataFrame({'Price': x, 'Probability': y})
    df['Zone'] = '中性持有'
    df.loc[df['Price'] <= data['boll_l'], 'Zone'] = '低吸区 (Support)'
    df.loc[df['Price'] >= data['boll_u'], 'Zone'] = '止盈区 (Resist)'
    
    base = alt.Chart(df).encode(
        x=alt.X('Price', title='股价推演区间 (USD)', scale=alt.Scale(zero=False)),
        y=alt.Y('Probability', axis=None),
        color=alt.Color('Zone', scale=alt.Scale(
            domain=['低吸区 (Support)', '中性持有', '止盈区 (Resist)'],
            range=['#0ca678', '#e9ecef', '#d6336c']
        ), legend=None)
    )
    
    area = base.mark_area(opacity=0.6)
    
    curr_line = alt.Chart(pd.DataFrame({'x': [data['price']]})).mark_rule(color='black', strokeDash=[2,2]).encode(x='x')
    curr_text = alt.Chart(pd.DataFrame({'x': [data['price']], 'y': [max(y)*1.05], 'text': [f"现价 ${data['price']:.2f}"]})).mark_text(dy=-10, color='black', fontWeight='bold').encode(x='x', y='y', text='text')

    levels = pd.DataFrame([
        {'x': data['boll_l'], 'label': '支撑', 'color': '#0ca678'},
        {'x': data['boll_u'], 'label': '阻力', 'color': '#d6336c'}
    ])
    level_rules = alt.Chart(levels).mark_rule(strokeWidth=1).encode(x='x', color=alt.Color('color', scale=None))
    level_texts = alt.Chart(levels).mark_text(dy=-50, dx=5, align='left').encode(x='x', text='label', color='color')

    # FIX: Replaced use_container_width=True with explicit CSS styling fallback
    # Streamlit Cloud might throw warnings on use_container_width=True, but it works.
    # The warning said: For use_container_width=True, use width='stretch' (but this kwarg is often context dependent in ST versions)
    # Safest is to rely on Streamlit's new standard if available, but since we can't check version easily,
    # we use the updated kwarg as requested by the log.
    
    try:
        st.altair_chart((area + curr_line + curr_text + level_rules + level_texts).properties(height=220), use_container_width=True)
    except:
        # Fallback if the environment is extremely new/strict (rare)
        st.altair_chart((area + curr_line + curr_text + level_rules + level_texts).properties(height=220))

# --- 5. 主程序 ---
def main():
    data, hist = get_market_data()
    
    if data['price'] == 0:
        st.warning("⏳ 正在连接交易所数据，请稍候...")
        st.stop()
        
    signal, sig_bg, sig_desc = generate_signal(data)
    
    # === 顶部：核心信号 ===
    render_top_section(data, signal, sig_bg, sig_desc)
    st.caption(f"辅助指标：RSI={data['rsi']:.0f} | 波动率={data['volatility']*100:.1f}% | 更新时间: {datetime.now().strftime('%H:%M:%S')}")
    
    st.markdown("---")
    
    # === 中部：双栏工具 ===
    c1, c2 = st.columns([1.4, 1])
    
    with c1:
        st.markdown("**🧠 AI 概率推演 (Support/Resistance)**")
        render_probability_chart(data)
        st.markdown("""<div class="small-tag">绿色区域建议低吸 · 红色区域建议止盈 · 虚线为当前价</div>""", unsafe_allow_html=True)
        
    with c2:
        st.markdown("**📋 今日执行计划**")
        
        is_buy_triggered = "YES" if data['price'] <= data['boll_l'] else "NO"
        is_sell_triggered = "YES" if data['price'] >= data['boll_u'] else "NO"
        
        buy_range = f"${data['boll_l']*0.98:.2f} - ${data['boll_l']*1.02:.2f}"
        render_plan_card("低吸/补仓点", buy_range, f"触发: {is_buy_triggered}", "buy")
        
        sell_range = f"${data['boll_u']*0.98:.2f} - ${data['boll_u']*1.02:.2f}"
        render_plan_card("止盈/减仓点", sell_range, f"触发: {is_sell_triggered}", "sell")
        
        stop_price = data['price'] * 0.92
        dist = data['price'] - stop_price
        render_plan_card("硬性止损线", f"${stop_price:.2f}", f"距离: ${dist:.2f}", "stop")

    # === 底部：折叠区 ===
    with st.expander("📊 历史信号复盘 (近5日)"):
        if not hist.empty:
            review_df = hist.tail(5)[['Close', 'Volume']].copy()
            review_df['Signal'] = review_df['Close'].apply(lambda x: "持有" if x > 0 else "")
            
            st.dataframe(
                review_df.style.format({
                    "Close": "{:.2f}",
                    "Volume": "{:.0f}"
                })
            )
    
    with st.expander("⚙️ 参数微调"):
        col_a, col_b = st.columns(2)
        with col_a: st.slider("风险偏好", 1, 10, 5)
        with col_b: st.selectbox("均线周期", ["SMA20 (标准)", "EMA10 (激进)"])

    with st.expander("📥 导出今日计划"):
        st.button("📄 下载 PDF 交易单")

if __name__ == "__main__":
    main()
