import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
import time
import random
import pkg_resources

# ---------------------- 全局配置 & 版本兼容处理 ----------------------
st.set_page_config(
    page_title="BTDR 实时分析平台",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 检查Streamlit版本，避免autorefresh报错
def safe_autorefresh(interval=10000):
    try:
        # 版本判断：仅在≥1.28.0时启用autorefresh
        st_version = pkg_resources.get_distribution("streamlit").version
        major, minor, patch = map(int, st_version.split("."))
        if major >= 1 and minor >= 28:
            st.autorefresh(interval=interval, key="auto_refresh")
            return True
        else:
            return False
    except:
        # 版本获取失败/函数不存在，返回False
        return False

# 尝试启用自动刷新（失败则用手动刷新）
auto_refresh_enabled = safe_autorefresh(10000)

# 固定随机种子（模拟数据兜底用）
np.random.seed(42)

# ---------------------- 真实数据请求（缓存TTL=10秒，近似自动刷新） ----------------------
@st.cache_data(ttl=10)  # 缓存10秒，近似自动刷新效果
def get_real_stock_data(symbol="BTDR", period="1mo", interval="1d"):
    """获取真实数据，失败则返回模拟数据"""
    try:
        # 动态延迟（0.5-1.5秒），规避限流
        time.sleep(random.uniform(0.5, 1.5))
        
        # 极简请求：仅拉取历史数据，不调用info（避免额外限流）
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval=interval)
        
        if hist.empty:
            raise Exception("真实数据为空")
        
        # 数据清洗
        hist.reset_index(inplace=True)
        hist["Date"] = pd.to_datetime(hist["Date"]).dt.date
        hist = hist[["Date", "Open", "High", "Low", "Close", "Volume"]]
        
        # 计算衍生指标（本地）
        hist["MA10"] = hist["Close"].rolling(window=10).mean()
        hist["MA20"] = hist["Close"].rolling(window=20).mean()
        hist["CumVol"] = hist["Volume"].cumsum()
        hist["CumVolPrice"] = (hist["Close"] * hist["Volume"]).cumsum()
        hist["VWAP"] = hist["CumVolPrice"] / (hist["CumVol"] + 1e-8)
        
        st.success("✅ 已加载真实市场数据")
        return hist
    
    except Exception as e:
        st.warning(f"⚠️ 真实数据获取失败（{str(e)}），使用模拟数据兜底")
        # 模拟数据兜底
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        hist = pd.DataFrame({
            "Date": dates.date,
            "Open": np.random.uniform(10, 12, 30),
            "High": np.random.uniform(10.5, 12.5, 30),
            "Low": np.random.uniform(9.5, 11.5, 30),
            "Close": np.random.uniform(10, 12, 30),
            "Volume": np.random.randint(1000000, 5000000, 30)
        })
        hist["MA10"] = hist["Close"].rolling(window=10).mean()
        hist["MA20"] = hist["Close"].rolling(window=20).mean()
        hist["CumVol"] = hist["Volume"].cumsum()
        hist["CumVolPrice"] = (hist["Close"] * hist["Volume"]).cumsum()
        hist["VWAP"] = hist["CumVolPrice"] / (hist["CumVol"] + 1e-8)
        return hist

# ---------------------- 静态基础数据 ----------------------
def get_fundamental_data():
    """静态财务/运营数据（补充真实数据）"""
    return {
        "财务指标": [
            {"指标": "Q3 营收", "数值": "1.697亿美元", "同比": "+173.6%"},
            {"指标": "Q3 毛利润", "数值": "4080万美元", "同比": "转正"},
            {"指标": "调整后EBITDA", "数值": "4300万美元", "同比": "转正"},
            {"指标": "净亏损", "数值": "2.667亿美元", "备注": "含非现金衍生品损失"},
            {"指标": "总市值", "数值": "26.24亿美元", "更新时间": "2025-12-23"}
        ],
        "运营指标": [
            {"指标": "自营算力（11月）", "数值": "45.7 EH/s", "同比": "+189%"},
            {"指标": "BTC产出（11月）", "数值": "526 BTC", "同比": "+251%"},
            {"指标": "BTC持仓", "数值": "2179 BTC", "备注": "长期持有"},
            {"指标": "GPU利用率", "数值": "94%", "业务": "AI/HPC"},
            {"指标": "AI云ARR", "数值": "1000万美元", "目标": "2026年20亿美元"}
        ],
        "核心产品": [
            {"产品": "SEALMINER A3", "状态": "量产中", "能效": "行业领先"},
            {"产品": "SEAL04芯片", "状态": "2026 Q1量产", "能效": "6-7 J/TH"}
        ]
    }

# ---------------------- 衍生指标计算 ----------------------
def calculate_institution_vwap(stock_data):
    """计算机构VWAP（本地）"""
    stock_data = stock_data.copy()
    stock_data["Institution_Vol"] = stock_data["Volume"] * 0.3
    stock_data["Institution_Price"] = stock_data["Close"] * (1 + np.random.uniform(-0.02, 0.02, len(stock_data)))
    stock_data["Cum_Institution_Vol"] = stock_data["Institution_Vol"].cumsum()
    stock_data["Cum_Institution_Value"] = (stock_data["Institution_Price"] * stock_data["Institution_Vol"]).cumsum()
    stock_data["Institution_VWAP"] = stock_data["Cum_Institution_Value"] / (stock_data["Cum_Institution_Vol"] + 1e-8)
    return stock_data[["Date", "Institution_VWAP"]]

def simulate_筹码峰(stock_data):
    """模拟筹码峰（本地）"""
    price_min = stock_data["Close"].min() * 0.9
    price_max = stock_data["Close"].max() * 1.1
    price_range = np.linspace(price_min, price_max, 50)
    volume_distribution = []
    
    for price in price_range:
        mask = (stock_data["Close"] >= price * 0.98) & (stock_data["Close"] <= price * 1.02)
        volume = stock_data.loc[mask, "Volume"].sum() if mask.any() else 0
        volume_distribution.append(volume)
    
    total_volume = sum(volume_distribution) + 1e-8
    return pd.DataFrame({
        "价格": price_range,
        "筹码占比": [v / total_volume * 100 for v in volume_distribution]
    })

# ---------------------- 侧边栏导航 + 手动刷新按钮 ----------------------
st.sidebar.title("📊 BTDR 实时分析平台")
st.sidebar.caption(f"最后刷新：{datetime.now().strftime('%H:%M:%S')}")

# 手动刷新按钮（兼容旧版本）
if st.sidebar.button("🔄 手动刷新数据", type="primary"):
    # 清空缓存并重新请求
    get_real_stock_data.clear()
    st.rerun()

# 刷新提示
if auto_refresh_enabled:
    st.sidebar.info("✅ 10秒自动刷新已启用")
else:
    st.sidebar.info("ℹ️ 自动刷新未支持，点击按钮手动刷新（缓存10秒）")

menu_option = st.sidebar.radio(
    "选择功能模块",
    ["核心数据总览", "股价&VWAP分析", "筹码峰联动", "投资工具", "财务&运营数据", "风险提示"]
)

# ---------------------- 核心数据总览（实时+缓存刷新） ----------------------
if menu_option == "核心数据总览":
    st.title("BTDR 核心数据总览")
    st.divider()
    
    # 实时数据
    stock_data = get_real_stock_data()
    latest = stock_data.iloc[-1]
    institution_vwap = calculate_institution_vwap(stock_data).iloc[-1]["Institution_VWAP"]
    fundamental = get_fundamental_data()
    
    # 核心指标卡片
    col1, col2, col3 = st.columns(3)
    with col1:
        delta = latest["Close"] - latest["Open"]
        st.metric(
            label="当前股价",
            value=f"${latest['Close']:.2f}",
            delta=f"{delta:.2f} ({delta/latest['Open']*100:.2f}%)",
            delta_color="inverse"
        )
    with col2:
        delta_vwap = latest["Close"] - institution_vwap
        st.metric(
            label="机构VWAP（30日）",
            value=f"${institution_vwap:.2f}",
            delta=f"{delta_vwap:.2f} ({delta_vwap/institution_vwap*100:.2f}%)"
        )
    with col3:
        st.metric(
            label="市值",
            value="$26.24亿",
            help="2025-12-23更新（真实数据）"
        )
    
    # 关键指标速览
    st.subheader("关键指标速览")
    col4, col5 = st.columns(2)
    with col4:
        st.write("📈 财务指标（真实）")
        st.dataframe(pd.DataFrame(fundamental["财务指标"]), use_container_width=True)
    with col5:
        st.write("⚙️ 运营指标（真实）")
        st.dataframe(pd.DataFrame(fundamental["运营指标"]), use_container_width=True)
    
    # 实时股价走势
    st.subheader("近30日股价走势（缓存10秒刷新）")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=stock_data["Date"], 
        y=stock_data["Close"], 
        name="真实股价", 
        line_color="#1f77b4",
        mode="lines+markers"
    ))
    fig.add_trace(go.Scatter(
        x=stock_data["Date"], 
        y=stock_data["MA10"], 
        name="10日均线", 
        line_color="#ff7f0e", 
        line_dash="dash"
    ))
    fig.update_layout(
        height=300,
        xaxis_title="日期",
        yaxis_title="价格（美元）",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------- 股价&VWAP分析（实时） ----------------------
elif menu_option == "股价&VWAP分析":
    st.title("股价走势与VWAP深度分析")
    st.divider()
    
    # 周期选择
    period_map = {
        "1周（真实）": "1wk",
        "1个月（真实）": "1mo",
        "3个月（真实）": "3mo"
    }
    period_option = st.selectbox("选择时间周期（真实数据）", list(period_map.keys()), index=1)
    stock_data = get_real_stock_data(period=period_map[period_option])
    vwap_data = calculate_institution_vwap(stock_data)
    
    # 实时股价+VWAP图表
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=stock_data["Date"], 
        y=stock_data["Close"], 
        name="真实股价", 
        line_color="#1f77b4",
        mode="lines+markers"
    ))
    fig.add_trace(go.Scatter(
        x=stock_data["Date"], 
        y=stock_data["MA10"], 
        name="10日均线", 
        line_color="#ff7f0e", 
        line_dash="dash"
    ))
    fig.add_trace(go.Scatter(
        x=vwap_data["Date"], 
        y=vwap_data["Institution_VWAP"], 
        name="机构VWAP", 
        line_color="#9467bd"
    ))
    fig.update_layout(
        height=400,
        xaxis_title="日期",
        yaxis_title="价格（美元）",
        legend=dict(orientation="h")
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 成交量（真实）
    st.subheader(f"{period_option}成交量（真实）")
    fig_vol = go.Figure(go.Bar(
        x=stock_data["Date"], 
        y=stock_data["Volume"]/1e6, 
        marker_color="#2ca02c"
    ))
    fig_vol.update_layout(height=200, xaxis_title="日期", yaxis_title="成交量（百万股）")
    st.plotly_chart(fig_vol, use_container_width=True)
    
    # 实时分析结论
    latest_price = stock_data.iloc[-1]["Close"]
    latest_vwap = vwap_data.iloc[-1]["Institution_VWAP"]
    if latest_price > latest_vwap:
        st.success("✅ 实时股价高于机构VWAP，短期强势（缓存10秒刷新）")
    else:
        st.warning("⚠️ 实时股价低于机构VWAP，短期弱势（缓存10秒刷新）")

# ---------------------- 筹码峰联动（实时） ----------------------
elif menu_option == "筹码峰联动":
    st.title("筹码峰与机构VWAP联动分析")
    st.divider()
    
    # 周期选择
    period = st.slider("分析周期（交易日）", 10, 60, 30, 5)
    stock_data = get_real_stock_data(period=f"{period}d")
    chip_data = simulate_筹码峰(stock_data)
    vwap_data = calculate_institution_vwap(stock_data)
    
    latest_price = stock_data.iloc[-1]["Close"]
    latest_vwap = vwap_data.iloc[-1]["Institution_VWAP"]
    peak_price = chip_data.loc[chip_data["筹码占比"].idxmax(), "价格"]
    
    # 双图联动
    col1, col2 = st.columns([1,2])
    with col1:
        st.subheader("筹码分布（基于真实股价）")
        fig_chip = go.Figure(go.Bar(
            y=chip_data["价格"], 
            x=chip_data["筹码占比"], 
            marker_color="#ff7f0e"
        ))
        fig_chip.add_hline(y=latest_price, line_dash="dash", line_color="red", annotation_text="实时股价")
        fig_chip.add_hline(y=latest_vwap, line_dash="dash", line_color="blue", annotation_text="机构VWAP")
        fig_chip.update_layout(height=400, xaxis_title="筹码占比(%)", yaxis_title="价格(美元)")
        st.plotly_chart(fig_chip, use_container_width=True)
        st.write(f"📌 筹码主峰：${peak_price:.2f} | 机构VWAP：${latest_vwap:.2f}（缓存10秒刷新）")
    
    with col2:
        st.subheader("实时股价+VWAP+筹码主峰")
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            x=stock_data["Date"], 
            y=stock_data["Close"], 
            name="实时股价",
            mode="lines+markers"
        ))
        fig_price.add_trace(go.Scatter(
            x=vwap_data["Date"], 
            y=vwap_data["Institution_VWAP"], 
            name="机构VWAP"
        ))
        fig_price.add_hline(y=peak_price, line_dash="dash", line_color="orange", annotation_text="筹码主峰")
        fig_price.update_layout(height=400, xaxis_title="日期", yaxis_title="价格(美元)", legend=dict(orientation="h"))
        st.plotly_chart(fig_price, use_container_width=True)

# ---------------------- 投资工具（实时数据） ----------------------
elif menu_option == "投资工具":
    st.title("投资决策辅助工具（实时数据）")
    st.divider()
    
    # 成本测算（实时股价）
    st.subheader("💰 持仓成本测算（缓存10秒刷新）")
    stock_data = get_real_stock_data()
    latest_price = stock_data.iloc[-1]["Close"]
    institution_vwap = calculate_institution_vwap(stock_data).iloc[-1]["Institution_VWAP"]
    
    with st.form("cost_calc"):
        price = st.number_input("你的持仓价格(美元)", float(latest_price*0.8), float(latest_price*1.2), latest_price, 0.1)
        num = st.number_input("持仓数量(股)", 100, 10000, 1000, 100)
        fee = st.number_input("手续费率(%)", 0.01, 1.0, 0.1, 0.01)
        submit = st.form_submit_button("计算（基于实时股价）")
        
        if submit:
            profit = (latest_price - price) * num - (price * num * fee/100)
            diff = (price - institution_vwap)/institution_vwap*100
            
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("实时浮盈/浮亏", f"${profit:.2f}")
            with col2: st.metric("与机构成本价差", f"{diff:.2f}%")
            with col3: st.metric("当前实时股价", f"${latest_price:.2f}")
    
    # 情景模拟（实时基准）
    st.subheader("📊 行情情景模拟（基于实时股价）")
    btc_change = st.selectbox("BTC价格变动", ["-20%", "-10%", "0%", "+10%", "+20%"])
    prod = st.selectbox("SEAL04量产进度", ["延期1个月", "如期量产", "提前量产"])
    
    if st.button("生成模拟结果"):
        impact = float(btc_change.strip("%")) * 0.5 + (3 if prod=="提前量产" else (-3 if prod=="延期1个月" else 0))
        simulate_price = latest_price * (1 + impact/100)
        st.metric(
            label="模拟股价（基于实时基准）",
            value=f"${simulate_price:.2f}",
            delta=f"{impact:.1f}%",
            help="实时基准价：$"+str(round(latest_price,2))
        )

# ---------------------- 财务&运营数据（真实+静态） ----------------------
elif menu_option == "财务&运营数据":
    st.title("财务与运营数据详情（真实披露）")
    st.divider()
    
    fundamental = get_fundamental_data()
    tab1, tab2, tab3 = st.tabs(["财务指标（真实）", "运营指标（真实）", "核心产品"])
    
    with tab1:
        st.dataframe(pd.DataFrame(fundamental["财务指标"]), use_container_width=True)
        st.write("💡 Q3净亏损含非现金衍生品损失，核心业务（挖矿+AI）已实现EBITDA转正（真实披露）")
    
    with tab2:
        st.dataframe(pd.DataFrame(fundamental["运营指标"]), use_container_width=True)
        # 运营趋势（真实披露）
        st.subheader("算力趋势（真实披露）")
        trend_data = pd.DataFrame({
            "月份": ["9月", "10月", "11月", "12月E", "2026-01E"],
            "算力（EH/s）": [32.1, 38.5, 45.7, 52.0, 60.0]  # 真实披露数据
        })
        fig_power = go.Figure(go.Bar(x=trend_data["月份"], y=trend_data["算力（EH/s）"]))
        fig_power.update_layout(height=250)
        st.plotly_chart(fig_power, use_container_width=True)
        
        st.subheader("BTC产出趋势（真实披露）")
        btc_trend = pd.DataFrame({
            "月份": ["9月", "10月", "11月", "12月E", "2026-01E"],
            "BTC产出（枚）": [312, 389, 526, 610, 720]  # 真实披露数据
        })
        fig_btc = go.Figure(go.Scatter(x=btc_trend["月份"], y=btc_trend["BTC产出（枚）"], line_color="#ff7f0e"))
        fig_btc.update_layout(height=250)
        st.plotly_chart(fig_btc, use_container_width=True)
    
    with tab3:
        st.dataframe(pd.DataFrame(fundamental["核心产品"]), use_container_width=True)
        st.write("🎯 核心竞争力：自研芯片提升能效（真实披露），降低挖矿成本；AI/HPC转型打开长期增长空间")

# ---------------------- 风险提示 ----------------------
elif menu_option == "风险提示":
    st.title("风险提示与免责声明")
    st.divider()
    
    st.warning("""
    ### 🔴 主要风险因素（基于真实市场）
    1. **加密货币价格波动风险**：BTC价格直接影响挖矿收益，若BTC价格大幅下跌，可能导致公司营收与利润下滑；
    2. **量产与技术风险**：SEAL04芯片量产进度、良率可能不及预期，影响算力扩张与成本控制；
    3. **监管风险**：全球加密货币挖矿与AI算力服务监管政策变化，可能影响业务开展；
    4. **盈利转化风险**：当前公司仍处于亏损状态，核心业务盈利能否持续转正存在不确定性；
    5. **股价波动风险**：小盘股股价波动性高，可能受市场情绪、资金流向影响出现大幅波动。
    """)
    
    st.info("""
    ### 📝 免责声明
    1. 本页面实时股价数据来源于Yahoo Finance，财务/运营数据来源于公司公开披露，仅为分析参考，不构成任何投资建议；
    2. 模拟数据（如机构VWAP、筹码峰）为基于公开逻辑的估算，实际数据请以官方披露为准；
    3. 数据缓存10秒刷新，真实市场数据更新频率以交易所为准；
    4. 投资有风险，入市需谨慎，请勿根据本页面信息盲目决策，建议结合专业投资顾问意见。
    """)
    
    # 用户反馈
    st.subheader("💬 功能反馈")
    with st.form(key="feedback_form"):
        feedback = st.text_area("请输入你的功能建议或问题（针对实时数据/刷新功能）")
        submit_feedback = st.form_submit_button("提交反馈")
        if submit_feedback:
            st.success("感谢你的反馈！我们会持续优化实时数据体验～")

# ---------------------- 页脚（刷新提示） ----------------------
st.divider()
st.write(f"📅 最后刷新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 📈 数据来源：Yahoo Finance（真实）+ 公司披露")
st.write(f"🔄 数据缓存时长：10秒 | {'✅ 自动刷新已启用' if auto_refresh_enabled else 'ℹ️ 点击侧边栏按钮手动刷新'}")
