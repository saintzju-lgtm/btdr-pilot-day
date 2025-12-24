import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ---------------------- 全局配置 ----------------------
st.set_page_config(
    page_title="BTDR 综合分析平台",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 固定模拟数据种子（保证数据稳定）
np.random.seed(42)

# ---------------------- 静态模拟数据（无外部请求） ----------------------
def get_static_stock_data(period_days=30):
    """生成纯静态模拟股价数据（无任何外部请求）"""
    dates = pd.date_range(end=datetime.now(), periods=period_days, freq='D')
    stock_data = pd.DataFrame({
        "Date": dates.date,
        "Open": np.random.uniform(10, 12, period_days),
        "High": np.random.uniform(10.5, 12.5, period_days),
        "Low": np.random.uniform(9.5, 11.5, period_days),
        "Close": np.random.uniform(10, 12, period_days),
        "Volume": np.random.randint(1000000, 5000000, period_days)
    })
    # 计算均线和VWAP（纯本地）
    stock_data["MA10"] = stock_data["Close"].rolling(window=10).mean()
    stock_data["MA20"] = stock_data["Close"].rolling(window=20).mean()
    stock_data["CumVol"] = stock_data["Volume"].cumsum()
    stock_data["CumVolPrice"] = (stock_data["Close"] * stock_data["Volume"]).cumsum()
    stock_data["VWAP"] = stock_data["CumVolPrice"] / (stock_data["CumVol"] + 1e-8)
    return stock_data

def get_static_fundamental_data():
    """静态财务/运营数据"""
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
        "运营趋势": {
            "月份": ["9月", "10月", "11月", "12月E", "2026-01E"],
            "算力（EH/s）": [32.1, 38.5, 45.7, 52.0, 60.0],
            "BTC产出（枚）": [312, 389, 526, 610, 720]
        },
        "核心产品": [
            {"产品": "SEALMINER A3", "状态": "量产中", "能效": "行业领先"},
            {"产品": "SEAL04芯片", "状态": "2026 Q1量产", "能效": "6-7 J/TH"}
        ]
    }

def calculate_institution_vwap(stock_data):
    """计算机构VWAP（纯本地）"""
    stock_data = stock_data.copy()
    stock_data["Institution_Vol"] = stock_data["Volume"] * 0.3
    stock_data["Institution_Price"] = stock_data["Close"] * (1 + np.random.uniform(-0.02, 0.02, len(stock_data)))
    stock_data["Cum_Institution_Vol"] = stock_data["Institution_Vol"].cumsum()
    stock_data["Cum_Institution_Value"] = (stock_data["Institution_Price"] * stock_data["Institution_Vol"]).cumsum()
    stock_data["Institution_VWAP"] = stock_data["Cum_Institution_Value"] / (stock_data["Cum_Institution_Vol"] + 1e-8)
    return stock_data[["Date", "Institution_VWAP"]]

def simulate_筹码峰(stock_data):
    """模拟筹码峰（极简版）"""
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

# ---------------------- 侧边栏导航 ----------------------
st.sidebar.title("📊 BTDR 分析导航")
menu_option = st.sidebar.radio(
    "选择功能模块",
    ["核心数据总览", "股价&VWAP分析", "筹码峰联动", "投资工具", "财务&运营数据", "风险提示"]
)

# ---------------------- 核心数据总览（无报错版） ----------------------
if menu_option == "核心数据总览":
    st.title("BTDR 核心数据总览")
    st.divider()
    
    # 静态数据
    stock_data = get_static_stock_data(30)
    latest = stock_data.iloc[-1]
    institution_vwap = calculate_institution_vwap(stock_data).iloc[-1]["Institution_VWAP"]
    fundamental = get_static_fundamental_data()
    
    # 核心指标卡片
    col1, col2, col3 = st.columns(3)
    with col1:
        delta = latest["Close"] - latest["Open"]
        st.metric("当前股价", f"${latest['Close']:.2f}", f"{delta:.2f} ({delta/latest['Open']*100:.2f}%)")
    with col2:
        delta_vwap = latest["Close"] - institution_vwap
        st.metric("机构VWAP（30日）", f"${institution_vwap:.2f}", f"{delta_vwap:.2f} ({delta_vwap/institution_vwap*100:.2f}%)")
    with col3:
        st.metric("市值", "$26.24亿", help="2025-12-23更新")
    
    # 关键指标速览
    st.subheader("关键指标速览")
    col4, col5 = st.columns(2)
    with col4:
        st.write("📈 财务指标")
        st.dataframe(pd.DataFrame(fundamental["财务指标"]), use_container_width=True)
    with col5:
        st.write("⚙️ 运营指标")
        st.dataframe(pd.DataFrame(fundamental["运营指标"]), use_container_width=True)
    
    # 极简股价走势（无副轴/复杂参数）
    st.subheader("近30日股价走势")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data["Date"], y=stock_data["Close"], name="股价", line_color="#1f77b4"))
    fig.add_trace(go.Scatter(x=stock_data["Date"], y=stock_data["MA10"], name="10日均线", line_color="#ff7f0e", line_dash="dash"))
    fig.update_layout(height=300, xaxis_title="日期", yaxis_title="价格（美元）", legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------- 股价&VWAP分析（极简版，无副轴） ----------------------
elif menu_option == "股价&VWAP分析":
    st.title("股价走势与VWAP深度分析")
    st.divider()
    
    # 周期选择
    period = st.selectbox("选择周期", ["1周(7天)", "1个月(30天)", "3个月(90天)"], index=1)
    days = {"1周(7天)":7, "1个月(30天)":30, "3个月(90天)":90}[period]
    stock_data = get_static_stock_data(days)
    vwap_data = calculate_institution_vwap(stock_data)
    
    # 极简图表（仅股价+VWAP，无成交量副轴）
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data["Date"], y=stock_data["Close"], name="股价", line_color="#1f77b4"))
    fig.add_trace(go.Scatter(x=stock_data["Date"], y=stock_data["MA10"], name="10日均线", line_color="#ff7f0e", line_dash="dash"))
    fig.add_trace(go.Scatter(x=vwap_data["Date"], y=vwap_data["Institution_VWAP"], name="机构VWAP", line_color="#9467bd"))
    fig.update_layout(height=400, xaxis_title="日期", yaxis_title="价格（美元）", legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    
    # 成交量单独展示（避免副轴报错）
    st.subheader(f"{period}成交量")
    fig_vol = go.Figure(go.Bar(x=stock_data["Date"], y=stock_data["Volume"]/1e6, marker_color="#2ca02c"))
    fig_vol.update_layout(height=200, xaxis_title="日期", yaxis_title="成交量（百万股）")
    st.plotly_chart(fig_vol, use_container_width=True)
    
    # 分析结论
    latest_price = stock_data.iloc[-1]["Close"]
    latest_vwap = vwap_data.iloc[-1]["Institution_VWAP"]
    if latest_price > latest_vwap:
        st.success("✅ 股价高于机构VWAP，短期强势")
    else:
        st.warning("⚠️ 股价低于机构VWAP，短期弱势")

# ---------------------- 筹码峰联动（极简版） ----------------------
elif menu_option == "筹码峰联动":
    st.title("筹码峰与机构VWAP联动分析")
    st.divider()
    
    # 周期选择
    period = st.slider("分析周期（交易日）", 10, 60, 30, 5)
    stock_data = get_static_stock_data(period)
    chip_data = simulate_筹码峰(stock_data)
    vwap_data = calculate_institution_vwap(stock_data)
    
    latest_price = stock_data.iloc[-1]["Close"]
    latest_vwap = vwap_data.iloc[-1]["Institution_VWAP"]
    peak_price = chip_data.loc[chip_data["筹码占比"].idxmax(), "价格"]
    
    # 双图联动（极简版）
    col1, col2 = st.columns([1,2])
    with col1:
        st.subheader("筹码分布")
        fig_chip = go.Figure(go.Bar(y=chip_data["价格"], x=chip_data["筹码占比"], marker_color="#ff7f0e"))
        fig_chip.add_hline(y=latest_price, line_dash="dash", line_color="red")
        fig_chip.add_hline(y=latest_vwap, line_dash="dash", line_color="blue")
        fig_chip.update_layout(height=400, xaxis_title="筹码占比(%)", yaxis_title="价格(美元)")
        st.plotly_chart(fig_chip, use_container_width=True)
        st.write(f"📌 筹码主峰：${peak_price:.2f} | 机构VWAP：${latest_vwap:.2f}")
    
    with col2:
        st.subheader("股价+VWAP+筹码主峰")
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=stock_data["Date"], y=stock_data["Close"], name="股价"))
        fig_price.add_trace(go.Scatter(x=vwap_data["Date"], y=vwap_data["Institution_VWAP"], name="机构VWAP"))
        fig_price.add_hline(y=peak_price, line_dash="dash", line_color="orange")
        fig_price.update_layout(height=400, xaxis_title="日期", yaxis_title="价格(美元)", legend=dict(orientation="h"))
        st.plotly_chart(fig_price, use_container_width=True)

# ---------------------- 投资工具（纯静态） ----------------------
elif menu_option == "投资工具":
    st.title("投资决策辅助工具")
    st.divider()
    
    # 成本测算
    st.subheader("💰 持仓成本测算")
    with st.form("cost_calc"):
        price = st.number_input("持仓价格(美元)", 10.0, 13.0, 11.0, 0.1)
        num = st.number_input("持仓数量(股)", 100, 10000, 1000, 100)
        fee = st.number_input("手续费率(%)", 0.01, 1.0, 0.1, 0.01)
        if st.form_submit_button("计算"):
            latest = get_static_stock_data(30).iloc[-1]["Close"]
            vwap = calculate_institution_vwap(get_static_stock_data(30)).iloc[-1]["Institution_VWAP"]
            profit = (latest - price) * num - (price * num * fee/100)
            diff = (price - vwap)/vwap*100
            
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("浮盈/浮亏", f"${profit:.2f}")
            with col2: st.metric("与机构成本价差", f"{diff:.2f}%")
            with col3: st.metric("当前股价", f"${latest:.2f}")
    
    # 情景模拟
    st.subheader("📊 行情情景模拟")
    btc_change = st.selectbox("BTC价格变动", ["-20%", "-10%", "0%", "+10%", "+20%"])
    prod = st.selectbox("SEAL04量产进度", ["延期1个月", "如期量产", "提前量产"])
    if st.button("生成模拟结果"):
        base = get_static_stock_data(30).iloc[-1]["Close"]
        impact = float(btc_change.strip("%")) * 0.5 + (3 if prod=="提前量产" else (-3 if prod=="延期1个月" else 0))
        st.metric("模拟股价", f"${base*(1+impact/100):.2f}", f"{impact:.1f}%")

# ---------------------- 财务&运营数据（极简图表） ----------------------
elif menu_option == "财务&运营数据":
    st.title("财务与运营数据详情")
    st.divider()
    
    fundamental = get_static_fundamental_data()
    tab1, tab2, tab3 = st.tabs(["财务指标", "运营指标", "核心产品"])
    
    with tab1:
        st.dataframe(pd.DataFrame(fundamental["财务指标"]), use_container_width=True)
        st.write("💡 Q3净亏损含非现金衍生品损失，核心业务已EBITDA转正")
    
    with tab2:
        st.dataframe(pd.DataFrame(fundamental["运营指标"]), use_container_width=True)
        # 极简运营趋势图（无副轴，分开展示）
        st.subheader("算力趋势（模拟）")
        fig_power = go.Figure(go.Bar(x=fundamental["运营趋势"]["月份"], y=fundamental["运营趋势"]["算力（EH/s）"]))
        fig_power.update_layout(height=250)
        st.plotly_chart(fig_power, use_container_width=True)
        
        st.subheader("BTC产出趋势（模拟）")
        fig_btc = go.Figure(go.Scatter(x=fundamental["运营趋势"]["月份"], y=fundamental["运营趋势"]["BTC产出（枚）"], line_color="#ff7f0e"))
        fig_btc.update_layout(height=250)
        st.plotly_chart(fig_btc, use_container_width=True)
    
    with tab3:
        st.dataframe(pd.DataFrame(fundamental["核心产品"]), use_container_width=True)

# ---------------------- 风险提示 ----------------------
elif menu_option == "风险提示":
    st.title("风险提示与免责声明")
    st.divider()
    
    st.warning("""
    ### 🔴 主要风险
    1. BTC价格波动直接影响挖矿收益；
    2. SEAL04芯片量产进度/良率不及预期；
    3. 加密货币/AI算力监管政策变化；
    4. 公司仍处于亏损状态，盈利转化存不确定性；
    5. 小盘股股价波动性高。
    """)
    
    st.info("""
    ### 📝 免责声明
    本页面数据为模拟/公开信息整理，仅作参考，不构成投资建议。
    投资有风险，入市需谨慎。
    """)
    
    # 反馈表单
    with st.form("feedback"):
        st.text_area("功能建议/问题")
        if st.form_submit_button("提交"):
            st.success("感谢反馈！")

# ---------------------- 页脚 ----------------------
st.divider()
st.write(f"📅 更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 🔧 数据来源：静态模拟（规避API限流）")
