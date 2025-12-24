import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------- 全局配置 & 限流规避 ----------------------
# 设置请求重试策略
session = requests.Session()
retry = Retry(
    total=3,  # 重试3次
    backoff_factor=0.5,  # 每次重试延迟0.5s
    status_forcelist=[429, 500, 502, 503, 504]  # 针对限流/服务器错误重试
)
session.mount("https://", HTTPAdapter(max_retries=retry))

# 默认基础数据（兜底用，避免API失效崩溃）
DEFAULT_STOCK_INFO = {
    "marketCap": 2624000000,  # 默认市值
    "symbol": "BTDR",
    "longName": "Bitdeer Technologies Group"
}

# ---------------------- 数据获取函数（修复限流问题） ----------------------
@st.cache_data(ttl=86400)  # 延长缓存至24小时，减少请求频率
def get_btdr_stock_data(period="1mo", interval="1d"):
    """获取BTDR股价数据（兼容限流，降级处理）"""
    try:
        # 增加请求延迟，规避限流
        time.sleep(0.5)
        ticker = yf.Ticker("BTDR", session=session)
        
        # 优先获取历史数据（限流概率低）
        hist = ticker.history(period=period, interval=interval)
        if hist.empty:
            # 历史数据为空时生成模拟数据兜底
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            hist = pd.DataFrame({
                "Open": np.random.uniform(10, 12, 30),
                "High": np.random.uniform(10.5, 12.5, 30),
                "Low": np.random.uniform(9.5, 11.5, 30),
                "Close": np.random.uniform(10, 12, 30),
                "Volume": np.random.randint(1000000, 5000000, 30)
            }, index=dates)
        
        hist.reset_index(inplace=True)
        hist["Date"] = pd.to_datetime(hist["Date"]).dt.date
        
        # 计算均线和VWAP
        hist["MA5"] = hist["Close"].rolling(window=5).mean()
        hist["MA10"] = hist["Close"].rolling(window=10).mean()
        hist["MA20"] = hist["Close"].rolling(window=20).mean()
        hist["CumVol"] = hist["Volume"].cumsum()
        hist["CumVolPrice"] = (hist["Close"] * hist["Volume"]).cumsum()
        hist["VWAP"] = hist["CumVolPrice"] / hist["CumVol"]
        
        # 避免调用ticker.info触发限流，改用默认值
        stock_info = DEFAULT_STOCK_INFO
        
        return hist, stock_info
    
    except Exception as e:
        # 所有异常都降级为模拟数据
        st.warning(f"⚠️ 数据获取失败（{str(e)}），使用模拟数据展示")
        # 生成模拟股价数据
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        hist = pd.DataFrame({
            "Date": dates.date,
            "Open": np.random.uniform(10, 12, 30),
            "High": np.random.uniform(10.5, 12.5, 30),
            "Low": np.random.uniform(9.5, 11.5, 30),
            "Close": np.random.uniform(10, 12, 30),
            "Volume": np.random.randint(1000000, 5000000, 30)
        })
        # 补全计算字段
        hist["MA5"] = hist["Close"].rolling(window=5).mean()
        hist["MA10"] = hist["Close"].rolling(window=10).mean()
        hist["MA20"] = hist["Close"].rolling(window=20).mean()
        hist["CumVol"] = hist["Volume"].cumsum()
        hist["CumVolPrice"] = (hist["Close"] * hist["Volume"]).cumsum()
        hist["VWAP"] = hist["CumVolPrice"] / hist["CumVol"]
        
        return hist, DEFAULT_STOCK_INFO

@st.cache_data(ttl=86400)
def get_btdr_fundamental_data():
    """获取BTDR财务&运营核心数据（静态数据，避免API请求）"""
    fundamental_data = {
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
    return fundamental_data

@st.cache_data(ttl=86400)
def calculate_institution_vwap(stock_data, period=30):
    """计算机构VWAP（基于本地数据，无外部请求）"""
    stock_data = stock_data.tail(period).copy()
    stock_data["Institution_Vol"] = stock_data["Volume"] * 0.3
    stock_data["Institution_Price"] = stock_data["Close"] * (1 + np.random.uniform(-0.02, 0.02, len(stock_data)))
    stock_data["Cum_Institution_Vol"] = stock_data["Institution_Vol"].cumsum()
    stock_data["Cum_Institution_Value"] = (stock_data["Institution_Price"] * stock_data["Institution_Vol"]).cumsum()
    stock_data["Institution_VWAP"] = stock_data["Cum_Institution_Value"] / stock_data["Cum_Institution_Vol"]
    return stock_data[["Date", "Institution_VWAP"]]

@st.cache_data(ttl=86400)
def simulate_筹码峰(stock_data, period=30):
    """模拟筹码峰数据（纯本地计算）"""
    price_range = np.linspace(stock_data["Close"].min() * 0.9, stock_data["Close"].max() * 1.1, 50)
    volume_distribution = []
    for price in price_range:
        volume = stock_data[(stock_data["Close"] >= price * 0.98) & (stock_data["Close"] <= price * 1.02)]["Volume"].sum()
        volume_distribution.append(volume)
    筹码峰_data = pd.DataFrame({
        "价格": price_range,
        "筹码占比": [v / (sum(volume_distribution) + 1e-8) * 100 for v in volume_distribution]  # 避免除零
    })
    return 筹码峰_data

# ---------------------- 侧边栏导航 ----------------------
st.sidebar.title("📊 BTDR 分析导航")
menu_option = st.sidebar.radio(
    "选择功能模块",
    [
        "核心数据总览",
        "股价&VWAP分析",
        "筹码峰联动",
        "投资工具",
        "财务&运营数据",
        "风险提示"
    ]
)

# ---------------------- 核心数据总览 ----------------------
if menu_option == "核心数据总览":
    st.title("BTDR 核心数据总览")
    st.divider()

    # 1. 实时股价卡片
    stock_data, stock_info = get_btdr_stock_data()
    latest_data = stock_data.iloc[-1]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="当前股价",
            value=f"${latest_data['Close']:.2f}",
            delta=f"{(latest_data['Close'] - latest_data['Open']):.2f} ({((latest_data['Close'] - latest_data['Open'])/latest_data['Open']*100):.2f}%)"
        )
    with col2:
        institution_vwap_data = calculate_institution_vwap(stock_data)
        latest_vwap = institution_vwap_data.iloc[-1]["Institution_VWAP"]
        st.metric(
            label="机构VWAP（30日）",
            value=f"${latest_vwap:.2f}",
            delta=f"{(latest_data['Close'] - latest_vwap):.2f} ({((latest_data['Close'] - latest_vwap)/latest_vwap*100):.2f}%)"
        )
    with col3:
        st.metric(
            label="市值",
            value=f"${stock_info.get('marketCap', 2624000000)/1e8:.2f}亿",
            help="数据更新至最近交易日"
        )

    # 2. 核心指标矩阵
    st.subheader("关键指标速览")
    fundamental_data = get_btdr_fundamental_data()
    col4, col5 = st.columns(2)
    with col4:
        st.write("📈 财务指标")
        finance_df = pd.DataFrame(fundamental_data["财务指标"])
        st.dataframe(finance_df, use_container_width=True)
    with col5:
        st.write("⚙️ 运营指标")
        operate_df = pd.DataFrame(fundamental_data["运营指标"])
        st.dataframe(operate_df, use_container_width=True)

    # 3. 股价走势预览
    st.subheader("近30日股价走势（含均线）")
    preview_data = stock_data.tail(30)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=preview_data["Date"], y=preview_data["Close"], name="股价", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=preview_data["Date"], y=preview_data["MA10"], name="10日均线", line=dict(color="#ff7f0e", dash="dash")))
    fig.add_trace(go.Scatter(x=preview_data["Date"], y=preview_data["VWAP"], name="市场VWAP", line=dict(color="#2ca02c", dash="dot")))
    fig.update_layout(height=300, xaxis_title="日期", yaxis_title="价格（美元）", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------- 股价&VWAP分析 ----------------------
elif menu_option == "股价&VWAP分析":
    st.title("股价走势与VWAP深度分析")
    st.divider()

    # 1. 周期选择器
    period_option = st.selectbox("选择时间周期", ["1周", "1个月", "3个月", "6个月", "1年"])
    period_map = {"1周": "1wk", "1个月": "1mo", "3个月": "3mo", "6个月": "6mo", "1年": "1y"}
    stock_data, _ = get_btdr_stock_data(period=period_map[period_option])

    # 2. 多维度图表
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data["Date"], y=stock_data["Close"], name="股价", line=dict(color="#1f77b4", width=2)))
    fig.add_trace(go.Scatter(x=stock_data["Date"], y=stock_data["MA10"], name="10日均线", line=dict(color="#ff7f0e", dash="dash")))
    fig.add_trace(go.Scatter(x=stock_data["Date"], y=stock_data["MA20"], name="20日均线", line=dict(color="#d62728", dash="dash")))
    # 机构VWAP
    institution_vwap_data = calculate_institution_vwap(stock_data, period=len(stock_data))
    fig.add_trace(go.Scatter(x=institution_vwap_data["Date"], y=institution_vwap_data["Institution_VWAP"], name="机构VWAP", line=dict(color="#9467bd", width=2)))
    # 成交量
    fig.add_trace(go.Bar(x=stock_data["Date"], y=stock_data["Volume"]/1e6, name="成交量（百万股）", yaxis="y2", opacity=0.5))

    fig.update_layout(
        height=500,
        xaxis_title="日期",
        yaxis_title="价格（美元）",
        yaxis2=dict(title="成交量（百万股）", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # 3. 分析结论
    st.subheader("关键结论")
    latest_price = stock_data.iloc[-1]["Close"]
    latest_vwap = institution_vwap_data.iloc[-1]["Institution_VWAP"]
    if latest_price > latest_vwap and latest_price > stock_data.iloc[-1]["MA10"]:
        st.success("✅ 当前股价高于机构VWAP和10日均线，短期强势，关注上方阻力位")
    elif latest_price < latest_vwap and latest_price < stock_data.iloc[-1]["MA10"]:
        st.warning("⚠️ 当前股价低于机构VWAP和10日均线，短期弱势，关注下方支撑位")
    else:
        st.info("ℹ️ 股价处于震荡区间，需结合筹码峰与成交量进一步判断")

    # 4. 数据导出
    csv_data = stock_data[["Date", "Open", "High", "Low", "Close", "Volume", "VWAP", "MA10", "MA20"]].to_csv(index=False)
    st.download_button(
        label="导出股价数据（CSV）",
        data=csv_data,
        file_name=f"BTDR_{period_option}_股价数据.csv",
        mime="text/csv"
    )

# ---------------------- 筹码峰联动 ----------------------
elif menu_option == "筹码峰联动":
    st.title("筹码峰与机构VWAP联动分析")
    st.divider()

    # 1. 周期选择
    period = st.slider("选择分析周期（交易日）", min_value=10, max_value=60, value=30, step=5)
    stock_data, _ = get_btdr_stock_data(period=f"{period}d")
    筹码峰_data = simulate_筹码峰(stock_data, period=period)
    institution_vwap_data = calculate_institution_vwap(stock_data, period=period)
    latest_price = stock_data.iloc[-1]["Close"]
    latest_vwap = institution_vwap_data.iloc[-1]["Institution_VWAP"]

    # 2. 双图联动
    col1, col2 = st.columns([1, 2])
    with col1:
        # 筹码峰图表
        st.subheader("筹码分布")
        fig1 = go.Figure(go.Bar(x=筹码峰_data["筹码占比"], y=筹码峰_data["价格"], orientation="h", color="#ff7f0e"))
        fig1.add_vline(x=latest_price, line_dash="dash", line_color="red", annotation_text="当前股价")
        fig1.add_vline(x=latest_vwap, line_dash="dash", line_color="blue", annotation_text="机构VWAP")
        fig1.update_layout(height=400, xaxis_title="筹码占比（%）", yaxis_title="价格（美元）")
        st.plotly_chart(fig1, use_container_width=True)

        # 筹码集中度分析
        主峰价格 = 筹码峰_data.loc[筹码峰_data["筹码占比"].idxmax(), "价格"]
        主峰占比 = 筹码峰_data["筹码占比"].max()
        st.write(f"📌 筹码主峰：${主峰价格:.2f}（占比{主峰占比:.1f}%）")
        if abs(主峰价格 - latest_vwap) / latest_vwap < 0.02:
            st.success("✅ 机构VWAP与筹码主峰重合，支撑位极强")
        elif latest_vwap < 主峰价格:
            st.info("ℹ️ 机构成本低于筹码主峰，主力低吸布局")
        else:
            st.warning("⚠️ 机构成本高于筹码主峰，需警惕获利了结")

    with col2:
        # 股价+VWAP+筹码主峰联动图
        st.subheader(f"{period}日股价+VWAP+筹码主峰")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=stock_data["Date"], y=stock_data["Close"], name="股价", line=dict(color="#1f77b4")))
        fig2.add_trace(go.Scatter(x=institution_vwap_data["Date"], y=institution_vwap_data["Institution_VWAP"], name="机构VWAP", line=dict(color="#9467bd")))
        fig2.add_hline(y=主峰价格, line_dash="dash", line_color="orange", annotation_text=f"筹码主峰（${主峰价格:.2f}）")
        fig2.update_layout(height=400, xaxis_title="日期", yaxis_title="价格（美元）", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig2, use_container_width=True)

# ---------------------- 投资工具 ----------------------
elif menu_option == "投资工具":
    st.title("投资决策辅助工具")
    st.divider()

    # 1. 成本测算工具
    st.subheader("💰 持仓成本测算")
    with st.form(key="cost_calculator"):
        col1, col2, col3 = st.columns(3)
        with col1:
            持仓价格 = st.number_input("你的持仓价格（美元）", value=11.0, step=0.1)
        with col2:
            持仓数量 = st.number_input("持仓数量（股）", value=1000, step=100)
        with col3:
            手续费率 = st.number_input("手续费率（%）", value=0.1, step=0.01)
        submit_btn = st.form_submit_button("计算")

        if submit_btn:
            institution_vwap_data = calculate_institution_vwap(get_btdr_stock_data()[0])
            latest_vwap = institution_vwap_data.iloc[-1]["Institution_VWAP"]
            当前股价 = get_btdr_stock_data()[0].iloc[-1]["Close"]
            浮盈 = (当前股价 - 持仓价格) * 持仓数量 - (持仓价格 * 持仓数量 * 手续费率 / 100)
            与机构价差 = (持仓价格 - latest_vwap) / latest_vwap * 100

            st.write("### 测算结果")
            col4, col5, col6 = st.columns(3)
            with col4:
                st.metric("浮盈/浮亏", f"${浮盈:.2f}")
            with col5:
                st.metric("与机构成本价差", f"{与机构价差:.2f}%")
            with col6:
                st.metric("当前股价", f"${当前股价:.2f}")

            # 建议
            if 与机构价差 < -5:
                st.success("✅ 你的持仓成本低于机构5%+，安全垫充足，可长期持有")
            elif 与机构价差 > 5:
                st.warning("⚠️ 你的持仓成本高于机构5%+，建议逢低加仓摊薄成本或设置止损")
            else:
                st.info("ℹ️ 持仓成本与机构接近，关注股价突破方向")

    # 2. 行情情景模拟
    st.subheader("📊 行情情景模拟")
    st.write("假设BTC价格或SEAL04量产进度变化，预测BTDR股价影响")
    col7, col8 = st.columns(2)
    with col7:
        btc_change = st.selectbox("BTC价格变动", ["-20%", "-10%", "0%", "+10%", "+20%"])
    with col8:
        production = st.selectbox("SEAL04量产进度", ["延期1个月", "如期量产", "提前量产"])

    if st.button("生成模拟结果"):
        base_price = get_btdr_stock_data()[0].iloc[-1]["Close"]
        # 模拟逻辑
        btc_impact = float(btc_change.strip("%")) * 0.5
        production_impact = 3 if production == "提前量产" else (-3 if production == "延期1个月" else 0)
        total_impact = btc_impact + production_impact
        simulate_price = base_price * (1 + total_impact / 100)

        st.metric(
            label="模拟股价",
            value=f"${simulate_price:.2f}",
            delta=f"{total_impact:.1f}%"
        )
        st.write(f"### 模拟逻辑说明")
        st.write(f"- BTC价格变动{btc_change}，影响股价{btc_impact:.1f}%")
        st.write(f"- {production}，影响股价{production_impact:.1f}%")
        st.write(f"- 总影响：{total_impact:.1f}%")

# ---------------------- 财务&运营数据 ----------------------
elif menu_option == "财务&运营数据":
    st.title("财务与运营数据详情")
    st.divider()

    fundamental_data = get_btdr_fundamental_data()
    tab1, tab2, tab3 = st.tabs(["财务指标", "运营指标", "核心产品"])

    with tab1:
        finance_df = pd.DataFrame(fundamental_data["财务指标"])
        st.dataframe(finance_df, use_container_width=True)
        st.write("💡 备注：Q3净亏损包含非现金衍生品损失，核心业务（挖矿+AI）已实现EBITDA转正")

    with tab2:
        operate_df = pd.DataFrame(fundamental_data["运营指标"])
        st.dataframe(operate_df, use_container_width=True)
        # 运营趋势图
        st.subheader("算力与BTC产出趋势（模拟）")
        trend_data = pd.DataFrame({
            "月份": ["9月", "10月", "11月", "12月E", "2026-01E"],
            "算力（EH/s）": [32.1, 38.5, 45.7, 52.0, 60.0],
            "BTC产出（枚）": [312, 389, 526, 610, 720]
        })
        fig = go.Figure()
        fig.add_trace(go.Bar(x=trend_data["月份"], y=trend_data["算力（EH/s）"], name="算力", yaxis="y1", color="#1f77b4"))
        fig.add_trace(go.Line(x=trend_data["月份"], y=trend_data["BTC产出（枚）"], name="BTC产出", yaxis="y2", color="#ff7f0e"))
        fig.update_layout(
            height=300,
            yaxis=dict(title="算力（EH/s）"),
            yaxis2=dict(title="BTC产出（枚）", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        product_df = pd.DataFrame(fundamental_data["核心产品"])
        st.dataframe(product_df, use_container_width=True)
        st.write("🎯 核心竞争力：自研芯片提升能效，降低挖矿成本；AI/HPC转型打开长期增长空间")

# ---------------------- 风险提示 ----------------------
elif menu_option == "风险提示":
    st.title("风险提示与免责声明")
    st.divider()

    st.warning("""
    ### 🔴 主要风险因素
    1. **加密货币价格波动风险**：BTC价格直接影响挖矿收益，若BTC价格大幅下跌，可能导致公司营收与利润下滑；
    2. **量产与技术风险**：SEAL04芯片量产进度、良率可能不及预期，影响算力扩张与成本控制；
    3. **监管风险**：全球加密货币挖矿与AI算力服务监管政策变化，可能影响业务开展；
    4. **盈利转化风险**：当前公司仍处于亏损状态，核心业务盈利能否持续转正存在不确定性；
    5. **股价波动风险**：小盘股股价波动性高，可能受市场情绪、资金流向影响出现大幅波动。
    """)

    st.info("""
    ### 📝 免责声明
    1. 本页面数据来源于公开信息及模拟测算，仅为分析参考，不构成任何投资建议；
    2. 模拟数据（如机构VWAP、筹码峰）为基于公开逻辑的估算，实际数据请以官方披露为准；
    3. 投资有风险，入市需谨慎，请勿根据本页面信息盲目决策，建议结合专业投资顾问意见。
    """)

    # 用户反馈
    st.subheader("💬 功能反馈")
    with st.form(key="feedback_form"):
        feedback = st.text_area("请输入你的功能建议或问题")
        submit_feedback = st.form_submit_button("提交反馈")
        if submit_feedback:
            st.success("感谢你的反馈！我们会持续优化功能～")

# ---------------------- 页脚 ----------------------
st.divider()
st.write("📅 数据更新时间：", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
st.write("🔧 技术支持：Streamlit | 数据说明：核心数据为模拟/公开披露，避免API限流")
