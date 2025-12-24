import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
import time
import random
import pytz  # 导入时区处理库

# 定义时区
beijing_tz = pytz.timezone('Asia/Shanghai')
new_york_tz = pytz.timezone('America/New_York')

def get_formatted_times():
    """获取当前北京时间与纽约时间"""
    now_utc = datetime.now(pytz.UTC)
    beijing_time = now_utc.astimezone(beijing_tz)
    new_york_time = now_utc.astimezone(new_york_tz)
    
    return {
        'beijing': beijing_time.strftime('%H:%M:%S'),
        'new_york': new_york_time.strftime('%H:%M:%S'),
        'beijing_date': beijing_time.strftime('%Y-%m-%d'),
        'new_york_date': new_york_time.strftime('%Y-%m-%d')
    }

# 从缓存中获取数据的函数
@st.cache_data(ttl=60)  # 缓存60秒
def load_data_cached():
    """加载缓存数据"""
    # Simulate loading data (replace with actual data loading logic)
    time.sleep(2)  # Simulate delay
    data = {
        'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='D'),
        'value': np.random.randn(100).cumsum()
    }
    return pd.DataFrame(data)

# 从API获取实时数据的函数
def load_data_real_time():
    """获取实时数据（模拟）"""
    # Simulate API call delay
    time.sleep(1)
    data = {
        'timestamp': [datetime.now()],
        'value': [random.random()]
    }
    return pd.DataFrame(data)

# 生成模拟数据的函数
def generate_mock_data():
    """生成模拟数据"""
    data = {
        'timestamp': pd.date_range(start='2023-07-01', periods=50, freq='H'),
        'value': np.random.randn(50).cumsum()
    }
    return pd.DataFrame(data)

# 生成模拟交易数据的函数
def generate_trading_data():
    """生成模拟交易数据"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    data = {
        'Date': dates,
        'Open': 100 + np.random.randn(100).cumsum(),
        'High': 100 + np.random.randn(100).cumsum() + np.random.uniform(0, 2, 100),
        'Low': 100 + np.random.randn(100).cumsum() - np.random.uniform(0, 2, 100),
        'Close': 100 + np.random.randn(100).cumsum(),
        'Volume': np.random.randint(1000, 5000, 100)
    }
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    return df

# 生成模拟财务数据的函数
def generate_financial_data():
    """生成模拟财务数据"""
    quarters = ['2022Q1', '2022Q2', '2022Q3', '2022Q4', '2023Q1', '2023Q2', '2023Q3', '2023Q4']
    revenue = np.random.uniform(1e6, 5e6, len(quarters))
    profit = revenue * np.random.uniform(0.1, 0.3, len(quarters)) # 10%-30%利润率
    return pd.DataFrame({
        'Quarter': quarters,
        'Revenue': revenue,
        'Profit': profit
    })

# ---------------------- 侧边栏导航 + 手动刷新按钮 ----------------------
st.sidebar.title("📊 BTDR 实时分析平台")

# 获取当前时间并显示
times = get_formatted_times()
st.sidebar.caption(f"最后刷新：{times['beijing']} (北京) | {times['new_york']} (纽约)")

if st.sidebar.button('手动刷新'):
    st.experimental_rerun()

# ---------------------- 主页面内容 ----------------------
menu_option = st.sidebar.selectbox("选择页面", [
    "核心数据总览",
    "交易数据监控",
    "财务数据概览",
    "实时数据流",
    "模拟数据展示"
])

# ---------------------- 核心数据总览（实时+缓存刷新） ----------------------
if menu_option == "核心数据总览":
    st.title("BTDR 核心数据总览")
    st.divider()

    # 获取当前时间
    times = get_formatted_times()

    # 显示时间
    st.markdown(f"**最后刷新时间：** {times['beijing']} (北京) | {times['new_york']} (纽约)")
    
    # 使用 st.spinner 显示加载状态
    with st.spinner("正在加载核心数据..."):
        data = load_data_cached()
        latest_value = data['value'].iloc[-1]

    col1, col2, col3 = st.columns(3)
    col1.metric("实时指标 1", f"{latest_value:.2f}", "1.2%")
    col2.metric("实时指标 2", "1234", "-0.5%")
    col3.metric("实时指标 3", "5678", "2.1%")

    st.subheader("核心数据图表")
    st.line_chart(data.set_index('timestamp')['value'])

    st.divider()
    st.write(f"页面最后更新时间：{times['beijing_date']} {times['beijing']} (北京) | {times['new_york_date']} {times['new_york']} (纽约)")

# ---------------------- 交易数据监控 ----------------------
elif menu_option == "交易数据监控":
    st.title("交易数据监控")
    st.divider()

    times = get_formatted_times()
    st.markdown(f"**最后刷新时间：** {times['beijing']} (北京) | {times['new_york']} (纽约)")

    with st.spinner("加载交易数据..."):
        trading_data = generate_trading_data()

    # 选择要显示的股票代码（模拟）
    symbol = st.selectbox("选择股票", ["BTDR", "AAPL", "GOOGL"])
    # 这里使用模拟数据，symbol 仅用于显示
    st.subheader(f"{symbol} 交易数据")

    # 显示K线图
    fig = go.Figure(data=go.Candlestick(
        x=trading_data.index,
        open=trading_data['Open'],
        high=trading_data['High'],
        low=trading_data['Low'],
        close=trading_data['Close']
    ))
    fig.update_layout(title=f"{symbol} K线图", xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig, use_container_width=True)

    # 显示交易数据表格
    st.dataframe(trading_data.tail(10))

    st.divider()
    st.write(f"页面最后更新时间：{times['beijing_date']} {times['beijing']} (北京) | {times['new_york_date']} {times['new_york']} (纽约)")

# ---------------------- 财务数据概览 ----------------------
elif menu_option == "财务数据概览":
    st.title("财务数据概览")
    st.divider()

    times = get_formatted_times()
    st.markdown(f"**最后刷新时间：** {times['beijing']} (北京) | {times['new_york']} (纽约)")

    with st.spinner("加载财务数据..."):
        financial_data = generate_financial_data()

    st.subheader("营收与利润")
    st.dataframe(financial_data)

    # 图表
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("营收趋势")
        st.line_chart(financial_data.set_index('Quarter')['Revenue'])

    with col2:
        st.subheader("利润趋势")
        st.bar_chart(financial_data.set_index('Quarter')['Profit'])

    st.divider()
    st.write(f"页面最后更新时间：{times['beijing_date']} {times['beijing']} (北京) | {times['new_york_date']} {times['new_york']} (纽约)")

# ---------------------- 实时数据流 ----------------------
elif menu_option == "实时数据流":
    st.title("实时数据流")
    st.divider()

    times = get_formatted_times()
    st.markdown(f"**最后刷新时间：** {times['beijing']} (北京) | {times['new_york']} (纽约)")

    # 创建一个空的图表容器
    chart_container = st.container()
    
    # 模拟实时数据更新
    placeholder = st.empty()
    for seconds in range(60):  # 模拟60秒的数据流
        with placeholder.container():
            times = get_formatted_times()
            st.write(f"当前时间：{times['beijing']} (北京) | {times['new_york']} (纽约)")
            
            # 生成新的实时数据点
            new_point = load_data_real_time()
            
            # 获取历史数据（这里简化为每次都重新生成，实际应用中应维护一个数据列表）
            # 为了演示，我们使用一个更长的历史数据集
            historical_data = generate_mock_data()
            # 模拟添加新点
            new_data = pd.concat([historical_data, new_point], ignore_index=True)
            
            # 显示图表
            with chart_container:
                st.subheader("实时数据图表")
                st.line_chart(new_data.set_index('timestamp')['value'])

        time.sleep(1) # 每秒更新一次

    st.divider()
    st.write(f"页面最后更新时间：{times['beijing_date']} {times['beijing']} (北京) | {times['new_york_date']} {times['new_york']} (纽约)")

# ---------------------- 模拟数据展示 ----------------------
elif menu_option == "模拟数据展示":
    st.title("模拟数据展示")
    st.divider()

    times = get_formatted_times()
    st.markdown(f"**最后刷新时间：** {times['beijing']} (北京) | {times['new_york']} (纽约)")

    with st.spinner("生成模拟数据..."):
        mock_data = generate_mock_data()

    st.subheader("模拟数据图表")
    st.area_chart(mock_data.set_index('timestamp'))

    st.subheader("模拟数据表格")
    st.dataframe(mock_data)

    st.divider()
    st.write(f"页面最后更新时间：{times['beijing_date']} {times['beijing']} (北京) | {times['new_york_date']} {times['new_york']} (纽约)")

# ---------------------- 底部信息 ----------------------
st.divider()
times = get_formatted_times()
st.write(f"页面最后更新时间：{times['beijing_date']} {times['beijing']} (北京) | {times['new_york_date']} {times['new_york']} (纽约)")
st.write("Powered by Streamlit & YFinance")
