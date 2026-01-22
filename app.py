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
st.set_page_config(page_title="BTDR Command Center v13.27 Sync", layout="centered")

# --- 核心常量锁定 (根据您的截图校准) ---
LOCKED_FLOAT_SHARES = 121100000 # 1.211亿 (流通股)
LOCKED_TOTAL_SHARES = 232000000 # 2.32亿 (总股本)

CUSTOM_CSS = """
<style>
    html { overflow-y: scroll; }
    .stApp > header { display: none; }
    .stApp { margin-top: -30px; background-color: #ffffff; }
    div[data-testid="stStatusWidget"] { visibility: hidden; }
    
    /* Font Fix */
    .metric-card, .miner-card, .factor-box, .action-banner, .intent-box, .scen-card, .time-bar, .chart-legend, .profile-bar {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        color: #212529 !important;
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
    
    .metric-label { font-size: 0.75rem; color: #666; margin-bottom: 2px; }
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
    
    /* Scenario Card Styles */
    .scen-card {
        background: #fff; border: 1px solid #eee; border-radius: 8px;
        padding: 12px; text-align: left; height: 100%; min-height: 110px;
        border-top: 3px solid #ccc;
        box-shadow: 0 2px 5px rgba(0,0,0,0.02); display: flex; flex-direction: column; justify-content: center;
    }
    .scen-bull { border-top-color: #0ca678; background: linear-gradient(to bottom, #fff, #f4fdf9); }
    .scen-base { border-top-color: #fab005; background: linear-gradient(to bottom, #fff, #fffbf0); }
    .scen-bear { border-top-color: #e03131; background: linear-gradient(to bottom, #fff, #fff5f5); }
    
    .scen-title { font-size: 0.75rem; font-weight: 700; text-transform: uppercase; margin-bottom: 4px; color: #555; display:flex; justify-content:space-between;}
    .scen-price { font-size: 1.2rem; font-weight: 800; color: #333; margin-bottom: 2px; }
    .scen-desc { font-size: 0.7rem; color: #666; line-height: 1.3; }
    .scen-prob { font-size: 0.65rem; background: rgba(0,0,0,0.05); padding: 1px 4px; border-radius: 3px; }

    .tag-smart { background: #228be6; color: white; padding: 1px 5px; border-radius: 4px; font-size: 0.6rem; vertical-align: middle; margin-left: 5px; }
    
    /* Intent Box */
    .intent-box {
        background-color: #fff; border-left: 4px solid #333;
        padding: 12px; margin-top: 8px; border-radius: 6px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    .intent-title { font-weight: bold; font-size: 0.9rem; margin-bottom: 4px; display: flex; align-items: center; gap: 6px; }
    .intent-desc { font-size: 0.8rem; color: #555; line-height: 1.5; }
    .tag-bull { color: #099268; background: #e6fcf5; padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; }
    .tag-bear { color: #c92a2a; background: #fff5f5; padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; }
    .tag-neu { color: #666; background: #f1f3f5; padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; }
    .tag-macro { color: #f76707; background: #fff4e6; padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; }
    
    .streamlit-expanderHeader {
        font-size: 0.8rem !important;
        color: #555 !important;
        background-color: #f8f9fa !important;
        border-radius: 6px !important;
        border: 1px solid #eee !important;
    }
    
    /* Profile Bar */
    .profile-bar {
        display: flex; justify-content: space-around; background: #343a40; color: #fff !important;
        padding: 10px; border-radius: 8px; margin-bottom: 15px; font-size: 0.8rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .profile-item { text-align: center; }
    .profile-lbl { font-size: 0.65rem; opacity: 0.9; text-transform: uppercase; color: #ced4da !important; font-weight: 600; letter-spacing: 0.5px;}
    .profile-val { font-weight: 800; font-size: 1.0rem; color: #fff !important; margin-top: 2px;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

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

# --- FIX: TIME SYNCED OPTIONS FETCH (Crucial Fix) ---
@st.cache_data(ttl=600)
def get_options_data(symbol, current_price, ref_date=None):
    try:
        tk = yf.Ticker(symbol)
        exps = tk.options
        if not exps: return None
        
        sorted_dates = sorted(exps)
        # 1. 使用市场数据的日期作为“今天”，解决时空错乱
        today = ref_date if ref_date else datetime.now()
        if today.tzinfo: today = today.replace(tzinfo=None)
        
        cutoff_date = today + timedelta(days=45) 
        
        calls_list = []
        puts_list = []
        valid_dates = []
        
        for d in sorted_dates:
            d_obj = datetime.strptime(d, '%Y-%m-%d')
            # 2. 只有当过期很久（比如 >180天）才过滤，保留最近过期的以防数据源延迟
            # 但为了准确，我们还是过滤掉 ref_date 之前的
            if d_obj < today: continue 
            if d_obj > cutoff_date: break 
            
            chain = tk.option_chain(d)
            if not chain.calls.empty: calls_list.append(chain.calls)
            if not chain.puts.empty: puts_list.append(chain.puts)
            valid_dates.append(d)
            
        if not calls_list or not puts_list: return None 

        all_calls = pd.concat(calls_list)
        all_puts = pd.concat(puts_list)
        
        lower_bound = current_price * 0.7
        upper_bound = current_price * 1.3
        
        c_clean = all_calls[(all_calls['strike'] >= lower_bound) & (all_calls['strike'] <= upper_bound)]
        p_clean = all_puts[(all_puts['strike'] >= lower_bound) & (all_puts['strike'] <= upper_bound)]
        
        if c_clean.empty: c_clean = all_calls
        if p_clean.empty: p_clean = all_puts
        
        total_call_vol = all_calls['volume'].sum() if not all_calls.empty else 1
        total_put_vol = all_puts['volume'].sum() if not all_puts.empty else 0
        pcr_vol = total_put_vol / total_call_vol
        
        strikes = set(c_clean['strike']).union(set(p_clean['strike']))
        min_loss = float('inf'); max_pain = current_price
        
        if strikes:
            for k in strikes:
                call_loss = c_clean[c_clean['strike'] < k].apply(lambda x: (k - x['strike']) * x['openInterest'], axis=1).sum()
                put_loss = p_clean[p_clean['strike'] > k].apply(lambda x: (x['strike'] - k) * x['openInterest'], axis=1).sum()
                total_loss = call_loss + put_loss
                if total_loss < min_loss: 
                    min_loss = total_loss; max_pain = k
        
        calls_above = c_clean[c_clean['strike'] > current_price]
        if not calls_above.empty:
            call_oi_map = calls_above.groupby('strike')['openInterest'].sum()
            call_wall = call_oi_map.idxmax()
        else: call_wall = current_price * 1.1

        puts_below = p_clean[p_clean['strike'] < current_price]
        if not puts_below.empty:
            put_oi_map = puts_below.groupby('strike')['openInterest'].sum()
            put_wall = put_oi_map.idxmax()
        else: put_wall = current_price * 0.9

        date_display = f"{len(valid_dates)} Exps (<45d)"

        return {
            "expiry": date_display, "pcr": pcr_vol, "max_pain": max_pain,
            "call_wall": call_wall, "put_wall": put_wall,
            "call_vol": total_call_vol, "put_vol": total_put_vol
        }
    except Exception as e:
        print(f"Options Error: {e}")
        return None

# --- Mining Data ---
@st.cache_data(ttl=3600)
def get_mining_metrics(btc_price):
    try:
        diff_url = "https://blockchain.info/q/getdifficulty"
        diff_resp = requests.get(diff_url, timeout=3)
        if diff_resp.status_code == 200:
            difficulty = float(diff_resp.text)
        else:
            difficulty = 80000000000000 
            
        btc_per_ph_day = (1e15 * 86400 * 3.125) / (difficulty * 4294967296)
        hashprice_real = btc_per_ph_day * btc_price
        
        return difficulty, hashprice_real
    except:
        return 0, 0

def get_macro_insight(hashprice, beta):
    if hashprice < 40:
        hp_tag = "🥶 极寒 (Survival)"
        hp_desc = "矿工收入极低，仅顶级算力盈利。"
    elif hashprice > 55:
        hp_tag = "🤑 暴利 (Money Printer)"
        hp_desc = "矿工印钞模式，基本面强力支撑。"
    else:
        hp_tag = "😐 平衡 (Balanced)"
        hp_desc = "收入处于平均水平。"

    if beta > 1.5:
        beta_tag = "🎰 强杠杆"
        beta_desc = "股价弹性极高，适合博弈BTC突破。"
    elif beta < 0.8:
        beta_tag = "🐢 弱势跟随"
        beta_desc = "相对BTC滞涨，需警惕动能衰竭。"
    else:
        beta_tag = "🔗 正常跟随"
        beta_desc = "与BTC走势同步。"

    title = f"{hp_tag} | {beta_tag}"
    desc = f"当前 Hashprice ${hashprice:.1f}/PH/Day ({hp_desc})。BTDR Beta {beta:.2f} ({beta_desc})。"
    return title, desc

# --- Intent Engine ---
def get_mm_intent(price, mp, cw, pw, pcr):
    gap_mp = (price - mp) / price
    
    if price >= cw * 0.98:
        title = "♟️ 庄家意图: 铁壁防守 (Defend Call Wall)"
        desc = f"股价逼近最大阻力位 ${cw}，做市商Gamma风险剧增。预计将出现强抛压以防守此位置，除非成交量剧增引发Gamma Squeeze。"
        color = "tag-bear"
        return title, desc, color
        
    if price <= pw * 1.02:
        title = "♟️ 庄家意图: 底部承接 (Support Put Wall)"
        desc = f"股价跌至最大支撑位 ${pw}，做市商需买入现货对冲Put头寸，此处易形成短期反弹。"
        color = "tag-bull"
        return title, desc, color

    if gap_mp > 0.15: 
        title = "♟️ 庄家意图: 诱多杀跌 (Suppress to Pain)"
        desc = f"现价(${price:.2f}) 远高于痛点(${mp:.1f})。做市商卖出的Call处于亏损边缘，有强烈动力打压股价，收割多头权利金。"
        color = "tag-bear"
    elif gap_mp < -0.15: 
        title = "♟️ 庄家意图: 诱空拉升 (Lift to Pain)"
        desc = f"现价(${price:.2f}) 远低于痛点(${mp:.1f})。做市商卖出的Put处于亏损边缘，有动力拉升股价，收割空头权利金。"
        color = "tag-bull"
    else: 
        title = "♟️ 庄家意图: 横盘收租 (Theta Burn)"
        desc = f"现价处于痛点(${mp:.1f}) 舒适区。做市商只需维持震荡，利用时间损耗 (Theta) 同时收割多空双方。"
        color = "tag-neu"
        
    return title, desc, color

@st.cache_data(ttl=300)
def run_grandmaster_analytics(live_price=None):
    default_model = {
        "high": {"intercept": 0, "beta_gap": 0.5, "beta_btc": 0.5, "beta_vol": 0},
        "low": {"intercept": 0, "beta_gap": 0.5, "beta_btc": 0.5, "beta_vol": 0},
        "ensemble_hist_h": 0.05, "ensemble_hist_l": -0.05,
        "ensemble_mom_h": 0.08, "ensemble_mom_l": -0.08,
        "top_peers": ["MARA", "RIOT", "CLSK", "CORZ", "IREN"]
    }
    default_factors = {"vwap": 0, "adx": 20, "regime": "Neutral", "beta_btc": 1.5, "beta_qqq": 1.2, "rsi": 50, "vol_base": 0.05, "atr_ratio": 0.05, "hurst": 0.5, "macd": 0, "macd_sig": 0, "boll_u": 0, "boll_l": 0, "boll_m": 0}

    try:
        tickers_str = "BTDR BTC-USD QQQ " + " ".join(MINER_POOL)
        data = yf.download(tickers_str, period="6mo", interval="1d", group_by='ticker', threads=True, progress=False)
        if data.empty: return default_model, default_factors, "No Data"

        btdr = data['BTDR'].dropna(); btc = data['BTC-USD'].dropna(); qqq = data['QQQ'].dropna()
        idx = btdr.index.intersection(btc.index).intersection(qqq.index)
        btdr, btc, qqq = btdr.loc[idx], btc.loc[idx], qqq.loc[idx]
        
        btdr.index = btdr.index.tz_localize(None)
        
        if live_price and live_price > 0:
            last_date = btdr.index[-1].date(); today = datetime.now().date()
            last_row = btdr.iloc[-1].copy()
            last_row['Close'] = live_price
            last_row['High'] = max(last_row['High'], live_price)
            last_row['Low'] = min(last_row['Low'], live_price)
            
            if last_date == today: btdr.iloc[-1] = last_row
            else:
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
        ret_btc = btc['Close'].pct_change().fillna(0).values
        ret_qqq = qqq['Close'].pct_change().fillna(0).values
        
        beta_btc = run_kalman_filter(ret_btdr, ret_btc, delta=1e-4)
        beta_qqq = run_kalman_filter(ret_btdr, ret_qqq, delta=1e-4)
        beta_btc = np.clip(beta_btc, -1, 5); beta_qqq = np.clip(beta_qqq, -1, 4)

        # Technicals
        close = btdr['Close']
        pv = (close * btdr['Volume'])
        vol_sum = btdr['Volume'].tail(30).sum()
        vwap_30d = pv.tail(30).sum() / vol_sum if vol_sum > 0 else btdr['Close'].mean()
        
        high, low, close = btdr['High'], btdr['Low'], btdr['Close']
        tr = np.maximum(high - low, np.abs(high - close.shift(1)))
        atr = tr.rolling(14).mean()
        
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9, adjust=False).mean()
        
        sma20 = close.rolling(window=20).mean(); std20 = close.rolling(window=20).std()
        boll_u = sma20 + (std20 * 2); boll_l = sma20 - (std20 * 2); boll_m = sma20
        
        up, down = high.diff(), -low.diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0); minus_dm = np.where((down > up) & (down > 0), down, 0)
        atr_s = pd.Series(atr.values, index=btdr.index)
        plus_di = 100 * (pd.Series(plus_dm, index=btdr.index).rolling(14).mean() / atr_s)
        minus_di = 100 * (pd.Series(minus_dm, index=btdr.index).rolling(14).mean() / atr_s)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(14).mean().iloc[-1]; adx = 20 if np.isnan(adx) else adx
        
        delta_p = close.diff()
        gain = delta_p.where(delta_p > 0, 0).rolling(14).mean(); loss = -delta_p.where(delta_p < 0, 0).rolling(14).mean()
        rs = gain / loss; rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        vol_base = ret_btdr.std()
        if len(ret_btdr) > 20: vol_base = pd.Series(ret_btdr).ewm(span=20).std().iloc[-1]
        atr_ratio = (atr / close).iloc[-1]
        hurst = calculate_hurst(btdr['Close'].values[-50:])
        regime = "Trend" if adx > 25 else ("MeanRev" if hurst < 0.4 else "Chop")

        last_close = close.iloc[-1]
        bu_val = boll_u.iloc[-1]; bl_val = boll_l.iloc[-1]; bm_val = boll_m.iloc[-1]
        
        if np.isnan(bu_val) or bu_val <= 0: bu_val = last_close * 1.10
        if np.isnan(bl_val) or bl_val <= 0: bl_val = last_close * 0.90
        if np.isnan(bm_val) or bm_val <= 0: bm_val = last_close

        factors = {
            "beta_btc": beta_btc, "beta_qqq": beta_qqq, "vwap": vwap_30d, 
            "adx": adx, "regime": regime, "rsi": rsi, 
            "vol_base": vol_base, "atr_ratio": atr_ratio, "hurst": hurst,
            "macd": macd.iloc[-1], "macd_sig": signal.iloc[-1], 
            "boll_u": bu_val, "boll_l": bl_val, "boll_m": bm_val
        }

        if live_price: btdr = btdr.iloc[:-1]
        df_reg = pd.DataFrame()
        df_reg['Target_High'] = (btdr['High'] - btdr['Close'].shift(1)) / btdr['Close'].shift(1)
        df_reg['Target_Low'] = (btdr['Low'] - btdr['Close'].shift(1)) / btdr['Close'].shift(1)
        df_reg = df_reg.dropna().tail(60)
        
        final_model = {
            "high": {"intercept": 0, "beta_gap": 0.5, "beta_btc": 0.5, "beta_vol": 0},
            "low": {"intercept": 0, "beta_gap": 0.5, "beta_btc": 0.5, "beta_vol": 0},
            "ensemble_hist_h": df_reg['Target_High'].tail(10).mean(), 
            "ensemble_hist_l": df_reg['Target_Low'].tail(10).mean(),
            "ensemble_mom_h": df_reg['Target_High'].tail(3).max(), 
            "ensemble_mom_l": df_reg['Target_Low'].tail(3).min(),
            "top_peers": default_model["top_peers"]
        }
        return final_model, factors, "v13.27 Sync"
    except Exception as e:
        print(f"Error: {e}")
        return default_model, default_factors, "Offline"

# --- FIX: DATA FETCHING ---
def get_realtime_data():
    tickers_list = "BTC-USD BTDR QQQ ^VIX " + " ".join(MINER_POOL)
    symbols = tickers_list.split()
    try:
        # 1. Fetch History (1 Year for robustness)
        btdr_full = yf.Ticker("BTDR").history(period="1y", interval="1d")
        btdr_full.index = btdr_full.index.tz_localize(None)
        
        # 2. Robust Profile Data Fetching
        try:
            btdr_obj = yf.Ticker("BTDR")
            info = btdr_obj.info
            fast = btdr_obj.fast_info
            
            # --- Use Locked Constants ---
            shares_total = LOCKED_TOTAL_SHARES
            
            last_p = fast.last_price
            if not last_p: last_p = btdr_full['Close'].iloc[-1]
            
            mkt_cap = last_p * shares_total
            
            # 52 Week Range
            h52 = fast.year_high
            l52 = fast.year_low
            if not h52 or pd.isna(h52):
                h52 = btdr_full['High'].max()
            if not l52 or pd.isna(l52):
                l52 = btdr_full['Low'].min()
            
            # Earnings Date Fix - Logic: Find next date AFTER "market today"
            # We assume market today is the last data point
            mkt_today = btdr_full.index[-1].date()
            
            try:
                cal = btdr_obj.calendar
                if isinstance(cal, dict) and 'Earnings Date' in cal:
                    dates = cal['Earnings Date']
                    future = [d for d in dates if d > mkt_today] # Compare with market time
                    if future: next_earn = future[0].strftime('%Y-%m-%d')
                    else: next_earn = "Est. Mid-May"
                elif isinstance(cal, pd.DataFrame) and not cal.empty:
                     vals = cal.iloc[0].values
                     next_earn = str(vals[0])[:10]
                else:
                    next_earn = "Est. Mid-May"
            except:
                next_earn = "Est. Mid-May"
            
            short_float = info.get('shortPercentOfFloat', 0)
            if short_float is None: short_float = 0
            
        except: 
            short_float = 0; mkt_cap = 0; h52 = 0; l52 = 0; next_earn = "TBD"
        
        quotes = {}
        tz_ny = pytz.timezone('America/New_York'); now_ny = datetime.now(tz_ny); state_tag, state_css = determine_market_state(now_ny)
        live_volatility = 0.01 

        for sym in symbols:
            try:
                t = yf.Ticker(sym)
                
                # --- FIX: Volume Source Priority ---
                vol = 0
                price_hist = 0
                
                # Priority 1: Regular Market Volume (Matches Broker)
                try:
                    i = t.info
                    vol = i.get('regularMarketVolume', 0)
                except: pass
                
                # Priority 2: Fast Info
                if vol == 0:
                    try: vol = t.fast_info.last_volume
                    except: pass
                
                # Priority 3: History
                try:
                    hist_day = t.history(period="1d")
                    if not hist_day.empty:
                        if vol == 0: vol = hist_day['Volume'].iloc[-1]
                        price_hist = hist_day['Close'].iloc[-1]
                except: pass
                
                try: 
                    price = t.fast_info['last_price']
                    prev = t.fast_info['previous_close']
                except:
                    price = price_hist
                    prev = t.info.get('previousClose', price)
                
                if price == 0 and price_hist > 0: price = price_hist
                if prev == 0: prev = price
                
                pct = ((price - prev) / prev) * 100 if prev > 0 else 0
                quotes[sym] = {"price": price, "pct": pct, "prev": prev, "open": price, "volume": vol, "tag": state_tag, "css": state_css, "is_open_today": True}
                
                if sym == 'BTDR':
                    live_volatility = btdr_full['Close'].pct_change().std()
                    if np.isnan(live_volatility): live_volatility = 0.05

            except Exception as e:
                quotes[sym] = {"price": 0, "pct": 0, "prev": 1, "open": 0, "volume": 0, "tag": "ERR", "css": "dot-closed", "is_open_today": False}
        
        try: fng = int(requests.get("https://api.alternative.me/fng/", timeout=1.0).json()['data'][0]['value'])
        except: fng = 50
        
        profile = {"mkt_cap": mkt_cap, "h52": h52, "l52": l52, "next_earn": next_earn}
        return quotes, fng, live_volatility, btdr_full, short_float, profile
    except: return {}, 50, 0.01, pd.DataFrame(), 0, {}

# --- 6. 绘图函数 ---
def draw_kline_chart(df, live_price):
    if df.empty: return alt.Chart(pd.DataFrame()).mark_text().encode(text=alt.value("No Data")), ""
    
    df = df.copy()
    if live_price > 0:
        last_idx = df.index[-1]; today = datetime.now().date()
        last_row = df.iloc[-1].to_dict()
        last_row['Close'] = live_price
        last_row['High'] = max(last_row['High'], live_price)
        last_row['Low'] = min(last_row['Low'], live_price)
        
        if last_idx.date() == today: df.iloc[-1] = pd.Series(last_row)
        else:
            new_idx = last_idx + timedelta(days=1)
            new_df = pd.DataFrame([last_row], index=[new_idx])
            df = pd.concat([df, new_df])

    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['BOLL_MID'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['BOLL_U'] = df['BOLL_MID'] + 2 * df['STD20']
    df['BOLL_L'] = df['BOLL_MID'] - 2 * df['STD20']
    
    last = df.iloc[-1]
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

    df = df.tail(80).reset_index()
    df.columns = ['T'] + list(df.columns[1:])
    
    base = alt.Chart(df).encode(x=alt.X('T:T', axis=alt.Axis(format='%m/%d', title=None)))
    rule = base.mark_rule().encode(y=alt.Y('Low:Q', scale=alt.Scale(zero=False)), y2='High:Q')
    bar = base.mark_bar(width=6).encode(y='Open:Q', y2='Close:Q', color=alt.condition("datum.Close >= datum.Open", alt.value("#e03131"), alt.value("#0ca678")))
    
    line_5 = base.mark_line(color='#228be6', size=1.5).encode(y='MA5:Q')
    line_mid = base.mark_line(color='#f59f00', size=1.5).encode(y='BOLL_MID:Q') 
    line_bu = base.mark_line(color='#adb5bd', strokeDash=[4,2], size=1).encode(y='BOLL_U:Q')
    line_bl = base.mark_line(color='#adb5bd', strokeDash=[4,2], size=1).encode(y='BOLL_L:Q')
    
    vol = base.mark_bar(opacity=0.3).encode(y=alt.Y('Volume:Q', axis=alt.Axis(title='Vol', labels=False, ticks=False)), color=alt.condition("datum.Close >= datum.Open", alt.value("#e03131"), alt.value("#0ca678"))).properties(height=60)
    
    chart = (rule + bar + line_5 + line_mid + line_bu + line_bl).properties(height=240)
    final_chart = alt.vconcat(chart, vol).resolve_scale(x='shared').interactive()
    
    return final_chart, legend_html

# --- 7. 仪表盘展示 ---
@st.fragment(run_every=15)
def show_live_dashboard():
    tz_ny = pytz.timezone('America/New_York')
    now_ny = datetime.now(tz_ny).strftime('%H:%M:%S')
    badge_class = "badge-ai"
    ai_status = "Init"
    act, reason, sub = "WAIT", "Initializing...", "Please wait"
    
    quotes, fng_val, live_vol_btdr, btdr_hist, short_float, profile = get_realtime_data()
    live_price = quotes.get('BTDR', {}).get('price', 0)
    
    if live_price <= 0:
        st.warning("⚠️ 市场数据暂不可用 (Market Data Unavailable)")
        time.sleep(3)
        st.rerun()
        return

    ai_model, factors, ai_status = run_grandmaster_analytics(live_price)
    
    # FIX: Pass market data time to align options expiry check
    last_market_date = btdr_hist.index[-1].replace(tzinfo=None)
    opt_data = get_options_data('BTDR', live_price, ref_date=last_market_date)
    
    # Macro
    btc_p = quotes.get('BTC-USD', {}).get('price', 90000)
    net_diff, hashprice = get_mining_metrics(btc_p)
    
    regime_tag = factors['regime']
    btc = quotes.get('BTC-USD', {'pct': 0, 'price': 0}); qqq = quotes.get('QQQ', {'pct': 0})
    btdr = quotes.get('BTDR', {'price': 0})

    vwap_val = factors['vwap']
    if vwap_val == 0 or np.isnan(vwap_val): vwap_val = btdr['price']
    dist_vwap = ((btdr['price'] - vwap_val) / vwap_val) * 100
    
    drift_est = (btc['pct']/100 * factors['beta_btc'] * 0.4) + (qqq['pct']/100 * factors['beta_qqq'] * 0.4)
    if abs(dist_vwap) > 10: drift_est -= (dist_vwap/100) * 0.05
    
    hist_h = ai_model['ensemble_hist_h']; hist_l = ai_model['ensemble_hist_l']
    p_high = btdr['price'] * (1 + hist_h + live_vol_btdr)
    p_low = btdr['price'] * (1 + hist_l - live_vol_btdr)

    act, reason, sub, score, macd_h, support_broken = get_signal_recommendation(btdr['price'], factors, p_low)

    curr_p = btdr['price']; atr_buffer = live_vol_btdr * 0.6
    if support_broken:
        support_label_color = "#e03131"; support_label_text = f"${p_low:.2f} (Broken)"
    else:
        support_label_color = "#ffffff"; support_label_text = f"${p_low:.2f}"

    # --- UI Rendering ---
    mkt_cap_str = f"${profile['mkt_cap']/1e9:.2f}B" if profile['mkt_cap'] else "N/A"
    range_str = f"${profile['l52']:.2f} - ${profile['h52']:.2f}" if profile['h52'] else "N/A"
    
    st.markdown(f"""
    <div class="profile-bar">
        <div class="profile-item"><div class="profile-lbl">Market Cap</div><div class="profile-val">{mkt_cap_str}</div></div>
        <div class="profile-item"><div class="profile-lbl">52-Wk Range</div><div class="profile-val">{range_str}</div></div>
        <div class="profile-item"><div class="profile-lbl">Next Earnings</div><div class="profile-val" style="color:#ffc9c9;">{profile['next_earn']}</div></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"<div class='time-bar'>美东 {now_ny} &nbsp;|&nbsp; AI 模式: <span class='{badge_class}'>{regime_tag}</span> &nbsp;|&nbsp; 引擎: <b>{ai_status}</b></div>", unsafe_allow_html=True)
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
        
    # --- Miner Macro ---
    st.markdown("---")
    st.markdown("<div style='margin-bottom: 8px; font-weight:bold; font-size:0.9rem;'>⛏️ 矿业宏观 (Miner Macro)</div>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    
    diff_str = f"{net_diff/1e12:.1f} T"
    hash_str = f"${hashprice:.2f}"
    beta_val = factors['beta_btc']
    
    with m1: st.markdown(card_html("全网难度 (Difficulty)", diff_str, None, 0, "Network"), unsafe_allow_html=True)
    with m2: st.markdown(card_html("挖矿收益 (Hashprice)", hash_str, "PH/Day", 1, "Est"), unsafe_allow_html=True)
    with m3: st.markdown(card_html("Beta vs BTC", f"{beta_val:.2f}", "High Beta" if beta_val>1.5 else "Low Beta", 1 if beta_val>1 else -1, "30d Kalman"), unsafe_allow_html=True)
    
    macro_t, macro_d = get_macro_insight(hashprice, beta_val)
    st.markdown(f"""
    <div class="intent-box" style="border-left-color: #f76707;">
        <div class="intent-title"><span class="tag-macro">Macro View</span> {macro_t}</div>
        <div class="intent-desc">{macro_d}</div>
    </div>
    <div style="margin-bottom:15px;"></div>
    """, unsafe_allow_html=True)
    
    # --- Liquidity & Sentiment (with Guide) ---
    st.markdown("<div style='margin-bottom: 8px; font-weight:bold; font-size:0.9rem;'>🌊 流动性与情绪 (Liquidity & Sentiment)</div>", unsafe_allow_html=True)
    
    vol_avg_10 = btdr_hist['Volume'].tail(10).mean()
    rvol = btdr['volume'] / vol_avg_10 if vol_avg_10 > 0 else 0
    
    l1, l2, l3 = st.columns(3)
    
    with l1: st.markdown(card_html("Short Interest", f"{short_float*100:.2f}%", "Squeeze?" if short_float>0.15 else "Normal", 1 if short_float>0.15 else 0), unsafe_allow_html=True)
    with l2: st.markdown(card_html("RVOL (量比)", f"{rvol:.2f}", "High Vol" if rvol>1.5 else "Low Vol", 1 if rvol>1.5 else 0), unsafe_allow_html=True)
    
    # Use LOCKED Float Shares for Turnover Calculation
    turnover = (btdr['volume'] / LOCKED_FLOAT_SHARES) * 100
    with l3: st.markdown(card_html("换手率 (Turnover)", f"{turnover:.2f}%", None, 0), unsafe_allow_html=True)
    
    with st.expander("🌊 如何解读流动性与情绪？(Sentiment Guide)"):
        st.markdown("""
        <div style='font-size: 0.85rem; color: #444; line-height: 1.6;'>
            <b>1. Short Interest (做空比例):</b><br>
            • <b>>20%:</b> 极高。如果股价上涨，空头将被迫回补，引发<b>“轧空 (Short Squeeze)”</b> 暴涨。<br>
            • <b><5%:</b> 正常水平，空头压力不大。<br><br>
            <b>2. RVOL (相对量能):</b><br>
            • <b>>1.5:</b> 成交量显著放大。若是上涨，说明是<b>机构进场</b>的真突破。<br>
            • <b><0.8:</b> 缩量整理。市场在观望，假突破概率高。<br><br>
            <b>3. Turnover (换手率):</b><br>
            • <b>>5%:</b> 交易极度活跃，通常伴随大波动（日内交易者的天堂）。<br>
            • <b><1%:</b> 交易清淡，流动性差。
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # --- OPTIONS RADAR + INTENT + EXPANDER ---
    if opt_data:
        st.markdown("<div style='margin-bottom: 8px; font-weight:bold; font-size:0.9rem;'>📡 期权雷达 (Real-Time Aggregate)</div>", unsafe_allow_html=True)
        
        pcr_color = "color-down" if opt_data['pcr'] > 1.0 else "color-up"
        mp_delta = opt_data['max_pain'] - btdr['price']
        
        op1, op2, op3, op4 = st.columns(4)
        with op1: st.markdown(card_html("近期合约 (45d)", f"{opt_data['expiry']}", None, 0, "Pooled"), unsafe_allow_html=True)
        with op2: st.markdown(card_html("最大痛点 (Max Pain)", f"${opt_data['max_pain']:.1f}", f"Gap: {mp_delta:+.2f}", mp_delta, tooltip_text="近期所有期权卖方亏损最小的价格 (加权)。"), unsafe_allow_html=True)
        with op3: 
            sentiment = "Bearish" if opt_data['pcr'] > 0.8 else ("Bullish" if opt_data['pcr'] < 0.5 else "Neutral")
            st.markdown(card_html("P/C Ratio (Vol)", f"{opt_data['pcr']:.2f}", sentiment, -1 if sentiment=="Bearish" else 1), unsafe_allow_html=True)
        with op4: st.markdown(card_html("期权墙 (OI Wall)", f"${opt_data['call_wall']:.1f}", f"Sup: ${opt_data['put_wall']:.1f}", 0, tooltip_text="现价上方最大阻力 & 下方最大支撑。"), unsafe_allow_html=True)

        total_vol = opt_data['call_vol'] + opt_data['put_vol']
        call_pct = (opt_data['call_vol'] / total_vol) * 100 if total_vol > 0 else 50
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; font-size:0.7rem; color:#666; margin-bottom:2px;">
            <span>🐂 Call Vol: {opt_data['call_vol']} ({call_pct:.0f}%)</span><span>🐻 Put Vol: {opt_data['put_vol']} ({100-call_pct:.0f}%)</span>
        </div>
        <div style="width:100%; height:6px; background:#f03e3e; border-radius:3px; overflow:hidden;">
            <div style="width:{call_pct}%; height:100%; background:#2f9e44;"></div>
        </div>
        """, unsafe_allow_html=True)

        i_title, i_desc, i_color = get_mm_intent(btdr['price'], opt_data['max_pain'], opt_data['call_wall'], opt_data['put_wall'], opt_data['pcr'])
        st.markdown(f"""
        <div class="intent-box">
            <div class="intent-title"><span class="{i_color}">Smart Money</span> {i_title}</div>
            <div class="intent-desc">{i_desc}</div>
        </div>
        <div style="margin-bottom:8px;"></div>
        """, unsafe_allow_html=True)

        with st.expander("💡 它是如何工作的？(实战解读指南)"):
            st.markdown(f"""
            <div style='font-size: 0.85rem; color: #444; line-height: 1.6;'>
                <b>1. 数据来源 (Data Source):</b><br>
                系统聚合了未来 <b>45天内</b> 所有有效到期日的期权持仓。这解决了单一周权流动性不足的问题，展示的是主力资金的<b>总体兵力部署</b>。<br><br>
                <b>2. 关键指标 (Key Metrics):</b><br>
                • <b>期权墙 (OI Wall - ${opt_data['call_wall']}):</b> 当前最强的<b>阻力位</b>。大量看涨期权堆积在此，做市商为了对冲风险，往往会在股价接近此位置时抛售现货，形成“铁板”。<br>
                • <b>最大痛点 (Max Pain - ${opt_data['max_pain']}):</b> 当前潜在的<b>引力位</b>。机构最希望结算的价格。若现价远高于痛点，股价会有向下的“地心引力”。<br><br>
                <b>3. 操盘剧本 (Action Plan):</b><br>
                • <b>冲高策略:</b> 若股价上攻至 <b>${opt_data['call_wall']}</b> 附近，抛压极大，建议<b>减仓或止盈</b> (除非放量突破)。<br>
                • <b>回调策略:</b> 若股价回调，下方第一支撑看 <b>${opt_data['max_pain']}</b> 附近，是相对安全的低吸区。
            </div>
            """, unsafe_allow_html=True)

    elif live_price > 0:
         st.markdown("---")
         st.info("⚠️ 近期 (45天内) 期权数据不足，暂不展示误导信息。请关注正股走势。")
    
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

    st.markdown("<div style='margin-bottom: 8px; font-weight:bold; font-size:0.9rem;'>🎯 情景推演 (Scenario Analysis)</div>", unsafe_allow_html=True)
    
    sc1, sc2, sc3 = st.columns(3)
    
    bull_target = p_high * (1 + live_vol_btdr)
    bull_prob = "Low" if score < 0 else "Med"
    with sc1:
        st.markdown(f"""
        <div class="scen-card scen-bull">
            <div class="scen-title">🐂 Bull Case (Breakout)<span class="scen-prob">{bull_prob}</span></div>
            <div class="scen-price">${bull_target:.2f}</div>
            <div class="scen-desc">若突破阻力位 <b>${p_high:.2f}</b>，动能释放看向ATR上轨。</div>
        </div>
        """, unsafe_allow_html=True)

    with sc2:
        st.markdown(f"""
        <div class="scen-card scen-base">
            <div class="scen-title">⚖️ Base Case (Range)<span class="scen-prob">High</span></div>
            <div class="scen-price">${p_low:.2f} - {p_high:.2f}</div>
            <div class="scen-desc">当前波动率下的震荡区间，围绕VWAP <b>${vwap_val:.2f}</b> 整理。</div>
        </div>
        """, unsafe_allow_html=True)

    bear_target = p_low * (1 - live_vol_btdr)
    bear_prob = "High" if score < -2 else "Low"
    with sc3:
        st.markdown(f"""
        <div class="scen-card scen-bear">
            <div class="scen-title">🐻 Bear Case (Breakdown)<span class="scen-prob">{bear_prob}</span></div>
            <div class="scen-price">${bear_target:.2f}</div>
            <div class="scen-desc">若跌破支撑位 <b>${p_low:.2f}</b>，止损盘触发寻找新低。</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="font-size:0.7rem; color:#888; margin-top:10px; margin-bottom:2px; display:flex; justify-content:space-between;">
        <span>🟦 Kalman (30%)</span><span>🟨 History (15%)</span><span>🟥 Momentum (5%)</span><span>🟪 AI Volatility (50%)</span>
    </div>
    <div class="ensemble-bar">
        <div class="bar-kalman" style="width: 30%"></div><div class="bar-hist" style="width: 15%"></div><div class="bar-mom" style="width: 5%"></div><div class="bar-ai" style="width: 50%"></div>
    </div><div style="margin-bottom:10px;"></div>""", unsafe_allow_html=True)
    
    col_h, col_l = st.columns(2)
    # FIX: Font color for prediction boxes is now dark for readability
    h_bg = "#e6fcf5" if btdr['price'] < p_high else "#0ca678"; h_txt = "#212529" 
    l_bg = "#fff5f5" if btdr['price'] > p_low else "#e03131"; l_txt = "#212529" 
    
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
    
    # Monte Carlo (Forecast)
    current_vol = factors['vol_base']; long_term_vol = 0.05; drift = drift_est
    sims, days, dt = 1500, 5, 1
    price_paths = np.zeros((sims, days + 1)); price_paths[:, 0] = btdr['price']
    kappa = 0.1; sim_vol = np.full(sims, current_vol)
    for t in range(1, days + 1):
        sim_vol = sim_vol + kappa * (long_term_vol - sim_vol); sim_vol = np.maximum(sim_vol, 0.01)
        shocks = np.random.standard_t(df=5, size=sims)
        daily_ret = np.exp((drift - 0.5 * sim_vol**2) * dt + sim_vol * np.sqrt(dt) * shocks)
        price_paths[:, t] = price_paths[:, t-1] * daily_ret
        
    percentiles = np.percentile(price_paths, [10, 50, 90], axis=0)
    chart_data = pd.DataFrame({"Day": np.arange(days+1), "P90": np.round(percentiles[2], 2), "P50": np.round(percentiles[1], 2), "P10": np.round(percentiles[0], 2)})
    
    base = alt.Chart(chart_data).encode(x=alt.X('Day:Q', title='Future Trading Days (T+)'))
    area = base.mark_area(opacity=0.1, color='#4dabf7').encode(y=alt.Y('P10', title='Forecast Price (USD)', scale=alt.Scale(zero=False)), y2='P90')
    l90 = base.mark_line(color='#0ca678', strokeDash=[5,5]).encode(y='P90')
    l50 = base.mark_line(color='#228be6', size=3).encode(y='P50')
    l10 = base.mark_line(color='#d6336c', strokeDash=[5,5]).encode(y='P10')
    
    st.altair_chart((area + l90 + l50 + l10).properties(height=220).interactive(), use_container_width=True)
    st.caption(f"AI Engine: v13.27 Sync | Score: {score:.1f} | Signal: {act}")

st.markdown("### ⚡ BTDR 领航员 v13.27 Sync")
show_live_dashboard()
