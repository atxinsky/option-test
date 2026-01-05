"""
中国股指期权量化系统 - Streamlit Web应用
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import OPTION_INSTRUMENTS, STRATEGY_DEFAULTS, IV_THRESHOLDS
from greeks import BlackScholes, ImpliedVolatility, calculate_greeks_batch
from iv_monitor import IVMonitor, IVAnalyzer, create_iv_dashboard_data
from data_manager import (
    init_database, download_underlying_daily, load_underlying_daily,
    download_option_realtime, get_option_chain, get_data_status
)
from engine import OptionBacktestEngine, run_backtest_with_strategy
from strategies import STRATEGIES, get_strategy, list_strategies

# 页面配置
st.set_page_config(
    page_title="期权量化系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS (白色主题)
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f2937;
    }
    .metric-label {
        color: #6b7280;
        font-size: 0.9rem;
    }
    .positive { color: #16a34a; }
    .negative { color: #dc2626; }
    .neutral { color: #ca8a04; }
</style>
""", unsafe_allow_html=True)


def main():
    """主函数"""
    # 侧边栏
    st.sidebar.title("📊 期权量化系统")
    st.sidebar.markdown("---")

    # 页面选择
    page = st.sidebar.radio(
        "功能导航",
        ["🏠 实时监控", "📈 IV分析", "🔬 策略回测", "📐 损益分析器", "💾 数据管理"]
    )

    # 品种选择
    st.sidebar.markdown("---")
    symbol = st.sidebar.selectbox(
        "选择品种",
        list(OPTION_INSTRUMENTS.keys()),
        format_func=lambda x: f"{x} - {OPTION_INSTRUMENTS[x]['name']}"
    )

    # 路由
    if page == "🏠 实时监控":
        page_dashboard(symbol)
    elif page == "📈 IV分析":
        page_iv_analysis(symbol)
    elif page == "🔬 策略回测":
        page_backtest(symbol)
    elif page == "📐 损益分析器":
        page_payoff_analyzer(symbol)
    elif page == "💾 数据管理":
        page_data_management(symbol)


@st.cache_data(ttl=300)  # 缓存5分钟
def fetch_underlying_price(symbol: str) -> dict:
    """获取标的指数实时价格"""
    import akshare as ak

    underlying_map = {
        "IO": ("000300", "沪深300"),
        "MO": ("000852", "中证1000"),
        "HO": ("000016", "上证50"),
    }

    code, name = underlying_map.get(symbol, ("000300", "沪深300"))

    try:
        # 获取最近的日线数据
        df = ak.index_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=(datetime.now() - timedelta(days=30)).strftime("%Y%m%d"),
            end_date=datetime.now().strftime("%Y%m%d")
        )

        if df is not None and not df.empty:
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else df.iloc[-1]

            close = float(latest['收盘'])
            prev_close = float(prev['收盘'])
            change = close - prev_close

            # 计算20日历史波动率
            if len(df) >= 20:
                returns = np.log(df['收盘'].astype(float) / df['收盘'].astype(float).shift(1))
                hv_20 = returns.tail(20).std() * np.sqrt(252)
            else:
                hv_20 = 0.18

            return {
                'price': close,
                'change': change,
                'change_pct': (change / prev_close) * 100 if prev_close else 0,
                'hv_20': hv_20,
                'success': True
            }
    except Exception as e:
        st.warning(f"获取标的数据失败: {e}")

    return {'price': 0, 'change': 0, 'change_pct': 0, 'hv_20': 0.18, 'success': False}


@st.cache_data(ttl=60)  # 缓存1分钟
def fetch_option_chain_data(symbol: str) -> pd.DataFrame:
    """获取期权链实时数据"""
    import akshare as ak

    option_board_map = {
        "IO": "沪深300股指期权",
        "MO": "中证1000股指期权",
        "HO": "上证50股指期权",
    }

    board_name = option_board_map.get(symbol, "沪深300股指期权")

    try:
        df = ak.option_finance_board(symbol=board_name)
        if df is not None and not df.empty:
            return df
    except Exception as e:
        st.warning(f"获取期权链失败: {e}")

    return pd.DataFrame()


def page_dashboard(symbol: str):
    """实时监控面板"""
    st.title("🏠 实时监控面板")

    config = OPTION_INSTRUMENTS[symbol]
    st.markdown(f"**{config['name']}** | 合约乘数: {config['multiplier']} | 交易时间: {config['trading_hours']}")

    # 获取真实数据
    with st.spinner("正在获取实时数据..."):
        underlying_data = fetch_underlying_price(symbol)
        option_df = fetch_option_chain_data(symbol)

    # 标的价格
    spot = underlying_data['price'] if underlying_data['success'] else 3900.0
    price_change = underlying_data['change'] if underlying_data['success'] else 0
    hv_20 = underlying_data['hv_20'] if underlying_data['success'] else 0.18

    # 计算ATM IV (从期权链数据估算)
    if not option_df.empty and spot > 0:
        try:
            # 找到最接近ATM的期权
            if '行权价' in option_df.columns:
                option_df['strike_diff'] = abs(option_df['行权价'].astype(float) - spot)
                atm_options = option_df.nsmallest(4, 'strike_diff')

                # 估算IV (简化处理：使用期权价格反推)
                if '最新价' in atm_options.columns:
                    atm_prices = atm_options['最新价'].astype(float).mean()
                    # 简单估算: IV ≈ 期权价格 / 标的价格 * 调整系数
                    current_iv = min(max((atm_prices / spot) * 8, 0.10), 0.50)
                else:
                    current_iv = 0.18
            else:
                current_iv = 0.18
        except:
            current_iv = 0.18
    else:
        current_iv = 0.18

    # IV百分位 (简化处理，实际应从历史数据计算)
    # 假设当前IV在15%-35%范围内对应0-100百分位
    iv_percentile = min(max((current_iv - 0.12) / 0.25 * 100, 0), 100)

    # 顶部指标卡片
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if underlying_data['success']:
            st.metric("标的现价", f"{spot:.2f}", f"{price_change:+.2f}")
        else:
            st.metric("标的现价", "获取失败", "请检查网络")

    with col2:
        iv_color = "🟢" if iv_percentile < 30 else ("🔴" if iv_percentile > 70 else "🟡")
        st.metric("ATM IV", f"{current_iv*100:.1f}%", f"{iv_color} P{iv_percentile:.0f}")

    with col3:
        st.metric("IV百分位", f"{iv_percentile:.0f}%",
                  "低" if iv_percentile < 30 else ("高" if iv_percentile > 70 else "中"))

    with col4:
        st.metric("HV(20)", f"{hv_20*100:.1f}%")

    with col5:
        iv_hv_ratio = current_iv / hv_20 if hv_20 > 0 else 1.0
        st.metric("IV/HV", f"{iv_hv_ratio:.2f}",
                  "溢价" if iv_hv_ratio > 1 else "折价")

    st.markdown("---")

    # 期权链展示
    st.subheader("📋 期权链 (T型报价)")

    # 使用真实数据或模拟数据
    if not option_df.empty:
        # 显示真实期权链
        display_cols = ['合约代码', '最新价', '涨跌幅', '成交量', '持仓量', '行权价'] if '合约代码' in option_df.columns else option_df.columns.tolist()
        available_cols = [col for col in display_cols if col in option_df.columns]
        st.dataframe(option_df[available_cols].head(20), use_container_width=True, hide_index=True)
    else:
        # 使用计算出的期权链
        strikes = [spot + i * 50 for i in range(-5, 6)]
        T = 30 / 365

        chain_data = []
        for strike in strikes:
            call_bs = BlackScholes(spot, strike, T, 0.02, current_iv, 0.025, 'call')
            put_bs = BlackScholes(spot, strike, T, 0.02, current_iv, 0.025, 'put')

            chain_data.append({
                'Call价格': f"{call_bs.price():.2f}",
                'Call Delta': f"{call_bs.delta():.3f}",
                'Call Theta': f"{call_bs.theta():.3f}",
                '行权价': f"{strike:.0f}",
                'Put Theta': f"{put_bs.theta():.3f}",
                'Put Delta': f"{put_bs.delta():.3f}",
                'Put价格': f"{put_bs.price():.2f}",
            })

        chain_df = pd.DataFrame(chain_data)
        st.dataframe(chain_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Greeks热力图
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Greeks分布")

        strikes = [spot + i * 50 for i in range(-5, 6)]
        T = 30 / 365
        greeks_data = calculate_greeks_batch(spot, strikes, T, 0.02, current_iv, 0.025)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=strikes, y=greeks_data['call_delta'],
                                  name='Call Delta', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=strikes, y=greeks_data['put_delta'],
                                  name='Put Delta', line=dict(color='red')))
        fig.update_layout(
            title='Delta vs 行权价',
            xaxis_title='行权价',
            yaxis_title='Delta',
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("策略建议")

        monitor = IVMonitor(symbol)
        iv_level = monitor.get_iv_level(iv_percentile)
        suggestions = monitor.get_strategy_suggestion(iv_percentile, iv_hv_ratio)

        st.info(f"**当前IV水平: {iv_level}**")

        for s in suggestions:
            st.markdown(f"• {s}")

    # 刷新按钮
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("🔄 刷新数据"):
            st.cache_data.clear()
            st.rerun()
    with col2:
        st.caption(f"数据更新时间: {datetime.now().strftime('%H:%M:%S')}")


def page_iv_analysis(symbol: str):
    """IV分析页面"""
    st.title("📈 IV分析")

    # 时间范围选择
    col1, col2 = st.columns(2)
    with col1:
        lookback = st.selectbox("回看周期", ["1个月", "3个月", "6个月", "1年", "2年"])

    lookback_days = {"1个月": 22, "3个月": 66, "6个月": 132, "1年": 252, "2年": 504}[lookback]

    # 生成模拟IV历史
    dates = pd.date_range(end=datetime.now(), periods=lookback_days, freq='D')
    iv_history = pd.Series(
        np.random.uniform(0.15, 0.35, lookback_days) +
        np.sin(np.arange(lookback_days) / 30) * 0.05,
        index=dates
    )

    # IV走势图
    st.subheader("IV历史走势")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3],
                        vertical_spacing=0.05)

    # IV曲线
    fig.add_trace(
        go.Scatter(x=dates, y=iv_history * 100, name='IV', line=dict(color='blue')),
        row=1, col=1
    )

    # 添加百分位区间
    iv_mean = iv_history.mean()
    iv_std = iv_history.std()
    fig.add_hline(y=iv_mean * 100, line_dash="dash", line_color="gray",
                  annotation_text=f"均值 {iv_mean*100:.1f}%", row=1, col=1)
    fig.add_hline(y=(iv_mean + iv_std) * 100, line_dash="dot", line_color="red",
                  annotation_text="+1 STD", row=1, col=1)
    fig.add_hline(y=(iv_mean - iv_std) * 100, line_dash="dot", line_color="green",
                  annotation_text="-1 STD", row=1, col=1)

    # IV百分位
    iv_percentile = iv_history.rolling(252, min_periods=20).apply(
        lambda x: (x < x.iloc[-1]).sum() / len(x) * 100
    )
    fig.add_trace(
        go.Scatter(x=dates, y=iv_percentile, name='IV百分位', fill='tozeroy',
                   line=dict(color='orange')),
        row=2, col=1
    )

    fig.update_layout(height=600, showlegend=True)
    fig.update_yaxes(title_text="IV (%)", row=1, col=1)
    fig.update_yaxes(title_text="百分位", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # IV统计
    st.subheader("IV统计")

    analyzer = IVAnalyzer()
    stats = analyzer.calculate_iv_stats(iv_history)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("当前IV", f"{stats.get('current', 0)*100:.1f}%")
    with col2:
        st.metric("平均IV", f"{stats.get('mean', 0)*100:.1f}%")
    with col3:
        st.metric("最低IV", f"{stats.get('min', 0)*100:.1f}%")
    with col4:
        st.metric("最高IV", f"{stats.get('max', 0)*100:.1f}%")

    # IV曲面（模拟）
    st.subheader("波动率曲面")

    strikes = np.arange(3600, 4200, 50)
    expiries = ['25%01', '2502', '2503', '2506']
    expiry_labels = ['1月', '2月', '3月', '6月']

    # 生成模拟曲面数据
    surface_data = np.zeros((len(strikes), len(expiries)))
    for i, strike in enumerate(strikes):
        for j, expiry in enumerate(expiries):
            # 简单的波动率微笑
            moneyness = strike / 3900
            smile = 0.02 * (moneyness - 1) ** 2
            term = 0.01 * j  # 期限结构
            surface_data[i, j] = (0.20 + smile + term) * 100

    fig = go.Figure(data=[go.Surface(z=surface_data.T, x=strikes, y=expiry_labels)])
    fig.update_layout(
        title='波动率曲面',
        scene=dict(
            xaxis_title='行权价',
            yaxis_title='到期月份',
            zaxis_title='IV (%)'
        ),
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)


def page_backtest(symbol: str):
    """策略回测页面"""
    st.title("🔬 策略回测")

    # 策略选择
    col1, col2 = st.columns(2)

    with col1:
        strategy_name = st.selectbox(
            "选择策略",
            list(STRATEGIES.keys()),
            format_func=lambda x: STRATEGIES[x].display_name
        )

    strategy_class = STRATEGIES[strategy_name]

    with col2:
        st.info(f"**{strategy_class.display_name}**\n\n{strategy_class.description}")

    st.markdown("---")

    # 参数设置
    st.subheader("策略参数")

    params = {}
    param_cols = st.columns(3)

    for i, param in enumerate(strategy_class.get_params()):
        with param_cols[i % 3]:
            if param.param_type == 'int':
                params[param.name] = st.slider(
                    param.label,
                    min_value=int(param.min_val) if param.min_val else 0,
                    max_value=int(param.max_val) if param.max_val else 100,
                    value=int(param.default),
                    step=int(param.step) if param.step else 1,
                    help=param.description
                )
            elif param.param_type == 'float':
                params[param.name] = st.slider(
                    param.label,
                    min_value=float(param.min_val) if param.min_val else 0.0,
                    max_value=float(param.max_val) if param.max_val else 1.0,
                    value=float(param.default),
                    step=float(param.step) if param.step else 0.01,
                    help=param.description
                )

    st.markdown("---")

    # 回测设置
    st.subheader("回测设置")

    col1, col2, col3 = st.columns(3)

    with col1:
        initial_capital = st.number_input("初始资金", value=1000000, step=100000)

    with col2:
        start_date = st.date_input("开始日期", value=datetime(2023, 1, 1))

    with col3:
        end_date = st.date_input("结束日期", value=datetime.now())

    # 运行回测
    if st.button("🚀 运行回测", type="primary"):
        with st.spinner("正在运行回测..."):
            # 生成模拟数据
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            np.random.seed(42)

            returns = np.random.normal(0.0002, 0.012, len(dates))
            prices = 3900 * np.cumprod(1 + returns)

            df = pd.DataFrame({
                'time': dates,
                'open': prices * (1 + np.random.uniform(-0.005, 0.005, len(dates))),
                'high': prices * (1 + np.random.uniform(0, 0.015, len(dates))),
                'low': prices * (1 - np.random.uniform(0, 0.015, len(dates))),
                'close': prices,
                'volume': np.random.uniform(1e9, 5e9, len(dates))
            })

            # 创建策略实例
            strategy = strategy_class(params=params, symbol=symbol)

            # 运行回测
            result = run_backtest_with_strategy(df, symbol, strategy, initial_capital)

            # 显示结果
            st.success("回测完成!")

            # 关键指标
            st.subheader("📊 关键指标")

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                color = "green" if result.total_return_pct > 0 else "red"
                st.metric("总收益率", f"{result.total_return_pct:.2f}%",
                         f"{result.total_pnl:,.0f}")

            with col2:
                st.metric("年化收益", f"{result.annual_return_pct:.2f}%")

            with col3:
                st.metric("最大回撤", f"{result.max_drawdown_pct:.2f}%")

            with col4:
                st.metric("夏普比率", f"{result.sharpe_ratio:.2f}")

            with col5:
                st.metric("胜率", f"{result.win_rate:.1f}%")

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric("交易次数", f"{result.trade_count}")

            with col2:
                st.metric("利润因子", f"{result.profit_factor:.2f}")

            with col3:
                st.metric("平均盈利", f"{result.avg_win:,.0f}")

            with col4:
                st.metric("平均亏损", f"{result.avg_loss:,.0f}")

            with col5:
                st.metric("总手续费", f"{result.total_commission:,.0f}")

            # 权益曲线
            st.subheader("📈 权益曲线")

            if result.equity_curve is not None:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=result.equity_curve['time'],
                    y=result.equity_curve['equity'],
                    name='账户权益',
                    line=dict(color='blue')
                ))
                fig.add_hline(y=initial_capital, line_dash="dash",
                             line_color="gray", annotation_text="初始资金")
                fig.update_layout(
                    title='账户权益曲线',
                    xaxis_title='日期',
                    yaxis_title='权益',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

            # 交易记录
            st.subheader("📝 交易记录")

            if result.trades:
                trades_df = pd.DataFrame([{
                    '交易ID': t.trade_id,
                    '入场时间': t.entry_time,
                    '出场时间': t.exit_time,
                    '入场价': f"{t.entry_underlying:.0f}",
                    '出场价': f"{t.exit_underlying:.0f}",
                    '损益': f"{t.pnl:,.0f}",
                    '收益率': f"{t.pnl_pct:.2f}%",
                    '出场原因': t.exit_reason,
                } for t in result.trades])

                st.dataframe(trades_df, use_container_width=True, hide_index=True)

            # 出场原因统计
            if result.exit_reason_stats:
                st.subheader("📊 出场原因统计")

                exit_df = pd.DataFrame([
                    {'原因': k, '次数': v['count'], '总损益': f"{v['pnl']:,.0f}"}
                    for k, v in result.exit_reason_stats.items()
                ])
                st.dataframe(exit_df, use_container_width=True, hide_index=True)


def page_payoff_analyzer(symbol: str):
    """损益分析器页面"""
    st.title("📐 期权损益分析器")

    # 标的价格
    col1, col2, col3 = st.columns(3)

    with col1:
        spot = st.number_input("标的现价", value=3900.0, step=10.0)

    with col2:
        iv = st.slider("隐含波动率 (%)", 10, 50, 20) / 100

    with col3:
        days = st.slider("距到期天数", 1, 90, 30)

    T = days / 365

    st.markdown("---")

    # 期权腿设置
    st.subheader("构建期权组合")

    legs = []

    for i in range(4):
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            enabled = st.checkbox(f"腿 {i+1}", value=(i < 2), key=f"leg_{i}_enabled")

        if enabled:
            with col2:
                opt_type = st.selectbox("类型", ["Call", "Put"], key=f"leg_{i}_type")

            with col3:
                position = st.selectbox("方向", ["买入", "卖出"], key=f"leg_{i}_pos")

            with col4:
                strike = st.number_input("行权价", value=spot, step=50.0, key=f"leg_{i}_strike")

            with col5:
                qty = st.number_input("数量", value=1, min_value=1, key=f"leg_{i}_qty")

            legs.append({
                'type': opt_type.lower(),
                'position': 1 if position == "买入" else -1,
                'strike': strike,
                'quantity': qty
            })

    if legs:
        st.markdown("---")

        # 计算损益图
        st.subheader("到期损益图")

        # 价格范围
        price_range = np.linspace(spot * 0.85, spot * 1.15, 100)

        # 计算到期损益
        payoff = np.zeros_like(price_range)

        for leg in legs:
            bs = BlackScholes(spot, leg['strike'], T, 0.02, iv, 0.025, leg['type'])
            entry_price = bs.price()

            for i, price in enumerate(price_range):
                if leg['type'] == 'call':
                    intrinsic = max(0, price - leg['strike'])
                else:
                    intrinsic = max(0, leg['strike'] - price)

                leg_pnl = (intrinsic - entry_price) * leg['position'] * leg['quantity']
                payoff[i] += leg_pnl

        # 绘图
        fig = go.Figure()

        # 损益曲线
        colors = np.where(payoff >= 0, 'green', 'red')
        fig.add_trace(go.Scatter(
            x=price_range,
            y=payoff,
            mode='lines',
            name='到期损益',
            line=dict(color='blue', width=2)
        ))

        # 零线
        fig.add_hline(y=0, line_dash="dash", line_color="gray")

        # 当前价格
        fig.add_vline(x=spot, line_dash="dot", line_color="orange",
                     annotation_text=f"现价 {spot:.0f}")

        # 盈亏平衡点
        for i in range(len(payoff) - 1):
            if payoff[i] * payoff[i+1] < 0:  # 穿越零线
                be_price = price_range[i]
                fig.add_vline(x=be_price, line_dash="dot", line_color="purple",
                             annotation_text=f"BEP {be_price:.0f}")

        fig.update_layout(
            title='期权组合到期损益图',
            xaxis_title='标的价格',
            yaxis_title='损益 (点)',
            height=500,
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

        # 组合统计
        st.subheader("组合统计")

        max_profit = payoff.max()
        max_loss = payoff.min()
        net_premium = sum(
            BlackScholes(spot, leg['strike'], T, 0.02, iv, 0.025, leg['type']).price() * leg['position']
            for leg in legs
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("净权利金", f"{net_premium:.2f}",
                     "收入" if net_premium > 0 else "支出")

        with col2:
            st.metric("最大盈利", f"{max_profit:.2f}" if max_profit < 1000 else "无限")

        with col3:
            st.metric("最大亏损", f"{max_loss:.2f}" if max_loss > -1000 else "无限")

        with col4:
            risk_reward = abs(max_profit / max_loss) if max_loss != 0 else float('inf')
            st.metric("盈亏比", f"{risk_reward:.2f}")

        # Greeks汇总
        st.subheader("组合Greeks")

        total_delta = sum(
            BlackScholes(spot, leg['strike'], T, 0.02, iv, 0.025, leg['type']).delta() * leg['position'] * leg['quantity']
            for leg in legs
        )
        total_gamma = sum(
            BlackScholes(spot, leg['strike'], T, 0.02, iv, 0.025, leg['type']).gamma() * leg['position'] * leg['quantity']
            for leg in legs
        )
        total_theta = sum(
            BlackScholes(spot, leg['strike'], T, 0.02, iv, 0.025, leg['type']).theta() * leg['position'] * leg['quantity']
            for leg in legs
        )
        total_vega = sum(
            BlackScholes(spot, leg['strike'], T, 0.02, iv, 0.025, leg['type']).vega() * leg['position'] * leg['quantity']
            for leg in legs
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Delta", f"{total_delta:.4f}")
        with col2:
            st.metric("Gamma", f"{total_gamma:.6f}")
        with col3:
            st.metric("Theta", f"{total_theta:.4f}")
        with col4:
            st.metric("Vega", f"{total_vega:.4f}")


def page_data_management(symbol: str):
    """数据管理页面"""
    st.title("💾 数据管理")

    # 初始化数据库
    if st.button("初始化数据库"):
        init_database()
        st.success("数据库初始化完成!")

    st.markdown("---")

    # 数据状态
    st.subheader("数据状态")

    status = get_data_status()
    if not status.empty:
        st.dataframe(status, use_container_width=True, hide_index=True)
    else:
        st.info("暂无数据，请先下载数据")

    st.markdown("---")

    # 数据下载
    st.subheader("下载数据")

    col1, col2 = st.columns(2)

    with col1:
        download_type = st.selectbox(
            "数据类型",
            ["标的指数日线", "期权实时行情"]
        )

    with col2:
        if download_type == "标的指数日线":
            underlying_code = {
                "IO": "000300",
                "MO": "000852",
                "HO": "000016"
            }.get(symbol, "000300")
            st.info(f"将下载 {symbol} 标的指数 ({underlying_code})")

    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input("开始日期", value=datetime(2022, 1, 1), key="dl_start")

    with col2:
        end_date = st.date_input("结束日期", value=datetime.now(), key="dl_end")

    if st.button("📥 下载数据"):
        with st.spinner("正在下载数据..."):
            if download_type == "标的指数日线":
                underlying_code = {
                    "IO": "000300",
                    "MO": "000852",
                    "HO": "000016"
                }.get(symbol, "000300")

                df = download_underlying_daily(
                    underlying_code,
                    start_date.strftime("%Y%m%d"),
                    end_date.strftime("%Y%m%d")
                )

                if df is not None:
                    st.success(f"下载完成! 共 {len(df)} 条记录")
                    st.dataframe(df.head(10), use_container_width=True)
                else:
                    st.error("下载失败，请检查网络连接")

            elif download_type == "期权实时行情":
                df = download_option_realtime(symbol)
                if df is not None:
                    st.success(f"获取成功! 共 {len(df)} 条记录")
                    st.dataframe(df.head(20), use_container_width=True)
                else:
                    st.error("获取失败")

    st.markdown("---")

    # 数据导出
    st.subheader("数据导出")

    export_type = st.selectbox("导出类型", ["标的指数数据"])

    if st.button("📤 导出CSV"):
        df = load_underlying_daily(
            {"IO": "000300", "MO": "000852", "HO": "000016"}.get(symbol, "000300")
        )

        if not df.empty:
            csv = df.to_csv(index=False)
            st.download_button(
                label="下载CSV文件",
                data=csv,
                file_name=f"{symbol}_underlying_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("无数据可导出")


if __name__ == "__main__":
    main()
