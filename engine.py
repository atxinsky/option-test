"""
期权策略回测引擎
支持多腿期权组合的回测、Greeks风险分析、统计报告
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from greeks import BlackScholes, calculate_historical_volatility
from iv_monitor import IVMonitor, create_iv_dashboard_data
from config import OPTION_INSTRUMENTS, RISK_FREE_RATE
from strategies.base import BaseOptionStrategy, Signal, OptionPosition


@dataclass
class Trade:
    """交易记录"""
    trade_id: int
    strategy: str
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_underlying: float
    exit_underlying: float
    entry_iv: float
    exit_iv: float
    legs: List[Tuple]          # [(type, strike, position, entry_price, exit_price), ...]
    pnl: float                 # 净损益（点数）
    pnl_pct: float             # 收益率
    commission: float          # 手续费
    holding_days: int
    exit_reason: str
    tag: str


@dataclass
class BacktestResult:
    """回测结果"""
    symbol: str
    strategy: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float

    # 交易列表
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.DataFrame = None
    daily_returns: pd.Series = None

    # 性能指标
    total_pnl: float = 0
    total_return_pct: float = 0
    annual_return_pct: float = 0
    max_drawdown_pct: float = 0
    max_drawdown_val: float = 0
    sharpe_ratio: float = 0
    sortino_ratio: float = 0
    calmar_ratio: float = 0
    win_rate: float = 0
    profit_factor: float = 0
    avg_win: float = 0
    avg_loss: float = 0
    max_win: float = 0
    max_loss: float = 0
    avg_holding_days: float = 0
    total_commission: float = 0
    trade_count: int = 0

    # 分组统计
    monthly_stats: pd.DataFrame = None
    exit_reason_stats: Dict = None


class OptionBacktestEngine:
    """期权回测引擎"""

    def __init__(self, symbol: str = "IO", initial_capital: float = 1000000):
        """
        初始化回测引擎

        Args:
            symbol: 期权品种
            initial_capital: 初始资金
        """
        self.symbol = symbol
        self.config = OPTION_INSTRUMENTS.get(symbol, {})
        self.multiplier = self.config.get('multiplier', 100)
        self.commission = self.config.get('commission_rate', 15)
        self.dividend_yield = self.config.get('dividend_yield', 0.025)

        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.equity_history = []

        self.trades: List[Trade] = []
        self.trade_id = 0

        self.iv_monitor = IVMonitor(symbol)

    def run_backtest(self, df: pd.DataFrame, strategy: BaseOptionStrategy,
                     iv_history: pd.Series = None) -> BacktestResult:
        """
        运行回测

        Args:
            df: 标的日线数据（含time, open, high, low, close, volume）
            strategy: 策略实例
            iv_history: IV历史数据（可选）

        Returns:
            回测结果
        """
        # 重置状态
        self.capital = self.initial_capital
        self.equity_history = []
        self.trades = []
        self.trade_id = 0

        # 计算策略指标
        df = strategy.calculate_indicators(df)

        # 模拟IV历史（如果未提供）
        if iv_history is None:
            df['hv_20'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252)
            iv_history = df['hv_20'] * 1.1  # 简单假设IV略高于HV

        # 逐根K线回测
        for idx in range(len(df)):
            bar = df.iloc[idx]
            spot = bar['close']

            # 计算当前IV数据
            iv_data = self._get_iv_data(df, idx, iv_history)

            # 生成模拟期权链（简化处理）
            option_chain = self._generate_mock_option_chain(
                spot, iv_data['current_iv'], iv_data['days_to_expiry']
            )

            # 调用策略
            signal = strategy.on_bar(idx, df, option_chain, iv_data)

            # 处理信号
            if signal:
                self._process_signal(signal, strategy, bar, iv_data)

            # 更新权益
            self._update_equity(strategy, spot, iv_data)

            # 记录权益历史
            self.equity_history.append({
                'time': bar['time'] if 'time' in bar else idx,
                'equity': self.capital,
                'underlying': spot,
                'position_pnl': strategy.position.total_pnl if strategy.position else 0
            })

        # 强制平仓剩余持仓
        if strategy.position:
            last_bar = df.iloc[-1]
            self._force_close(strategy, last_bar)

        # 计算统计指标
        return self._calculate_statistics(df, strategy)

    def _get_iv_data(self, df: pd.DataFrame, idx: int,
                     iv_history: pd.Series) -> Dict:
        """获取IV相关数据"""
        # 当前IV
        if idx < len(iv_history):
            current_iv = iv_history.iloc[idx] if not pd.isna(iv_history.iloc[idx]) else 0.2
        else:
            current_iv = 0.2

        # IV百分位（基于历史）
        if idx > 252:
            past_iv = iv_history.iloc[max(0, idx-504):idx]
            iv_percentile = (past_iv < current_iv).sum() / len(past_iv) * 100
        else:
            iv_percentile = 50

        # HV
        hv_20 = df.iloc[idx].get('hv_20', current_iv * 0.9) if 'hv_20' in df.columns else current_iv * 0.9

        # 假设到期天数（简化）
        days_to_expiry = 30

        return {
            'current_iv': current_iv,
            'iv_percentile': iv_percentile,
            'hv_20': hv_20,
            'hv_60': hv_20,
            'days_to_expiry': days_to_expiry,
        }

    def _generate_mock_option_chain(self, spot: float, iv: float,
                                    days_to_expiry: int) -> pd.DataFrame:
        """生成模拟期权链"""
        # 生成行权价序列
        base_strike = round(spot / 50) * 50
        strikes = [base_strike + i * 50 for i in range(-5, 6)]

        T = days_to_expiry / 365
        expiry = (datetime.now() + timedelta(days=days_to_expiry)).strftime("%y%m")

        chain_data = []
        for strike in strikes:
            # 计算Call价格
            call_bs = BlackScholes(spot, strike, T, RISK_FREE_RATE, iv,
                                   self.dividend_yield, 'call')
            put_bs = BlackScholes(spot, strike, T, RISK_FREE_RATE, iv,
                                  self.dividend_yield, 'put')

            # Call
            chain_data.append({
                'strike': strike,
                'expiry': expiry,
                'option_type': 'call',
                'last_price': call_bs.price(),
                'close': call_bs.price(),
                'delta': call_bs.delta(),
                'gamma': call_bs.gamma(),
                'theta': call_bs.theta(),
                'vega': call_bs.vega(),
            })

            # Put
            chain_data.append({
                'strike': strike,
                'expiry': expiry,
                'option_type': 'put',
                'last_price': put_bs.price(),
                'close': put_bs.price(),
                'delta': put_bs.delta(),
                'gamma': put_bs.gamma(),
                'theta': put_bs.theta(),
                'vega': put_bs.vega(),
            })

        return pd.DataFrame(chain_data)

    def _process_signal(self, signal: Signal, strategy: BaseOptionStrategy,
                        bar: pd.Series, iv_data: Dict):
        """处理交易信号"""
        if signal.action == 'open' and not strategy.position:
            # 开仓
            strategy.open_position(
                legs=signal.legs,
                underlying_price=signal.underlying_price,
                iv=signal.iv,
                tag=signal.tag
            )

            # 扣除手续费（开仓）
            commission = len(signal.legs) * self.commission
            self.capital -= commission

        elif signal.action == 'close' and strategy.position:
            # 平仓
            trade_record = strategy.close_position(reason=signal.reason)

            if trade_record:
                # 计算损益
                pnl_points = trade_record.get('pnl', 0)
                pnl_value = pnl_points * self.multiplier

                # 手续费（平仓）
                commission = len(trade_record.get('legs', [])) * self.commission
                net_pnl = pnl_value - commission

                self.capital += net_pnl

                # 记录交易
                self.trade_id += 1
                trade = Trade(
                    trade_id=self.trade_id,
                    strategy=strategy.name,
                    symbol=self.symbol,
                    entry_time=trade_record.get('entry_time'),
                    exit_time=trade_record.get('exit_time'),
                    entry_underlying=trade_record.get('entry_underlying', 0),
                    exit_underlying=bar['close'],
                    entry_iv=trade_record.get('entry_iv', 0),
                    exit_iv=iv_data.get('current_iv', 0),
                    legs=trade_record.get('legs', []),
                    pnl=net_pnl,
                    pnl_pct=net_pnl / self.initial_capital * 100,
                    commission=commission,
                    holding_days=0,  # 简化
                    exit_reason=trade_record.get('exit_reason', ''),
                    tag=trade_record.get('tag', '')
                )
                self.trades.append(trade)

    def _update_equity(self, strategy: BaseOptionStrategy, spot: float,
                       iv_data: Dict):
        """更新权益（含未实现盈亏）"""
        # 如果有持仓，更新当前价格和Greeks
        if strategy.position and strategy.position.legs:
            T = iv_data.get('days_to_expiry', 30) / 365
            current_iv = iv_data.get('current_iv', 0.2)

            for leg in strategy.position.legs:
                bs = BlackScholes(spot, leg.strike, T, RISK_FREE_RATE,
                                  current_iv, self.dividend_yield, leg.option_type)
                leg.current_price = bs.price()

    def _force_close(self, strategy: BaseOptionStrategy, bar: pd.Series):
        """强制平仓"""
        if strategy.position:
            # 模拟平仓
            for leg in strategy.position.legs:
                leg.current_price = leg.entry_price  # 简化处理

            trade_record = strategy.close_position(reason="回测结束强制平仓")

            if trade_record:
                pnl_points = trade_record.get('pnl', 0)
                pnl_value = pnl_points * self.multiplier
                commission = len(trade_record.get('legs', [])) * self.commission

                self.capital += pnl_value - commission

                self.trade_id += 1
                trade = Trade(
                    trade_id=self.trade_id,
                    strategy=strategy.name,
                    symbol=self.symbol,
                    entry_time=trade_record.get('entry_time'),
                    exit_time=bar.get('time', datetime.now()),
                    entry_underlying=trade_record.get('entry_underlying', 0),
                    exit_underlying=bar['close'],
                    entry_iv=trade_record.get('entry_iv', 0),
                    exit_iv=0,
                    legs=trade_record.get('legs', []),
                    pnl=pnl_value - commission,
                    pnl_pct=(pnl_value - commission) / self.initial_capital * 100,
                    commission=commission,
                    holding_days=0,
                    exit_reason="回测结束",
                    tag="force_close"
                )
                self.trades.append(trade)

    def _calculate_statistics(self, df: pd.DataFrame,
                              strategy: BaseOptionStrategy) -> BacktestResult:
        """计算统计指标"""
        equity_df = pd.DataFrame(self.equity_history)

        result = BacktestResult(
            symbol=self.symbol,
            strategy=strategy.name,
            start_date=df.iloc[0].get('time', datetime.now()),
            end_date=df.iloc[-1].get('time', datetime.now()),
            initial_capital=self.initial_capital,
            final_capital=self.capital,
            trades=self.trades,
            equity_curve=equity_df
        )

        # 基础指标
        result.total_pnl = self.capital - self.initial_capital
        result.total_return_pct = result.total_pnl / self.initial_capital * 100
        result.trade_count = len(self.trades)

        # 年化收益率
        if len(df) > 0:
            days = len(df)
            result.annual_return_pct = result.total_return_pct * 252 / days

        # 最大回撤
        if not equity_df.empty:
            equity = equity_df['equity']
            running_max = equity.expanding().max()
            drawdown = (running_max - equity) / running_max * 100
            result.max_drawdown_pct = drawdown.max()
            result.max_drawdown_val = (running_max - equity).max()

        # 夏普比率
        if not equity_df.empty and len(equity_df) > 1:
            daily_returns = equity_df['equity'].pct_change().dropna()
            if len(daily_returns) > 0 and daily_returns.std() > 0:
                result.sharpe_ratio = (daily_returns.mean() * 252 - 0.02) / (daily_returns.std() * np.sqrt(252))
                result.daily_returns = daily_returns

        # 索提诺比率
        if result.daily_returns is not None:
            negative_returns = result.daily_returns[result.daily_returns < 0]
            if len(negative_returns) > 0 and negative_returns.std() > 0:
                result.sortino_ratio = (result.daily_returns.mean() * 252 - 0.02) / (negative_returns.std() * np.sqrt(252))

        # 卡尔玛比率
        if result.max_drawdown_pct > 0:
            result.calmar_ratio = result.annual_return_pct / result.max_drawdown_pct

        # 交易统计
        if self.trades:
            wins = [t for t in self.trades if t.pnl > 0]
            losses = [t for t in self.trades if t.pnl <= 0]

            result.win_rate = len(wins) / len(self.trades) * 100 if self.trades else 0

            total_wins = sum(t.pnl for t in wins)
            total_losses = abs(sum(t.pnl for t in losses))

            result.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            result.avg_win = total_wins / len(wins) if wins else 0
            result.avg_loss = total_losses / len(losses) if losses else 0
            result.max_win = max(t.pnl for t in self.trades) if self.trades else 0
            result.max_loss = min(t.pnl for t in self.trades) if self.trades else 0
            result.total_commission = sum(t.commission for t in self.trades)

            # 出场原因统计
            exit_reasons = {}
            for t in self.trades:
                reason = t.exit_reason or 'unknown'
                if reason not in exit_reasons:
                    exit_reasons[reason] = {'count': 0, 'pnl': 0}
                exit_reasons[reason]['count'] += 1
                exit_reasons[reason]['pnl'] += t.pnl
            result.exit_reason_stats = exit_reasons

        return result


def run_backtest_with_strategy(df: pd.DataFrame, symbol: str,
                               strategy: BaseOptionStrategy,
                               initial_capital: float = 1000000,
                               iv_history: pd.Series = None) -> BacktestResult:
    """
    便捷函数：运行策略回测

    Args:
        df: 标的日线数据
        symbol: 期权品种
        strategy: 策略实例
        initial_capital: 初始资金
        iv_history: IV历史数据

    Returns:
        回测结果
    """
    engine = OptionBacktestEngine(symbol, initial_capital)
    return engine.run_backtest(df, strategy, iv_history)


if __name__ == "__main__":
    # 测试回测引擎
    print("期权回测引擎测试")
    print("=" * 50)

    # 生成模拟数据
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)

    # 模拟沪深300指数走势
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

    # 导入策略
    from strategies.bull_call_spread import BullCallSpread

    strategy = BullCallSpread(symbol="IO")

    # 运行回测
    result = run_backtest_with_strategy(df, "IO", strategy)

    print(f"\n回测结果:")
    print(f"  初始资金: {result.initial_capital:,.0f}")
    print(f"  最终资金: {result.final_capital:,.0f}")
    print(f"  总收益: {result.total_pnl:,.0f} ({result.total_return_pct:.2f}%)")
    print(f"  年化收益: {result.annual_return_pct:.2f}%")
    print(f"  最大回撤: {result.max_drawdown_pct:.2f}%")
    print(f"  夏普比率: {result.sharpe_ratio:.2f}")
    print(f"  交易次数: {result.trade_count}")
    print(f"  胜率: {result.win_rate:.1f}%")
    print(f"  利润因子: {result.profit_factor:.2f}")
