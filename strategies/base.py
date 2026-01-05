"""
期权策略基类
定义策略框架、持仓管理、信号生成等基础功能
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from greeks import BlackScholes, ImpliedVolatility
from config import OPTION_INSTRUMENTS, RISK_FREE_RATE


@dataclass
class StrategyParam:
    """策略参数定义"""
    name: str               # 参数名
    label: str              # 显示标签
    default: Any            # 默认值
    min_val: Any = None     # 最小值
    max_val: Any = None     # 最大值
    step: Any = None        # 步长
    param_type: str = "float"  # 参数类型: int, float, bool, select
    options: List = None    # select类型的选项
    description: str = ""   # 参数说明


@dataclass
class OptionLeg:
    """期权腿（组合中的单个期权）"""
    option_type: str        # 'call' 或 'put'
    strike: float           # 行权价
    expiry: str             # 到期日
    position: int           # 头寸方向：1=买入，-1=卖出
    quantity: int = 1       # 数量
    entry_price: float = 0  # 入场价格
    entry_time: datetime = None
    current_price: float = 0
    iv: float = 0           # 隐含波动率
    delta: float = 0
    gamma: float = 0
    theta: float = 0
    vega: float = 0

    @property
    def is_long(self) -> bool:
        return self.position > 0

    @property
    def is_short(self) -> bool:
        return self.position < 0

    @property
    def pnl(self) -> float:
        """计算该腿的损益"""
        return (self.current_price - self.entry_price) * self.position * self.quantity

    def update_greeks(self, spot: float, T: float, r: float, sigma: float, q: float = 0):
        """更新Greeks"""
        bs = BlackScholes(spot, self.strike, T, r, sigma, q, self.option_type)
        self.delta = bs.delta() * self.position * self.quantity
        self.gamma = bs.gamma() * self.position * self.quantity
        self.theta = bs.theta() * self.position * self.quantity
        self.vega = bs.vega() * self.position * self.quantity


@dataclass
class OptionPosition:
    """期权持仓（可包含多腿）"""
    legs: List[OptionLeg] = field(default_factory=list)
    entry_time: datetime = None
    entry_underlying_price: float = 0
    entry_iv: float = 0
    strategy_name: str = ""
    tag: str = ""

    @property
    def net_delta(self) -> float:
        return sum(leg.delta for leg in self.legs)

    @property
    def net_gamma(self) -> float:
        return sum(leg.gamma for leg in self.legs)

    @property
    def net_theta(self) -> float:
        return sum(leg.theta for leg in self.legs)

    @property
    def net_vega(self) -> float:
        return sum(leg.vega for leg in self.legs)

    @property
    def total_pnl(self) -> float:
        return sum(leg.pnl for leg in self.legs)

    @property
    def max_profit(self) -> float:
        """最大盈利（需要根据策略类型计算）"""
        # 这个需要在具体策略中实现
        return float('inf')

    @property
    def max_loss(self) -> float:
        """最大亏损（需要根据策略类型计算）"""
        return float('inf')

    def get_entry_cost(self) -> float:
        """计算入场成本（净权利金）"""
        return sum(leg.entry_price * leg.position * leg.quantity for leg in self.legs)


@dataclass
class Signal:
    """交易信号"""
    action: str             # 'open', 'close', 'adjust'
    legs: List[OptionLeg] = field(default_factory=list)
    tag: str = ""           # 信号标签
    reason: str = ""        # 信号原因
    timestamp: datetime = None
    underlying_price: float = 0
    iv: float = 0


class BaseOptionStrategy(ABC):
    """期权策略基类"""

    # 策略元信息
    name: str = "base"
    display_name: str = "基础策略"
    description: str = "期权策略基类"
    version: str = "1.0"
    author: str = ""
    strategy_type: str = "neutral"  # bullish, bearish, neutral, volatility

    # 预热期
    warmup_bars: int = 50

    def __init__(self, params: Dict = None, symbol: str = "IO"):
        """
        初始化策略

        Args:
            params: 策略参数
            symbol: 期权品种
        """
        self.symbol = symbol
        self.config = OPTION_INSTRUMENTS.get(symbol, {})
        self.multiplier = self.config.get('multiplier', 100)
        self.dividend_yield = self.config.get('dividend_yield', 0.025)

        # 合并默认参数和传入参数
        self.params = self._get_default_params()
        if params:
            self.params.update(params)

        # 持仓状态
        self.position: Optional[OptionPosition] = None
        self.trade_history: List[Dict] = []

        # 状态变量
        self.current_bar: int = 0
        self.current_time: datetime = None

    @classmethod
    @abstractmethod
    def get_params(cls) -> List[StrategyParam]:
        """获取策略参数定义"""
        pass

    def _get_default_params(self) -> Dict:
        """获取默认参数值"""
        params = {}
        for p in self.get_params():
            params[p.name] = p.default
        return params

    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算策略所需指标

        Args:
            df: 包含OHLCV的DataFrame

        Returns:
            添加了指标的DataFrame
        """
        pass

    @abstractmethod
    def generate_signal(self, idx: int, df: pd.DataFrame,
                        option_chain: pd.DataFrame,
                        iv_data: Dict) -> Optional[Signal]:
        """
        生成交易信号

        Args:
            idx: 当前bar索引
            df: 标的数据（含指标）
            option_chain: 期权链数据
            iv_data: IV相关数据

        Returns:
            交易信号或None
        """
        pass

    def on_bar(self, idx: int, df: pd.DataFrame,
               option_chain: pd.DataFrame,
               iv_data: Dict,
               capital: float) -> Optional[Signal]:
        """
        每根K线调用的主函数

        Args:
            idx: 当前bar索引
            df: 标的数据
            option_chain: 期权链数据
            iv_data: IV数据
            capital: 当前资金

        Returns:
            交易信号或None
        """
        self.current_bar = idx
        self.current_time = df.iloc[idx]['time'] if 'time' in df.columns else None

        # 预热期不交易
        if idx < self.warmup_bars:
            return None

        # 更新持仓Greeks（如果有持仓）
        if self.position:
            self._update_position_greeks(df.iloc[idx], option_chain)

        # 生成信号
        return self.generate_signal(idx, df, option_chain, iv_data)

    def _update_position_greeks(self, bar: pd.Series, option_chain: pd.DataFrame):
        """更新持仓的Greeks"""
        if not self.position or not self.position.legs:
            return

        spot = bar['close']

        for leg in self.position.legs:
            # 计算距到期天数
            expiry_date = datetime.strptime(f"20{leg.expiry}15", "%Y%m%d")
            days = max(1, (expiry_date - datetime.now()).days)
            T = days / 365

            # 获取当前IV（简化处理，使用入场IV）
            sigma = leg.iv if leg.iv > 0 else 0.2

            leg.update_greeks(spot, T, RISK_FREE_RATE, sigma, self.dividend_yield)

    def open_position(self, legs: List[OptionLeg], underlying_price: float,
                      iv: float, tag: str = "") -> OptionPosition:
        """
        开仓

        Args:
            legs: 期权腿列表
            underlying_price: 标的价格
            iv: 当前IV
            tag: 标签

        Returns:
            新建的持仓
        """
        self.position = OptionPosition(
            legs=legs,
            entry_time=self.current_time,
            entry_underlying_price=underlying_price,
            entry_iv=iv,
            strategy_name=self.name,
            tag=tag
        )
        return self.position

    def close_position(self, reason: str = "") -> Dict:
        """
        平仓

        Args:
            reason: 平仓原因

        Returns:
            交易记录
        """
        if not self.position:
            return {}

        trade_record = {
            'strategy': self.name,
            'entry_time': self.position.entry_time,
            'exit_time': self.current_time,
            'entry_underlying': self.position.entry_underlying_price,
            'entry_iv': self.position.entry_iv,
            'legs': [(leg.option_type, leg.strike, leg.position,
                     leg.entry_price, leg.current_price) for leg in self.position.legs],
            'pnl': self.position.total_pnl,
            'exit_reason': reason,
            'tag': self.position.tag,
        }

        self.trade_history.append(trade_record)
        self.position = None

        return trade_record

    def select_strike(self, spot: float, option_chain: pd.DataFrame,
                      target_delta: float, option_type: str,
                      T: float, sigma: float) -> float:
        """
        根据目标Delta选择行权价

        Args:
            spot: 标的现价
            option_chain: 期权链
            target_delta: 目标Delta
            option_type: 期权类型
            T: 到期时间
            sigma: 波动率

        Returns:
            选中的行权价
        """
        if option_chain is None or option_chain.empty:
            # 无期权链数据时，使用理论计算
            from greeks import find_strike_by_delta
            return find_strike_by_delta(spot, T, RISK_FREE_RATE, sigma,
                                        target_delta, self.dividend_yield, option_type)

        # 从期权链中选择
        strikes = sorted(option_chain['strike'].unique())

        best_strike = strikes[0]
        min_diff = float('inf')

        for strike in strikes:
            bs = BlackScholes(spot, strike, T, RISK_FREE_RATE, sigma,
                             self.dividend_yield, option_type)
            delta = bs.delta()
            diff = abs(delta - target_delta)

            if diff < min_diff:
                min_diff = diff
                best_strike = strike

        return best_strike

    def get_option_price(self, strike: float, option_type: str,
                         option_chain: pd.DataFrame,
                         spot: float, T: float, sigma: float) -> float:
        """
        获取期权价格

        Args:
            strike: 行权价
            option_type: 期权类型
            option_chain: 期权链
            spot: 标的现价
            T: 到期时间
            sigma: 波动率

        Returns:
            期权价格
        """
        if option_chain is not None and not option_chain.empty:
            # 从期权链获取实际价格
            match = option_chain[
                (option_chain['strike'] == strike) &
                (option_chain['option_type'] == option_type)
            ]
            if not match.empty:
                price = match.iloc[0].get('last_price', match.iloc[0].get('close', 0))
                if price > 0:
                    return price

        # 使用BS理论价格
        bs = BlackScholes(spot, strike, T, RISK_FREE_RATE, sigma,
                         self.dividend_yield, option_type)
        return bs.price()

    def calculate_spread_metrics(self, buy_strike: float, sell_strike: float,
                                 buy_price: float, sell_price: float,
                                 option_type: str = 'call') -> Dict:
        """
        计算价差组合的关键指标

        Args:
            buy_strike: 买入腿行权价
            sell_strike: 卖出腿行权价
            buy_price: 买入腿价格
            sell_price: 卖出腿价格
            option_type: 期权类型

        Returns:
            指标字典
        """
        net_premium = buy_price - sell_price  # 净权利金支出
        spread_width = abs(sell_strike - buy_strike)

        if option_type == 'call':
            # Bull Call Spread
            max_profit = spread_width - net_premium
            max_loss = net_premium
        else:
            # Bear Put Spread
            max_profit = spread_width - net_premium
            max_loss = net_premium

        return {
            'net_premium': net_premium,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'risk_reward': max_profit / max_loss if max_loss > 0 else float('inf'),
            'breakeven': buy_strike + net_premium if option_type == 'call' else buy_strike - net_premium,
        }

    @staticmethod
    def calculate_ema(series: pd.Series, period: int) -> pd.Series:
        """计算EMA"""
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_ma(series: pd.Series, period: int) -> pd.Series:
        """计算MA"""
        return series.rolling(window=period).mean()

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算ATR"""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算ADX"""
        high = df['high']
        low = df['low']
        close = df['close']

        # +DM and -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        plus_dm[(plus_dm < minus_dm)] = 0
        minus_dm[(minus_dm < plus_dm)] = 0

        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        # +DI and -DI
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        # DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return adx
