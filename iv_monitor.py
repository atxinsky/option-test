"""
IV监控模块
隐含波动率分析、百分位计算、波动率曲面等功能
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

from greeks import BlackScholes, ImpliedVolatility, calculate_historical_volatility
from config import OPTION_INSTRUMENTS, RISK_FREE_RATE, IV_THRESHOLDS


@dataclass
class IVSnapshot:
    """IV快照数据"""
    symbol: str
    timestamp: datetime
    atm_iv: float           # ATM隐含波动率
    iv_percentile: float    # IV百分位（0-100）
    hv_20: float            # 20日历史波动率
    hv_60: float            # 60日历史波动率
    iv_hv_ratio: float      # IV/HV比率
    vix: float              # VIX指数
    skew: float             # 波动率偏斜
    term_structure: Dict    # 期限结构


class IVMonitor:
    """IV监控器"""

    def __init__(self, symbol: str = "IO"):
        """
        初始化监控器

        Args:
            symbol: 期权品种代码
        """
        self.symbol = symbol
        self.config = OPTION_INSTRUMENTS.get(symbol, {})
        self.dividend_yield = self.config.get('dividend_yield', 0.025)

    def calculate_atm_iv(self, spot: float, option_chain: pd.DataFrame,
                         days_to_expiry: int) -> float:
        """
        计算ATM隐含波动率

        Args:
            spot: 标的现价
            option_chain: 期权链数据
            days_to_expiry: 距到期天数

        Returns:
            ATM IV
        """
        if option_chain is None or option_chain.empty:
            return 0.0

        # 找到最接近现价的行权价
        strikes = option_chain['strike'].unique()
        atm_strike = min(strikes, key=lambda x: abs(x - spot))

        # 获取ATM期权价格
        atm_options = option_chain[option_chain['strike'] == atm_strike]

        T = days_to_expiry / 365

        ivs = []

        for _, row in atm_options.iterrows():
            price = row.get('last_price', row.get('close', 0))
            if price <= 0:
                continue

            opt_type = row.get('option_type', 'call')

            iv = ImpliedVolatility.calculate(
                S=spot,
                K=atm_strike,
                T=T,
                r=RISK_FREE_RATE,
                market_price=price,
                q=self.dividend_yield,
                option_type=opt_type
            )

            if iv is not None and 0.01 < iv < 2.0:
                ivs.append(iv)

        return np.mean(ivs) if ivs else 0.0

    def calculate_iv_percentile(self, current_iv: float,
                                 iv_history: pd.Series,
                                 window: int = 504) -> float:
        """
        计算IV百分位（基于2年历史，约504个交易日）

        Args:
            current_iv: 当前IV
            iv_history: IV历史序列
            window: 计算窗口

        Returns:
            百分位（0-100）
        """
        if iv_history is None or len(iv_history) == 0:
            return 50.0  # 无历史数据时返回中位数

        # 使用最近window个数据
        if len(iv_history) > window:
            recent_iv = iv_history[-window:]
        else:
            recent_iv = iv_history

        # 计算百分位
        percentile = (recent_iv < current_iv).sum() / len(recent_iv) * 100

        return percentile

    def calculate_iv_surface(self, spot: float, option_chain: pd.DataFrame,
                             expiries: List[str]) -> pd.DataFrame:
        """
        计算波动率曲面

        Args:
            spot: 标的现价
            option_chain: 期权链数据
            expiries: 到期月份列表

        Returns:
            波动率曲面DataFrame（行=行权价，列=到期日）
        """
        if option_chain is None or option_chain.empty:
            return pd.DataFrame()

        # 获取所有行权价
        strikes = sorted(option_chain['strike'].unique())

        surface_data = {}

        for expiry in expiries:
            expiry_options = option_chain[option_chain['expiry'] == expiry]
            if expiry_options.empty:
                continue

            # 计算距到期天数（假设月底到期）
            expiry_date = datetime.strptime(f"20{expiry}01", "%Y%m%d")
            # 找到该月第四个周五（股指期权到期日）
            expiry_date = self._get_expiry_friday(expiry_date)
            days = (expiry_date - datetime.now()).days

            if days <= 0:
                continue

            T = days / 365
            iv_column = []

            for strike in strikes:
                strike_options = expiry_options[expiry_options['strike'] == strike]

                if strike_options.empty:
                    iv_column.append(np.nan)
                    continue

                # 优先使用Call计算IV
                call_row = strike_options[strike_options['option_type'] == 'call']
                if not call_row.empty:
                    price = call_row.iloc[0].get('last_price', 0)
                    opt_type = 'call'
                else:
                    put_row = strike_options[strike_options['option_type'] == 'put']
                    if not put_row.empty:
                        price = put_row.iloc[0].get('last_price', 0)
                        opt_type = 'put'
                    else:
                        iv_column.append(np.nan)
                        continue

                if price <= 0:
                    iv_column.append(np.nan)
                    continue

                iv = ImpliedVolatility.calculate(
                    S=spot, K=strike, T=T, r=RISK_FREE_RATE,
                    market_price=price, q=self.dividend_yield,
                    option_type=opt_type
                )

                iv_column.append(iv if iv else np.nan)

            surface_data[expiry] = iv_column

        return pd.DataFrame(surface_data, index=strikes)

    def calculate_term_structure(self, spot: float, option_chain: pd.DataFrame,
                                  expiries: List[str]) -> Dict[str, float]:
        """
        计算期限结构（不同到期日的ATM IV）

        Args:
            spot: 标的现价
            option_chain: 期权链数据
            expiries: 到期月份列表

        Returns:
            期限结构字典
        """
        term_structure = {}

        for expiry in expiries:
            expiry_options = option_chain[option_chain['expiry'] == expiry]
            if expiry_options.empty:
                continue

            expiry_date = self._get_expiry_friday(
                datetime.strptime(f"20{expiry}01", "%Y%m%d")
            )
            days = (expiry_date - datetime.now()).days

            if days <= 0:
                continue

            iv = self.calculate_atm_iv(spot, expiry_options, days)
            if iv > 0:
                term_structure[expiry] = iv

        return term_structure

    def calculate_skew(self, spot: float, option_chain: pd.DataFrame,
                       days_to_expiry: int) -> float:
        """
        计算波动率偏斜（25Delta Put IV - 25Delta Call IV）

        Args:
            spot: 标的现价
            option_chain: 期权链数据
            days_to_expiry: 距到期天数

        Returns:
            偏斜值
        """
        if option_chain is None or option_chain.empty:
            return 0.0

        T = days_to_expiry / 365
        strikes = sorted(option_chain['strike'].unique())

        # 找到大约25 Delta的Put和Call
        atm_strike = min(strikes, key=lambda x: abs(x - spot))
        atm_idx = strikes.index(atm_strike)

        # 25 Delta Put大约是ATM下方1-2个档位
        # 25 Delta Call大约是ATM上方1-2个档位
        put_idx = max(0, atm_idx - 2)
        call_idx = min(len(strikes) - 1, atm_idx + 2)

        put_strike = strikes[put_idx]
        call_strike = strikes[call_idx]

        # 计算IV
        put_iv = None
        call_iv = None

        put_options = option_chain[
            (option_chain['strike'] == put_strike) &
            (option_chain['option_type'] == 'put')
        ]
        if not put_options.empty:
            price = put_options.iloc[0].get('last_price', 0)
            if price > 0:
                put_iv = ImpliedVolatility.calculate(
                    S=spot, K=put_strike, T=T, r=RISK_FREE_RATE,
                    market_price=price, q=self.dividend_yield,
                    option_type='put'
                )

        call_options = option_chain[
            (option_chain['strike'] == call_strike) &
            (option_chain['option_type'] == 'call')
        ]
        if not call_options.empty:
            price = call_options.iloc[0].get('last_price', 0)
            if price > 0:
                call_iv = ImpliedVolatility.calculate(
                    S=spot, K=call_strike, T=T, r=RISK_FREE_RATE,
                    market_price=price, q=self.dividend_yield,
                    option_type='call'
                )

        if put_iv and call_iv:
            return put_iv - call_iv

        return 0.0

    def get_iv_level(self, iv_percentile: float) -> str:
        """
        根据IV百分位返回波动率水平

        Args:
            iv_percentile: IV百分位

        Returns:
            波动率水平描述
        """
        if iv_percentile <= IV_THRESHOLDS['extreme_low']:
            return "极低"
        elif iv_percentile <= IV_THRESHOLDS['low']:
            return "低"
        elif iv_percentile <= IV_THRESHOLDS['medium']:
            return "中等"
        elif iv_percentile <= IV_THRESHOLDS['high']:
            return "较高"
        else:
            return "极高"

    def get_strategy_suggestion(self, iv_percentile: float,
                                 hv_iv_ratio: float,
                                 trend: str = "neutral") -> List[str]:
        """
        根据IV状态给出策略建议

        Args:
            iv_percentile: IV百分位
            hv_iv_ratio: HV/IV比率
            trend: 趋势判断 ('bullish', 'bearish', 'neutral')

        Returns:
            策略建议列表
        """
        suggestions = []

        # 低IV环境
        if iv_percentile < IV_THRESHOLDS['low']:
            suggestions.append("买入波动率策略（跨式/宽跨式）")
            if trend == 'bullish':
                suggestions.append("牛市Call价差（买入腿成本低）")
            elif trend == 'bearish':
                suggestions.append("熊市Put价差（买入腿成本低）")

        # 高IV环境
        elif iv_percentile > IV_THRESHOLDS['high']:
            suggestions.append("卖出波动率策略（Iron Condor）")
            suggestions.append("Wheel策略（权利金丰厚）")
            if trend == 'bullish':
                suggestions.append("牛市Put价差（卖方策略）")
            elif trend == 'bearish':
                suggestions.append("熊市Call价差（卖方策略）")

        # 中等IV环境
        else:
            if trend == 'bullish':
                suggestions.append("牛市Call价差")
            elif trend == 'bearish':
                suggestions.append("熊市Put价差")
            else:
                suggestions.append("等待更好的入场机会")

        # HV > IV时可能有波动率回升机会
        if hv_iv_ratio > 1.2:
            suggestions.append("注意：HV > IV，实际波动可能高于预期")

        return suggestions

    def _get_expiry_friday(self, date: datetime) -> datetime:
        """
        获取期权到期日（每月第三个周五）

        Args:
            date: 月份的第一天

        Returns:
            该月期权到期日
        """
        # 找到该月第一个周五
        first_day = date.replace(day=1)
        days_until_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_until_friday)

        # 第三个周五
        third_friday = first_friday + timedelta(weeks=2)

        return third_friday


class IVAnalyzer:
    """IV分析器，用于历史分析和统计"""

    def __init__(self):
        pass

    @staticmethod
    def calculate_iv_stats(iv_series: pd.Series) -> Dict:
        """
        计算IV统计指标

        Args:
            iv_series: IV时间序列

        Returns:
            统计指标字典
        """
        if iv_series is None or len(iv_series) == 0:
            return {}

        return {
            'current': iv_series.iloc[-1] if len(iv_series) > 0 else 0,
            'mean': iv_series.mean(),
            'std': iv_series.std(),
            'min': iv_series.min(),
            'max': iv_series.max(),
            'median': iv_series.median(),
            'percentile_25': iv_series.quantile(0.25),
            'percentile_75': iv_series.quantile(0.75),
        }

    @staticmethod
    def calculate_iv_change(iv_series: pd.Series) -> Dict:
        """
        计算IV变化

        Args:
            iv_series: IV时间序列

        Returns:
            变化指标字典
        """
        if iv_series is None or len(iv_series) < 2:
            return {}

        current = iv_series.iloc[-1]

        return {
            'daily_change': current - iv_series.iloc[-2] if len(iv_series) >= 2 else 0,
            'weekly_change': current - iv_series.iloc[-5] if len(iv_series) >= 5 else 0,
            'monthly_change': current - iv_series.iloc[-22] if len(iv_series) >= 22 else 0,
            'daily_pct': (current / iv_series.iloc[-2] - 1) * 100 if len(iv_series) >= 2 else 0,
        }

    @staticmethod
    def detect_iv_regime(iv_series: pd.Series, window: int = 60) -> str:
        """
        检测IV状态（上升/下降/震荡）

        Args:
            iv_series: IV时间序列
            window: 检测窗口

        Returns:
            状态描述
        """
        if iv_series is None or len(iv_series) < window:
            return "unknown"

        recent = iv_series[-window:]

        # 计算趋势（线性回归斜率）
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent.values, 1)[0]

        # 计算波动（标准差）
        volatility = recent.std() / recent.mean()

        if slope > 0.001:
            return "rising"
        elif slope < -0.001:
            return "falling"
        elif volatility > 0.1:
            return "volatile"
        else:
            return "stable"

    @staticmethod
    def find_iv_extremes(iv_series: pd.Series, threshold: float = 2.0) -> pd.DataFrame:
        """
        找到IV极值点（偏离均值超过threshold个标准差）

        Args:
            iv_series: IV时间序列
            threshold: 标准差阈值

        Returns:
            极值点DataFrame
        """
        if iv_series is None or len(iv_series) == 0:
            return pd.DataFrame()

        mean = iv_series.mean()
        std = iv_series.std()

        extremes = []

        for idx, iv in iv_series.items():
            z_score = (iv - mean) / std
            if abs(z_score) > threshold:
                extremes.append({
                    'date': idx,
                    'iv': iv,
                    'z_score': z_score,
                    'type': 'high' if z_score > 0 else 'low'
                })

        return pd.DataFrame(extremes)


def create_iv_dashboard_data(symbol: str, spot: float,
                              option_chain: pd.DataFrame,
                              iv_history: pd.Series,
                              price_history: pd.Series) -> Dict:
    """
    创建IV监控面板数据

    Args:
        symbol: 期权品种
        spot: 标的现价
        option_chain: 期权链数据
        iv_history: IV历史数据
        price_history: 价格历史数据

    Returns:
        面板数据字典
    """
    monitor = IVMonitor(symbol)
    analyzer = IVAnalyzer()

    # 获取到期月份
    if option_chain is not None and not option_chain.empty:
        expiries = sorted(option_chain['expiry'].unique())
        nearest_expiry = expiries[0] if expiries else None
    else:
        expiries = []
        nearest_expiry = None

    # 计算当前ATM IV
    if nearest_expiry:
        expiry_date = monitor._get_expiry_friday(
            datetime.strptime(f"20{nearest_expiry}01", "%Y%m%d")
        )
        days = max(1, (expiry_date - datetime.now()).days)
        current_iv = monitor.calculate_atm_iv(spot, option_chain, days)
    else:
        current_iv = 0.0
        days = 30

    # 计算IV百分位
    iv_percentile = monitor.calculate_iv_percentile(current_iv, iv_history)

    # 计算历史波动率
    hv_20 = calculate_historical_volatility(price_history.tolist(), 20) if price_history is not None else 0
    hv_60 = calculate_historical_volatility(price_history.tolist(), 60) if price_history is not None else 0

    # 计算IV/HV比率
    iv_hv_ratio = current_iv / hv_20 if hv_20 > 0 else 1.0

    # IV统计
    iv_stats = analyzer.calculate_iv_stats(iv_history) if iv_history is not None else {}

    # IV变化
    iv_change = analyzer.calculate_iv_change(iv_history) if iv_history is not None else {}

    # IV状态
    iv_regime = analyzer.detect_iv_regime(iv_history) if iv_history is not None else "unknown"

    # 波动率水平
    iv_level = monitor.get_iv_level(iv_percentile)

    # 期限结构
    term_structure = monitor.calculate_term_structure(
        spot, option_chain, expiries
    ) if option_chain is not None else {}

    # 偏斜
    skew = monitor.calculate_skew(
        spot, option_chain, days
    ) if option_chain is not None else 0

    # 策略建议
    suggestions = monitor.get_strategy_suggestion(iv_percentile, iv_hv_ratio)

    return {
        'symbol': symbol,
        'spot': spot,
        'timestamp': datetime.now(),
        'current_iv': current_iv,
        'iv_percentile': iv_percentile,
        'iv_level': iv_level,
        'hv_20': hv_20,
        'hv_60': hv_60,
        'iv_hv_ratio': iv_hv_ratio,
        'iv_stats': iv_stats,
        'iv_change': iv_change,
        'iv_regime': iv_regime,
        'term_structure': term_structure,
        'skew': skew,
        'suggestions': suggestions,
        'days_to_expiry': days,
        'nearest_expiry': nearest_expiry,
    }


if __name__ == "__main__":
    # 测试
    print("IV Monitor Test")
    print("=" * 50)

    # 模拟数据
    spot = 3900
    iv_history = pd.Series(np.random.uniform(0.15, 0.35, 504))
    price_history = pd.Series(np.random.uniform(3700, 4100, 60))

    monitor = IVMonitor("IO")

    # 测试IV百分位计算
    current_iv = 0.22
    percentile = monitor.calculate_iv_percentile(current_iv, iv_history)
    print(f"Current IV: {current_iv:.2%}")
    print(f"IV Percentile: {percentile:.1f}%")
    print(f"IV Level: {monitor.get_iv_level(percentile)}")

    # 测试策略建议
    print(f"\nStrategy Suggestions:")
    suggestions = monitor.get_strategy_suggestion(percentile, 1.1, 'bullish')
    for s in suggestions:
        print(f"  - {s}")
