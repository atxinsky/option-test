"""
工具函数模块
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple, Optional


def get_expiry_friday(year: int, month: int) -> datetime:
    """
    获取期权到期日（每月第三个周五）

    Args:
        year: 年份
        month: 月份

    Returns:
        到期日datetime
    """
    # 找到该月第一天
    first_day = datetime(year, month, 1)

    # 找到第一个周五
    days_until_friday = (4 - first_day.weekday()) % 7
    first_friday = first_day + timedelta(days=days_until_friday)

    # 第三个周五
    third_friday = first_friday + timedelta(weeks=2)

    return third_friday


def days_to_expiry(expiry: str, current_date: datetime = None) -> int:
    """
    计算距到期日的天数

    Args:
        expiry: 到期月份，格式 'YYMM'，如 '2501'
        current_date: 当前日期

    Returns:
        天数
    """
    if current_date is None:
        current_date = datetime.now()

    year = 2000 + int(expiry[:2])
    month = int(expiry[2:])

    expiry_date = get_expiry_friday(year, month)

    return (expiry_date - current_date).days


def format_number(value: float, decimals: int = 2) -> str:
    """格式化数字显示"""
    if abs(value) >= 1e6:
        return f"{value/1e6:.{decimals}f}M"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.{decimals}f}K"
    else:
        return f"{value:.{decimals}f}"


def calculate_kelly_fraction(win_rate: float, avg_win: float,
                             avg_loss: float) -> float:
    """
    计算凯利公式最优仓位

    Args:
        win_rate: 胜率 (0-1)
        avg_win: 平均盈利
        avg_loss: 平均亏损（正值）

    Returns:
        最优仓位比例
    """
    if avg_loss == 0:
        return 0

    b = avg_win / avg_loss  # 赔率
    p = win_rate
    q = 1 - p

    kelly = (b * p - q) / b

    return max(0, min(kelly, 1))  # 限制在0-1之间


def calculate_position_size(capital: float, risk_per_trade: float,
                           stop_loss_points: float, multiplier: float) -> int:
    """
    计算头寸大小

    Args:
        capital: 总资金
        risk_per_trade: 单笔风险比例（如0.02表示2%）
        stop_loss_points: 止损点数
        multiplier: 合约乘数

    Returns:
        手数
    """
    risk_amount = capital * risk_per_trade
    point_value = multiplier
    max_lots = risk_amount / (stop_loss_points * point_value)

    return max(1, int(max_lots))


def calculate_margin(price: float, multiplier: float, margin_rate: float,
                     num_contracts: int = 1) -> float:
    """
    计算保证金

    Args:
        price: 标的价格
        multiplier: 合约乘数
        margin_rate: 保证金比例
        num_contracts: 合约数量

    Returns:
        保证金金额
    """
    return price * multiplier * margin_rate * num_contracts


def annualize_return(total_return: float, days: int) -> float:
    """
    年化收益率

    Args:
        total_return: 总收益率（如0.1表示10%）
        days: 持有天数

    Returns:
        年化收益率
    """
    if days <= 0:
        return 0

    return (1 + total_return) ** (252 / days) - 1


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    计算夏普比率

    Args:
        returns: 日收益率序列
        risk_free_rate: 无风险利率（年化）

    Returns:
        夏普比率
    """
    if returns is None or len(returns) < 2:
        return 0

    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf

    if excess_returns.std() == 0:
        return 0

    return (excess_returns.mean() * 252) / (excess_returns.std() * np.sqrt(252))


def calculate_max_drawdown(equity_curve: pd.Series) -> Tuple[float, int, int]:
    """
    计算最大回撤

    Args:
        equity_curve: 权益曲线

    Returns:
        (最大回撤比例, 开始索引, 结束索引)
    """
    if equity_curve is None or len(equity_curve) < 2:
        return 0, 0, 0

    running_max = equity_curve.expanding().max()
    drawdown = (running_max - equity_curve) / running_max

    max_dd = drawdown.max()
    end_idx = drawdown.idxmax()

    # 找到开始点
    start_idx = equity_curve[:end_idx].idxmax()

    return max_dd, start_idx, end_idx


def generate_strike_range(spot: float, num_strikes: int = 11,
                          step: float = 50) -> List[float]:
    """
    生成行权价序列

    Args:
        spot: 标的现价
        num_strikes: 行权价数量
        step: 行权价间距

    Returns:
        行权价列表
    """
    # 找到最接近的整数行权价
    base_strike = round(spot / step) * step

    # 生成上下对称的行权价
    half = num_strikes // 2
    strikes = [base_strike + i * step for i in range(-half, half + 1)]

    return strikes


def parse_option_symbol(symbol: str) -> Optional[dict]:
    """
    解析期权合约代码

    Args:
        symbol: 如 'IO2501-C-3900' 或 'io2501C3900'

    Returns:
        解析结果字典
    """
    import re

    symbol = symbol.upper()

    # 格式1: IO2501-C-3900
    if '-' in symbol:
        parts = symbol.split('-')
        if len(parts) >= 3:
            base = parts[0]
            product = ''.join(filter(str.isalpha, base))
            expiry = ''.join(filter(str.isdigit, base))
            opt_type = 'call' if parts[1] == 'C' else 'put'
            strike = float(parts[2])

            return {
                'product': product,
                'expiry': expiry,
                'type': opt_type,
                'strike': strike
            }

    # 格式2: IO2501C3900
    match = re.match(r'([A-Z]+)(\d{4})([CP])(\d+)', symbol)
    if match:
        return {
            'product': match.group(1),
            'expiry': match.group(2),
            'type': 'call' if match.group(3) == 'C' else 'put',
            'strike': float(match.group(4))
        }

    return None


def format_option_symbol(product: str, expiry: str, opt_type: str,
                         strike: float) -> str:
    """
    格式化期权合约代码

    Args:
        product: 品种代码
        expiry: 到期月份
        opt_type: 'call' 或 'put'
        strike: 行权价

    Returns:
        合约代码
    """
    type_char = 'C' if opt_type.lower() == 'call' else 'P'
    return f"{product}{expiry}-{type_char}-{int(strike)}"
