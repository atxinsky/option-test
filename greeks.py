"""
Greeks计算模块
基于Black-Scholes模型的期权定价和希腊字母计算
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class OptionGreeks:
    """期权Greeks数据类"""
    price: float        # 期权理论价格
    delta: float        # Delta
    gamma: float        # Gamma
    theta: float        # Theta (日衰减)
    vega: float         # Vega (1%波动率变化)
    rho: float          # Rho (1%利率变化)

    def to_dict(self) -> Dict:
        return {
            'price': self.price,
            'delta': self.delta,
            'gamma': self.gamma,
            'theta': self.theta,
            'vega': self.vega,
            'rho': self.rho
        }


class BlackScholes:
    """
    Black-Scholes期权定价模型
    支持考虑分红率的欧式期权定价和Greeks计算
    """

    def __init__(self, S: float, K: float, T: float, r: float,
                 sigma: float, q: float = 0.0, option_type: str = 'call'):
        """
        初始化期权参数

        Args:
            S: 标的资产现价
            K: 行权价
            T: 到期时间（年），如30天 = 30/365
            r: 无风险利率（年化），如2% = 0.02
            sigma: 波动率（年化），如25% = 0.25
            q: 连续分红率（年化），如3% = 0.03
            option_type: 'call' 或 'put'
        """
        self.S = S
        self.K = K
        self.T = max(T, 1e-10)  # 避免除零
        self.r = r
        self.sigma = max(sigma, 1e-10)  # 避免除零
        self.q = q
        self.option_type = option_type.lower()

        # 预计算d1和d2
        self._d1, self._d2 = self._calc_d1_d2()

    def _calc_d1_d2(self) -> Tuple[float, float]:
        """计算d1和d2"""
        sqrt_T = np.sqrt(self.T)
        d1 = (np.log(self.S / self.K) +
              (self.r - self.q + 0.5 * self.sigma**2) * self.T) / \
             (self.sigma * sqrt_T)
        d2 = d1 - self.sigma * sqrt_T
        return d1, d2

    def price(self) -> float:
        """计算期权理论价格"""
        if self.option_type == 'call':
            price = (self.S * np.exp(-self.q * self.T) * norm.cdf(self._d1) -
                     self.K * np.exp(-self.r * self.T) * norm.cdf(self._d2))
        else:  # put
            price = (self.K * np.exp(-self.r * self.T) * norm.cdf(-self._d2) -
                     self.S * np.exp(-self.q * self.T) * norm.cdf(-self._d1))
        return max(price, 0)

    def delta(self) -> float:
        """
        计算Delta - 期权价格对标的价格的敏感度
        Call Delta: 0 ~ 1
        Put Delta: -1 ~ 0
        """
        if self.option_type == 'call':
            return np.exp(-self.q * self.T) * norm.cdf(self._d1)
        else:
            return -np.exp(-self.q * self.T) * norm.cdf(-self._d1)

    def gamma(self) -> float:
        """
        计算Gamma - Delta对标的价格的敏感度
        Call和Put的Gamma相同，总是正值
        """
        return (np.exp(-self.q * self.T) * norm.pdf(self._d1)) / \
               (self.S * self.sigma * np.sqrt(self.T))

    def theta(self) -> float:
        """
        计算Theta - 期权价格对时间的敏感度（日衰减）
        通常为负值（期权随时间贬值）
        """
        sqrt_T = np.sqrt(self.T)
        pdf_d1 = norm.pdf(self._d1)

        # 第一项：时间价值衰减
        term1 = -self.S * np.exp(-self.q * self.T) * pdf_d1 * self.sigma / (2 * sqrt_T)

        if self.option_type == 'call':
            # 第二项：利率影响
            term2 = -self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self._d2)
            # 第三项：分红影响
            term3 = self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(self._d1)
        else:
            term2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self._d2)
            term3 = -self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(-self._d1)

        # 转换为日衰减
        return (term1 + term2 + term3) / 365

    def vega(self) -> float:
        """
        计算Vega - 期权价格对波动率的敏感度
        返回波动率变化1%（0.01）对期权价格的影响
        Call和Put的Vega相同，总是正值
        """
        return self.S * np.exp(-self.q * self.T) * norm.pdf(self._d1) * np.sqrt(self.T) / 100

    def rho(self) -> float:
        """
        计算Rho - 期权价格对利率的敏感度
        返回利率变化1%（0.01）对期权价格的影响
        """
        if self.option_type == 'call':
            return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self._d2) / 100
        else:
            return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self._d2) / 100

    def get_all_greeks(self) -> OptionGreeks:
        """获取所有Greeks"""
        return OptionGreeks(
            price=self.price(),
            delta=self.delta(),
            gamma=self.gamma(),
            theta=self.theta(),
            vega=self.vega(),
            rho=self.rho()
        )


class ImpliedVolatility:
    """隐含波动率计算"""

    @staticmethod
    def calculate(S: float, K: float, T: float, r: float,
                  market_price: float, q: float = 0.0,
                  option_type: str = 'call',
                  method: str = 'newton',
                  max_iter: int = 100,
                  tol: float = 1e-6) -> Optional[float]:
        """
        计算隐含波动率

        Args:
            S: 标的资产现价
            K: 行权价
            T: 到期时间（年）
            r: 无风险利率
            market_price: 市场期权价格
            q: 分红率
            option_type: 'call' 或 'put'
            method: 'newton'（牛顿法）或 'bisection'（二分法）
            max_iter: 最大迭代次数
            tol: 收敛精度

        Returns:
            隐含波动率，如果无法收敛返回None
        """
        if method == 'newton':
            return ImpliedVolatility._newton_method(
                S, K, T, r, market_price, q, option_type, max_iter, tol)
        else:
            return ImpliedVolatility._bisection_method(
                S, K, T, r, market_price, q, option_type, tol)

    @staticmethod
    def _newton_method(S: float, K: float, T: float, r: float,
                       market_price: float, q: float,
                       option_type: str, max_iter: int,
                       tol: float) -> Optional[float]:
        """牛顿迭代法计算隐含波动率"""
        # 初始猜测
        sigma = 0.3

        for _ in range(max_iter):
            bs = BlackScholes(S, K, T, r, sigma, q, option_type)
            bs_price = bs.price()
            vega = bs.vega()

            # 检查收敛
            diff = bs_price - market_price
            if abs(diff) < tol:
                return sigma

            # Vega太小时停止
            if abs(vega) < 1e-10:
                break

            # 牛顿迭代（vega已经是1%变化的影响，需要调整）
            sigma = sigma - diff / (vega * 100)

            # 限制波动率范围
            sigma = max(0.001, min(sigma, 3.0))

        return sigma if abs(bs_price - market_price) < 0.01 else None

    @staticmethod
    def _bisection_method(S: float, K: float, T: float, r: float,
                          market_price: float, q: float,
                          option_type: str, tol: float) -> Optional[float]:
        """二分法计算隐含波动率（更稳定）"""
        low, high = 0.001, 3.0

        # 检查边界
        bs_low = BlackScholes(S, K, T, r, low, q, option_type)
        bs_high = BlackScholes(S, K, T, r, high, q, option_type)

        if market_price < bs_low.price() or market_price > bs_high.price():
            return None

        while high - low > tol:
            mid = (low + high) / 2
            bs = BlackScholes(S, K, T, r, mid, q, option_type)
            bs_price = bs.price()

            if bs_price < market_price:
                low = mid
            else:
                high = mid

        return (low + high) / 2


def calculate_greeks_batch(spot: float, strikes: list, T: float, r: float,
                           sigma: float, q: float = 0.0) -> Dict:
    """
    批量计算期权链的Greeks

    Args:
        spot: 标的现价
        strikes: 行权价列表
        T: 到期时间
        r: 无风险利率
        sigma: 波动率
        q: 分红率

    Returns:
        包含call和put Greeks的字典
    """
    results = {
        'strikes': strikes,
        'call_price': [],
        'call_delta': [],
        'call_gamma': [],
        'call_theta': [],
        'call_vega': [],
        'put_price': [],
        'put_delta': [],
        'put_gamma': [],
        'put_theta': [],
        'put_vega': [],
    }

    for K in strikes:
        # Call
        call = BlackScholes(spot, K, T, r, sigma, q, 'call')
        results['call_price'].append(call.price())
        results['call_delta'].append(call.delta())
        results['call_gamma'].append(call.gamma())
        results['call_theta'].append(call.theta())
        results['call_vega'].append(call.vega())

        # Put
        put = BlackScholes(spot, K, T, r, sigma, q, 'put')
        results['put_price'].append(put.price())
        results['put_delta'].append(put.delta())
        results['put_gamma'].append(put.gamma())
        results['put_theta'].append(put.theta())
        results['put_vega'].append(put.vega())

    return results


def find_strike_by_delta(spot: float, T: float, r: float, sigma: float,
                         target_delta: float, q: float = 0.0,
                         option_type: str = 'call',
                         strike_range: Tuple[float, float] = None) -> float:
    """
    根据目标Delta找到对应的行权价

    Args:
        spot: 标的现价
        T: 到期时间
        r: 无风险利率
        sigma: 波动率
        target_delta: 目标Delta值
        q: 分红率
        option_type: 'call' 或 'put'
        strike_range: 行权价搜索范围，默认为现价的0.7-1.3倍

    Returns:
        对应的行权价
    """
    if strike_range is None:
        strike_range = (spot * 0.7, spot * 1.3)

    low, high = strike_range
    tol = spot * 0.001  # 容差为现价的0.1%

    while high - low > tol:
        mid = (low + high) / 2
        bs = BlackScholes(spot, mid, T, r, sigma, q, option_type)
        delta = bs.delta()

        if option_type == 'call':
            # Call Delta随行权价上升而下降
            if delta > target_delta:
                low = mid
            else:
                high = mid
        else:
            # Put Delta随行权价上升而上升（从-1到0）
            if delta < target_delta:
                low = mid
            else:
                high = mid

    return (low + high) / 2


def calculate_historical_volatility(prices: list, window: int = 20) -> float:
    """
    计算历史波动率（年化）

    Args:
        prices: 价格序列
        window: 计算窗口

    Returns:
        年化历史波动率
    """
    if len(prices) < window + 1:
        return 0.0

    prices = np.array(prices[-window-1:])
    log_returns = np.log(prices[1:] / prices[:-1])

    # 日波动率 * sqrt(252) = 年化波动率
    return np.std(log_returns) * np.sqrt(252)


if __name__ == "__main__":
    # 测试示例
    print("=" * 50)
    print("Black-Scholes期权定价测试")
    print("=" * 50)

    # 沪深300股指期权示例
    # 假设当前指数3900点，行权价3900，30天到期
    S = 3900      # 标的现价
    K = 3900      # 行权价（ATM）
    T = 30/365    # 30天到期
    r = 0.02      # 2%无风险利率
    sigma = 0.20  # 20%波动率
    q = 0.025     # 2.5%分红率

    # 计算Call
    call = BlackScholes(S, K, T, r, sigma, q, 'call')
    call_greeks = call.get_all_greeks()

    print(f"\n看涨期权 (ATM Call):")
    print(f"  理论价格: {call_greeks.price:.2f} 点")
    print(f"  Delta: {call_greeks.delta:.4f}")
    print(f"  Gamma: {call_greeks.gamma:.6f}")
    print(f"  Theta: {call_greeks.theta:.4f} 点/天")
    print(f"  Vega: {call_greeks.vega:.4f} 点/1%IV")
    print(f"  Rho: {call_greeks.rho:.4f} 点/1%利率")

    # 计算Put
    put = BlackScholes(S, K, T, r, sigma, q, 'put')
    put_greeks = put.get_all_greeks()

    print(f"\n看跌期权 (ATM Put):")
    print(f"  理论价格: {put_greeks.price:.2f} 点")
    print(f"  Delta: {put_greeks.delta:.4f}")
    print(f"  Gamma: {put_greeks.gamma:.6f}")
    print(f"  Theta: {put_greeks.theta:.4f} 点/天")
    print(f"  Vega: {put_greeks.vega:.4f} 点/1%IV")

    # 测试隐含波动率计算
    print(f"\n隐含波动率计算测试:")
    market_price = call_greeks.price  # 使用理论价格作为市场价格
    iv = ImpliedVolatility.calculate(S, K, T, r, market_price, q, 'call')
    print(f"  市场价格: {market_price:.2f}")
    print(f"  反推IV: {iv:.4f} ({iv*100:.2f}%)")
    print(f"  原始sigma: {sigma:.4f} ({sigma*100:.2f}%)")

    # 测试Delta行权价查找
    print(f"\n按Delta查找行权价:")
    strike_50d = find_strike_by_delta(S, T, r, sigma, 0.50, q, 'call')
    strike_25d = find_strike_by_delta(S, T, r, sigma, 0.25, q, 'call')
    print(f"  Delta=0.50 Call行权价: {strike_50d:.0f}")
    print(f"  Delta=0.25 Call行权价: {strike_25d:.0f}")
