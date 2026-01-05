"""
买入宽跨式策略
IV极低时买入OTM Call + OTM Put，成本更低但需要更大波动
"""

import pandas as pd
from typing import List, Optional, Dict
from datetime import datetime

from .base import (
    BaseOptionStrategy, StrategyParam, Signal, OptionLeg
)


class LongStrangle(BaseOptionStrategy):
    """买入宽跨式策略"""

    name = "long_strangle"
    display_name = "买入宽跨式"
    description = "IV极低时买入OTM Call + OTM Put，成本低但需要更大波动获利"
    version = "1.0"
    strategy_type = "volatility"
    warmup_bars = 50

    @classmethod
    def get_params(cls) -> List[StrategyParam]:
        return [
            StrategyParam("iv_percentile_max", "IV百分位上限", 20, 10, 35, 5, "int"),
            StrategyParam("delta_call", "Call腿Delta", 0.25, 0.15, 0.35, 0.05, "float"),
            StrategyParam("delta_put", "Put腿Delta", -0.25, -0.35, -0.15, 0.05, "float"),
            StrategyParam("days_to_expiry_min", "最小到期天数", 14, 7, 21, 7, "int"),
            StrategyParam("days_to_expiry_max", "最大到期天数", 45, 30, 60, 5, "int"),
            StrategyParam("iv_rise_target", "IV上涨止盈", 0.40, 0.25, 0.60, 0.05, "float"),
            StrategyParam("price_move_target", "价格波动止盈", 0.06, 0.04, 0.10, 0.01, "float"),
            StrategyParam("time_decay_stop", "时间止损", 0.50, 0.30, 0.70, 0.1, "float"),
            StrategyParam("days_before_expiry_close", "到期前平仓天数", 7, 5, 14, 1, "int"),
        ]

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算指标"""
        df = df.copy()

        df['returns'] = df['close'].pct_change()
        df['hv_20'] = df['returns'].rolling(20).std() * (252 ** 0.5)

        df['ma20'] = df['close'].rolling(20).mean()
        df['std20'] = df['close'].rolling(20).std()
        df['bb_width'] = df['std20'] / df['ma20']

        return df

    def generate_signal(self, idx: int, df: pd.DataFrame,
                        option_chain: pd.DataFrame,
                        iv_data: Dict) -> Optional[Signal]:
        """生成交易信号"""
        p = self.params
        bar = df.iloc[idx]
        spot = bar['close']

        iv_percentile = iv_data.get('iv_percentile', 50)
        current_iv = iv_data.get('current_iv', 0.2)
        days_to_expiry = iv_data.get('days_to_expiry', 30)

        if self.position is None:
            return self._check_entry(bar, spot, option_chain, iv_percentile,
                                     current_iv, days_to_expiry)

        return self._check_exit(bar, spot, option_chain, iv_data, days_to_expiry)

    def _check_entry(self, bar, spot, option_chain, iv_percentile,
                     current_iv, days_to_expiry) -> Optional[Signal]:
        """检查入场条件"""
        p = self.params

        # 条件1: IV百分位极低
        if iv_percentile > p['iv_percentile_max']:
            return None

        # 条件2: 到期天数合适
        if not (p['days_to_expiry_min'] <= days_to_expiry <= p['days_to_expiry_max']):
            return None

        T = days_to_expiry / 365

        # 选择OTM Call行权价
        call_strike = self.select_strike(spot, option_chain, p['delta_call'],
                                         'call', T, current_iv)

        # 选择OTM Put行权价
        put_strike = self.select_strike(spot, option_chain, p['delta_put'],
                                        'put', T, current_iv)

        # 获取价格
        call_price = self.get_option_price(call_strike, 'call', option_chain,
                                           spot, T, current_iv)
        put_price = self.get_option_price(put_strike, 'put', option_chain,
                                          spot, T, current_iv)

        legs = [
            OptionLeg(
                option_type='call',
                strike=call_strike,
                expiry=self._get_nearest_expiry(option_chain),
                position=1,
                quantity=1,
                entry_price=call_price,
                entry_time=self.current_time,
                iv=current_iv
            ),
            OptionLeg(
                option_type='put',
                strike=put_strike,
                expiry=self._get_nearest_expiry(option_chain),
                position=1,
                quantity=1,
                entry_price=put_price,
                entry_time=self.current_time,
                iv=current_iv
            )
        ]

        return Signal(
            action='open',
            legs=legs,
            tag='long_strangle',
            reason=f"IV百分位={iv_percentile:.0f}%极低",
            timestamp=self.current_time,
            underlying_price=spot,
            iv=current_iv
        )

    def _check_exit(self, bar, spot, option_chain, iv_data,
                    days_to_expiry) -> Optional[Signal]:
        """检查出场条件"""
        p = self.params

        if not self.position or not self.position.legs:
            return None

        T = max(1, days_to_expiry) / 365
        current_iv = iv_data.get('current_iv', 0.2)

        for leg in self.position.legs:
            leg.current_price = self.get_option_price(
                leg.strike, leg.option_type, option_chain,
                spot, T, current_iv
            )

        entry_cost = abs(self.position.get_entry_cost())
        current_value = sum(leg.current_price for leg in self.position.legs)
        pnl = current_value - entry_cost

        # 条件1: IV大幅上升
        entry_iv = self.position.entry_iv
        if entry_iv > 0:
            iv_change = (current_iv - entry_iv) / entry_iv
            if iv_change >= p['iv_rise_target']:
                return Signal(
                    action='close',
                    legs=self.position.legs,
                    tag='iv_rise',
                    reason=f"IV上涨{iv_change*100:.0f}%",
                    timestamp=self.current_time,
                    underlying_price=spot
                )

        # 条件2: 标的大幅波动
        entry_price = self.position.entry_underlying_price
        price_move = abs(spot - entry_price) / entry_price
        if price_move >= p['price_move_target']:
            return Signal(
                action='close',
                legs=self.position.legs,
                tag='price_move',
                reason=f"标的波动{price_move*100:.1f}%",
                timestamp=self.current_time,
                underlying_price=spot
            )

        # 条件3: 盈利
        if pnl > entry_cost * 0.5:
            return Signal(
                action='close',
                legs=self.position.legs,
                tag='profit_take',
                reason=f"盈利{pnl/entry_cost*100:.0f}%",
                timestamp=self.current_time,
                underlying_price=spot
            )

        # 条件4: 时间价值损耗
        if entry_cost > 0:
            loss_ratio = -pnl / entry_cost
            if loss_ratio > p['time_decay_stop']:
                return Signal(
                    action='close',
                    legs=self.position.legs,
                    tag='time_decay_stop',
                    reason=f"时间价值损耗{loss_ratio*100:.0f}%",
                    timestamp=self.current_time,
                    underlying_price=spot
                )

        # 条件5: 临近到期
        if days_to_expiry <= p['days_before_expiry_close']:
            return Signal(
                action='close',
                legs=self.position.legs,
                tag='near_expiry',
                reason=f"距到期仅{days_to_expiry}天",
                timestamp=self.current_time,
                underlying_price=spot
            )

        return None

    def _get_nearest_expiry(self, option_chain) -> str:
        if option_chain is not None and not option_chain.empty:
            expiries = sorted(option_chain['expiry'].unique())
            if expiries:
                return expiries[0]
        now = datetime.now()
        if now.day > 15:
            month = now.month + 2
        else:
            month = now.month + 1
        year = now.year
        if month > 12:
            month -= 12
            year += 1
        return f"{year % 100:02d}{month:02d}"
