"""
熊市Put价差策略
趋势确认后开仓，买入ATM Put + 卖出OTM Put
"""

import pandas as pd
from typing import List, Optional, Dict
from datetime import datetime

from .base import (
    BaseOptionStrategy, StrategyParam, Signal, OptionLeg
)


class BearPutSpread(BaseOptionStrategy):
    """熊市Put价差策略"""

    name = "bear_put_spread"
    display_name = "熊市Put价差"
    description = "趋势看跌时，买入ATM Put + 卖出OTM Put，风险有限盈利有限"
    version = "1.0"
    strategy_type = "bearish"
    warmup_bars = 50

    @classmethod
    def get_params(cls) -> List[StrategyParam]:
        return [
            StrategyParam("ema_fast", "快速EMA", 9, 5, 20, 1, "int"),
            StrategyParam("ema_slow", "慢速EMA", 21, 15, 50, 1, "int"),
            StrategyParam("ma_len", "MA均线", 20, 10, 60, 5, "int"),
            StrategyParam("iv_percentile_max", "IV百分位上限", 50, 20, 80, 5, "int"),
            StrategyParam("delta_buy", "买入腿Delta", -0.50, -0.60, -0.40, 0.05, "float"),
            StrategyParam("delta_sell", "卖出腿Delta", -0.30, -0.40, -0.20, 0.05, "float"),
            StrategyParam("spread_width", "价差宽度", 100, 50, 200, 25, "int"),
            StrategyParam("days_to_expiry_min", "最小到期天数", 20, 10, 30, 5, "int"),
            StrategyParam("days_to_expiry_max", "最大到期天数", 45, 30, 60, 5, "int"),
            StrategyParam("profit_target", "止盈比例", 0.80, 0.50, 1.0, 0.1, "float"),
            StrategyParam("time_decay_stop", "时间止损", 0.50, 0.30, 0.70, 0.1, "float"),
            StrategyParam("days_before_expiry_close", "到期前平仓天数", 5, 3, 10, 1, "int"),
        ]

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算趋势指标"""
        p = self.params

        df = df.copy()
        df['ema_fast'] = self.calculate_ema(df['close'], p['ema_fast'])
        df['ema_slow'] = self.calculate_ema(df['close'], p['ema_slow'])
        df['ma'] = self.calculate_ma(df['close'], p['ma_len'])

        # 金叉死叉信号
        df['ema_cross'] = 0
        df.loc[(df['ema_fast'] > df['ema_slow']) &
               (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1)), 'ema_cross'] = 1
        df.loc[(df['ema_fast'] < df['ema_slow']) &
               (df['ema_fast'].shift(1) >= df['ema_slow'].shift(1)), 'ema_cross'] = -1

        # 趋势状态
        df['trend'] = 0
        df.loc[(df['ema_fast'] > df['ema_slow']) & (df['close'] > df['ma']), 'trend'] = 1
        df.loc[(df['ema_fast'] < df['ema_slow']) & (df['close'] < df['ma']), 'trend'] = -1

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

        # 条件1: EMA死叉且收盘价<MA（下跌趋势）
        if bar['trend'] != -1:
            return None

        # 条件2: IV百分位不能太高
        if iv_percentile > p['iv_percentile_max']:
            return None

        # 条件3: 到期天数合适
        if not (p['days_to_expiry_min'] <= days_to_expiry <= p['days_to_expiry_max']):
            return None

        T = days_to_expiry / 365

        # 买入腿：接近ATM Put
        buy_strike = self.select_strike(spot, option_chain, p['delta_buy'],
                                        'put', T, current_iv)

        # 卖出腿：OTM Put（更低的行权价）
        sell_strike = buy_strike - p['spread_width']

        # 获取价格
        buy_price = self.get_option_price(buy_strike, 'put', option_chain,
                                          spot, T, current_iv)
        sell_price = self.get_option_price(sell_strike, 'put', option_chain,
                                           spot, T, current_iv)

        legs = [
            OptionLeg(
                option_type='put',
                strike=buy_strike,
                expiry=self._get_nearest_expiry(option_chain),
                position=1,
                quantity=1,
                entry_price=buy_price,
                entry_time=self.current_time,
                iv=current_iv
            ),
            OptionLeg(
                option_type='put',
                strike=sell_strike,
                expiry=self._get_nearest_expiry(option_chain),
                position=-1,
                quantity=1,
                entry_price=sell_price,
                entry_time=self.current_time,
                iv=current_iv
            )
        ]

        return Signal(
            action='open',
            legs=legs,
            tag='bear_put_spread',
            reason=f"EMA死叉，趋势看跌，IV={iv_percentile:.0f}%",
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

        entry_cost = self.position.get_entry_cost()
        current_value = sum(leg.current_price * leg.position for leg in self.position.legs)
        pnl = current_value - entry_cost

        spread_width = abs(self.position.legs[0].strike - self.position.legs[1].strike)
        max_profit = spread_width + entry_cost

        # 条件1: 达到目标盈利
        if max_profit > 0 and pnl >= max_profit * p['profit_target']:
            return Signal(
                action='close',
                legs=self.position.legs,
                tag='profit_target',
                reason=f"达到目标盈利 {p['profit_target']*100:.0f}%",
                timestamp=self.current_time,
                underlying_price=spot
            )

        # 条件2: 趋势反转（EMA金叉）
        if bar['ema_cross'] == 1:
            return Signal(
                action='close',
                legs=self.position.legs,
                tag='trend_reverse',
                reason="EMA金叉，趋势反转",
                timestamp=self.current_time,
                underlying_price=spot
            )

        # 条件3: 时间价值损耗过多且方向不利
        if entry_cost < 0:
            loss_ratio = pnl / entry_cost
            if loss_ratio > p['time_decay_stop'] and spot > self.position.entry_underlying_price:
                return Signal(
                    action='close',
                    legs=self.position.legs,
                    tag='time_decay_stop',
                    reason=f"时间价值损耗>{p['time_decay_stop']*100:.0f}%且方向不利",
                    timestamp=self.current_time,
                    underlying_price=spot
                )

        # 条件4: 临近到期
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
