"""
买入跨式策略
IV低位时买入ATM Call + ATM Put，赌波动率上升或大幅波动
"""

import pandas as pd
from typing import List, Optional, Dict
from datetime import datetime

from .base import (
    BaseOptionStrategy, StrategyParam, Signal, OptionLeg
)


class LongStraddle(BaseOptionStrategy):
    """买入跨式策略"""

    name = "long_straddle"
    display_name = "买入跨式"
    description = "IV低位时买入ATM Call + ATM Put，期待大幅波动或IV回升"
    version = "1.0"
    strategy_type = "volatility"
    warmup_bars = 50

    @classmethod
    def get_params(cls) -> List[StrategyParam]:
        return [
            StrategyParam("iv_percentile_max", "IV百分位上限", 25, 10, 40, 5, "int",
                         description="IV百分位低于此值才入场"),
            StrategyParam("hv_iv_ratio_min", "HV/IV比率下限", 1.0, 0.8, 1.5, 0.1, "float",
                         description="历史波动率/隐含波动率比率"),
            StrategyParam("days_to_expiry_min", "最小到期天数", 14, 7, 21, 7, "int"),
            StrategyParam("days_to_expiry_max", "最大到期天数", 45, 30, 60, 5, "int"),
            StrategyParam("iv_rise_target", "IV上涨止盈", 0.30, 0.20, 0.50, 0.05, "float",
                         description="IV上涨此比例时止盈"),
            StrategyParam("price_move_target", "价格波动止盈", 0.05, 0.03, 0.08, 0.01, "float",
                         description="标的波动超过此比例时止盈"),
            StrategyParam("time_decay_stop", "时间止损", 0.50, 0.30, 0.70, 0.1, "float"),
            StrategyParam("days_after_event", "事件后持有天数", 3, 1, 7, 1, "int",
                         description="重大事件后最多持有天数"),
            StrategyParam("days_before_expiry_close", "到期前平仓天数", 7, 5, 14, 1, "int"),
        ]

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算指标"""
        df = df.copy()

        # 计算历史波动率
        df['returns'] = df['close'].pct_change()
        df['hv_20'] = df['returns'].rolling(20).std() * (252 ** 0.5)
        df['hv_60'] = df['returns'].rolling(60).std() * (252 ** 0.5)

        # 计算ATR
        df['atr'] = self.calculate_atr(df, 14)

        # 波动率收缩指标（布林带宽度）
        df['ma20'] = df['close'].rolling(20).mean()
        df['std20'] = df['close'].rolling(20).std()
        df['bb_width'] = df['std20'] / df['ma20']  # 布林带宽度/价格

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
        hv_20 = bar.get('hv_20', 0.2) if 'hv_20' in bar else iv_data.get('hv_20', 0.2)

        if self.position is None:
            return self._check_entry(bar, spot, option_chain, iv_percentile,
                                     current_iv, days_to_expiry, hv_20)

        return self._check_exit(bar, spot, option_chain, iv_data, days_to_expiry)

    def _check_entry(self, bar, spot, option_chain, iv_percentile,
                     current_iv, days_to_expiry, hv_20) -> Optional[Signal]:
        """检查入场条件"""
        p = self.params

        # 条件1: IV百分位极低
        if iv_percentile > p['iv_percentile_max']:
            return None

        # 条件2: HV > IV（历史波动率高于隐含波动率）
        hv_iv_ratio = hv_20 / current_iv if current_iv > 0 else 1.0
        if hv_iv_ratio < p['hv_iv_ratio_min']:
            return None

        # 条件3: 到期天数合适
        if not (p['days_to_expiry_min'] <= days_to_expiry <= p['days_to_expiry_max']):
            return None

        T = days_to_expiry / 365

        # 选择ATM行权价
        if option_chain is not None and not option_chain.empty:
            strikes = sorted(option_chain['strike'].unique())
            atm_strike = min(strikes, key=lambda x: abs(x - spot))
        else:
            # 四舍五入到最近的50点
            atm_strike = round(spot / 50) * 50

        # 获取价格
        call_price = self.get_option_price(atm_strike, 'call', option_chain,
                                           spot, T, current_iv)
        put_price = self.get_option_price(atm_strike, 'put', option_chain,
                                          spot, T, current_iv)

        legs = [
            OptionLeg(
                option_type='call',
                strike=atm_strike,
                expiry=self._get_nearest_expiry(option_chain),
                position=1,
                quantity=1,
                entry_price=call_price,
                entry_time=self.current_time,
                iv=current_iv
            ),
            OptionLeg(
                option_type='put',
                strike=atm_strike,
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
            tag='long_straddle',
            reason=f"IV百分位={iv_percentile:.0f}%极低，HV/IV={hv_iv_ratio:.2f}",
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

        # 条件3: 盈利超过入场成本（盈亏平衡后继续盈利）
        if pnl > entry_cost * 0.5:  # 盈利超过50%
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
