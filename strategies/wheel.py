"""
Wheel策略（循环收租）
阶段1：卖Put收权利金，被行权后进入阶段2
阶段2：卖Call收权利金，被行权后回到阶段1
"""

import pandas as pd
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum

from .base import (
    BaseOptionStrategy, StrategyParam, Signal, OptionLeg
)


class WheelPhase(Enum):
    """Wheel策略阶段"""
    IDLE = 0            # 空闲，无持仓
    SELL_PUT = 1        # 卖Put阶段
    HOLD_UNDERLYING = 2  # 持有标的（被行权后）
    SELL_CALL = 3       # 卖Call阶段


class WheelStrategy(BaseOptionStrategy):
    """Wheel循环收租策略"""

    name = "wheel"
    display_name = "Wheel策略"
    description = "循环卖Put和卖Call收取时间价值，适合长期看好的标的"
    version = "1.0"
    strategy_type = "neutral"
    warmup_bars = 50

    @classmethod
    def get_params(cls) -> List[StrategyParam]:
        return [
            StrategyParam("iv_percentile_min", "IV百分位下限", 30, 20, 50, 5, "int",
                         description="IV百分位高于此值才入场"),
            StrategyParam("delta_put", "卖Put Delta", -0.30, -0.40, -0.20, 0.05, "float",
                         description="卖出Put的目标Delta"),
            StrategyParam("delta_call", "卖Call Delta", 0.30, 0.20, 0.40, 0.05, "float",
                         description="卖出Call的目标Delta"),
            StrategyParam("days_to_expiry_min", "最小到期天数", 30, 21, 45, 7, "int"),
            StrategyParam("days_to_expiry_max", "最大到期天数", 45, 35, 60, 5, "int"),
            StrategyParam("profit_target", "止盈比例", 0.50, 0.30, 0.80, 0.1, "float",
                         description="权利金获得此比例时提前平仓"),
            StrategyParam("roll_days", "展期天数", 7, 5, 14, 1, "int",
                         description="距到期此天数考虑展期"),
            StrategyParam("underlying_stop_loss", "标的止损", 0.15, 0.10, 0.25, 0.05, "float",
                         description="标的下跌超过此比例止损"),
        ]

    def __init__(self, params: Dict = None, symbol: str = "IO"):
        super().__init__(params, symbol)
        self.phase = WheelPhase.IDLE
        self.underlying_cost = 0  # 标的成本（被行权价格）
        self.total_premium_collected = 0  # 累计收取的权利金

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算指标"""
        df = df.copy()

        # 简单趋势判断
        df['ma50'] = df['close'].rolling(50).mean()
        df['ma200'] = df['close'].rolling(200).mean()

        # 长期趋势
        df['long_trend'] = 0
        df.loc[df['ma50'] > df['ma200'], 'long_trend'] = 1
        df.loc[df['ma50'] < df['ma200'], 'long_trend'] = -1

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

        # 根据当前阶段处理
        if self.phase == WheelPhase.IDLE:
            return self._check_sell_put_entry(bar, spot, option_chain,
                                              iv_percentile, current_iv, days_to_expiry)

        elif self.phase == WheelPhase.SELL_PUT:
            return self._check_sell_put_exit(bar, spot, option_chain,
                                             iv_data, days_to_expiry)

        elif self.phase == WheelPhase.HOLD_UNDERLYING:
            return self._check_sell_call_entry(bar, spot, option_chain,
                                               iv_percentile, current_iv, days_to_expiry)

        elif self.phase == WheelPhase.SELL_CALL:
            return self._check_sell_call_exit(bar, spot, option_chain,
                                              iv_data, days_to_expiry)

        return None

    def _check_sell_put_entry(self, bar, spot, option_chain, iv_percentile,
                              current_iv, days_to_expiry) -> Optional[Signal]:
        """检查卖Put入场条件"""
        p = self.params

        # 条件1: IV百分位不能太低
        if iv_percentile < p['iv_percentile_min']:
            return None

        # 条件2: 长期趋势不能太弱（可选）
        long_trend = bar.get('long_trend', 0)
        if long_trend < 0:  # 长期下跌趋势中谨慎
            return None

        # 条件3: 到期天数合适
        if not (p['days_to_expiry_min'] <= days_to_expiry <= p['days_to_expiry_max']):
            return None

        T = days_to_expiry / 365

        # 选择卖Put行权价
        put_strike = self.select_strike(spot, option_chain, p['delta_put'],
                                        'put', T, current_iv)

        # 获取价格
        put_price = self.get_option_price(put_strike, 'put', option_chain,
                                          spot, T, current_iv)

        legs = [
            OptionLeg(
                option_type='put',
                strike=put_strike,
                expiry=self._get_nearest_expiry(option_chain),
                position=-1,  # 卖出
                quantity=1,
                entry_price=put_price,
                entry_time=self.current_time,
                iv=current_iv
            )
        ]

        self.phase = WheelPhase.SELL_PUT
        self.total_premium_collected += put_price

        return Signal(
            action='open',
            legs=legs,
            tag='wheel_sell_put',
            reason=f"卖Put收租，IV={iv_percentile:.0f}%，行权价={put_strike:.0f}",
            timestamp=self.current_time,
            underlying_price=spot,
            iv=current_iv
        )

    def _check_sell_put_exit(self, bar, spot, option_chain, iv_data,
                             days_to_expiry) -> Optional[Signal]:
        """检查卖Put出场条件"""
        p = self.params

        if not self.position or not self.position.legs:
            return None

        T = max(1, days_to_expiry) / 365
        current_iv = iv_data.get('current_iv', 0.2)

        leg = self.position.legs[0]
        leg.current_price = self.get_option_price(
            leg.strike, leg.option_type, option_chain,
            spot, T, current_iv
        )

        # 收入的权利金
        entry_premium = leg.entry_price
        # 当前平仓成本
        current_cost = leg.current_price

        # 盈利 = 收入 - 平仓成本
        pnl = entry_premium - current_cost

        # 条件1: 达到目标盈利（权利金衰减）
        if entry_premium > 0 and pnl >= entry_premium * p['profit_target']:
            self.phase = WheelPhase.IDLE
            self.position = None
            return Signal(
                action='close',
                legs=[leg],
                tag='profit_target',
                reason=f"权利金获利{pnl/entry_premium*100:.0f}%",
                timestamp=self.current_time,
                underlying_price=spot
            )

        # 条件2: 需要展期
        if days_to_expiry <= p['roll_days']:
            # 模拟被行权（如果价格低于行权价）
            if spot < leg.strike:
                # 被行权，进入持有标的阶段
                self.phase = WheelPhase.HOLD_UNDERLYING
                self.underlying_cost = leg.strike  # 成本为行权价
                self.position = None
                return Signal(
                    action='close',
                    legs=[leg],
                    tag='assigned',
                    reason=f"Put被行权，持有标的成本={leg.strike:.0f}",
                    timestamp=self.current_time,
                    underlying_price=spot
                )
            else:
                # 未被行权，可以展期或到期获利
                self.phase = WheelPhase.IDLE
                self.position = None
                return Signal(
                    action='close',
                    legs=[leg],
                    tag='expiry_profit',
                    reason=f"Put到期作废，获得全部权利金",
                    timestamp=self.current_time,
                    underlying_price=spot
                )

        return None

    def _check_sell_call_entry(self, bar, spot, option_chain, iv_percentile,
                               current_iv, days_to_expiry) -> Optional[Signal]:
        """检查卖Call入场条件（持有标的后）"""
        p = self.params

        # 条件1: IV不能太低
        if iv_percentile < p['iv_percentile_min']:
            return None

        # 条件2: 到期天数合适
        if not (p['days_to_expiry_min'] <= days_to_expiry <= p['days_to_expiry_max']):
            return None

        T = days_to_expiry / 365

        # 选择卖Call行权价（高于成本价）
        # 目标Delta约0.30，但行权价至少要高于成本
        call_strike = self.select_strike(spot, option_chain, p['delta_call'],
                                         'call', T, current_iv)

        # 确保行权价高于成本
        if call_strike < self.underlying_cost:
            call_strike = self.underlying_cost

        # 获取价格
        call_price = self.get_option_price(call_strike, 'call', option_chain,
                                           spot, T, current_iv)

        legs = [
            OptionLeg(
                option_type='call',
                strike=call_strike,
                expiry=self._get_nearest_expiry(option_chain),
                position=-1,  # 卖出
                quantity=1,
                entry_price=call_price,
                entry_time=self.current_time,
                iv=current_iv
            )
        ]

        self.phase = WheelPhase.SELL_CALL
        self.total_premium_collected += call_price

        return Signal(
            action='open',
            legs=legs,
            tag='wheel_sell_call',
            reason=f"卖Call（备兑），行权价={call_strike:.0f}，成本={self.underlying_cost:.0f}",
            timestamp=self.current_time,
            underlying_price=spot,
            iv=current_iv
        )

    def _check_sell_call_exit(self, bar, spot, option_chain, iv_data,
                              days_to_expiry) -> Optional[Signal]:
        """检查卖Call出场条件"""
        p = self.params

        if not self.position or not self.position.legs:
            return None

        T = max(1, days_to_expiry) / 365
        current_iv = iv_data.get('current_iv', 0.2)

        leg = self.position.legs[0]
        leg.current_price = self.get_option_price(
            leg.strike, leg.option_type, option_chain,
            spot, T, current_iv
        )

        entry_premium = leg.entry_price
        current_cost = leg.current_price
        pnl = entry_premium - current_cost

        # 条件1: 达到目标盈利
        if entry_premium > 0 and pnl >= entry_premium * p['profit_target']:
            # 继续持有标的，可以再卖Call
            self.phase = WheelPhase.HOLD_UNDERLYING
            self.position = None
            return Signal(
                action='close',
                legs=[leg],
                tag='profit_target',
                reason=f"Call权利金获利{pnl/entry_premium*100:.0f}%",
                timestamp=self.current_time,
                underlying_price=spot
            )

        # 条件2: 需要展期或到期
        if days_to_expiry <= p['roll_days']:
            if spot > leg.strike:
                # 被行权，卖出标的，回到IDLE
                self.phase = WheelPhase.IDLE
                profit = leg.strike - self.underlying_cost + self.total_premium_collected
                self.underlying_cost = 0
                self.total_premium_collected = 0
                self.position = None
                return Signal(
                    action='close',
                    legs=[leg],
                    tag='assigned',
                    reason=f"Call被行权，完成一轮Wheel，总利润约{profit:.0f}点",
                    timestamp=self.current_time,
                    underlying_price=spot
                )
            else:
                # 未被行权，继续持有标的
                self.phase = WheelPhase.HOLD_UNDERLYING
                self.position = None
                return Signal(
                    action='close',
                    legs=[leg],
                    tag='expiry_profit',
                    reason=f"Call到期作废，继续持有标的",
                    timestamp=self.current_time,
                    underlying_price=spot
                )

        # 条件3: 标的大幅下跌止损
        if self.underlying_cost > 0:
            underlying_loss = (self.underlying_cost - spot) / self.underlying_cost
            if underlying_loss > p['underlying_stop_loss']:
                # 止损，平掉Call和标的
                self.phase = WheelPhase.IDLE
                self.underlying_cost = 0
                self.total_premium_collected = 0
                self.position = None
                return Signal(
                    action='close',
                    legs=[leg],
                    tag='stop_loss',
                    reason=f"标的下跌{underlying_loss*100:.0f}%，止损",
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

    def get_status(self) -> Dict:
        """获取Wheel策略当前状态"""
        return {
            'phase': self.phase.name,
            'underlying_cost': self.underlying_cost,
            'total_premium_collected': self.total_premium_collected,
            'has_position': self.position is not None,
        }
