"""
策略模块
"""

from .base import BaseOptionStrategy, OptionLeg, OptionPosition, StrategyParam, Signal

# 导入所有策略
from .bull_call_spread import BullCallSpread
from .bear_put_spread import BearPutSpread
from .long_straddle import LongStraddle
from .long_strangle import LongStrangle
from .iron_condor import IronCondor
from .wheel import WheelStrategy

# 策略注册表
STRATEGIES = {
    'bull_call_spread': BullCallSpread,
    'bear_put_spread': BearPutSpread,
    'long_straddle': LongStraddle,
    'long_strangle': LongStrangle,
    'iron_condor': IronCondor,
    'wheel': WheelStrategy,
}


def get_strategy(name: str):
    """获取策略类"""
    return STRATEGIES.get(name)


def list_strategies():
    """列出所有可用策略"""
    return list(STRATEGIES.keys())


def get_strategy_info(name: str):
    """获取策略信息"""
    strategy_class = STRATEGIES.get(name)
    if strategy_class:
        return {
            'name': strategy_class.name,
            'display_name': strategy_class.display_name,
            'description': strategy_class.description,
            'strategy_type': strategy_class.strategy_type,
        }
    return None
