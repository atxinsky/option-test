"""
期权品种配置
中国股指期权（IO/MO/HO）配置参数
"""

# 股指期权品种配置
OPTION_INSTRUMENTS = {
    "IO": {
        "name": "沪深300股指期权",
        "underlying": "沪深300",
        "underlying_code": "000300",
        "exchange": "CFFEX",
        "multiplier": 100,          # 合约乘数
        "price_tick": 0.2,          # 最小变动单位
        "margin_rate": 0.12,        # 保证金比例（卖方）
        "commission_rate": 15,      # 手续费（元/手）
        "exercise_type": "european", # 欧式期权
        "settlement": "cash",       # 现金交割
        "trading_hours": "09:30-11:30,13:00-15:00",
        "night_trade": False,
        "dividend_yield": 0.025,    # 估算分红率
    },
    "MO": {
        "name": "中证1000股指期权",
        "underlying": "中证1000",
        "underlying_code": "000852",
        "exchange": "CFFEX",
        "multiplier": 100,
        "price_tick": 0.2,
        "margin_rate": 0.15,
        "commission_rate": 15,
        "exercise_type": "european",
        "settlement": "cash",
        "trading_hours": "09:30-11:30,13:00-15:00",
        "night_trade": False,
        "dividend_yield": 0.015,
    },
    "HO": {
        "name": "上证50股指期权",
        "underlying": "上证50",
        "underlying_code": "000016",
        "exchange": "CFFEX",
        "multiplier": 100,
        "price_tick": 0.2,
        "margin_rate": 0.12,
        "commission_rate": 15,
        "exercise_type": "european",
        "settlement": "cash",
        "trading_hours": "09:30-11:30,13:00-15:00",
        "night_trade": False,
        "dividend_yield": 0.030,
    },
}

# ETF期权配置（备用）
ETF_OPTION_INSTRUMENTS = {
    "510050": {
        "name": "50ETF期权",
        "underlying": "华夏上证50ETF",
        "exchange": "SSE",
        "multiplier": 10000,
        "price_tick": 0.0001,
        "margin_rate": 0.12,
        "commission_rate": 1.5,
        "exercise_type": "european",
        "dividend_yield": 0.025,
    },
    "510300": {
        "name": "300ETF期权(华泰柏瑞)",
        "underlying": "华泰柏瑞沪深300ETF",
        "exchange": "SSE",
        "multiplier": 10000,
        "price_tick": 0.0001,
        "margin_rate": 0.12,
        "commission_rate": 1.5,
        "exercise_type": "european",
        "dividend_yield": 0.020,
    },
}

# AKShare数据接口映射
AKSHARE_MAPPING = {
    "IO": {
        "option_board": "沪深300股指期权",
        "history_func": "option_cffex_hs300_daily_sina",
        "list_func": "option_cffex_hs300_list_sina",
        "spot_func": "option_cffex_hs300_spot_sina",
    },
    "MO": {
        "option_board": "中证1000股指期权",
        "history_func": "option_cffex_zz1000_daily_sina",
        "list_func": "option_cffex_zz1000_list_sina",
        "spot_func": "option_cffex_zz1000_spot_sina",
    },
    "HO": {
        "option_board": "上证50股指期权",
        "history_func": "option_cffex_sz50_daily_sina",
        "list_func": "option_cffex_sz50_list_sina",
        "spot_func": "option_cffex_sz50_spot_sina",
    },
}

# 无风险利率（参考1年期国债或SHIBOR）
RISK_FREE_RATE = 0.02

# IV百分位阈值配置
IV_THRESHOLDS = {
    "extreme_low": 10,      # 极低波动率
    "low": 25,              # 低波动率
    "medium": 50,           # 中等波动率
    "high": 75,             # 高波动率
    "extreme_high": 90,     # 极高波动率
}

# 策略参数默认值
STRATEGY_DEFAULTS = {
    "bull_call_spread": {
        "ema_fast": 9,
        "ema_slow": 21,
        "ma_len": 20,
        "iv_percentile_max": 50,
        "delta_buy": 0.50,
        "delta_sell": 0.30,
        "spread_width": 100,        # 行权价间距（点）
        "days_to_expiry_min": 20,
        "days_to_expiry_max": 45,
        "profit_target": 0.80,      # 最大盈利80%止盈
        "time_decay_stop": 0.50,    # 时间价值损耗50%止损
        "days_before_expiry_close": 5,
    },
    "bear_put_spread": {
        "ema_fast": 9,
        "ema_slow": 21,
        "ma_len": 20,
        "iv_percentile_max": 50,
        "delta_buy": -0.50,
        "delta_sell": -0.30,
        "spread_width": 100,
        "days_to_expiry_min": 20,
        "days_to_expiry_max": 45,
        "profit_target": 0.80,
        "time_decay_stop": 0.50,
        "days_before_expiry_close": 5,
    },
    "long_straddle": {
        "iv_percentile_max": 25,
        "hv_iv_ratio_min": 1.0,     # HV > IV
        "days_to_expiry_min": 14,
        "days_to_expiry_max": 45,
        "iv_rise_target": 0.30,     # IV上升30%止盈
        "time_decay_stop": 0.50,
        "days_after_event": 3,
    },
    "long_strangle": {
        "iv_percentile_max": 20,
        "delta_call": 0.25,
        "delta_put": -0.25,
        "days_to_expiry_min": 14,
        "days_to_expiry_max": 45,
        "time_decay_stop": 0.50,
    },
    "iron_condor": {
        "iv_percentile_min": 50,
        "adx_max": 25,
        "delta_sell_call": 0.15,
        "delta_sell_put": -0.15,
        "delta_buy_call": 0.05,
        "delta_buy_put": -0.05,
        "days_to_expiry_min": 20,
        "days_to_expiry_max": 45,
        "profit_target": 0.50,
        "loss_stop": 0.50,
        "days_before_expiry_close": 7,
    },
    "wheel": {
        "iv_percentile_min": 30,
        "delta_put": -0.30,
        "days_to_expiry_min": 30,
        "days_to_expiry_max": 45,
    },
}

# 数据库配置
DB_PATH = "data/option_data.db"

# 期权论坛数据源
OPTBBS_URL = "http://1.optbbs.com"
