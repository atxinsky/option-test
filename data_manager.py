"""
数据管理模块
支持股指期权行情、标的指数、IV历史数据的获取和存储
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
import requests
from bs4 import BeautifulSoup
import time

try:
    import akshare as ak
except ImportError:
    ak = None
    print("Warning: akshare not installed. Run: pip install akshare")

from config import (
    OPTION_INSTRUMENTS, AKSHARE_MAPPING, DB_PATH,
    OPTBBS_URL, RISK_FREE_RATE
)


# ============== 数据库操作 ==============

def get_db_path() -> str:
    """获取数据库完整路径"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, DB_PATH)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return db_path


def get_connection() -> sqlite3.Connection:
    """获取数据库连接"""
    return sqlite3.connect(get_db_path())


def init_database():
    """初始化数据库表"""
    conn = get_connection()
    cursor = conn.cursor()

    # 标的指数日线数据表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS underlying_daily (
            symbol TEXT NOT NULL,
            time DATE NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            amount REAL,
            PRIMARY KEY (symbol, time)
        )
    """)

    # 期权日线数据表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS option_daily (
            symbol TEXT NOT NULL,
            contract TEXT NOT NULL,
            strike REAL,
            expiry DATE,
            option_type TEXT,
            time DATE NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            settle REAL,
            volume REAL,
            open_interest REAL,
            iv REAL,
            PRIMARY KEY (symbol, contract, time)
        )
    """)

    # IV历史数据表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS iv_history (
            symbol TEXT NOT NULL,
            time DATE NOT NULL,
            iv REAL,
            iv_percentile REAL,
            hv_20 REAL,
            hv_60 REAL,
            vix REAL,
            skew REAL,
            PRIMARY KEY (symbol, time)
        )
    """)

    # 期权链快照表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS option_chain (
            symbol TEXT NOT NULL,
            expiry DATE NOT NULL,
            strike REAL NOT NULL,
            time DATETIME NOT NULL,
            call_price REAL,
            put_price REAL,
            call_iv REAL,
            put_iv REAL,
            call_delta REAL,
            put_delta REAL,
            call_volume REAL,
            put_volume REAL,
            call_oi REAL,
            put_oi REAL,
            PRIMARY KEY (symbol, expiry, strike, time)
        )
    """)

    # 数据更新记录表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS data_update_log (
            symbol TEXT NOT NULL,
            data_type TEXT NOT NULL,
            last_update DATETIME,
            start_date DATE,
            end_date DATE,
            record_count INTEGER,
            PRIMARY KEY (symbol, data_type)
        )
    """)

    conn.commit()
    conn.close()
    print("Database initialized successfully.")


# ============== AKShare数据获取 ==============

def download_underlying_daily(symbol: str, start_date: str = None,
                               end_date: str = None) -> Optional[pd.DataFrame]:
    """
    下载标的指数日线数据

    Args:
        symbol: 指数代码，如 '000300'（沪深300）
        start_date: 开始日期，格式 'YYYYMMDD'
        end_date: 结束日期

    Returns:
        DataFrame或None
    """
    if ak is None:
        print("Error: akshare not available")
        return None

    try:
        # 使用AKShare获取指数数据
        df = ak.index_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date or "20200101",
            end_date=end_date or datetime.now().strftime("%Y%m%d")
        )

        if df is not None and not df.empty:
            # 重命名列
            df = df.rename(columns={
                '日期': 'time',
                '开盘': 'open',
                '最高': 'high',
                '最低': 'low',
                '收盘': 'close',
                '成交量': 'volume',
                '成交额': 'amount'
            })
            df['symbol'] = symbol
            df['time'] = pd.to_datetime(df['time']).dt.date

            return df[['symbol', 'time', 'open', 'high', 'low', 'close', 'volume', 'amount']]

    except Exception as e:
        print(f"Error downloading underlying data for {symbol}: {e}")

    return None


def save_underlying_daily(df: pd.DataFrame):
    """保存标的指数日线数据到数据库"""
    if df is None or df.empty:
        return

    conn = get_connection()
    df.to_sql('underlying_daily', conn, if_exists='replace', index=False,
              method='multi')
    conn.close()


def load_underlying_daily(symbol: str, start_date: str = None,
                          end_date: str = None) -> pd.DataFrame:
    """从数据库加载标的指数日线数据"""
    conn = get_connection()

    query = "SELECT * FROM underlying_daily WHERE symbol = ?"
    params = [symbol]

    if start_date:
        query += " AND time >= ?"
        params.append(start_date)
    if end_date:
        query += " AND time <= ?"
        params.append(end_date)

    query += " ORDER BY time"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if not df.empty:
        df['time'] = pd.to_datetime(df['time'])

    return df


def download_option_realtime(symbol: str = "IO",
                              end_month: str = None) -> Optional[pd.DataFrame]:
    """
    获取期权实时行情

    Args:
        symbol: 期权品种，'IO'/'MO'/'HO'
        end_month: 到期月份，如 '2501'

    Returns:
        DataFrame或None
    """
    if ak is None:
        print("Error: akshare not available")
        return None

    try:
        mapping = AKSHARE_MAPPING.get(symbol)
        if not mapping:
            print(f"Unknown option symbol: {symbol}")
            return None

        # 获取实时行情
        df = ak.option_finance_board(
            symbol=mapping['option_board'],
            end_month=end_month or ""
        )

        if df is not None and not df.empty:
            return df

    except Exception as e:
        print(f"Error downloading option realtime for {symbol}: {e}")

    return None


def download_option_contracts(symbol: str = "IO") -> Optional[pd.DataFrame]:
    """
    获取期权合约列表

    Args:
        symbol: 期权品种，'IO'/'MO'/'HO'

    Returns:
        DataFrame或None
    """
    if ak is None:
        return None

    try:
        mapping = AKSHARE_MAPPING.get(symbol)
        if not mapping:
            return None

        # 获取合约列表
        if symbol == "IO":
            df = ak.option_cffex_hs300_list_sina()
        elif symbol == "MO":
            df = ak.option_cffex_zz1000_list_sina()
        elif symbol == "HO":
            df = ak.option_cffex_sz50_list_sina()
        else:
            return None

        return df

    except Exception as e:
        print(f"Error downloading option contracts for {symbol}: {e}")
        return None


def download_option_history(contract: str, symbol: str = "IO") -> Optional[pd.DataFrame]:
    """
    下载单个期权合约的历史数据

    Args:
        contract: 合约代码，如 'io2501C3900'
        symbol: 期权品种

    Returns:
        DataFrame或None
    """
    if ak is None:
        return None

    try:
        if symbol == "IO":
            df = ak.option_cffex_hs300_daily_sina(symbol=contract)
        elif symbol == "MO":
            df = ak.option_cffex_zz1000_daily_sina(symbol=contract)
        elif symbol == "HO":
            df = ak.option_cffex_sz50_daily_sina(symbol=contract)
        else:
            return None

        if df is not None and not df.empty:
            df['contract'] = contract
            df['symbol'] = symbol
            return df

    except Exception as e:
        print(f"Error downloading history for {contract}: {e}")

    return None


# ============== IV数据获取（期权论坛） ==============

def fetch_optbbs_iv_data(symbol: str = "300etf") -> Optional[pd.DataFrame]:
    """
    从期权论坛获取IV数据

    Args:
        symbol: 标的，如 '300etf', '50etf', '300index'

    Returns:
        DataFrame或None
    """
    try:
        url = f"{OPTBBS_URL}/s/hv.shtml"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = 'utf-8'

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'lxml')
            # 解析页面数据
            # 注意：实际解析逻辑需要根据网页结构调整
            tables = soup.find_all('table')

            if tables:
                # 尝试解析第一个表格
                df = pd.read_html(str(tables[0]))[0]
                return df

    except Exception as e:
        print(f"Error fetching IV data from optbbs: {e}")

    return None


def calculate_iv_percentile(iv_series: pd.Series, current_iv: float,
                            window: int = 252) -> float:
    """
    计算IV百分位

    Args:
        iv_series: IV历史序列
        current_iv: 当前IV
        window: 计算窗口（默认252个交易日=1年）

    Returns:
        IV百分位（0-100）
    """
    if len(iv_series) < window:
        recent_iv = iv_series
    else:
        recent_iv = iv_series[-window:]

    percentile = (recent_iv < current_iv).sum() / len(recent_iv) * 100
    return percentile


def save_iv_history(df: pd.DataFrame):
    """保存IV历史数据"""
    if df is None or df.empty:
        return

    conn = get_connection()

    for _, row in df.iterrows():
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO iv_history
            (symbol, time, iv, iv_percentile, hv_20, hv_60, vix, skew)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row.get('symbol'),
            row.get('time'),
            row.get('iv'),
            row.get('iv_percentile'),
            row.get('hv_20'),
            row.get('hv_60'),
            row.get('vix'),
            row.get('skew')
        ))

    conn.commit()
    conn.close()


def load_iv_history(symbol: str, start_date: str = None,
                    end_date: str = None) -> pd.DataFrame:
    """加载IV历史数据"""
    conn = get_connection()

    query = "SELECT * FROM iv_history WHERE symbol = ?"
    params = [symbol]

    if start_date:
        query += " AND time >= ?"
        params.append(start_date)
    if end_date:
        query += " AND time <= ?"
        params.append(end_date)

    query += " ORDER BY time"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if not df.empty:
        df['time'] = pd.to_datetime(df['time'])

    return df


# ============== 期权链数据 ==============

def get_option_chain(symbol: str = "IO",
                     expiry: str = None) -> Optional[pd.DataFrame]:
    """
    获取期权链数据（T型报价）

    Args:
        symbol: 期权品种
        expiry: 到期月份，如 '2501'

    Returns:
        DataFrame或None
    """
    df = download_option_realtime(symbol, expiry)

    if df is None or df.empty:
        return None

    # 解析合约信息
    # 合约格式: IO2501-C-3900
    chain_data = []

    for _, row in df.iterrows():
        contract = row.get('instrument', row.get('合约代码', ''))
        if not contract:
            continue

        # 解析合约
        parts = contract.split('-')
        if len(parts) >= 3:
            base = parts[0]
            opt_type = parts[1]  # C或P
            strike = float(parts[2])

            # 提取到期月份
            expiry_month = ''.join(filter(str.isdigit, base))

            chain_data.append({
                'contract': contract,
                'expiry': expiry_month,
                'strike': strike,
                'option_type': 'call' if opt_type == 'C' else 'put',
                'last_price': row.get('lastprice', row.get('最新价', 0)),
                'volume': row.get('volume', row.get('成交量', 0)),
                'open_interest': row.get('position', row.get('持仓量', 0)),
                'bid_price': row.get('bprice', row.get('买价', 0)),
                'ask_price': row.get('sprice', row.get('卖价', 0)),
            })

    if chain_data:
        return pd.DataFrame(chain_data)

    return None


def format_option_chain_ttype(chain_df: pd.DataFrame,
                               expiry: str = None) -> pd.DataFrame:
    """
    格式化为T型报价格式

    Args:
        chain_df: 期权链数据
        expiry: 到期月份筛选

    Returns:
        T型报价DataFrame
    """
    if chain_df is None or chain_df.empty:
        return pd.DataFrame()

    if expiry:
        chain_df = chain_df[chain_df['expiry'] == expiry]

    # 分离Call和Put
    calls = chain_df[chain_df['option_type'] == 'call'].copy()
    puts = chain_df[chain_df['option_type'] == 'put'].copy()

    # 合并为T型报价
    calls = calls.set_index('strike')
    puts = puts.set_index('strike')

    t_type = pd.DataFrame(index=sorted(set(calls.index) | set(puts.index)))

    # Call列
    t_type['call_price'] = calls['last_price']
    t_type['call_volume'] = calls['volume']
    t_type['call_oi'] = calls['open_interest']

    # 行权价
    t_type['strike'] = t_type.index

    # Put列
    t_type['put_price'] = puts['last_price']
    t_type['put_volume'] = puts['volume']
    t_type['put_oi'] = puts['open_interest']

    return t_type.reset_index(drop=True)


# ============== 数据状态查询 ==============

def get_data_status() -> pd.DataFrame:
    """获取所有数据的状态"""
    conn = get_connection()

    # 检查表是否存在
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]

    status_data = []

    # 标的数据状态
    if 'underlying_daily' in tables:
        cursor.execute("""
            SELECT symbol, MIN(time), MAX(time), COUNT(*)
            FROM underlying_daily GROUP BY symbol
        """)
        for row in cursor.fetchall():
            status_data.append({
                'data_type': '标的指数',
                'symbol': row[0],
                'start_date': row[1],
                'end_date': row[2],
                'records': row[3]
            })

    # IV数据状态
    if 'iv_history' in tables:
        cursor.execute("""
            SELECT symbol, MIN(time), MAX(time), COUNT(*)
            FROM iv_history GROUP BY symbol
        """)
        for row in cursor.fetchall():
            status_data.append({
                'data_type': 'IV历史',
                'symbol': row[0],
                'start_date': row[1],
                'end_date': row[2],
                'records': row[3]
            })

    # 期权数据状态
    if 'option_daily' in tables:
        cursor.execute("""
            SELECT symbol, MIN(time), MAX(time), COUNT(*)
            FROM option_daily GROUP BY symbol
        """)
        for row in cursor.fetchall():
            status_data.append({
                'data_type': '期权日线',
                'symbol': row[0],
                'start_date': row[1],
                'end_date': row[2],
                'records': row[3]
            })

    conn.close()

    return pd.DataFrame(status_data) if status_data else pd.DataFrame()


def download_all_data(symbols: List[str] = None,
                      start_date: str = None,
                      end_date: str = None):
    """
    下载所有数据

    Args:
        symbols: 要下载的品种列表，默认全部
        start_date: 开始日期
        end_date: 结束日期
    """
    if symbols is None:
        symbols = ['IO', 'MO', 'HO']

    # 初始化数据库
    init_database()

    # 下载标的指数数据
    underlying_map = {
        'IO': '000300',  # 沪深300
        'MO': '000852',  # 中证1000
        'HO': '000016',  # 上证50
    }

    for sym in symbols:
        underlying_code = underlying_map.get(sym)
        if underlying_code:
            print(f"Downloading underlying data for {sym} ({underlying_code})...")
            df = download_underlying_daily(underlying_code, start_date, end_date)
            if df is not None:
                save_underlying_daily(df)
                print(f"  Saved {len(df)} records")

    print("Data download completed.")


# ============== 工具函数 ==============

def get_trading_days(start_date: str, end_date: str) -> List[str]:
    """获取交易日列表"""
    if ak is None:
        return []

    try:
        df = ak.tool_trade_date_hist_sina()
        df['trade_date'] = pd.to_datetime(df['trade_date'])

        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        mask = (df['trade_date'] >= start) & (df['trade_date'] <= end)
        trading_days = df.loc[mask, 'trade_date'].dt.strftime('%Y-%m-%d').tolist()

        return trading_days

    except Exception as e:
        print(f"Error getting trading days: {e}")
        return []


def days_to_expiry(expiry_date: str, current_date: str = None) -> int:
    """计算距到期日的天数"""
    if current_date is None:
        current = datetime.now().date()
    else:
        current = pd.to_datetime(current_date).date()

    expiry = pd.to_datetime(expiry_date).date()

    return (expiry - current).days


def parse_option_contract(contract: str) -> Dict:
    """
    解析期权合约代码

    Args:
        contract: 合约代码，如 'IO2501-C-3900' 或 'io2501C3900'

    Returns:
        解析后的字典
    """
    contract = contract.upper()

    # 格式1: IO2501-C-3900
    if '-' in contract:
        parts = contract.split('-')
        if len(parts) >= 3:
            base = parts[0]
            opt_type = 'call' if parts[1] == 'C' else 'put'
            strike = float(parts[2])

            symbol = ''.join(filter(str.isalpha, base))
            expiry = ''.join(filter(str.isdigit, base))

            return {
                'symbol': symbol,
                'expiry': expiry,
                'option_type': opt_type,
                'strike': strike
            }

    # 格式2: io2501C3900
    import re
    match = re.match(r'([A-Z]+)(\d+)([CP])(\d+)', contract)
    if match:
        return {
            'symbol': match.group(1),
            'expiry': match.group(2),
            'option_type': 'call' if match.group(3) == 'C' else 'put',
            'strike': float(match.group(4))
        }

    return {}


if __name__ == "__main__":
    # 测试
    print("Initializing database...")
    init_database()

    print("\nTesting underlying data download...")
    df = download_underlying_daily("000300", "20240101", "20241231")
    if df is not None:
        print(f"Downloaded {len(df)} records for 沪深300")
        print(df.tail())

    print("\nTesting option realtime...")
    chain = download_option_realtime("IO")
    if chain is not None:
        print(f"Got {len(chain)} option contracts")
        print(chain.head())

    print("\nData status:")
    status = get_data_status()
    print(status)
