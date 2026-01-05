# 中国股指期权量化系统

个人期权量化交易系统，支持IO(沪深300)、MO(中证1000)、HO(上证50)股指期权。

## 功能特性

- **IV监控**: 实时波动率监控、IV百分位、波动率曲面
- **Greeks计算**: Black-Scholes模型、Delta/Gamma/Theta/Vega/Rho
- **策略回测**: 支持多腿期权组合回测
- **损益分析器**: 期权组合构建、到期损益图

## 内置策略

| 策略 | 类型 | 说明 |
|------|------|------|
| 牛市Call价差 | 趋势 | EMA金叉后买入ATM Call + 卖出OTM Call |
| 熊市Put价差 | 趋势 | EMA死叉后买入ATM Put + 卖出OTM Put |
| 买入跨式 | 波动率 | IV低位买入ATM Call + ATM Put |
| 买入宽跨式 | 波动率 | IV低位买入OTM Call + OTM Put |
| Iron Condor | 收租 | 高IV震荡市卖出宽跨式+保护翼 |
| Wheel | 收租 | 循环卖Put/卖Call收取时间价值 |

## 快速开始

### Docker启动（推荐）

```bash
docker-compose up -d --build
```

访问 http://localhost:8503

### 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py --server.port 8503
```

## 技术栈

- Python 3.11
- Streamlit (Web界面)
- AKShare (数据源)
- SQLite (数据存储)
- Docker (部署)

## 项目结构

```
├── app.py              # Streamlit主应用
├── engine.py           # 回测引擎
├── greeks.py           # Greeks计算
├── iv_monitor.py       # IV监控
├── data_manager.py     # 数据管理
├── config.py           # 配置
├── strategies/         # 策略文件夹
│   ├── base.py
│   ├── bull_call_spread.py
│   ├── bear_put_spread.py
│   ├── long_straddle.py
│   ├── long_strangle.py
│   ├── iron_condor.py
│   └── wheel.py
├── Dockerfile
└── docker-compose.yml
```

## License

MIT
