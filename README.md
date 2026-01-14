# 币安事件合约交易信号机器人

基于技术指标的币安事件合约交易信号生成和回测系统。

## 项目功能

- **交易信号生成**：基于多种技术指标策略生成交易信号
- **回测系统**：支持历史数据回测，评估策略性能
- **参数优化**：自动优化策略参数，提高收益率
- **实时信号**：实时监控市场，生成实时交易信号
- **AutoX集成**：支持通过AutoX执行自动化交易
- **前端界面**：提供Web界面查看信号和配置系统

## 技术栈

- **后端**：Python 3.12, FastAPI, Uvicorn
- **前端**：HTML, CSS, JavaScript
- **数据处理**：Pandas, NumPy
- **交易所API**：币安WebSocket API
- **数据库**：SQLite

## 安装步骤

1. **克隆项目**

```bash
git clone https://github.com/xzxljt/Eventcontract.git
cd Eventcontract
```

2. **创建虚拟环境**

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows
```

3. **安装依赖**

```bash
pip install -r requirements.txt
```

4. **配置环境变量**

复制 `.env.example` 文件为 `.env` 并填写相应的配置：

```bash
cp .env.example .env
# 编辑 .env 文件，填写API密钥等信息
```

## 运行服务

```bash
python main.py
```

服务将在 `http://0.0.0.0:3100` 启动。

## 项目结构

```
├── main.py              # 主服务文件
├── backtester.py        # 回测系统
├── strategies.py        # 交易策略
├── investment_strategies.py  # 投资策略
├── optimization_engine.py    # 参数优化引擎
├── market_client.py     # 市场数据客户端
├── gate_client.py       #  gate.io客户端
├── timezone_utils.py    # 时区工具
├── config/              # 配置文件
├── data/                # 数据文件
├── logs/                # 日志文件
├── frontend/            # 前端界面
├── autoxjs/             # AutoX脚本
├── templates/           # 模板文件
├── .kline_cache/        # K线缓存
├── backup_script.sh     # 备份脚本
└── README.md            # 项目文档
```

## 使用说明

### 1. 查看交易信号

访问 `http://localhost:3100` 查看实时交易信号。

### 2. 运行回测

在前端界面选择交易对、时间周期和策略参数，点击"开始回测"按钮。

### 3. 优化策略参数

访问 `http://localhost:3100/optimization` 页面，设置优化参数范围，点击"开始优化"按钮。

### 4. 配置AutoX

1. 在 `autoxjs` 目录中配置AutoX脚本
2. 访问 `http://localhost:3100/autox` 页面管理AutoX客户端
3. 启用AutoX功能，系统将自动发送交易指令到客户端

## 策略说明

### 预测策略

- **simple_rsi**：基于RSI指标的简单策略
- **ma_crossover**：基于移动平均线交叉的策略
- **bollinger_bands**：基于布林带的策略
- **macd**：基于MACD指标的策略

### 投资策略

- **fixed**：固定金额投资策略
- **percent**：百分比投资策略
- **martingale**：马丁格尔投资策略
- **anti_martingale**：反马丁格尔投资策略

## 备份系统

项目包含自动备份脚本 `backup_script.sh`，用于备份所有重要的文件和数据：

```bash
./backup_script.sh
```

备份文件存储在 `备份文件/` 目录中，每个备份都有唯一的时间戳。

## 日志系统

- **服务日志**：存储在 `logs/service.log`
- **交易日志**：存储在 `logs/trade_logs/` 目录
- **回测日志**：存储在 `logs/` 目录

## 常见问题

### 1. 服务无法启动

- 检查 `.env` 文件配置是否正确
- 检查网络连接是否正常
- 检查币安API密钥是否有效

### 2. 没有交易信号

- 检查策略参数是否合理
- 检查市场数据是否正常获取
- 检查时间周期设置是否正确

### 3. AutoX连接失败

- 检查AutoX客户端是否运行
- 检查WebSocket连接是否正常
- 检查客户端ID是否正确配置

## 贡献指南

欢迎提交Issue和Pull Request，帮助改进项目。

## 许可证

MIT License
