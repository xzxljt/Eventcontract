# 币安事件合约交易信号机器人

这是一个基于技术指标的币安事件合约交易信号生成和回测系统，可以帮助交易者分析市场趋势，生成交易信号，并进行回测验证。

## 项目结构

- `backtester.py`: 回测引擎，用于模拟历史数据上的交易信号表现
- `strategies.py`: 策略引擎，包含各种交易策略的实现
- `binance_client.py`: 币安API客户端，用于获取历史K线数据
- `main.py`: FastAPI后端服务，提供API接口
- `frontend/index.html`: 前端界面，用于可视化回测结果

## 功能特点

- 支持多种技术指标策略（RSI、布林带、EMA等）
- 可配置的事件合约周期（10分钟、30分钟、1小时、1天）
- 可调整的信号置信度阈值
- 详细的回测统计（总体胜率、做多/做空分析）
- 直观的前端界面展示回测结果

## 安装步骤

1. 克隆项目并进入项目目录

2. 创建并激活虚拟环境（推荐）
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. 安装依赖包
   ```
   pip install fastapi uvicorn pandas numpy python-dotenv requests
   ```

4. 配置币安API密钥
   - 在项目根目录创建`.env`文件
   - 添加以下内容（替换为你的API密钥）：
     ```
     BINANCE_API_KEY=你的币安API密钥
     BINANCE_API_SECRET=你的币安API密钥
     ```

## 使用方法

1. 启动后端服务
   ```
   python main.py
   ```
   服务将在 http://localhost:8000 运行

2. 打开前端界面
   - 直接在浏览器中打开 `frontend/index.html` 文件
   - 或者使用简单的HTTP服务器：
     ```
     cd frontend
     python -m http.server
     ```
     然后访问 http://localhost:8000

3. 使用界面进行回测
   - 选择交易对（如BTCUSDT）
   - 设置K线周期和时间范围
   - 选择策略并调整参数
   - 设置事件合约周期和置信度阈值
   - 点击"运行回测"按钮

## API文档

启动服务后，可以访问 http://localhost:8000/docs 查看完整的API文档。

主要API端点：
- `GET /api/symbols`: 获取可用的交易对列表
- `GET /api/strategies`: 获取可用的策略列表
- `POST /api/backtest`: 运行回测
- `GET /api/klines`: 获取K线数据

## 扩展策略

要添加新的交易策略，请在 `strategies.py` 文件中：

1. 创建一个继承自 `Strategy` 基类的新类
2. 实现 `calculate_indicators` 和 `generate_signals` 方法
3. 在 `get_available_strategies` 函数中注册新策略

## 注意事项

- 本项目仅供学习和研究使用，不构成投资建议
- 使用真实API密钥时请注意资金安全
- 回测结果不代表未来表现