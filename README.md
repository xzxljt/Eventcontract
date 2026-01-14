# 币安事件合约交易信号机器人

原作者仓库：[https://github.com/R1ckyyyyy/Eventcontract](https://github.com/R1ckyyyyy/Eventcontract)

这是一个功能全面的币安事件合约交易信号生成、回测和管理系统。项目集成了多种技术分析策略和投资策略，能够通过币安 API 获取实时和历史市场数据，进行详细的策略回测，并通过交互式 Web 界面进行实时信号监控、自动交易客户端管理和配置。**生成的交易信号会通过 WebSocket 推送到连接的 AutoX 客户端，从而在有合适的客户端代码时实现自动交易。** 项目采用 FastAPI 构建后端服务，并通过环境变量和配置文件进行灵活管理。

## 功能特点

-   **多策略支持:** 集成多种预测策略（如 RSI, MACD, 布林带等）和投资策略（如固定金额、马丁格尔、反马丁格尔等）。
-   **市场数据获取:** 通过币安 API 高效获取历史 K 线数据和实时市场数据。
-   **高级回测引擎:** 提供详细的回测功能，模拟策略在历史数据上的表现，并生成全面的统计报告，包括总体胜率、盈亏分析、最大回撤等。
-   **实时信号监控:** 通过 WebSocket 连接向前端推送实时生成的交易信号。
-   **自动交易集成:** 支持连接和管理自动交易客户端。**生成的交易信号会实时推送到连接的 AutoX 客户端，使得在客户端实现相应的交易逻辑后，可以实现自动执行交易。**
-   **灵活配置:** 支持通过 `.env` 文件配置敏感信息（如 API 密钥）和通过 `config/` 目录下的 JSON 文件配置策略参数、活动测试和自动交易客户端数据。
-   **Web 交互界面:** 提供直观的前端界面，包括主页、实时测试页面和自动交易客户端管理页面。
-   **RESTful API:** 提供基于 FastAPI 的 API 接口，支持数据获取、回测触发、配置管理等功能。
-   **跨平台兼容性:** 考虑并处理了 Windows 和 Linux 环境之间的文件路径、大小写敏感性和行尾符差异。
-   **时区处理:** 使用 [`timezone_utils.py`](timezone_utils.py) 模块处理时区转换，确保时间数据的准确性。

## 项目结构

-   `main.py`: 项目主入口，初始化 FastAPI 应用，定义 API 端点和 WebSocket 端点，并启动后端服务。
-   `backtester.py`: 实现核心回测逻辑，包括信号处理、交易模拟和统计计算。
-   `strategies.py`: 包含各种预测交易策略的基类和具体实现。
-   `investment_strategies.py`: 包含各种投资策略的基类和具体实现。
-   `binance_client.py`: 封装与币安 API 的交互，处理数据获取、WebSocket 连接和数据缓存。
-   `timezone_utils.py`: 提供时区相关的工具函数，用于处理时间数据的转换和格式化。
-   `requirements.txt`: 项目所需的 Python 依赖列表。
-   `.env.example`: 环境变量配置示例文件，说明需要配置的环境变量。
-   `.env`: 实际的环境变量配置文件（**不应提交到版本控制**）。
-   `.gitignore`: Git 忽略文件配置，确保敏感文件（如 `.env`）不被提交。
-   `config/`: 存放项目配置相关的 JSON 文件：
    -   `active_test_config.json`: 存储当前活动的实时测试配置。
    -   `autox_clients_data.json`: 存储自动交易客户端的持久化数据。
    -   `strategy_parameters.json`: 存储各种策略的参数配置。
-   `frontend/`: 包含前端 Web 界面文件和静态资源：
    -   `index.html`: 项目主页。
    -   `autox-manager.html`: 自动交易客户端管理页面。
    -   `live-test.html`: 实时信号测试和监控页面。
    -   `static/`: 存放 CSS, JavaScript, 图片等静态资源。
-   `templates/`: 存放 HTML 模板文件，例如 `navbar.html`。

## 设置和运行

1.  **克隆项目并进入项目目录**

2.  **创建并激活虚拟环境（推荐）**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Linux/macOS
    source .venv/bin/activate
    ```

3.  **安装依赖包**
    项目依赖列在 [`requirements.txt`](requirements.txt) 文件中。使用 pip 安装：
    ```bash
    pip install -r requirements.txt
    ```

4.  **配置环境变量**
    项目使用 `.env` 文件管理环境变量，特别是敏感信息如 API 密钥。
    请复制项目根目录下的 `.env.example` 文件并将其重命名为 `.env`：
    ```bash
    cp .env.example .env # Linux/macOS
    copy .env.example .env # Windows
    ```
    然后编辑新创建的 `.env` 文件，根据 `.env.example` 中的示例填入你的实际配置。至少需要配置币安 API 密钥：
    ```dotenv
    BINANCE_API_KEY=你的币安API密钥
    BINANCE_API_SECRET=你的币安API密钥
    # 根据 .env.example 补充其他必要的配置项
    ```
    **重要提示:** `.env` 文件包含敏感信息，不应提交到版本控制中。`.gitignore` 文件已配置忽略 `.env` 文件。

5.  **启动应用服务**
    在项目根目录下运行：
    ```bash
    uvicorn main:app
    ```
    服务将在 `.env` 文件或环境变量中配置的主机和端口上运行 (默认为 `http://127.0.0.1:8000`)。

6.  **访问前端界面**
    启动应用服务后，前端界面即可通过浏览器访问。访问地址为主机和端口，例如 `http://127.0.0.1:8000` (如果使用默认配置)。

## API文档

启动应用服务后，可以访问 `/docs` 路径查看由 FastAPI 自动生成的完整 API 文档（Swagger UI）。访问地址为主机和端口加上 `/docs`，例如 `http://127.0.0.1:8000/docs` (如果使用默认配置)。

主要 API 端点包括：

-   `/api/symbols`: 获取可用的交易对列表。
-   `/api/prediction-strategies`: 获取可用的预测策略列表。
-   `/api/investment-strategies`: 获取可用的投资策略列表。
-   `/api/backtest`: 运行回测并获取结果。
-   `/api/live-signals`: 获取实时信号。
-   `/api/test-signal`: 生成测试信号。
-   `/api/load_all_strategy_parameters`: 加载所有策略参数配置。
-   `/api/save_strategy_parameter_set`: 保存策略参数配置。
-   `/api/autox/trade_logs`: 获取自动交易日志。
-   `/api/autox/clients/...`: 自动交易客户端相关的 API 端点（发送指令、更新备注等）。
-   `/api/config/...`: 配置相关的 API 端点。

## WebSocket 端点

WebSocket 端点与应用服务在同一主机和端口上。

-   `/ws/live-test`: 用于实时信号监控的前端 WebSocket 连接。
-   `/ws/autox_control`: 用于自动交易客户端连接和接收指令的 WebSocket 连接。
-   `/ws/autox_status`: 用于前端监控自动交易客户端状态的 WebSocket 连接。

## 扩展指南

-   **扩展预测策略:** 在 [`strategies.py`](strategies.py) 中创建新的策略类，继承 `Strategy` 基类并实现必要方法，然后在 `get_available_strategies` 函数中注册。
-   **扩展投资策略:** 在 [`investment_strategies.py`](investment_strategies.py) 中创建新的投资策略类，继承 `BaseInvestmentStrategy` 基类并实现必要方法，然后在 `get_available_investment_strategies` 函数中注册。
-   **添加新的 API 端点:** 在 [`main.py`](main.py) 中使用 FastAPI 的装饰器 (`@app.get`, `@app.post` 等) 定义新的 API 路由和处理函数。
-   **修改前端界面:** 修改 `frontend/` 目录下的 HTML, CSS, JavaScript 文件。

## 注意事项

-   本项目仅供学习和研究使用，不构成投资建议。
-   使用真实 API 密钥时请注意资金安全。
-   回测结果不代表未来表现。

## 原作者信息

本项目基于 [https://github.com/R1ckyyyyy/Eventcontract](https://github.com/R1ckyyyyy/Eventcontract) 开发，感谢原作者的贡献。

### 原作者仓库
- **GitHub仓库**: [https://github.com/R1ckyyyyy/Eventcontract](https://github.com/R1ckyyyyy/Eventcontract)

### 项目致谢
- 感谢原作者开发的基础框架和核心功能
- 感谢原作者提供的技术分析策略和投资策略实现
- 感谢原作者的文档和代码注释，为项目维护和扩展提供了帮助