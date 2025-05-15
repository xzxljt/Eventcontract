# 币安事件合约交易信号机器人

这是一个用于币安事件合约的交易信号生成和回测系统。它基于技术指标分析市场数据，生成交易信号，并允许用户通过回测验证策略在历史数据上的表现。

## 项目结构

- `backtester.py`: 回测引擎，用于模拟历史数据上的交易信号表现。
- `strategies.py`: 策略引擎，包含各种交易策略的实现。
- `binance_client.py`: 币安API客户端，用于获取历史K线数据。
- `main.py`: FastAPI后端服务，提供API接口。
- `frontend/`: 包含前端界面的文件。
- `.gitignore`: 指定 Git 应该忽略的文件和目录，例如敏感配置或编译生成的文件。

## 功能特点

- 支持多种技术指标策略（RSI、布林带、EMA等）。
- 可配置的事件合约周期（10分钟、30分钟、1小时、1天）。
- 可调整的信号置信度阈值。
- 详细的回测统计（总体胜率、做多/做空分析）。
- 直观的前端界面展示回测结果。

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
    在项目根目录创建 `.env` 文件，用于存储敏感信息和配置。请参考 `.env.example`（如果存在，否则直接说明格式）或在 `.env` 文件中添加以下内容（替换为你的实际API密钥）：
    ```dotenv
    BINANCE_API_KEY=你的币安API密钥
    BINANCE_API_SECRET=你的币安API密钥
    ```

5.  **启动后端服务**
    ```bash
    python main.py
    ```
    服务将在 `http://localhost:8000` 运行。

6.  **打开前端界面**
    - 直接在浏览器中打开 `frontend/index.html` 文件。
    - 或者使用简单的HTTP服务器（例如 Python 的 `http.server`）：
      ```bash
      cd frontend
      python -m http.server
      ```
      然后访问 `http://localhost:8000`。

## 跨平台注意事项 (Windows 到 Linux)

在将项目从 Windows 环境迁移到 Linux 环境时，请注意以下几点：

-   **文件系统大小写敏感性**: Linux 文件系统是大小写敏感的。确保代码中引用的文件名和目录名与实际文件系统中的大小写完全匹配。例如，`main.py` 和 `Main.py` 在 Linux 中是不同的文件。
-   **文件路径**: 文件路径分隔符在 Windows 中通常是反斜杠 `\`，而在 Linux 中是正斜杠 `/`。Python 的 `os` 模块通常可以处理跨平台路径，但在手动构建路径时请注意这一点，或者使用 `os.path.join()`。
-   **行尾符**: Windows 使用 CRLF (回车+换行) 作为行尾符，而 Linux 使用 LF (换行)。这可能会导致一些脚本文件在不同系统上执行问题。Git 可以配置自动处理行尾符（`core.autocrlf` 设置），或者可以使用工具（如 `dos2unix`）进行转换。

## API文档

启动服务后，可以访问 `http://localhost:8000/docs` 查看完整的API文档。

主要API端点：

-   `GET /api/symbols`: 获取可用的交易对列表。
-   `GET /api/strategies`: 获取可用的策略列表。
-   `POST /api/backtest`: 运行回测。
-   `GET /api/klines`: 获取K线数据。

## 扩展策略

要添加新的交易策略，请在 [`strategies.py`](strategies.py) 文件中：

1.  创建一个继承自 `Strategy` 基类的新类。
2.  实现 `calculate_indicators` 和 `generate_signals` 方法。
3.  在 `get_available_strategies` 函数中注册新策略。

## 注意事项

-   本项目仅供学习和研究使用，不构成投资建议。
-   使用真实API密钥时请注意资金安全。
-   回测结果不代表未来表现。