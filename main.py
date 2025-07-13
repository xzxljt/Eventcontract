# --- START OF FILE main.py ---

# 导入必要的库
import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv(override=True) # 强制覆盖已存在的同名环境变量
from pandas import Timestamp # 确保明确导入 Timestamp 类型
from fastapi import FastAPI, HTTPException, Query, Depends, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Any, Optional
from datetime import date, datetime, timedelta, timezone # 明确导入 date 类
import uvicorn
import numpy as np
import json
import time
import asyncio # 确保导入 asyncio
shutdown_event_async = asyncio.Event() # 全局关闭事件
import threading
from queue import Queue, Empty
import random
import traceback
import uuid # 唯一ID生成

import logging

# 导入优化引擎
from optimization_engine import get_optimization_engine

# 配置日志记录
# 可以配置输出到文件和控制台
import sys

# 配置日志记录
# 可以配置输出到文件和控制台
import sys
import os # Already imported at line 4, but good to be explicit here
import logging.handlers # Already implicitly used, but good to be explicit

# 从环境变量读取日志配置
LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_FILENAME = os.getenv("LOG_FILENAME", "service.log")
LOG_ROTATION_WHEN = os.getenv("LOG_ROTATION_WHEN", "midnight")
LOG_ROTATION_INTERVAL = int(os.getenv("LOG_ROTATION_INTERVAL", "1"))
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "7"))

# 确保日志目录存在
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# --- AutoX 交易日志配置 ---
AUTOX_TRADE_LOG_DIR = os.path.join(LOG_DIR, "trade_logs")

def get_autox_trade_log_file_path(date: Optional[datetime.date] = None) -> str:
    """获取指定日期或今天的 AutoX 交易日志文件路径。"""
    if date is None: date = datetime.date.today()
    return os.path.join(AUTOX_TRADE_LOG_DIR, f"trade_logs_{date.strftime('%Y-%m-%d')}.json")

# 构建完整的日志文件路径
log_file_path = os.path.join(LOG_DIR, LOG_FILENAME)

logging.basicConfig(
    level=logging.INFO, # 设置最低日志级别，例如 logging.DEBUG 可以看到更详细的日志
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        # 使用 TimedRotatingFileHandler 实现按配置轮转和保留历史日志
        logging.handlers.TimedRotatingFileHandler(
            log_file_path,
            when=LOG_ROTATION_WHEN, # 轮转周期
            interval=LOG_ROTATION_INTERVAL, # 轮转间隔
            backupCount=LOG_BACKUP_COUNT,   # 保留历史日志文件数量
            encoding='utf-8' # 指定编码
        ),
        logging.StreamHandler(sys.stdout) # 保留：输出到控制台
    ]
)

# 获取一个 Logger 实例，通常使用 __name__
logger = logging.getLogger(__name__)
# 导入项目组件
from strategies import Strategy, get_available_strategies
from investment_strategies import BaseInvestmentStrategy, get_available_investment_strategies
from backtester import Backtester
from timezone_utils import to_china_timezone, to_utc, now_china, now_utc, format_for_display, parse_frontend_datetime, CHINA_TIMEZONE
from binance_client import BinanceClient

# --- 全局变量和配置 ---
# 存储策略参数的配置字典
strategy_parameters_config: Dict[str, Any] = {
    "prediction_strategies": {},
    "investment_strategies": {}
}

# 回测任务管理
active_backtest_tasks: Dict[str, Dict[str, Any]] = {}  # task_id -> task_info
backtest_cancellation_flags: Dict[str, bool] = {}  # task_id -> should_cancel
STRATEGY_PARAMS_FILE = "config/strategy_parameters.json" # 策略参数配置文件路径
AUTOX_CLIENTS_FILE = "config/autox_clients_data.json" # AutoX 客户端数据文件路径

# 用于存储从文件加载的持久化AutoX客户端数据
# 这个字典将以 client_id 为键，存储客户端的注册信息和最新状态
persistent_autox_clients_data: Dict[str, Dict[str, Any]] = {}



# 初始化 FastAPI 应用
app = FastAPI(
    title="币安事件合约交易信号机器人", # 应用标题
    description="基于技术指标的币安事件合约交易信号生成和回测系统", # 应用描述
    version="1.4.3" # 应用版本
)

# 添加信号处理，确保服务能够正确响应系统信号
import signal
import platform

def handle_sigterm(signum, frame):
    """处理SIGTERM信号，确保服务能够优雅关闭。"""
    logger.warning(f"收到系统信号 {signum}，准备优雅关闭服务...")
    # 这里不需要做任何事情，因为信号会传递给uvicorn，
    # uvicorn会触发FastAPI的shutdown事件，我们在shutdown_event中处理清理工作
    # 记录日志以便于调试
    logger.warning("信号已处理，等待uvicorn触发shutdown事件...")

# 只在非Windows系统上注册信号处理函数
# Windows下的信号处理机制与Unix/Linux不同，可能导致问题
if platform.system() != "Windows":
    try:
        signal.signal(signal.SIGTERM, handle_sigterm)
        signal.signal(signal.SIGINT, handle_sigterm)
        logger.info("已注册系统信号处理函数")
    except Exception as e:
        logger.warning(f"注册信号处理函数失败: {e}，这在某些环境下是正常的")

# 添加FastAPI应用关闭事件处理函数

# --- WebSocket 连接管理器 (保持不变) ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    async def connect(self, websocket: WebSocket):
        await websocket.accept(); self.active_connections.append(websocket)
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections: self.active_connections.remove(websocket)
    async def _send_to_connection(self, connection: WebSocket, data: dict, client_id: str, timeout_sec: float):
        """辅助函数：向单个WebSocket连接发送数据并处理异常。"""
        send_start_time = time.time()
        try:
            await asyncio.wait_for(connection.send_json(data), timeout=timeout_sec)
            send_elapsed = time.time() - send_start_time
            logger.debug(f"WebSocket广播成功到客户端 {client_id}，耗时: {send_elapsed:.4f}秒")
            return True
        except asyncio.TimeoutError:
            logger.warning(f"WebSocket广播到客户端 {client_id} 超时 (>{timeout_sec}秒)")
            self.disconnect(connection) # 超时后断开连接
            return False
        except WebSocketDisconnect:
            logger.debug(f"WebSocket广播时发现客户端 {client_id} 已断开连接")
            self.disconnect(connection)
            return False
        except Exception as e:
            logger.error(f"WebSocket广播到客户端 {client_id} 失败: {e}")
            # 根据具体错误类型决定是否断开连接，这里暂时不断开，除非是特定IO错误
            # self.disconnect(connection)
            return False

    async def broadcast_json(self, data: dict, filter_func=None, timeout_sec: float = 5.0):
        """异步广播JSON数据到所有活跃的WebSocket连接。
        为每个客户端的发送操作创建一个独立的asyncio.create_task()。
        
        Args:
            data: 要广播的JSON数据
            filter_func: 可选的过滤函数，决定哪些连接接收广播
            timeout_sec: 每个WebSocket发送操作的超时时间（秒）
        """
        broadcast_overall_start_time = time.time()
        # logger.info(f"开始异步WebSocket广播 - 时间: {datetime.now().isoformat()}")
        
        active_connections_copy = list(self.active_connections) # 迭代副本以允许在广播时断开连接
        
        tasks = []
        connections_to_send_to = []

        for idx, connection in enumerate(active_connections_copy):
            if filter_func is None or filter_func(connection):
                client_id = getattr(connection, 'client', f'未知客户端-{idx}')
                connections_to_send_to.append({"conn": connection, "id": client_id})

        if not connections_to_send_to:
            logger.debug("没有符合条件的WebSocket连接需要广播。")
            broadcast_overall_elapsed = time.time() - broadcast_overall_start_time
            # logger.info(f"异步WebSocket广播完成 (无连接) - 总耗时: {broadcast_overall_elapsed:.4f}秒")
            return {"total_targeted": 0, "success": 0, "error": 0}

        logger.debug(f"准备向 {len(connections_to_send_to)} 个客户端创建发送任务...")

        for conn_info in connections_to_send_to:
            task = asyncio.create_task(
                self._send_to_connection(conn_info["conn"], data, conn_info["id"], timeout_sec)
            )
            tasks.append(task)
        
        results = []
        if tasks:
            logger.debug(f"等待 {len(tasks)} 个WebSocket发送任务完成...")
            # 使用 asyncio.gather 等待所有任务完成，return_exceptions=True 使得gather不会因单个任务失败而停止
            results = await asyncio.gather(*tasks, return_exceptions=True)
            logger.debug(f"所有 {len(tasks)} 个WebSocket发送任务已处理。")
        
        success_count = 0
        error_count = 0
        
        for i, result in enumerate(results):
            client_id_for_log = connections_to_send_to[i]["id"] # 获取对应任务的客户端ID
            if isinstance(result, Exception): # asyncio.gather 中 return_exceptions=True 时，异常会作为结果返回
                error_count += 1
                logger.error(f"WebSocket发送任务到客户端 {client_id_for_log} 失败 (异常由gather捕获): {result}")
            elif result is True:
                success_count += 1
            else: # result is False or some other unexpected non-Exception value
                error_count += 1
                # _send_to_connection 内部已经记录了具体错误，这里只记录聚合结果
                logger.warning(f"WebSocket发送任务到客户端 {client_id_for_log} 返回失败状态。")

        total_targeted_connections = len(connections_to_send_to)
        
        broadcast_overall_elapsed = time.time() - broadcast_overall_start_time
        logger.info(
            f"异步WebSocket广播完成 - 总耗时: {broadcast_overall_elapsed:.4f}秒, "
            f"目标客户端数: {total_targeted_connections}, 成功: {success_count}, 失败: {error_count}"
        )
        
        if broadcast_overall_elapsed > 1.0:
            logger.warning(
                # f"异步WebSocket广播耗时较长: {broadcast_overall_elapsed:.4f}秒, "
                f"成功/目标: {success_count}/{total_targeted_connections}"
            )
        
        return {"total_targeted": total_targeted_connections, "success": success_count, "error": error_count}

# --- Pydantic 模型定义 (保持不变) ---
class InvestmentStrategySettings(BaseModel):
    model_config = {'arbitrary_types_allowed': True} # 允许任意类型
    amount: float = Field(20.0, description="基础投资金额或固定金额")
    strategy_id: str = Field("fixed", description="投资策略ID")
    minAmount: float = Field(5.0, description="最小投资金额")
    maxAmount: float = Field(250.0, description="最大投资金额")
    percentageOfBalance: Optional[float] = Field(None, description="账户百分比策略的百分比值 (1-100)")
    profitRate: float = Field(80.0, description="事件合约获胜收益率 (%)")
    lossRate: float = Field(100.0, description="事件合约失败损失率 (%)")
    simulatedBalance: Optional[float] = Field(None, description="模拟账户总资金 (用于百分比投资策略计算)")
    min_trade_interval_minutes: Optional[float] = Field(0, description="最小开单间隔（分钟），0表示不限制")
    investment_strategy_specific_params: Optional[Dict[str, Any]] = Field(None, description="选定投资策略的特定参数")


class BacktestInvestmentSettings(BaseModel):
    model_config = {'arbitrary_types_allowed': True} # 允许任意类型
    initial_balance: float = Field(1000.0, description="回测初始模拟资金")
    investment_strategy_id: str = Field("fixed", description="投资策略ID")
    investment_strategy_specific_params: Optional[Dict[str, Any]] = Field(None, description="选定投资策略的特定参数")
    min_investment_amount: float = Field(5.0, description="单次最小投资额")
    max_investment_amount: float = Field(250.0, description="单次最大投资额")
    profit_rate_pct: float = Field(80.0, description="事件合约盈利百分比 (%)")
    loss_rate_pct: float = Field(100.0, description="事件合约亏损百分比 (%)")
    min_trade_interval_minutes: Optional[float] = Field(0, description="最小开单间隔（分钟），0表示不限制")

class BacktestRequest(BaseModel):
    model_config = {'arbitrary_types_allowed': True} # 允许任意类型
    symbol: str = Field(..., description="交易对")
    interval: str = Field(..., description="K线周期")
    start_time: datetime = Field(..., description="回测开始时间 (中国时区)")
    end_time: datetime = Field(..., description="回测结束时间 (中国时区)")
    prediction_strategy_id: str = Field(..., description="预测策略ID")
    prediction_strategy_params: Optional[Dict[str, Any]] = Field(None, description="预测策略参数")
    event_period: str = Field(..., description="事件合约周期")
    confidence_threshold: float = Field(0, description="预测置信度阈值, 0-100")
    task_id: Optional[str] = Field(None, description="回测任务唯一标识符")
    investment: BacktestInvestmentSettings = Field(..., description="投资设置")

class SymbolInfo(BaseModel):
    model_config = {'arbitrary_types_allowed': True} # 允许任意类型
    symbol: str; base_asset: str; quote_asset: str
class StrategyParameterSet(BaseModel):
    model_config = {'arbitrary_types_allowed': True} # 允许任意类型
    strategy_type: str = Field(..., description="策略类型: 'prediction' or 'investment'")
    strategy_id: str = Field(..., description="策略ID")
    params: Dict[str, Any] = Field(..., description="策略参数")

class AutoXClientInfo(BaseModel):
    model_config = {'arbitrary_types_allowed': True} # 允许任意类型
    client_id: str
    status: str = "idle"
    supported_symbols: List[str] = ["BTCUSDT"]
    last_seen: Optional[datetime] = None
    connected_at: datetime = Field(default_factory=now_utc)
    notes: Optional[str] = Field(None, description="管理员为客户端添加的备注")

class AutoXTradeLogEntry(BaseModel):
    model_config = {'arbitrary_types_allowed': True} # 允许任意类型
    log_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    client_id: str
    signal_id: Optional[str] = None
    command_type: str
    command_payload: Dict[str, Any]
    status: str
    details: Optional[str] = None
    error_message: Optional[str] = None
    timestamp: datetime = Field(default_factory=now_utc)

class TriggerAutoXTradePayload(BaseModel):
    model_config = {'arbitrary_types_allowed': True} # 允许任意类型
    symbol: str = "ETHUSDT"
    direction: str = Query(..., pattern="^(up|down)$")
    amount: str = "5"
    signal_id: Optional[str] = None

class ClientNotesPayload(BaseModel):
    model_config = {'arbitrary_types_allowed': True} # 允许任意类型
    notes: Optional[str] = Field(None, max_length=255)

class DeleteSignalsRequest(BaseModel):
    model_config = {'arbitrary_types_allowed': True} # 允许任意类型
    signal_ids: List[str]

class CancelBacktestRequest(BaseModel):
    model_config = {'arbitrary_types_allowed': True} # 允许任意类型
    task_id: str = Field(..., description="要取消的回测任务ID")

# --- CORS, StaticFiles, BinanceClient ---
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
app.mount("/templates", StaticFiles(directory="templates"), name="templates") # Add this line to serve templates directory
# app.mount("/autoxjs", StaticFiles(directory="autoxjs"), name="autoxjs") # Add this line to serve autoxjs directory
binance_client = BinanceClient()

# --- WebSocket 连接管理器 ---
manager = ConnectionManager()
autox_manager = ConnectionManager()
autox_status_manager = ConnectionManager()

# --- WebSocket Connection Manager for Optimization ---
class OptimizationConnectionManager:
    def __init__(self):
        # A dictionary to map optimization_id to a list of active WebSocket connections
        self.connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, optimization_id: str):
        """Accepts a new connection and associates it with an optimization ID."""
        await websocket.accept()
        if optimization_id not in self.connections:
            self.connections[optimization_id] = []
        self.connections[optimization_id].append(websocket)
        logger.info(f"WebSocket connected for optimization_id: {optimization_id}")

    def disconnect(self, websocket: WebSocket, optimization_id: str):
        """Handles a disconnected connection."""
        if optimization_id in self.connections:
            if websocket in self.connections[optimization_id]:
                self.connections[optimization_id].remove(websocket)
                logger.info(f"WebSocket disconnected for optimization_id: {optimization_id}")
                # If no more connections for this optimization_id, clean up the entry
                if not self.connections[optimization_id]:
                    del self.connections[optimization_id]

    async def send_update(self, optimization_id: str, data: dict):
        """Sends progress data to all clients associated with a specific optimization ID."""
        if optimization_id in self.connections:
            # Create a list of tasks to send updates concurrently
            connections_to_send = self.connections[optimization_id][:] # Iterate over a copy
            tasks = [connection.send_json(data) for connection in connections_to_send]
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                # Log errors from gather
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Error sending update to a client for optimization_id {optimization_id}: {result}")
                        # The connection might be stale, disconnect it
                        self.disconnect(connections_to_send[i], optimization_id)

                successful_sends = sum(1 for r in results if not isinstance(r, Exception))
                if successful_sends > 0:
                    logger.info(f"Sent update to {successful_sends} clients for optimization_id: {optimization_id}")


# Global instance for optimization WebSocket connections
optimization_manager = OptimizationConnectionManager()


# --- 实时信号与队列 ---
live_signals: List[Dict[str, Any]] = [] # 类型提示明确
LIVE_SIGNALS_FILE = "live_signals.json"
live_signals_lock = threading.Lock() # 用于保护 live_signals 的直接访问
signals_queue = Queue()

# --- 后台持续运行的实时测试核心状态 (保持不变) ---
active_kline_streams: Dict[str, int] = {}
active_kline_streams_lock = threading.Lock()
running_live_test_configs: Dict[str, Dict[str, Any]] = {}
running_live_test_configs_lock = threading.Lock()
websocket_to_config_id_map: Dict[WebSocket, str] = {}

# 新增: 全局唯一活动配置ID
active_live_test_config_id: Optional[str] = None
active_live_test_config_lock = threading.Lock()
ACTIVE_TEST_CONFIG_FILE = "config/active_test_config.json" # 活动测试配置文件路径

# 新增: 全局运行账户余额
global_running_balance: float = 1000.0
global_running_balance_lock = threading.Lock()

# --- AutoX.js 控制相关全局变量 (保持不变) ---
active_autox_clients: Dict[WebSocket, Dict[str, Any]] = {} 
autox_clients_lock = threading.Lock() # 用于保护 active_autox_clients 和 persistent_autox_clients_data
autox_trade_logs: List[Dict[str, Any]] = []
autox_trade_logs_lock = threading.Lock() # 用于保护 autox_trade_logs
MAX_AUTOX_LOG_ENTRIES = 200
current_autox_trade_log_date: date = date.today() # 当前日志对应的日期

# --- 辅助函数：确保数据是JSON可序列化的 (保持不变) ---
def ensure_json_serializable(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: ensure_json_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [ensure_json_serializable(i) for i in data]
    elif isinstance(data, np.integer): return int(data)
    elif isinstance(data, np.floating): return float(data)
    elif isinstance(data, np.ndarray): return data.tolist()
    elif isinstance(data, (datetime, pd.Timestamp)): return data.isoformat()
    return data

# --- 新增：文件写入锁管理器，解决并发写入冲突 ---
file_write_locks: Dict[str, threading.Lock] = {}
file_write_locks_lock = threading.Lock() # 用于保护 file_write_locks 字典本身

def get_lock_for_file(file_path: str) -> threading.Lock:
    """为指定的文件路径获取或创建一把锁，确保文件操作的线程安全。"""
    with file_write_locks_lock:
        if file_path not in file_write_locks:
            file_write_locks[file_path] = threading.Lock()
        return file_write_locks[file_path]


# --- 用于执行阻塞文件操作的辅助函数 ---
def _blocking_save_json_to_file(file_path: str, data_to_save: Any):
    """
    一个同步的辅助函数，用于将数据保存到JSON文件。
    它会创建临时文件并进行原子替换，以确保数据完整性。
    这个函数应该通过 asyncio.to_thread 来调用。
    """
    file_lock = get_lock_for_file(file_path) # 获取文件对应的锁
    with file_lock: # 使用锁来保护整个文件读写和替换过程
        temp_file_path = file_path + ".tmp"
        try:
            # 确保目录存在
            file_dir = os.path.dirname(file_path)
            if file_dir and not os.path.exists(file_dir):
                os.makedirs(file_dir)

            serializable_data = ensure_json_serializable(data_to_save) # 确保数据可序列化
            with open(temp_file_path, "w", encoding="utf-8") as f:
                json.dump(serializable_data, f, indent=4)
                f.flush()  # 确保所有内部缓冲区的数据都写入文件
                os.fsync(f.fileno()) # 强制将文件写入磁盘
            
            os.replace(temp_file_path, file_path) # 原子性地替换旧文件
            # print(f"数据已同步保存到: {file_path}") # 日志可以在调用方打印
        except Exception as e:
            print(f"同步保存文件 {file_path} 失败: {e}\n{traceback.format_exc()}")
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception as rm_err:
                    print(f"清理临时文件 {temp_file_path} 失败: {rm_err}")
            raise # 重新抛出异常，让调用者知道保存失败

def _blocking_load_json_from_file(file_path: str, default_value: Any = None) -> Any:
    """
    一个同步的辅助函数，用于从JSON文件加载数据。
    这个函数应该通过 asyncio.to_thread 来调用。
    """
    try:
        # 确保目录存在（主要用于后续创建文件时，加载时若目录不存在则文件也不存在）
        params_dir = os.path.dirname(file_path)
        if params_dir and not os.path.exists(params_dir):
            # 如果文件路径的目录不存在，那文件肯定也不存在
            if default_value is not None: return default_value
            return {} # 或者根据期望返回空列表等

        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                if content.strip(): # 确保文件内容不为空
                    return json.loads(content)
        
        # 文件不存在或为空，返回默认值
        if default_value is not None: return default_value
        return {} # 或者根据期望返回空列表，比如 [] for live_signals
    except json.JSONDecodeError:
        print(f"错误: 文件 {file_path} 包含无效JSON。将返回默认值。")
        if default_value is not None: return default_value
        return {}
    except Exception as e:
        print(f"从文件 {file_path} 加载数据时发生错误: {e}\n{traceback.format_exc()}。将返回默认值。")
        if default_value is not None: return default_value
        return {}

# --- 修改后的持久化函数 ---
async def load_autox_trade_logs():
    """在应用启动时加载当天的 AutoX 交易日志。"""
    global autox_trade_logs, current_autox_trade_log_date

    current_date = date.today() # 使用导入的 date 类
    log_file_path = get_autox_trade_log_file_path(current_date)
    current_autox_trade_log_date = current_date # 记录当前加载的日志日期

    print(f"尝试从文件 {log_file_path} 加载当天的 AutoX 交易日志...")

    loaded_data = await asyncio.to_thread(
        _blocking_load_json_from_file,
        log_file_path,
        default_value=[] # 交易日志是列表
    )

    # 验证加载的数据结构
    valid_logs = []
    if isinstance(loaded_data, list):
        for log_entry_dict in loaded_data:
            try:
                # 使用 Pydantic 模型验证并转换为字典
                log_model = AutoXTradeLogEntry(**log_entry_dict)
                valid_logs.append(log_model.model_dump(mode='json'))
            except Exception as e_val:
                print(f"加载AutoX交易日志时验证失败: {e_val}. 跳过此日志条目。")

    with autox_trade_logs_lock: # 保护对全局变量的写入
        autox_trade_logs = valid_logs

async def load_active_test_config():
    """从文件异步加载活动测试配置。"""
    global active_live_test_config_id, running_live_test_configs

    print(f"尝试从文件 {ACTIVE_TEST_CONFIG_FILE} 加载活动测试配置...") # 新增日志

    loaded_data = await asyncio.to_thread(
        _blocking_load_json_from_file,
        ACTIVE_TEST_CONFIG_FILE,
        default_value={"active_config_id": None, "config_data": None, "last_updated": None}
    )

    if not isinstance(loaded_data, dict):
        print(f"警告: {ACTIVE_TEST_CONFIG_FILE} 包含无效数据格式，将使用默认空配置。")
        return

    active_config_id = loaded_data.get("active_config_id")
    config_data = loaded_data.get("config_data")

    if active_config_id and config_data:
        with active_live_test_config_lock:
            active_live_test_config_id = active_config_id

        with running_live_test_configs_lock:
            running_live_test_configs[active_config_id] = config_data
            # 从文件加载配置后，重新创建策略实例
            # 注意：这里需要确保 _create_and_store_investment_strategy_instance 函数已经被定义
            # 我们将把它的定义放在 handle_kline_data 之前
            # 在创建实例之前，将配置ID添加到数据中
            config_data_with_id = config_data.copy()
            config_data_with_id['_config_id'] = active_config_id
            inv_instance = _create_and_store_investment_strategy_instance(config_data_with_id)
            if inv_instance:
                running_live_test_configs[active_config_id]['investment_strategy_instance'] = inv_instance

        print(f"成功从 {ACTIVE_TEST_CONFIG_FILE} 加载活动测试配置 (ID: {active_config_id})。") # 修改日志
        # 可以选择在这里打印加载的配置数据，但可能比较冗长
        # print(f"加载的配置数据: {config_data}")

    else:
        print(f"{ACTIVE_TEST_CONFIG_FILE} 未找到活动测试配置或配置不完整。") # 修改日志

async def save_autox_trade_logs_async():
    """异步保存 AutoX 交易日志到当天的文件。"""
    global autox_trade_logs, current_autox_trade_log_date

    log_file_path = get_autox_trade_log_file_path(current_autox_trade_log_date)
    logs_copy = []
    with autox_trade_logs_lock: # 获取锁以安全地复制数据
        logs_copy = [log.copy() for log in autox_trade_logs] # 创建副本以传递给线程

    try:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        await asyncio.to_thread(_blocking_save_json_to_file, log_file_path, logs_copy)
        # print(f"AutoX 交易日志已异步保存到文件 {log_file_path}。") # 保存函数内部会打印
    except Exception as e:
        print(f"异步保存 AutoX 交易日志到 {log_file_path} 失败 (从主调函数看): {e}\n{traceback.format_exc()}")


async def save_active_test_config():
    """异步保存当前活动测试配置到文件。"""
    global active_live_test_config_id, running_live_test_configs

    config_to_save = {
        "active_config_id": None,
        "config_data": None,
        "last_updated": format_for_display(now_utc())
    }

    with active_live_test_config_lock:
        config_id = active_live_test_config_id

    if config_id:
        with running_live_test_configs_lock:
            if config_id in running_live_test_configs:
                config_data = running_live_test_configs[config_id].copy()
                # 在保存到文件前，移除不可被JSON序列化的策略实例
                config_data.pop('investment_strategy_instance', None)
                config_to_save["active_config_id"] = config_id
                config_to_save["config_data"] = config_data
    
    try:
        await asyncio.to_thread(_blocking_save_json_to_file, ACTIVE_TEST_CONFIG_FILE, config_to_save)
        print(f"活动测试配置已保存到 {ACTIVE_TEST_CONFIG_FILE}")
    except Exception as e:
        print(f"保存活动测试配置失败: {e}\n{traceback.format_exc()}")

async def load_autox_clients_from_file():
    """从文件异步加载 AutoX 客户端数据到 persistent_autox_clients_data。"""
    global persistent_autox_clients_data
    
    loaded_data = await asyncio.to_thread(
        _blocking_load_json_from_file, 
        AUTOX_CLIENTS_FILE, 
        default_value={} # 默认返回空字典
    )

    # 验证加载的数据结构
    valid_clients = {}
    if isinstance(loaded_data, dict):
        for client_id, client_info_dict in loaded_data.items():
            try:
                # 使用 Pydantic 模型验证，并确保所有字段都是预期的
                client_model = AutoXClientInfo(**client_info_dict)
                # 确保 last_seen 和 connected_at 是 datetime 对象（如果它们是字符串）
                # Pydantic 模型应该已经处理了ISO格式字符串到datetime的转换
                valid_clients[client_id] = client_model.model_dump(mode='json') # 保存为字典
            except Exception as e_val:
                print(f"加载AutoX客户端数据时验证失败 (ID: {client_id}): {e_val}. 跳过此客户端。")

    with autox_clients_lock: # 保护对全局变量的写入
        persistent_autox_clients_data = valid_clients
    
    if persistent_autox_clients_data:
        print(f"AutoX客户端数据已从 {AUTOX_CLIENTS_FILE} 异步加载。加载了 {len(persistent_autox_clients_data)} 个有效客户端。")
    else:
        print(f"{AUTOX_CLIENTS_FILE} 未找到或为空/无效，AutoX客户端数据为空。")


async def save_autox_clients_to_file():
    """将当前活动和持久化的 AutoX 客户端数据异步保存到文件。"""
    global active_autox_clients, persistent_autox_clients_data
    
    all_clients_to_save_copy = {}

    with autox_clients_lock: # 保护对 active_autox_clients 和 persistent_autox_clients_data 的读取
        # 从持久化数据开始，确保所有已知客户端（包括离线的）都被考虑
        # 进行深拷贝以避免在保存过程中全局变量被修改
        for client_id, p_info in persistent_autox_clients_data.items():
            all_clients_to_save_copy[client_id] = p_info.copy() if isinstance(p_info, dict) else p_info


        # 更新或添加活动客户端信息，活动信息优先
        for ws, active_info_dict in active_autox_clients.items():
            client_id = active_info_dict.get('client_id')
            if client_id:
                try:
                    # 确保数据通过Pydantic模型进行序列化前的验证和转换
                    # active_info_dict 已经是 model_dump() 后的结果，但再次校验无害
                    client_model = AutoXClientInfo(**active_info_dict)
                    all_clients_to_save_copy[client_id] = client_model.model_dump(mode='json')
                except Exception as e_val:
                    print(f"准备保存AutoX客户端数据时验证失败 (ID: {client_id}): {e_val}. 跳过此活动客户端。")
    
    try:
        await asyncio.to_thread(_blocking_save_json_to_file, AUTOX_CLIENTS_FILE, all_clients_to_save_copy)
        print(f"AutoX客户端数据已异步保存到文件 {AUTOX_CLIENTS_FILE}。")
    except Exception as e:
        # _blocking_save_json_to_file 内部已经打印了详细错误
        print(f"异步保存AutoX客户端数据到 {AUTOX_CLIENTS_FILE} 失败 (从主调函数看)。")


async def load_strategy_parameters_from_file():
    """异步加载策略参数。"""
    global strategy_parameters_config
    
    default_config = {"prediction_strategies": {}, "investment_strategies": {}}
    loaded_params = await asyncio.to_thread(
        _blocking_load_json_from_file, 
        STRATEGY_PARAMS_FILE, 
        default_value=default_config
    )
    
    # 更新全局配置，确保类型正确
    strategy_parameters_config["prediction_strategies"] = ensure_json_serializable(
        loaded_params.get("prediction_strategies", {})
    )
    strategy_parameters_config["investment_strategies"] = ensure_json_serializable(
        loaded_params.get("investment_strategies", {})
    )
    print(f"策略参数已从 {STRATEGY_PARAMS_FILE} 异步加载。")


async def save_strategy_parameters_to_file():
    """异步保存策略参数。"""
    global strategy_parameters_config
    # 创建要保存的数据的副本，以防在保存过程中被修改
    config_copy = {
        "prediction_strategies": ensure_json_serializable(strategy_parameters_config.get("prediction_strategies", {}).copy()),
        "investment_strategies": ensure_json_serializable(strategy_parameters_config.get("investment_strategies", {}).copy())
    }
    try:
        await asyncio.to_thread(_blocking_save_json_to_file, STRATEGY_PARAMS_FILE, config_copy)
        # print(f"策略参数已异步保存到文件 {STRATEGY_PARAMS_FILE}。") # 保存函数内部会打印
    except Exception as e:
        print(f"异步保存策略参数到 {STRATEGY_PARAMS_FILE} 失败 (从主调函数看)。")


async def load_live_signals_async():
    """异步加载历史实时信号。"""
    global live_signals
    
    loaded_data = await asyncio.to_thread(
        _blocking_load_json_from_file, 
        LIVE_SIGNALS_FILE, 
        default_value=[] # live_signals 是列表
    )
    
    with live_signals_lock: # 保护对全局 live_signals 的写入
        live_signals = loaded_data if isinstance(loaded_data, list) else []
        
    if live_signals:
        print(f"已异步加载 {len(live_signals)} 个历史实时信号从 {LIVE_SIGNALS_FILE}")
    else:
        print(f"{LIVE_SIGNALS_FILE} 未找到或为空/无效，实时信号列表为空。")


async def save_live_signals_async():
    """异步保存实时信号。"""
    global live_signals
    signals_copy = []
    with live_signals_lock: # 获取锁以安全地复制数据
        signals_copy = [s.copy() for s in live_signals] # 创建副本以传递给线程

    try:
        await asyncio.to_thread(_blocking_save_json_to_file, LIVE_SIGNALS_FILE, signals_copy)
        # print(f"实时信号已异步保存到文件 {LIVE_SIGNALS_FILE}。") # 保存函数内部会打印
    except Exception as e:
        print(f"异步保存实时信号到 {LIVE_SIGNALS_FILE} 失败 (从主调函数看)。")


# --- 核心业务逻辑函数 ---
async def process_kline_queue(): # 基本不变，但其调用的 handle_kline_data 会改变
    while not shutdown_event_async.is_set():
        try:
            # signals_queue.get_nowait() 是同步的，但通常很快
            # 如果队列为空，它会立即抛出 Empty 异常，然后 asyncio.sleep(0.01) 释放控制权
            kline_data = signals_queue.get_nowait()
            if shutdown_event_async.is_set(): # 在处理前再次检查
                logger.info("process_kline_queue: shutdown_event_async set, exiting.")
                break
            await handle_kline_data(kline_data) # handle_kline_data 现在内部会有异步操作
        except Empty:
            if shutdown_event_async.is_set():
                logger.info("process_kline_queue: shutdown_event_async set during empty queue, exiting.")
                break
            await asyncio.sleep(0.01) # 队列为空时短暂休眠，避免CPU空转
        except Exception as e:
            print(f"处理K线队列时出错: {e}\n{traceback.format_exc()}");
            if shutdown_event_async.is_set():
                logger.info("process_kline_queue: shutdown_event_async set during exception, exiting.")
                break
            await asyncio.sleep(1) # 出错时稍长休眠
    logger.info("process_kline_queue task gracefully shut down.")

async def background_signal_verifier():
    global live_signals, global_running_balance, global_running_balance_lock, \
           running_live_test_configs, running_live_test_configs_lock # 确保全局变量可用
    while True:
        await asyncio.sleep(60) 
        verified_something = False; current_time_utc = now_utc()
        signals_to_verify_copy = []
        
        with live_signals_lock: 
            for signal_idx in range(len(live_signals) -1, -1, -1):
                signal = live_signals[signal_idx]
                if not signal.get('verified'):
                    try:
                        expected_end_time_str = signal.get('expected_end_time')
                        if not expected_end_time_str: 
                            live_signals[signal_idx].update({'verified': True, 'result': False, 'actual_end_price': -2, 'verify_time': format_for_display(current_time_utc)})
                            verified_something = True; continue
                        
                        expected_end_time_utc = parse_frontend_datetime(expected_end_time_str)
                        if not expected_end_time_utc.tzinfo: 
                            expected_end_time_utc = expected_end_time_utc.replace(tzinfo=timezone.utc)
                        else: 
                            expected_end_time_utc = expected_end_time_utc.astimezone(timezone.utc)

                        if current_time_utc >= expected_end_time_utc:
                            signals_to_verify_copy.append(signal.copy()) 
                    except Exception as e:
                        print(f"后台验证：解析信号 {signal.get('id')} 时间出错: {e}")
                        live_signals[signal_idx].update({'verified': True, 'result': False, 'actual_end_price': -1, 'verify_time': format_for_display(current_time_utc)})
                        verified_something = True
        
        for signal_copy_to_verify in signals_to_verify_copy:
            logger.info(f"[background_signal_verifier] Verifying signal ID: {signal_copy_to_verify.get('id')}")
            try:
                end_time_utc = parse_frontend_datetime(signal_copy_to_verify['expected_end_time'])
                if not end_time_utc.tzinfo: end_time_utc = end_time_utc.replace(tzinfo=timezone.utc)
                else: end_time_utc = end_time_utc.astimezone(timezone.utc)

                # Calculate the correct start time for the kline that ends at expected_end_time
                # For a 1m interval, if expected_end_time is 10:05:00 (meaning the 10:04:00-10:04:59 kline has just closed),
                # we need the kline that starts at 10:04:00.
                kline_start_time_utc = end_time_utc - timedelta(minutes=1)

                # --- MODIFICATION: Use Index Price for validation (OPEN price with precise time range and retry) ---
                logger.info(f"Verifying signal {signal_copy_to_verify['id']} using index price.")
                actual_price = None
                max_verify_retries = 3

                # Wait for historical data to be available (2 minutes after verification time)
                current_time = datetime.now(timezone.utc)
                time_since_verify = (current_time - kline_start_time_utc).total_seconds()

                if time_since_verify < 120:  # Less than 2 minutes
                    wait_time = 120 - time_since_verify
                    logger.info(f"验证 {signal_copy_to_verify['id']}: 等待 {wait_time:.1f} 秒让历史数据可用")
                    await asyncio.sleep(wait_time)

                for verify_attempt in range(max_verify_retries):
                    try:
                        logger.info(f"验证 {signal_copy_to_verify['id']}: 尝试 {verify_attempt + 1}/{max_verify_retries}")

                        # Use precise time range to get the exact verification minute kline
                        verify_start_time_ms = int(kline_start_time_utc.timestamp() * 1000)
                        verify_end_time_ms = int((kline_start_time_utc + timedelta(minutes=1)).timestamp() * 1000)

                        logger.info(f"验证 {signal_copy_to_verify['id']}: 获取验证时间 {kline_start_time_utc} 的指数K线")

                        kline_df = await asyncio.to_thread(
                            binance_client.get_index_price_klines,
                            signal_copy_to_verify['symbol'], '1m',
                            start_time=verify_start_time_ms,
                            end_time=verify_end_time_ms,
                            limit=1
                        )

                        if not kline_df.empty:
                            # Use OPEN price instead of CLOSE price for verification
                            actual_price = float(kline_df.iloc[0]['open'])
                            verify_time_str = kline_df.index[0].strftime('%H:%M')
                            logger.info(f"验证 {signal_copy_to_verify['id']}: 使用 {verify_time_str} 验证时间点指数开盘价 {actual_price}")
                            break  # Success, exit retry loop
                        else:
                            logger.warning(f"验证 {signal_copy_to_verify['id']}: 未能获取到 {kline_start_time_utc.isoformat()} 的指数价格K线 (尝试 {verify_attempt + 1})")
                            if verify_attempt < max_verify_retries - 1:
                                await asyncio.sleep(2 ** verify_attempt)  # Exponential backoff
                                continue
                            else:
                                logger.error(f"验证 {signal_copy_to_verify['id']}: 经过 {max_verify_retries} 次尝试后仍无法获取指数价格K线")

                    except Exception as e_idx_verify:
                        logger.error(f"验证 {signal_copy_to_verify['id']} 时获取指数价格失败 (尝试 {verify_attempt + 1}): {e_idx_verify}")
                        if verify_attempt < max_verify_retries - 1:
                            await asyncio.sleep(2 ** verify_attempt)  # Exponential backoff
                            continue
                        else:
                            logger.error(f"验证 {signal_copy_to_verify['id']}: 经过 {max_verify_retries} 次尝试后仍然失败")
                # --- END MODIFICATION ---

                if actual_price is None:
                    print(f"无法获取 {signal_copy_to_verify['symbol']} 的指数价格进行验证。信号ID: {signal_copy_to_verify['id']}")
                    with live_signals_lock:
                        for i, sig_live in enumerate(live_signals):
                            if sig_live.get('id') == signal_copy_to_verify['id'] and not sig_live.get('verified'):
                                update_fields_on_failure = {
                                    'verified': False,
                                    'verification_status': 'failed_retrying',
                                    'actual_end_price': None,
                                    'price_change_pct': None,
                                    'result': None,
                                    'pnl_pct': None,
                                    'actual_profit_loss_amount': None,
                                    'verify_notes': '获取价格失败，等待重试...',
                                    'verify_time': format_for_display(now_utc())
                                }
                                live_signals[i].update(update_fields_on_failure)
                                verified_something = True; break
                    continue

                signal_price_val = signal_copy_to_verify['signal_price']
                if signal_price_val is None: 
                    print(f"信号 {signal_copy_to_verify['id']} 缺少 signal_price。"); 
                    with live_signals_lock:
                        for i, sig_live in enumerate(live_signals):
                            if sig_live.get('id') == signal_copy_to_verify['id'] and not sig_live.get('verified'):
                                live_signals[i].update({'verified': True, 'result': False, 'actual_end_price': actual_price, 'verify_notes': '缺少信号价格', 'verify_time': format_for_display(current_time_utc)})
                                verified_something = True; break
                    continue

                change_pct = ((actual_price - signal_price_val) / signal_price_val) * 100 if signal_price_val != 0 else 0
                correct = (signal_copy_to_verify['signal'] == 1 and actual_price > signal_price_val) or \
                          (signal_copy_to_verify['signal'] == -1 and actual_price < signal_price_val)
                pnl = change_pct if signal_copy_to_verify['signal'] == 1 else -change_pct 

                inv_amt = signal_copy_to_verify.get('investment_amount', 0)
                profit_r = signal_copy_to_verify.get('profit_rate_pct', 80.0) / 100.0
                loss_r = signal_copy_to_verify.get('loss_rate_pct', 100.0) / 100.0
                actual_pnl_amt = inv_amt * profit_r if correct else -(inv_amt * loss_r) if inv_amt > 0 else 0.0

                if actual_pnl_amt is not None:
                    with global_running_balance_lock: # 更新全局余额
                        global_running_balance += actual_pnl_amt

                update_fields = {
                    'actual_end_price': actual_price, 'price_change_pct': round(change_pct, 4),
                    'result': correct, 'pnl_pct': round(pnl, 4),
                    'actual_profit_loss_amount': round(actual_pnl_amt, 2),
                    'verified': True, 'verify_time': format_for_display(current_time_utc),
                    'verification_status': 'success'
                }
                logger.debug(f"[background_signal_verifier] Signal {signal_copy_to_verify.get('id')}: Created update_fields: {json.dumps(ensure_json_serializable(update_fields), indent=2)}")

                original_updated = False
                with live_signals_lock:
                    for i, sig_live in enumerate(live_signals):
                        if sig_live.get('id') == signal_copy_to_verify['id'] and not sig_live.get('verified'): 
                            live_signals[i].update(update_fields)
                            original_updated = True; verified_something = True
                            signal_copy_to_verify.update(update_fields) 
                            print(f"信号 {signal_copy_to_verify['id']} 已验证: 结果={'正确' if correct else '错误'}, 实际结束价={actual_price}, 盈亏额={actual_pnl_amt:.2f}")
                            break
                
                if original_updated:
                    # --- MODIFICATION FOR CONFIG-SPECIFIC BALANCE UPDATE ---
                    config_id_of_signal = signal_copy_to_verify.get('origin_config_id')
                    if config_id_of_signal and actual_pnl_amt is not None:
                        updated_config_data = None
                        with running_live_test_configs_lock: # 保护对 running_live_test_configs 的写操作
                            if config_id_of_signal in running_live_test_configs:
                                config_entry = running_live_test_configs[config_id_of_signal]
                                
                                # 更新当前余额 (每次交易后的余额)
                                current_config_bal = config_entry.get('current_balance')
                                if current_config_bal is None:
                                     sim_bal_from_settings = config_entry.get('investment_settings', {}).get('simulatedBalance')
                                     current_config_bal = sim_bal_from_settings if sim_bal_from_settings is not None else 1000.0 # Fallback
                                     print(f"警告: Config ID {config_id_of_signal} 在验证时缺少 current_balance, 已从 simulatedBalance 或默认值回退。")
                                config_entry['current_balance'] = current_config_bal + (signal_copy_to_verify.get('investment_amount', 0) + actual_pnl_amt)
                                
                                # 累加总盈亏额
                                current_total_pnl = config_entry.get('total_profit_loss_amount', 0.0) # 如果不存在，初始化为0
                                config_entry['total_profit_loss_amount'] = current_total_pnl + actual_pnl_amt
                                
                                updated_config_data = config_entry.copy() # 复制更新后的数据用于广播
                        
                        if updated_config_data:
                            updated_config_balance_payload = {
                                "type": "config_specific_balance_update",
                                "data": {
                                    "config_id": config_id_of_signal,
                                    "new_balance": round(updated_config_data.get('current_balance', 0.0), 2), # 发送更新后的余额
                                    "last_pnl_amount": round(actual_pnl_amt, 2), # 发送单次交易盈亏
                                    "total_profit_loss_amount": round(updated_config_data.get('total_profit_loss_amount', 0.0), 2) # 新增：发送累加总盈亏
                                }
                            }
                            await manager.broadcast_json(
                                updated_config_balance_payload,
                                filter_func=lambda c: websocket_to_config_id_map.get(c) == config_id_of_signal
                            )
                    # --- END MODIFICATION ---
                    
                    # Broadcast the next investment amount after state change
                    await broadcast_next_investment_amount(config_id_of_signal)
                    
                    logger.debug(f"[background_signal_verifier] Broadcasting 'verified_signal' for signal {signal_copy_to_verify.get('id')}: {json.dumps(signal_copy_to_verify, indent=2)}")
                    logger.debug(f"[background_signal_verifier] Broadcasting 'verified_signal' for signal {signal_copy_to_verify.get('id')}: {json.dumps(ensure_json_serializable(signal_copy_to_verify), indent=2)}")
                    await manager.broadcast_json(
                        {"type": "verified_signal", "data": signal_copy_to_verify},
                        filter_func=lambda c: websocket_to_config_id_map.get(c) == signal_copy_to_verify.get('origin_config_id')
                    )
            except Exception as e:
                print(f"后台验证信号 {signal_copy_to_verify.get('id', 'N/A')} 时出错: {e}\n{traceback.format_exc()}")
                with live_signals_lock:
                    for i, sig_live in enumerate(live_signals):
                        if sig_live.get('id') == signal_copy_to_verify.get('id') and not sig_live.get('verified'):
                            live_signals[i].update({'verified': True, 'result': False, 'actual_end_price': -4, 'verify_notes': f'验证异常: {str(e)[:50]}', 'verify_time': format_for_display(current_time_utc)})
                            verified_something = True; break

        if verified_something: await save_live_signals_async()

        signals_for_stats = []
        with live_signals_lock: 
            signals_for_stats = [s.copy() for s in live_signals]
        
        verified_list_global = [s for s in signals_for_stats if s.get('verified')]
        total_verified_global = len(verified_list_global)
        total_correct_global = sum(1 for s in verified_list_global if s.get('result'))
        total_actual_profit_loss_amount_global = sum(s.get('actual_profit_loss_amount', 0.0) for s in verified_list_global if s.get('actual_profit_loss_amount') is not None)
        total_pnl_pct_sum_global = sum(s.get('pnl_pct', 0.0) for s in verified_list_global if s.get('pnl_pct') is not None)
        average_pnl_pct_global = total_pnl_pct_sum_global / total_verified_global if total_verified_global > 0 else 0
        stats_payload = {
            "total_signals": len(signals_for_stats), 
            "total_verified": total_verified_global, "total_correct": total_correct_global,
            "win_rate": round(total_correct_global / total_verified_global * 100 if total_verified_global > 0 else 0, 2),
            "total_pnl_pct": round(total_pnl_pct_sum_global, 2), "average_pnl_pct": round(average_pnl_pct_global, 2), 
            "total_profit_amount": round(total_actual_profit_loss_amount_global, 2),
            "current_balance": round(global_running_balance, 2) # 全局余额
        }
        await manager.broadcast_json({"type": "stats_update", "data": stats_payload})
    logger.info("background_signal_verifier task gracefully shut down.")
    logger.info("background_signal_verifier task gracefully shut down.")


async def _send_autox_command(client_ws: WebSocket, command: Dict[str, Any]): # 基本不变
    try:
        await client_ws.send_json(command)
        
        # 从 active_autox_clients 获取 client_id
        client_id_for_log = "未知"
        with autox_clients_lock: # 保护对 active_autox_clients 的读取
            client_info = active_autox_clients.get(client_ws)
            if client_info:
                client_id_for_log = client_info.get('client_id', '未知')
        
        print(f"已向AutoX客户端 {client_id_for_log} 发送指令: {command.get('type')}")
        
        log_entry_data = {
            "client_id": client_id_for_log,
            "signal_id": command.get("payload", {}).get("signal_id"),
            "command_type": command.get("type"),
            "command_payload": command.get("payload"),
            "status": "command_sent_to_client",
            "details": f"指令已发送给客户端。",
        }
        with autox_trade_logs_lock: # 保护对 autox_trade_logs 的写入
            autox_trade_logs.append(AutoXTradeLogEntry(**log_entry_data).model_dump(mode='json'))
            if len(autox_trade_logs) > MAX_AUTOX_LOG_ENTRIES:
                autox_trade_logs.pop(0)
        await save_autox_trade_logs_async() # 新增：保存交易日志
    except Exception as e:
        print(f"向AutoX客户端发送指令失败: {e}")


def _create_and_store_investment_strategy_instance(config_data: Dict[str, Any]) -> Optional[BaseInvestmentStrategy]:
    """
    根据配置数据创建投资策略实例。
    这个函数是同步的，因为它主要处理字典操作和对象实例化，不涉及IO。
    """
    config_id_for_log = config_data.get('_config_id', 'N/A')
    try:
        inv_settings = config_data.get("investment_settings", {})
        inv_strat_id = inv_settings.get("strategy_id")
        if not inv_strat_id:
            logger.warning(f"配置 {config_id_for_log} 中缺少 investment_settings.strategy_id，无法创建策略实例。")
            return None

        inv_strat_def = next((s for s in get_available_investment_strategies() if s['id'] == inv_strat_id), None)
        if not inv_strat_def:
            logger.error(f"创建投资策略实例失败 (Config: {config_id_for_log}): 未找到策略定义 {inv_strat_id}")
            return None

        # 整合参数 (运行时特定配置 > 全局保存 > 策略定义默认)
        default_params = {p['name']: p['default'] for p in inv_strat_def.get('parameters', []) if not p.get('readonly')}
        global_params = strategy_parameters_config.get("investment_strategies", {}).get(inv_strat_id, {})
        
        # inv_settings 包含了所有运行时配置，包括 minAmount, maxAmount, percentageOfBalance 等
        final_params = {**default_params, **global_params, **inv_settings}
        
        instance = inv_strat_def['class'](params=final_params)
        logger.info(f"为配置 {config_id_for_log} 成功创建了投资策略实例: {instance.name}")
        return instance
    except Exception as e:
        logger.error(f"为配置 {config_id_for_log} 创建投资策略实例时出错: {e}\n{traceback.format_exc()}")
        return None


async def update_signal_entry_price(signal_id: str, symbol: str, signal_time_dt: datetime, interval: str, max_retries: int = 3):
    """
    异步更新信号的入场价格为下一分钟的指数开盘价
    支持重试机制，等待历史数据可用
    """
    # Wait for historical data to be available (2 minutes after signal time)
    entry_time = signal_time_dt.replace(second=0, microsecond=0)
    current_time = datetime.now(timezone.utc)
    time_since_entry = (current_time - entry_time).total_seconds()

    if time_since_entry < 120:  # Less than 2 minutes
        wait_time = 120 - time_since_entry
        # logger.info(f"[update_signal_entry_price] Waiting {wait_time:.1f} seconds for historical data to be available for signal {signal_id}")
        await asyncio.sleep(wait_time)

    for attempt in range(max_retries):
        try:
            # logger.info(f"[update_signal_entry_price] Attempt {attempt + 1}/{max_retries} - Starting price update for signal {signal_id}")
            # logger.info(f"[update_signal_entry_price] Signal time: {signal_time_dt}, Symbol: {symbol}, Interval: {interval}")

            # The signal_time_dt is already the entry time (e.g., 7:52:01 for a signal triggered at 7:51 close)
            # We need to get the index open price for this exact minute, so round down to minute boundary
            entry_time_ms = int(entry_time.timestamp() * 1000)

            # logger.info(f"[update_signal_entry_price] Entry time (rounded to minute): {entry_time}, timestamp: {entry_time_ms}")
            # logger.info(f"[update_signal_entry_price] Original signal time: {signal_time_dt}")

            # Use historical klines to get the exact open price for the entry minute
            # logger.info(f"[update_signal_entry_price] Fetching historical kline for {symbol} at {entry_time}")

            # Calculate the end time (entry_time + 1 minute)
            from datetime import timedelta
            end_time = entry_time + timedelta(minutes=1)
            end_time_ms = int(end_time.timestamp() * 1000)

            index_price_df = await asyncio.to_thread(
                binance_client.get_index_price_klines,
                symbol, interval,
                start_time=entry_time_ms,
                end_time=end_time_ms,
                limit=1
            )

            # logger.info(f"[update_signal_entry_price] Fetched kline data: empty={index_price_df.empty}, shape={index_price_df.shape if not index_price_df.empty else 'N/A'}")

            if not index_price_df.empty:
                # Use the OPEN price of the entry minute as entry price
                new_entry_price = float(index_price_df.iloc[0]['open'])
                # logger.info(f"[update_signal_entry_price] New entry price (entry minute open): {new_entry_price}")

                # Update the signal in live_signals
                signal_found = False
                with live_signals_lock:
                    for i, signal in enumerate(live_signals):
                        if signal.get('id') == signal_id:
                            old_price = signal.get('signal_price')
                            live_signals[i]['signal_price'] = new_entry_price
                            signal_found = True
                            # logger.info(f"[update_signal_entry_price] Updated entry price for signal {signal_id}: {old_price} -> {new_entry_price}")
                            break

                if signal_found:
                    # Broadcast the price update to clients
                    # logger.info(f"[update_signal_entry_price] Broadcasting price update to clients")
                    await manager.broadcast_json({
                        "type": "signal_price_update",
                        "data": {
                            "signal_id": signal_id,
                            "new_entry_price": new_entry_price,
                            "old_entry_price": old_price
                        }
                    })
                    # logger.info(f"[update_signal_entry_price] Price update completed successfully")
                    return  # Success, exit retry loop
                else:
                    # logger.warning(f"[update_signal_entry_price] Signal {signal_id} not found in live_signals")
                    return  # Signal not found, no point retrying
            else:
                # logger.warning(f"[update_signal_entry_price] Could not fetch index price data for signal {signal_id} (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    # logger.error(f"[update_signal_entry_price] Failed to fetch price after {max_retries} attempts. Entry price remains as trigger price.")
                    return

        except Exception as e:
            logger.error(f"[update_signal_entry_price] Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                continue
            else:
                logger.error(f"[update_signal_entry_price] Final attempt failed for signal {signal_id}")
                import traceback
                logger.error(f"[update_signal_entry_price] Traceback: {traceback.format_exc()}")
                return


async def broadcast_next_investment_amount(config_id: str):
    """
    Calculates and broadcasts the next potential investment amount for a given config.
    """
    if not config_id:
        return

    with running_live_test_configs_lock:
        config_data = running_live_test_configs.get(config_id)
        if not config_data:
            return
        
        inv_instance = config_data.get('investment_strategy_instance')
        if not inv_instance:
            logger.warning(f"Config ID {config_id}: No investment strategy instance found for broadcasting next investment amount.")
            return

        # Get the latest trade result for this config
        last_verified_signal = None
        with live_signals_lock:
            config_signals = [s for s in live_signals if s.get('origin_config_id') == config_id and s.get('verified')]
            if config_signals:
                last_verified_signal = max(config_signals, key=lambda s: s.get('verify_time') or s.get('signal_time'))
        
        previous_trade_result = last_verified_signal.get('result') if last_verified_signal else None
        
        current_balance = config_data.get('current_balance', 0.0)
        
        # Get base investment from settings for the calculate_investment call
        base_investment_from_settings = config_data.get('investment_settings', {}).get('amount', 20.0)

        next_amount = inv_instance.calculate_investment(
            current_balance=current_balance,
            previous_trade_result=previous_trade_result,
            base_investment_from_settings=base_investment_from_settings
        )
        
        # Apply min/max bounds from the config
        min_amount = config_data.get('investment_settings', {}).get('minAmount', 5.0)
        max_amount = config_data.get('investment_settings', {}).get('maxAmount', 250.0)
        next_amount = max(min_amount, min(max_amount, next_amount))

    payload = {
        "type": "next_investment_update",
        "data": {
            "config_id": config_id,
            "next_amount": round(next_amount, 2)
        }
    }
    
    await manager.broadcast_json(
        payload,
        filter_func=lambda c: websocket_to_config_id_map.get(c) == config_id
    )
    logger.info(f"Broadcasted next investment amount for config {config_id}: {next_amount:.2f}")


async def handle_kline_data(kline_data: dict):
    global live_signals, strategy_parameters_config, running_live_test_configs, active_autox_clients, binance_client
    try:
        kline_symbol = kline_data.get('symbol')
        kline_interval = kline_data.get('interval')
        is_kline_closed = kline_data.get('is_kline_closed', False)
        if not (kline_symbol and kline_interval and is_kline_closed):
            return

        # Extract WebSocket kline data for use as the latest kline
        ws_kline_data = {
            'open_time': datetime.fromtimestamp(kline_data.get('kline_start_time', 0) / 1000, tz=timezone.utc),
            'open': kline_data.get('open', 0),
            'high': kline_data.get('high', 0),
            'low': kline_data.get('low', 0),
            'close': kline_data.get('close', 0),
            'volume': kline_data.get('volume', 0),
            'close_time': datetime.fromtimestamp(kline_data.get('kline_close_time', 0) / 1000, tz=timezone.utc),
            'quote_asset_volume': kline_data.get('quote_asset_volume', 0),
            'number_of_trades': kline_data.get('number_of_trades', 0),
            'taker_buy_base_asset_volume': kline_data.get('taker_buy_base_asset_volume', 0),
            'taker_buy_quote_asset_volume': kline_data.get('taker_buy_quote_asset_volume', 0)
        }

        logger.debug(f"[WS_KLINE] {kline_symbol}_{kline_interval}: "
                    f"Time={ws_kline_data['open_time'].strftime('%H:%M:%S')}, "
                    f"OHLC={ws_kline_data['open']:.4f}/{ws_kline_data['high']:.4f}/"
                    f"{ws_kline_data['low']:.4f}/{ws_kline_data['close']:.4f}")

        active_test_configs_for_this_kline = []
        with running_live_test_configs_lock: # 保护读取
            for config_id, config_content in running_live_test_configs.items():
                if config_content.get('symbol') == kline_symbol and \
                   config_content.get('interval') == kline_interval:
                    config_content_copy = config_content.copy() # 对每个配置创建副本进行操作
                    config_content_copy['_should_autox_trigger'] = config_content.get('autox_enabled', True)
                    # 确保副本中包含 current_balance，如果源配置中没有（理论上应该有），则设置一个默认值
                    if 'current_balance' not in config_content_copy:
                        sim_bal = config_content_copy.get('investment_settings', {}).get('simulatedBalance', 1000.0)
                        config_content_copy['current_balance'] = sim_bal if sim_bal is not None else 1000.0
                    
                    active_test_configs_for_this_kline.append({**config_content_copy, '_config_id': config_id})


        if not active_test_configs_for_this_kline:
            return

        for live_test_config_data in active_test_configs_for_this_kline: # live_test_config_data 是一个包含 current_balance 的副本
            pred_strat_id = live_test_config_data['prediction_strategy_id']
            pred_params_from_config = live_test_config_data.get('prediction_strategy_params')
            
            final_pred_params = None
            if pred_params_from_config is not None:
                final_pred_params = pred_params_from_config
            else:
                final_pred_params = strategy_parameters_config.get("prediction_strategies", {}).get(pred_strat_id, {})
            
            if not final_pred_params: # 如果还是没有，尝试从策略定义中获取默认参数
                pred_def = next((s for s in get_available_strategies() if s['id'] == pred_strat_id), None)
                if pred_def and 'parameters' in pred_def:
                    final_pred_params = {p['name']: p['default'] for p in pred_def['parameters']}
            
            if final_pred_params is None: final_pred_params = {} # 最终回退

            # --- MODIFICATION: Hybrid approach - API for history + WebSocket for latest ---
            # Get 99 historical klines from API (stable data)
            # Use WebSocket kline as the 100th (latest) kline for consistency

            df_historical = await asyncio.to_thread(
                binance_client.get_historical_klines,
                live_test_config_data['symbol'],
                live_test_config_data['interval'],
                None, None, 99  # Get 99 historical klines
            )

            if df_historical.empty:
                logger.warning(f"Config {live_test_config_data['_config_id']}: No historical klines available")
                continue

            # Create a new row from WebSocket data for the latest kline
            # Ensure data types match the API DataFrame structure
            ws_row_data = {
                'open': float(ws_kline_data['open']),
                'high': float(ws_kline_data['high']),
                'low': float(ws_kline_data['low']),
                'close': float(ws_kline_data['close']),
                'volume': float(ws_kline_data['volume']),
                'close_time': ws_kline_data['close_time'],
                'quote_asset_volume': float(ws_kline_data['quote_asset_volume']),
                'number_of_trades': int(ws_kline_data['number_of_trades']),
                'taker_buy_base_asset_volume': float(ws_kline_data['taker_buy_base_asset_volume']),
                'taker_buy_quote_asset_volume': float(ws_kline_data['taker_buy_quote_asset_volume']),
                'ignore': 0
            }

            # Create a new DataFrame row with WebSocket data, matching API structure
            ws_df = pd.DataFrame([ws_row_data], index=[ws_kline_data['open_time']])
            ws_df.index.name = 'open_time'  # Ensure index name matches

            # Ensure column order matches the historical DataFrame
            if not df_historical.empty:
                ws_df = ws_df.reindex(columns=df_historical.columns, fill_value=0)

            # Combine historical data with WebSocket data
            df_klines = pd.concat([df_historical, ws_df], ignore_index=False)
            df_klines = df_klines.sort_index()  # Ensure proper time ordering

            # Remove any duplicate timestamps (in case API already included the latest kline)
            df_klines = df_klines[~df_klines.index.duplicated(keep='last')]

            logger.info(f"[HYBRID_DATA] Config {live_test_config_data['_config_id']}: "
                       f"API klines: {len(df_historical)}, WebSocket kline: 1, Final total: {len(df_klines)}. "
                       f"Latest: {ws_kline_data['open_time'].strftime('%H:%M:%S')} Close={ws_kline_data['close']:.4f}")

            # Debug: Check if WebSocket kline was actually added
            if len(df_klines) == len(df_historical):
                logger.warning(f"[HYBRID_DATA] WebSocket kline may have been dropped due to duplicate timestamp!")

                if not df_historical.empty:
                    hist_last_time = df_historical.index[-1]
                    ws_time = ws_kline_data['open_time']
                    logger.debug(f"[HYBRID_DATA] Historical last time: {hist_last_time}")
                    logger.debug(f"[HYBRID_DATA] WebSocket time: {ws_time}")

                    # Check if timestamps are exactly the same
                    if hist_last_time == ws_time:
                        # Replace the last historical kline with WebSocket data
                        df_klines.iloc[-1] = ws_df.iloc[0]
                        logger.info(f"[HYBRID_DATA] Replaced duplicate timestamp kline with WebSocket data. Total: {len(df_klines)}")
                    else:
                        # Different timestamps, force add
                        df_klines = pd.concat([df_klines, ws_df], ignore_index=False)
                        df_klines = df_klines.sort_index()
                        logger.info(f"[HYBRID_DATA] Force added WebSocket kline with different timestamp. New total: {len(df_klines)}")
                else:
                    # No historical data, just use WebSocket data
                    df_klines = ws_df.copy()
                    logger.info(f"[HYBRID_DATA] No historical data, using only WebSocket kline. Total: {len(df_klines)}")
            # --- END MODIFICATION ---
            
            if df_klines.empty:
                print(f"Config ID {live_test_config_data['_config_id']}: 获取历史K线数据为空，跳过。")
                continue

            pred_strat_info = next((s for s in get_available_strategies() if s['id'] == pred_strat_id), None)
            if not pred_strat_info:
                print(f"Config ID {live_test_config_data['_config_id']}: 未找到预测策略 {pred_strat_id}，跳过。")
                continue
            
            # Debug: Check input data before strategy calculation
            logger.debug(f"[STRATEGY_INPUT] Config {live_test_config_data['_config_id']}: "
                        f"Input klines count: {len(df_klines)}, "
                        f"Columns: {list(df_klines.columns)}, "
                        f"Last close: {df_klines['close'].iloc[-1]:.4f}")

            signal_df = pred_strat_info['class'](params=final_pred_params).generate_signals(df_klines.copy()) # 策略计算（如 generate_signals）通常是CPU密集型操作。如果耗时较长，建议后续使用 await asyncio.to_thread() 将其移出事件循环执行，以避免阻塞。

            # Debug: Check output data after strategy calculation
            logger.debug(f"[STRATEGY_OUTPUT] Config {live_test_config_data['_config_id']}: "
                        f"Output signal_df count: {len(signal_df) if not signal_df.empty else 0}, "
                        f"Columns: {list(signal_df.columns) if not signal_df.empty else 'EMPTY'}")

# --- START LOGGING LATEST K-LINE DATA ---
            try:
                if not signal_df.empty and 'close' in signal_df.columns:
                    # Find RSI column (could be 'rsi', 'RSI_14', etc.)
                    rsi_col = None
                    for col in signal_df.columns:
                        if col.lower() == 'rsi' or col.startswith('RSI_'):
                            rsi_col = col
                            break

                    if rsi_col:
                        latest_data = signal_df.tail(3)
                        log_message = f"[{live_test_config_data.get('_config_id', 'N/A')}] 最新3条K线数据:"
                        for index, row in latest_data.iterrows():
                            timestamp_str = index.strftime('%Y-%m-%d %H:%M:%S') if isinstance(index, pd.Timestamp) else str(index)
                            log_message += f"\n  - 时间: {timestamp_str}, 价格: {row['close']:.4f}, RSI: {row[rsi_col]:.2f}"
                        logger.info(log_message)
                    else:
                        logger.warning(f"[{live_test_config_data.get('_config_id', 'N/A')}] No RSI column found. "
                                     f"Available columns: {list(signal_df.columns)}")
                else:
                    logger.warning(f"[{live_test_config_data.get('_config_id', 'N/A')}] K-line data logging skipped: "
                                 f"signal_df empty: {signal_df.empty}, "
                                 f"has 'close': {'close' in signal_df.columns if not signal_df.empty else False}")
            except Exception as e:
                logger.error(f"[{live_test_config_data.get('_config_id', 'N/A')}] Error during K-line data logging: {e}", exc_info=True)
            # --- END LOGGING LATEST K-LINE DATA ---
            
            if signal_df.empty or 'signal' not in signal_df.columns or 'confidence' not in signal_df.columns:
                continue
            
            latest_sig_data = signal_df.iloc[-1]
            sig_val = int(latest_sig_data.get('signal', 0))
            conf_val = float(latest_sig_data.get('confidence', 0))
            current_confidence_threshold = live_test_config_data.get('confidence_threshold', 0)

            # --- COMPREHENSIVE SIGNAL DEBUGGING ---
            if sig_val != 0 and conf_val >= current_confidence_threshold and not signal_df.empty:
                # Signal triggered - log detailed information
                last_row = signal_df.iloc[-1]
                last_time = signal_df.index[-1].strftime('%H:%M:%S') if hasattr(signal_df.index[-1], 'strftime') else str(signal_df.index[-1])

                debug_info = f"[SIGNAL_TRIGGERED] {live_test_config_data['symbol']}_{live_test_config_data['interval']} at {last_time}:\n"
                debug_info += f"  Signal={sig_val}, Confidence={conf_val:.2f} (threshold: {current_confidence_threshold})\n"
                debug_info += f"  Strategy: {pred_strat_id}, Params: {final_pred_params}\n"
                debug_info += f"  Last kline: Close={last_row.get('close', 'N/A'):.4f}"

                if 'rsi' in last_row:
                    debug_info += f", RSI={last_row.get('rsi', 'N/A'):.2f}"
                if 'volume' in last_row:
                    debug_info += f", Volume={last_row.get('volume', 'N/A'):.0f}"

                # Add data source information for debugging
                debug_info += f"\n  Data info: Total klines={len(signal_df)}, "
                debug_info += f"First={signal_df.index[0].strftime('%H:%M:%S')}, "
                debug_info += f"Last={signal_df.index[-1].strftime('%H:%M:%S')}"

                # Add WebSocket trigger info
                current_time = datetime.now(timezone.utc)
                trigger_delay = (current_time - signal_df.index[-1]).total_seconds()
                debug_info += f"\n  Trigger delay: {trigger_delay:.1f}s from kline close"

                logger.info(debug_info)

            elif sig_val != 0:  # Signal generated but below threshold
                logger.debug(f"[SIGNAL_BELOW_THRESHOLD] {live_test_config_data['symbol']}_{live_test_config_data['interval']}: "
                           f"Signal={sig_val}, Conf={conf_val:.2f} < {current_confidence_threshold}")
            # --- END SIGNAL DEBUGGING ---

            if sig_val != 0 and conf_val >= current_confidence_threshold:
                # --- START: 时间过滤逻辑 ---
                # 修复字段名不一致问题：兼容两种字段名格式
                trade_start_time_str = live_test_config_data.get("tradeStartTime") or live_test_config_data.get("trade_start_time")
                trade_end_time_str = live_test_config_data.get("tradeEndTime") or live_test_config_data.get("trade_end_time")
                excluded_weekdays = live_test_config_data.get("excluded_weekdays", []) or live_test_config_data.get("excludedWeekdays", [])

                # 只有在设置了有效的过滤条件时才执行
                if (trade_start_time_str and trade_end_time_str) or (excluded_weekdays):
                    now_in_china = now_china()
                    current_weekday = now_in_china.weekday() # Monday is 0 and Sunday is 6

                    # 1. 检查星期
                    # 检查星期 (修正：将Python的weekday 0-6 转换为前端的 1-7 进行比较)
                    if excluded_weekdays and (current_weekday + 1) in excluded_weekdays:
                        logger.info(f"信号 (Config: {live_test_config_data.get('_config_id', 'N/A')}) 被星期过滤规则跳过。今天是星期 {current_weekday + 1}，在排除列表 {excluded_weekdays} 中。")
                        continue # 使用 continue 跳到下一个配置

                    # 2. 检查时间段
                    if trade_start_time_str and trade_end_time_str:
                        try:
                            start_time = datetime.strptime(trade_start_time_str, '%H:%M').time()
                            end_time = datetime.strptime(trade_end_time_str, '%H:%M').time()
                            current_time = now_in_china.time()

                            is_in_time_range = False
                            if start_time <= end_time:
                                # 时间段在同一天内 (e.g., 09:00 - 17:00)
                                if start_time <= current_time <= end_time:
                                    is_in_time_range = True
                            else:
                                # 时间段跨越午夜 (e.g., 22:00 - 04:00)
                                if current_time >= start_time or current_time <= end_time:
                                    is_in_time_range = True
                            
                            if not is_in_time_range:
                                logger.info(f"信号 (Config: {live_test_config_data.get('_config_id', 'N/A')}) 被时间段过滤规则跳过。当前时间 {current_time.strftime('%H:%M')} 不在允许的 {trade_start_time_str}-{trade_end_time_str} 范围内。")
                                continue # 使用 continue 跳到下一个配置
                        except ValueError:
                            logger.error(f"时间过滤错误 (Config: {live_test_config_data.get('_config_id', 'N/A')}): 无法解析时间字符串 '{trade_start_time_str}' 或 '{trade_end_time_str}'。")
                            # 如果时间格式错误，默认不进行过滤，让信号通过
                            pass

                # --- END: 时间过滤逻辑 ---

                sig_time_dt = now_utc()
                
                # --- 新增：最小开单间隔检查 ---
                min_interval_minutes = live_test_config_data.get("investment_settings", {}).get("min_trade_interval_minutes", 0)
                if min_interval_minutes > 0:
                    last_actual_trade_time = None
                    with live_signals_lock:
                        # 倒序查找属于同一个配置的、投资额大于0的最近一笔信号
                        for s in reversed(live_signals):
                            if s.get('origin_config_id') == live_test_config_data['_config_id'] and s.get('investment_amount', 0) > 0:
                                last_actual_trade_time = parse_frontend_datetime(s.get('signal_time'))
                                break
                    
                    if last_actual_trade_time:
                        time_diff_minutes = (sig_time_dt - last_actual_trade_time).total_seconds() / 60
                        if time_diff_minutes < min_interval_minutes:
                            logger.info(f"信号 (Config: {live_test_config_data['_config_id']}) 被最小开单间隔规则跳过。间隔: {time_diff_minutes:.2f} < {min_interval_minutes} 分钟。")
                            return # 直接返回，彻底忽略此信号

                event_period_minutes = {'3m': 3, '10m': 10, '30m': 30, '1h': 60, '1d': 1440}.get(
                    live_test_config_data.get("event_period", "10m"), 10
                )
                
                exp_end_time_dt = sig_time_dt + timedelta(minutes=event_period_minutes)
                
                # --- MODIFICATION: Use trigger price first, then update with next minute's index open price ---
                # Use trigger price as initial entry price (current kline's close price)
                sig_price = float(latest_sig_data['close']) # Initial value using trigger price
                # logger.info(f"[handle_kline_data] Initial signal price set to trigger price: {sig_price}")
                # --- END MODIFICATION ---
                
                inv_amount = 20.0; profit_pct = 80.0; loss_pct = 100.0
                inv_settings_from_config = live_test_config_data.get("investment_settings")
                if inv_settings_from_config:
                    try:
                        live_inv_model = InvestmentStrategySettings(**inv_settings_from_config)
                        profit_pct = live_inv_model.profitRate; loss_pct = live_inv_model.lossRate
                        inv_strat_id_cfg = live_inv_model.strategy_id
                        
                        # 构建策略实例所需的参数 (整合全局保存、策略定义默认、本次运行时特定配置)
                        # 1. 策略定义中的默认 (非只读、非通用字段)
                        inv_strat_def_cfg = next((s for s in get_available_investment_strategies() if s['id'] == inv_strat_id_cfg), None)
                        default_inv_params_from_def = {}
                        if inv_strat_def_cfg and 'parameters' in inv_strat_def_cfg:
                            default_inv_params_from_def = {
                                p['name']: p['default'] for p in inv_strat_def_cfg['parameters']
                                if not p.get('readonly') and p.get('name') not in ['amount', 'minAmount', 'maxAmount']
                            }
                        # 2. 全局保存的该投资策略参数
                        final_inv_calc_params_from_global = strategy_parameters_config.get("investment_strategies", {}).get(inv_strat_id_cfg, {})
                        # 3. 本次运行时配置中的投资设置 (live_inv_model 已经解析了大部分, 其他特定参数可能需要从 inv_settings_from_config 获取)
                        #    这里假设 inv_settings_from_config 可能包含策略类期望的、未在 InvestmentStrategySettings 标准字段中的参数
                        
                        # 策略实例的参数应该是: 运行时特定配置 > 全局保存 > 策略定义默认
                        strategy_specific_params_for_instance = {
                            **default_inv_params_from_def,
                            **(final_inv_calc_params_from_global or {}),
                            **(inv_settings_from_config or {}) # 覆盖
                        }
                        # 确保 InvestmentStrategySettings 中的标准字段也传递给策略实例（如果策略需要它们作为 params 的一部分）
                        strategy_specific_params_for_instance.update({
                            'amount': live_inv_model.amount, # Pydantic模型中的基础投资额
                            'minAmount': live_inv_model.minAmount,
                            'maxAmount': live_inv_model.maxAmount,
                            'percentageOfBalance': live_inv_model.percentageOfBalance,
                            # 可以添加 profitRate, lossRate 如果策略内部需要
                        })
                        
                        if inv_strat_def_cfg:
                            # inv_instance = inv_strat_def_cfg['class'](params=strategy_specific_params_for_instance)
                            inv_instance = live_test_config_data.get('investment_strategy_instance')

                            if not inv_instance:
                                logger.error(f"Config ID {live_test_config_data['_config_id']}: 未找到预先创建的投资策略实例，将使用默认投资额。")
                                inv_amount = max(live_inv_model.minAmount, min(live_inv_model.maxAmount, live_inv_model.amount))
                            else:
                                # --- 核心状态管理修改：获取上一次交易结果 ---
                                last_verified_signal = None
                                with live_signals_lock:
                                    # 筛选出属于此配置、且已验证的信号
                                    config_signals = [
                                        s for s in live_signals
                                        if s.get('origin_config_id') == live_test_config_data['_config_id'] and s.get('verified')
                                    ]
                                    if config_signals:
                                        # 按验证时间（如果不存在则按信号时间）降序排序，找到最新的一个
                                        last_verified_signal = max(config_signals, key=lambda s: s.get('verify_time') or s.get('signal_time'))
                                
                                prev_trade_result = last_verified_signal.get('result') if last_verified_signal else None
                                if last_verified_signal:
                                    logger.info(f"Config ID {live_test_config_data['_config_id']}: 找到上一个已验证信号 {last_verified_signal.get('id')}, 结果: {prev_trade_result}")
                                # --- 结束核心状态管理修改 ---

                                # --- MODIFICATION FOR CURRENT BALANCE ---
                                current_balance_for_calc = live_test_config_data.get('current_balance')
                                if current_balance_for_calc is None:
                                    current_balance_for_calc = live_inv_model.simulatedBalance
                                if current_balance_for_calc is None:
                                    if 'percentage' in inv_instance.name.lower():
                                        logger.warning(f"Config ID {live_test_config_data['_config_id']}: 百分比策略缺少余额，将使用默认1000。")
                                        current_balance_for_calc = 1000.0
                                    else:
                                        current_balance_for_calc = live_inv_model.amount
                                # --- END MODIFICATION FOR CURRENT BALANCE ---
                                
                                inv_amount = inv_instance.calculate_investment(
                                    current_balance=current_balance_for_calc,
                                    previous_trade_result=prev_trade_result, # 传递真实的上一次交易结果
                                    base_investment_from_settings=live_inv_model.amount
                                )
                                inv_amount = max(live_inv_model.minAmount, min(live_inv_model.maxAmount, inv_amount))
                        else:
                            inv_amount = max(live_inv_model.minAmount, min(live_inv_model.maxAmount, live_inv_model.amount)) # Fallback
                    except Exception as e_inv_calc:
                        print(f"实时信号投资金额计算错误 (Config ID: {live_test_config_data['_config_id']}): {e_inv_calc}\n{traceback.format_exc()}")
                        inv_amount = live_test_config_data.get("investment_settings", {}).get("amount", 20.0)

                signal_id_str = f"{live_test_config_data['symbol']}_{live_test_config_data['interval']}_{pred_strat_id}_{int(sig_time_dt.timestamp())}_{random.randint(100,999)}"
                
                new_live_signal = {
                    'id': signal_id_str,
                    'symbol': live_test_config_data['symbol'], 'interval': live_test_config_data['interval'],
                    'prediction_strategy_id': pred_strat_id,
                    'prediction_strategy_params': final_pred_params,
                    'signal_time': format_for_display(sig_time_dt),
                    'signal': sig_val, 'confidence': round(conf_val, 2),
                    'signal_price': sig_price, 'event_period': live_test_config_data.get("event_period", "10m"),
                    'expected_end_time': format_for_display(exp_end_time_dt),
                    'investment_amount': round(inv_amount, 2),
                    'profit_rate_pct': profit_pct, 'loss_rate_pct': loss_pct,
                    'verified': False, 'origin_config_id': live_test_config_data['_config_id'],
                    'verification_status': 'pending',
                    'autox_triggered_info': []
                }
                logger.debug(f"[handle_kline_data] Created new_live_signal object: {json.dumps(ensure_json_serializable(new_live_signal), indent=2)}")

                # --- MODIFICATION: Deduct investment amount immediately ---
                config_id_for_balance_update = live_test_config_data.get('_config_id')
                investment_amount_deducted = new_live_signal.get('investment_amount', 0)

                if config_id_for_balance_update and investment_amount_deducted > 0:
                    new_config_balance_after_deduction = None
                    with running_live_test_configs_lock:
                        if config_id_for_balance_update in running_live_test_configs:
                            current_config_bal = running_live_test_configs[config_id_for_balance_update].get('current_balance')
                            if current_config_bal is not None:
                                running_live_test_configs[config_id_for_balance_update]['current_balance'] = current_config_bal - investment_amount_deducted
                                new_config_balance_after_deduction = running_live_test_configs[config_id_for_balance_update]['current_balance']
                                print(f"Config ID {config_id_for_balance_update}: 信号触发，扣除投资额 {investment_amount_deducted:.2f}。新余额: {new_config_balance_after_deduction:.2f}")
                            else:
                                print(f"警告: Config ID {config_id_for_balance_update} 在信号触发时缺少 current_balance。无法扣除投资额。")

                    if new_config_balance_after_deduction is not None:
                        # Broadcast the balance update to the specific client
                        updated_config_balance_payload = {
                            "type": "config_specific_balance_update",
                            "data": {
                                "config_id": config_id_for_balance_update,
                                "new_balance": new_config_balance_after_deduction,
                                "last_pnl_amount": -investment_amount_deducted # 信号触发时，盈亏是负的投资额
                            }
                        }
                        await manager.broadcast_json(
                            updated_config_balance_payload,
                            filter_func=lambda c: websocket_to_config_id_map.get(c) == config_id_for_balance_update
                        )
                # --- END MODIFICATION ---

                # After deducting, calculate and broadcast the *next* investment amount
                await broadcast_next_investment_amount(config_id_for_balance_update)

                should_trigger_autox = live_test_config_data.get('_should_autox_trigger', False)
                AUTOX_GLOBAL_ENABLED = True

                if should_trigger_autox and AUTOX_GLOBAL_ENABLED:
                    clients_to_send_command_to = []
                    with autox_clients_lock:
                        for ws_client, client_info_dict in active_autox_clients.items():
                            if client_info_dict.get('status') == 'idle' and \
                               new_live_signal['symbol'] in client_info_dict.get('supported_symbols', []):
                                client_info_dict['status'] = 'processing_trade'
                                client_info_dict['last_signal_id'] = new_live_signal['id']
                                target_client_id = client_info_dict.get('client_id')
                                trade_command_payload = {
                                    "signal_id": new_live_signal['id'], "symbol": new_live_signal['symbol'],
                                    "direction": "up" if new_live_signal['signal'] == 1 else "down",
                                    "amount": str(new_live_signal['investment_amount']),
                                    "timestamp": new_live_signal['signal_time']
                                }
                                command_to_send = {"type": "execute_trade", "payload": trade_command_payload}
                                clients_to_send_command_to.append({
                                    "ws": ws_client, "command": command_to_send, "client_id": target_client_id
                                })
                    if clients_to_send_command_to:
                        tasks_for_sending = []
                        for client_trigger_details in clients_to_send_command_to:
                            tasks_for_sending.append(
                                _send_autox_command(client_trigger_details["ws"], client_trigger_details["command"])
                            )
                            new_live_signal['autox_triggered_info'].append({
                                "client_id": client_trigger_details["client_id"],
                                "sent_at": format_for_display(now_utc()), "status": "command_sent"
                            })
                        if tasks_for_sending:
                            await asyncio.gather(*tasks_for_sending); await save_autox_clients_to_file()
                        if not new_live_signal['autox_triggered_info']:
                             new_live_signal['autox_triggered_info'] = {"status": "command_preparation_failed_or_no_client_info_logged"}
                    else:
                        print(f"信号 {new_live_signal['id']} 符合AutoX触发条件，但未找到合适的空闲AutoX客户端进行广播。")
                        new_live_signal['autox_triggered_info'] = {"status": "no_available_client_for_broadcast"}

                with live_signals_lock: live_signals.append(new_live_signal)
                await save_live_signals_async()
                
                num_triggered = 0
                if isinstance(new_live_signal.get('autox_triggered_info'), list):
                    num_triggered = len(new_live_signal['autox_triggered_info'])
                
                print(f"有效交易信号 (Config ID: {live_test_config_data['_config_id']}): ID={new_live_signal['id']}, 交易对={new_live_signal['symbol']}_{new_live_signal['interval']}, 方向={'上涨' if new_live_signal['signal'] == 1 else '下跌'}, 置信度={new_live_signal['confidence']:.2f}, 投资额={new_live_signal['investment_amount']:.2f}, AutoX触发客户端数: {num_triggered}")
                
                logger.debug(f"[handle_kline_data] Broadcasting 'new_signal': {json.dumps(new_live_signal, indent=2)}")
                logger.debug(f"[handle_kline_data] Broadcasting 'new_signal': {json.dumps(ensure_json_serializable(new_live_signal), indent=2)}")
                await manager.broadcast_json(
                    {"type": "new_signal", "data": new_live_signal},
                    filter_func=lambda c: websocket_to_config_id_map.get(c) == new_live_signal['origin_config_id']
                )

                # --- MODIFICATION: Schedule async update of entry price to next minute's index open price ---
                asyncio.create_task(update_signal_entry_price(
                    signal_id_str,
                    live_test_config_data['symbol'],
                    sig_time_dt,
                    live_test_config_data['interval']
                ))
                # --- END MODIFICATION ---
    except Exception as e:
        print(f"处理K线数据时发生严重错误: {e}\n{traceback.format_exc()}")

def kline_callback_wrapper(kline_data): # 基本不变
    try: signals_queue.put_nowait(kline_data)
    except Exception as e: print(f"kline_callback_wrapper 中发生错误: {e}")

# --- K线流管理辅助函数 (保持不变) ---
async def start_kline_websocket_if_needed(symbol: str, interval: str):
    stream_key = f"{symbol}_{interval}"
    with active_kline_streams_lock:
        current_refs = active_kline_streams.get(stream_key, 0)
        if current_refs == 0:
            try:
                print(f"准备为 {symbol} {interval} 启动K线流 (首次需要)...")
                binance_client.start_kline_websocket(symbol, interval, kline_callback_wrapper)
                print(f"已为 {symbol} {interval} 启动K线流的请求已发送。")
            except Exception as e:
                print(f"为 {symbol} {interval} 启动K线流失败: {e}")
                # 如果启动失败，不应该增加引用计数，或者应该有错误处理机制
                # 此处暂不修改，但标记为潜在改进点
                raise # 将异常抛出，让调用者处理
        active_kline_streams[stream_key] = current_refs + 1
        print(f"K线流 {stream_key} 当前引用计数: {active_kline_streams[stream_key]}")


async def stop_kline_websocket_if_not_needed(symbol: str, interval: str):
    stream_key = f"{symbol}_{interval}"
    with active_kline_streams_lock:
        current_refs = active_kline_streams.get(stream_key, 0)
        if current_refs <= 0:
            print(f"警告: 尝试减少 {stream_key} 的引用计数，但已为0或更少。")
            if stream_key in active_kline_streams: del active_kline_streams[stream_key]
            return
        
        new_refs = current_refs - 1
        active_kline_streams[stream_key] = new_refs
        print(f"K线流 {stream_key} 新的引用计数: {new_refs}")

        if new_refs == 0:
            try:
                print(f"准备停止 {symbol} {interval} 的K线流 (不再需要)...")
                binance_client.stop_kline_websocket(symbol, interval)
                print(f"已发送停止 {symbol} {interval} 的K线流的请求。")
            except Exception as e:
                print(f"停止 {symbol} {interval} 的K线流失败: {e}")
            if stream_key in active_kline_streams:
                 del active_kline_streams[stream_key]


# --- FastAPI 事件和API端点 ---
@app.on_event("startup")
async def startup_event():
    # 所有加载函数已改为异步
    await load_autox_trade_logs() # 新增：加载当天的交易日志
    await load_live_signals_async()
    await load_strategy_parameters_from_file()
    await load_autox_clients_from_file()
    await load_active_test_config() # 新增：加载活动测试配置

    # 如果存在活动测试配置，启动相应的K线流
    current_active_config_id = None
    with active_live_test_config_lock:
        current_active_config_id = active_live_test_config_id

    if current_active_config_id:
        current_active_config = None
        with running_live_test_configs_lock:
            if current_active_config_id in running_live_test_configs:
                current_active_config = running_live_test_configs[current_active_config_id]

        if current_active_config:
            symbol = current_active_config.get('symbol')
            interval = current_active_config.get('interval')
            if symbol and interval and symbol != 'all' and interval != 'all':
                try:
                    await start_kline_websocket_if_needed(symbol, interval)
                    print(f"应用启动：已根据活动配置 {current_active_config_id} 启动 {symbol} {interval} 的K线流。")
                except Exception as e:
                    print(f"应用启动：根据活动配置 {current_active_config_id} 启动K线流失败: {e}")
            else:
                 print(f"应用启动：活动配置 {current_active_config_id} 的交易对或周期无效，未启动K线流。")
        else:
             print(f"应用启动：未在 running_live_test_configs 中找到 ID 为 {current_active_config_id} 的活动配置数据。")
    else:
        print("应用启动：未找到活动测试配置。")


    # 创建后台任务
    asyncio.create_task(process_kline_queue())
    asyncio.create_task(background_signal_verifier())
    print("应用启动完成。后台任务已启动。")

    # 新增：启动每日日志切换和清理任务
    asyncio.create_task(daily_log_rotation_and_cleanup())

# 新增：每日日志轮转和清理任务 (占位符)
async def daily_log_rotation_and_cleanup():
    """后台任务：每日检查并轮转 AutoX 交易日志，清理旧文件。"""
    logger.info("每日日志轮转和清理任务已启动。")
    while not shutdown_event_async.is_set():
        try:
            await asyncio.sleep(3600) # 每小时检查一次 (可以调整)
        except asyncio.CancelledError:
            logger.info("daily_log_rotation_and_cleanup: sleep cancelled, likely due to shutdown.")
            break # 如果睡眠被取消，也退出循环

        if shutdown_event_async.is_set():
            logger.info("daily_log_rotation_and_cleanup: shutdown_event_async set, exiting.")
            break
        # TODO: 实现日志轮转和清理逻辑
        # 例如：检查当前日期是否与日志文件日期一致，不一致则切换并清理旧文件
        logger.debug("daily_log_rotation_and_cleanup: hourly check.")
    logger.info("daily_log_rotation_and_cleanup task gracefully shut down.")


# --- WebSocket 端点 for Optimization ---
@app.websocket("/ws/optimization/{optimization_id}")
async def websocket_optimization_endpoint(websocket: WebSocket, optimization_id: str):
    """WebSocket endpoint for optimization status updates."""
    await optimization_manager.connect(websocket, optimization_id)
    try:
        while True:
            # Keep the connection alive to receive server-pushed messages.
            # The client does not need to send any messages.
            # We can add a timeout to detect stale connections if needed.
            await websocket.receive_text() # This will wait for a message, effectively keeping it alive.
    except WebSocketDisconnect:
        logger.info(f"Client for optimization_id {optimization_id} disconnected.")
    except Exception as e:
        logger.error(f"Error in optimization websocket for {optimization_id}: {e}", exc_info=True)
    finally:
        optimization_manager.disconnect(websocket, optimization_id)


# --- WebSocket 端点 for Web UI (/ws/live-test) (逻辑微调，确保使用锁和副本) ---
@app.websocket("/ws/live-test")
async def websocket_endpoint(websocket: WebSocket):
    global active_live_test_config_id
    global active_live_test_config_id # 声明为全局变量
    global running_live_test_configs # 也需要声明 running_live_test_configs 为全局变量
    global websocket_to_config_id_map # 也需要声明 websocket_to_config_id_map 为全局变量
    await manager.connect(websocket)
    try:
        # 发送初始信号数据
        initial_signals_to_send = []
        with live_signals_lock: 
            sorted_signals = sorted(
                [s.copy() for s in live_signals], 
                key=lambda s: s.get('signal_time', ''), 
                reverse=True
            )
            initial_signals_to_send = sorted_signals[:50]
        await websocket.send_json({"type": "initial_signals", "data": initial_signals_to_send})

        # 发送初始统计数据
        stats_payload_init_data = []
        with live_signals_lock: 
            stats_payload_init_data = [s.copy() for s in live_signals]
        
        verified_list_global_init = [s for s in stats_payload_init_data if s.get('verified')]
        total_verified_global_init = len(verified_list_global_init)
        total_correct_global_init = sum(1 for s in verified_list_global_init if s.get('result'))
        total_actual_profit_loss_amount_global_init = sum(s.get('actual_profit_loss_amount', 0.0) for s in verified_list_global_init if s.get('actual_profit_loss_amount') is not None)
        stats_payload_init = {
            "total_signals": len(stats_payload_init_data),
            "total_verified": total_verified_global_init,
            "total_correct": total_correct_global_init,
            "win_rate": round(total_correct_global_init / total_verified_global_init * 100 if total_verified_global_init > 0 else 0, 2),
            "total_pnl_pct": round(sum(s.get('pnl_pct', 0.0) for s in verified_list_global_init if s.get('pnl_pct') is not None), 2),
            "average_pnl_pct": round(sum(s.get('pnl_pct', 0.0) for s in verified_list_global_init if s.get('pnl_pct') is not None) / total_verified_global_init if total_verified_global_init > 0 else 0, 2),
            "total_profit_amount": round(total_actual_profit_loss_amount_global_init, 2),
             "current_balance": round(global_running_balance, 2) # 初始也发送全局余额
        }
        await websocket.send_json({"type": "initial_stats", "data": stats_payload_init})
        
        # 检查是否有活动的测试配置，如果有，立即发送给新连接的客户端
        current_active_config_id = None
        current_active_config = None
        with active_live_test_config_lock:
            current_active_config_id = active_live_test_config_id
        
        if current_active_config_id:
            with running_live_test_configs_lock:
                if current_active_config_id in running_live_test_configs:
                    current_active_config = running_live_test_configs[current_active_config_id].copy()
        
        if current_active_config:
            # 将当前WebSocket关联到活动配置ID
            websocket_to_config_id_map[websocket] = current_active_config_id
            # 在通过WebSocket发送前，移除不可序列化的策略实例
            config_for_broadcast = current_active_config.copy()
            config_for_broadcast.pop('investment_strategy_instance', None)

            # 确保新字段有默认值，统一使用下划线格式
            config_for_broadcast.setdefault('trade_start_time', '')
            config_for_broadcast.setdefault('trade_end_time', '')
            config_for_broadcast.setdefault('excluded_weekdays', [])

            # 发送活动配置信息给新连接的客户端
            await websocket.send_json({
                "type": "active_config_notification",
                "data": {
                    "config_id": current_active_config_id,
                    "config": config_for_broadcast,
                    "message": "当前有活动的测试配置"
                }
            })
        
        while True: 
            data = await websocket.receive_json()
            message_type = data.get('type')

            if message_type == 'restore_session':
                client_config_id = data.get('data', {}).get('config_id')
                restored_config_data = None
                if client_config_id:
                    with running_live_test_configs_lock: 
                        config_dict = running_live_test_configs.get(client_config_id)
                        if config_dict: restored_config_data = config_dict.copy() 
                
                if restored_config_data:
                    # 确保恢复的配置有 current_balance，并且它是一个有效的数值
                    current_balance_val = restored_config_data.get('current_balance')
                    if current_balance_val is None: # 包括 'current_balance' 键不存在，或者键存在但值为 None
                        logger.info(f"恢复会话 {client_config_id}: 'current_balance' 为 None 或不存在于 restored_config_data。尝试从 investment_settings.simulatedBalance 回退。")
                        sim_bal = restored_config_data.get('investment_settings', {}).get('simulatedBalance')
                        if sim_bal is not None:
                            restored_config_data['current_balance'] = float(sim_bal) # 确保是 float
                            logger.info(f"恢复会话 {client_config_id}: 'current_balance' 已从 simulatedBalance ({sim_bal}) 设置。")
                        else:
                            restored_config_data['current_balance'] = 1000.0 # 默认值
                            logger.info(f"恢复会话 {client_config_id}: 'current_balance' 已设置为默认值 1000.0 (simulatedBalance 也为 None)。")
                    elif not isinstance(current_balance_val, (int, float)):
                        logger.warning(f"恢复会话 {client_config_id}: 'current_balance' ({current_balance_val}) 不是有效数值类型。将尝试回退。")
                        sim_bal = restored_config_data.get('investment_settings', {}).get('simulatedBalance')
                        if sim_bal is not None:
                            restored_config_data['current_balance'] = float(sim_bal)
                            logger.info(f"恢复会话 {client_config_id}: 'current_balance' 因类型无效已从 simulatedBalance ({sim_bal}) 重置。")
                        else:
                            restored_config_data['current_balance'] = 1000.0
                            logger.info(f"恢复会话 {client_config_id}: 'current_balance' 因类型无效已重置为默认值 1000.0。")
                    else:
                        # 如果 current_balance 存在且是有效数值，确保它是 float 类型以保持一致性
                        restored_config_data['current_balance'] = float(current_balance_val)
                        # logger.info(f"恢复会话 {client_config_id}: 'current_balance' ({current_balance_val}) 已存在且有效。")

                    # 确保 total_profit_loss_amount 也存在且为数值，如果不存在则初始化为 0.0
                    total_pnl_val = restored_config_data.get('total_profit_loss_amount')
                    if total_pnl_val is None:
                        restored_config_data['total_profit_loss_amount'] = 0.0
                        logger.info(f"恢复会话 {client_config_id}: 'total_profit_loss_amount' 为 None 或不存在，已初始化为 0.0。")
                    elif not isinstance(total_pnl_val, (int, float)):
                        restored_config_data['total_profit_loss_amount'] = 0.0
                        logger.warning(f"恢复会话 {client_config_id}: 'total_profit_loss_amount' ({total_pnl_val}) 不是有效数值类型，已重置为 0.0。")
                    else:
                        restored_config_data['total_profit_loss_amount'] = float(total_pnl_val)
                    
                    websocket_to_config_id_map[websocket] = client_config_id
                    # 在通过WebSocket发送前，移除不可序列化的策略实例
                    config_for_broadcast = restored_config_data.copy()
                    config_for_broadcast.pop('investment_strategy_instance', None)
                    
                    # 确保新字段有默认值，统一使用下划线格式
                    config_for_broadcast.setdefault('trade_start_time', '')
                    config_for_broadcast.setdefault('trade_end_time', '')
                    config_for_broadcast.setdefault('excluded_weekdays', [])

                    # 发送给客户端的 config_for_broadcast 不再包含实例
                    await websocket.send_json({"type": "session_restored", "data": {"config_id": client_config_id, "config_details": config_for_broadcast}})
                    # After restoring session, immediately calculate and send the next investment amount
                    await broadcast_next_investment_amount(client_config_id)
                    # logger.info(f"会话已恢复 {client_config_id}，发送的 config_details: {json.dumps(ensure_json_serializable(restored_config_data), indent=2)}")
                else:
                    await websocket.send_json({"type": "session_not_found", "data": {"config_id": client_config_id}})
            
            elif message_type == 'set_runtime_config':
                logger.info(f"[DEBUG] Received 'set_runtime_config' message. Raw data: {json.dumps(data, indent=2)}")
                config_payload_data = data.get('data', {})

                # 修正：在验证之前，手动合并策略特定参数到 investment_settings 中
                if 'investment_settings' in config_payload_data and 'investment_strategy_specific_params' in config_payload_data['investment_settings']:
                    specific_params = config_payload_data['investment_settings'].pop('investment_strategy_specific_params')
                    if specific_params:
                        logger.info(f"[DEBUG] Merging specific params into investment_settings: {specific_params}")
                        config_payload_data['investment_settings'].update(specific_params)
                        logger.info(f"[DEBUG] Final investment_settings before validation: {config_payload_data['investment_settings']}")

                try:
                    required_keys = ["symbol", "interval", "prediction_strategy_id", "confidence_threshold", "event_period", "investment_settings"]
                    if not all(key in config_payload_data for key in required_keys):
                        await websocket.send_json({"type": "error", "data": {"message": "运行时配置缺少必要字段。"}})
                        continue
                    # Pydantic 模型验证 investment_settings
                    validated_investment_settings = InvestmentStrategySettings(**config_payload_data["investment_settings"]).model_dump(mode='json')
                    config_payload_data["investment_settings"] = validated_investment_settings # 使用验证后的数据
                except ValidationError as e_val:
                    # Extract detailed errors from Pydantic ValidationError
                    error_details = e_val.errors()
                    await websocket.send_json({
                        "type": "error",
                        "data": {
                            "message": "配置数据验证失败", # 通用错误消息
                            "details": error_details # 详细错误列表
                        }
                    })
                    continue # 继续处理下一个消息
                except Exception as e_val:
                    # Handle other exceptions during config processing
                    print(f"处理 set_runtime_config 时发生未知错误: {e_val}\n{traceback.format_exc()}")
                    await websocket.send_json({"type": "error", "data": {"message": f"应用配置时发生未知错误: {str(e_val)}"}})
                    continue # 继续处理下一个消息

                # --- 新逻辑：无论是否已有活动配置，都允许覆盖并保存新配置 ---
                # 先移除当前WebSocket的旧配置ID映射
                existing_config_id = websocket_to_config_id_map.pop(websocket, None)
                # 生成/复用活动配置ID
                new_config_id = None
                with active_live_test_config_lock:
                    if active_live_test_config_id:
                        new_config_id = active_live_test_config_id
                    else:
                        new_config_id = uuid.uuid4().hex
                        active_live_test_config_id = new_config_id

                new_symbol = config_payload_data['symbol']
                new_interval = config_payload_data['interval']
                initial_simulated_balance = config_payload_data["investment_settings"].get("simulatedBalance")
                initial_current_balance = initial_simulated_balance if initial_simulated_balance is not None else 1000.0

                # 构造完整配置，保留当前余额和盈亏（如果已有配置）
                with running_live_test_configs_lock:
                    old_config = running_live_test_configs.get(new_config_id)
                    
                    # config_payload_data["investment_settings"] 已经被 validated_investment_settings (Pydantic model_dump) 替换
                    # validated_investment_settings 在大约 line 1415 定义并赋值给 config_payload_data["investment_settings"]
                    current_payload_investment_settings = config_payload_data["investment_settings"].copy() # 使用副本
                    newly_input_simulated_balance = current_payload_investment_settings.get("simulatedBalance")

                    final_current_balance: float
                    final_total_profit_loss_amount: float
                    # final_investment_settings_to_store 将是 current_payload_investment_settings，因为它已包含最新的 simulatedBalance
                    final_investment_settings_to_store: dict = current_payload_investment_settings
                    created_at_to_use: str

                    if old_config: # 配置已存在
                        existing_total_profit_loss = old_config.get('total_profit_loss_amount', 0.0)
                        
                        # 使用新输入的 simulatedBalance (可能为None) 进行计算，如果为 None，则计算 current_balance 时视为 0.0
                        simulated_balance_for_calc = newly_input_simulated_balance if newly_input_simulated_balance is not None else 0.0
                        
                        final_current_balance = simulated_balance_for_calc + existing_total_profit_loss
                        final_total_profit_loss_amount = existing_total_profit_loss # PnL 保持不变
                        
                        # final_investment_settings_to_store 已经包含了来自 payload 的 investment_settings,
                        # 其中 simulatedBalance 就是 newly_input_simulated_balance (用户新输入的值).
                        
                        created_at_to_use = old_config.get("created_at", format_for_display(now_utc()))
                        logger.info(f"Updating existing config {new_config_id}. New simulatedBalance: {newly_input_simulated_balance}, existing PnL: {existing_total_profit_loss}, new current_balance: {final_current_balance}")

                    else: # 新配置
                        # initial_current_balance (在前面约 line 1448 定义)
                        # 已经是基于 newly_input_simulated_balance (即 initial_simulated_balance from payload) 或默认值1000.0计算的
                        final_current_balance = initial_current_balance
                        final_total_profit_loss_amount = 0.0
                        # final_investment_settings_to_store 已经包含了来自 payload 的 investment_settings.
                        
                        created_at_to_use = format_for_display(now_utc())
                        logger.info(f"Creating new config {new_config_id}. Initial/New simulatedBalance: {newly_input_simulated_balance}, initial current_balance: {final_current_balance}")

                    full_config_to_store = {
                        "_config_id": new_config_id,
                        "symbol": new_symbol, # new_symbol, new_interval 在前面约 line 1445-1446 定义
                        "interval": new_interval,
                        "prediction_strategy_id": config_payload_data["prediction_strategy_id"],
                        "prediction_strategy_params": config_payload_data.get("prediction_strategy_params"),
                        "confidence_threshold": config_payload_data["confidence_threshold"],
                        "event_period": config_payload_data["event_period"],
                        "investment_settings": final_investment_settings_to_store,
                        "autox_enabled": config_payload_data.get("autox_enabled", True),
                        "current_balance": round(final_current_balance, 2),
                        "total_profit_loss_amount": round(final_total_profit_loss_amount, 2),
                        "created_at": created_at_to_use,
                        # 新增：保存时间过滤字段，统一使用下划线格式
                        "trade_start_time": config_payload_data.get("trade_start_time", ""),
                        "trade_end_time": config_payload_data.get("trade_end_time", ""),
                        "excluded_weekdays": config_payload_data.get("excluded_weekdays", [])
                    }
                    running_live_test_configs[new_config_id] = full_config_to_store
                    
                    # 创建并存储投资策略实例
                    inv_instance = _create_and_store_investment_strategy_instance(full_config_to_store)
                    if inv_instance:
                        # 将实例直接添加到内存中的配置字典里
                        running_live_test_configs[new_config_id]['investment_strategy_instance'] = inv_instance
                    else:
                        logger.error(f"未能为新配置 {new_config_id} 创建投资策略实例。")

                websocket_to_config_id_map[websocket] = new_config_id

                try:
                    if new_symbol != 'all' and new_interval != 'all':
                        await start_kline_websocket_if_needed(new_symbol, new_interval)
                    # 保存活动配置到文件
                    await save_active_test_config()
                    # 在通过WebSocket发送前，移除不可序列化的策略实例
                    config_for_broadcast = full_config_to_store.copy()
                    config_for_broadcast.pop('investment_strategy_instance', None)
                    # 发送给客户端的 config_for_broadcast 不再包含实例
                    await websocket.send_json({"type": "config_set_confirmation", "data": {"success": True, "message": "运行时配置已应用。", "config_id": new_config_id, "applied_config": config_for_broadcast}})
                    
                    # After setting/updating config, broadcast the initial next investment amount
                    await broadcast_next_investment_amount(new_config_id)
                except Exception as e_start_stream:
                    await websocket.send_json({"type": "error", "data": {"message": f"应用配置时启动K线流失败: {str(e_start_stream)}"}})
                    with running_live_test_configs_lock:
                        running_live_test_configs.pop(new_config_id, None)
                    websocket_to_config_id_map.pop(websocket, None) 
            
            elif message_type == 'stop_current_test':
                config_id_to_stop = websocket_to_config_id_map.pop(websocket, None)
                stopped_config_content = None
                
                if config_id_to_stop:
                    with running_live_test_configs_lock:
                        stopped_config_content = running_live_test_configs.pop(config_id_to_stop, None)
                    
                    # 清除全局活动配置ID
                    with active_live_test_config_lock:
                        if active_live_test_config_id == config_id_to_stop:
                            active_live_test_config_id = None
                            print(f"已清除全局活动配置ID: {config_id_to_stop}")
                    
                    # 保存更新后的活动配置状态 (应为空或新的活动配置)
                    await save_active_test_config()

                if stopped_config_content:
                    await stop_kline_websocket_if_not_needed(stopped_config_content['symbol'], stopped_config_content['interval'])
                    await websocket.send_json({"type": "test_stopped_confirmation", "data": {"success": True, "message": "当前测试配置已停止。", "stopped_config_id": config_id_to_stop}})
                else:
                    await websocket.send_json({"type": "error", "data": {"message": "未找到活动的测试配置来停止。"}})
            
            await asyncio.sleep(0.1)

    except WebSocketDisconnect: 
        print(f"客户端 {getattr(websocket, 'client', 'N/A')} 断开连接。")
        # 移除WebSocket到配置ID的映射，但保留配置数据以支持会话恢复
        config_id = websocket_to_config_id_map.pop(websocket, None)
        
        # 检查是否还有其他WebSocket连接到同一配置
        if config_id:
            other_connections_to_same_config = False
            for ws, cfg_id in websocket_to_config_id_map.items():
                if cfg_id == config_id:
                    other_connections_to_same_config = True
                    break
            
            # 如果没有其他连接使用此配置，记录日志但不删除配置，保留以支持会话恢复
            if not other_connections_to_same_config:
                print(f"配置 {config_id} 的最后一个WebSocket连接已断开，但配置将保留以支持会话恢复。")
    except Exception as e: print(f"WebSocket端点错误 ({getattr(websocket, 'client', 'N/A')}): {e}\n{traceback.format_exc()}")
    finally:
        manager.disconnect(websocket)
        # 只移除WebSocket映射，但不清理配置，以支持会话恢复
        # 配置将保留在running_live_test_configs中，直到用户显式停止测试
        # 或者通过API/UI操作清理

# --- WebSocket 端点 for AutoX Clients (/ws/autox_control) (确保保存操作是异步的) ---
@app.websocket("/ws/autox_control")
async def autox_websocket_endpoint(websocket: WebSocket):
    # 在 accept 之前，我们不能确定这个连接的身份
    logger.info(f"新的 AutoX WebSocket 连接请求进入，等待 accept... (Client: {websocket.client.host}:{websocket.client.port})")
    await autox_manager.connect(websocket)
    logger.info(f"AutoX WebSocket 连接已 accept。 (Client: {websocket.client.host}:{websocket.client.port})")
    client_id_local: Optional[str] = None # 用于在 finally 中记录
    
    try:
        while True:
            try:
                # 处理不同类型的WebSocket消息
                data = await websocket.receive_json()
                logger.debug(f"收到AutoX客户端消息 (JSON): {data}")
                message_type = data.get("type")
                payload = data.get("payload", {})

                # 处理文本消息的逻辑
                if message_type == "register":
                            client_id_from_payload = payload.get("client_id")
                            supported_symbols_list = payload.get("supported_symbols", ["BTCUSDT"])
                            client_notes_from_payload = payload.get("notes") # JS脚本目前不发送这个，但可以保留

                            if not client_id_from_payload:
                                await websocket.send_json({"type": "error", "message": "注册失败：client_id 不能为空。"})
                                await websocket.close(code=1008)
                                return

                            client_id_local = client_id_from_payload

                            client_info_to_store_dict: Optional[Dict[str, Any]] = None

                            with autox_clients_lock:
                                existing_persistent_info = persistent_autox_clients_data.get(client_id_local)
                                
                                old_ws_to_close = None
                                for ws_iter, active_info_iter in active_autox_clients.items():
                                    if active_info_iter.get('client_id') == client_id_local and ws_iter != websocket:
                                        logger.warning(f"AutoX客户端 {client_id_local} 重复连接 (不同WebSocket)，准备关闭旧连接。")
                                        old_ws_to_close = ws_iter
                                        break
                                
                                if old_ws_to_close:
                                    logger.info(f"开始处理旧连接的关闭流程 for client {client_id_local}...")
                                    # 不直接调用 close()，因为它可能在另一个任务中已经被关闭
                                    # 而是从我们的管理器中移除它，让 FastAPI/Starlette 的底层来处理物理关闭
                                    autox_manager.disconnect(old_ws_to_close)
                                    active_autox_clients.pop(old_ws_to_close, None)
                                    logger.info(f"旧的 WebSocket 对象 for client {client_id_local} 已从 active_autox_clients 和 autox_manager 中移除。")

                                if existing_persistent_info:
                                    final_notes = client_notes_from_payload if client_notes_from_payload is not None else existing_persistent_info.get('notes')
                                    client_info_model = AutoXClientInfo(
                                        client_id=client_id_local,
                                        status='idle',
                                        supported_symbols=supported_symbols_list,
                                        last_seen=now_utc(),
                                        connected_at=parse_frontend_datetime(existing_persistent_info.get('connected_at')) if existing_persistent_info.get('connected_at') else now_utc(),
                                        notes=final_notes
                                    )
                                else:
                                    client_info_model = AutoXClientInfo(
                                        client_id=client_id_local,
                                        status='idle',
                                        supported_symbols=supported_symbols_list,
                                        last_seen=now_utc(),
                                        connected_at=now_utc(),
                                        notes=client_notes_from_payload
                                    )
                                
                                client_info_to_store_dict = client_info_model.model_dump(mode='json')
                                # 在 active_autox_clients 中记录最后一次收到 pong 的时间
                                client_info_to_store_dict['last_pong_time'] = now_utc().isoformat()
                                active_autox_clients[websocket] = client_info_to_store_dict
                                persistent_autox_clients_data[client_id_local] = client_info_to_store_dict

                            print(f"AutoX客户端已注册/更新: ID={client_id_local}, 支持交易对={supported_symbols_list}, 备注='{client_info_to_store_dict.get('notes', '') if client_info_to_store_dict else ''}'")
                            await websocket.send_json({"type": "registered", "message": "客户端注册成功。", "client_info": client_info_to_store_dict})
                            
                            await debounced_broadcast_autox_clients_status()
                            await save_autox_clients_to_file()
 
                elif message_type == "status_update":
                    if not client_id_local:
                        await websocket.send_json({"type": "error", "message": "未注册的客户端不能发送状态更新。"})
                        continue

                    client_reported_status = payload.get("status")
                    if not client_reported_status:
                        print(f"AutoX客户端 {client_id_local} 发送的状态更新中缺少 status 字段。")
                        continue

                    updated_info_for_broadcast = None
                    
                    # 定义哪些JS上报的状态表示一个交易周期的结束，之后客户端应变为空闲
                    # 这些状态是JS脚本通过sendStatusUpdate的tradeStatus参数发送的
                    TERMINAL_AND_RESET_TO_IDLE_STATUSES = {
                        "trade_execution_failed",    # JS: 交易执行失败
                        "manual_confirmation_pending", # JS: 脚本已设置好参数，等待人工确认 (自动化流程结束)
                        "internal_error",            # JS: 客户端内部错误
                        "trade_execution_completed", # 新增：交易执行完成
                        "test_command_received", # 新增此行
                        # "trade_execution_succeeded" # 如果JS将来会报告明确的成功，也应加进来
                    }

                    # 记录状态更新处理开始时间
                    status_update_start_time = time.time()
                    logger.debug(f"开始处理客户端 {client_id_local} 状态更新 - {datetime.now().isoformat()}")
                    
                    # 在锁内准备数据，但不执行耗时操作
                    updated_info_for_broadcast = None
                    should_continue = False
                    
                    with autox_clients_lock:
                        if websocket not in active_autox_clients:
                            logger.warning(f"警告: 收到来自未知/已断开 WebSocket (client_id登记为: {client_id_local}) 的状态更新。忽略。")
                            should_continue = True # 标记需要跳过后续处理
                        
                        if not should_continue: # 只有在websocket有效时才继续处理
                            current_client_info_active = active_autox_clients[websocket]
                            current_client_info_persistent = persistent_autox_clients_data.get(client_id_local)

                            # 默认情况下，客户端的下一个状态是它自己报告的状态
                            next_client_status_to_set = client_reported_status

                            # 如果报告的状态是终端状态，则将客户端的最终状态设置为 'idle'
                            if client_reported_status in TERMINAL_AND_RESET_TO_IDLE_STATUSES:
                                next_client_status_to_set = "idle"
                                logger.info(f"AutoX客户端 {client_id_local} 报告状态 '{client_reported_status}', 将其重置为 'idle'。")
                                # 对于这些状态，也清除 last_signal_id
                                current_client_info_active.pop('last_signal_id', None)
                                if current_client_info_persistent:
                                    current_client_info_persistent.pop('last_signal_id', None)
                            elif client_reported_status == "idle":
                                # 如果客户端明确报告自己是idle (例如，可能是手动干预或脚本逻辑)
                                current_client_info_active.pop('last_signal_id', None)
                                if current_client_info_persistent:
                                    current_client_info_persistent.pop('last_signal_id', None)
                            # 对于其他中间状态 (如 command_received, amount_set_attempted, direction_selected)，
                            # 客户端状态将设置为这些中间状态。服务器在派发新任务时检查的是 'idle'。

                            current_client_info_active['status'] = next_client_status_to_set
                            current_client_info_active['last_seen'] = now_utc().isoformat()
                            
                            if current_client_info_persistent:
                                current_client_info_persistent['status'] = next_client_status_to_set
                                current_client_info_persistent['last_seen'] = now_utc().isoformat()
                            else:
                                # 理论上注册时就应该有了，但作为保障
                                persistent_autox_clients_data[client_id_local] = current_client_info_active.copy()
                            
                            updated_info_for_broadcast = current_client_info_active.copy()
                    
                    # 记录锁释放时间
                    lock_release_time = time.time()
                    logger.debug(f"状态更新锁内处理完成，耗时: {(lock_release_time - status_update_start_time):.4f}秒 - {datetime.now().isoformat()}")
                    
                    # 如果需要跳过后续处理，直接返回
                    if should_continue:
                        continue
                    
                    # 锁外执行日志记录部分
                    log_payload = {
                        "client_id": client_id_local, "signal_id": payload.get("signal_id"),
                        "command_type": "status_from_client", "command_payload": payload,
                        "status": client_reported_status, # 日志中记录JS上报的原始状态
                        "details": payload.get("details"), "error_message": payload.get("error_message"),
                    }
                    with autox_trade_logs_lock:
                        autox_trade_logs.append(AutoXTradeLogEntry(**log_payload).model_dump(mode='json'))
                        if len(autox_trade_logs) > MAX_AUTOX_LOG_ENTRIES:
                            autox_trade_logs.pop(0)

                    # 锁外执行耗时的IO操作
                    await save_autox_trade_logs_async() # 保存交易日志
                    final_set_status = updated_info_for_broadcast.get('status') if updated_info_for_broadcast else 'N/A'
                    logger.info(f"收到AutoX客户端 {client_id_local} 状态更新: '{client_reported_status}' (原始), Signal ID: {payload.get('signal_id')}. "
                          f"客户端最终状态设置为: '{final_set_status}'")
                    
                    # 记录日志处理完成时间
                    log_process_time = time.time()
                    logger.debug(f"状态更新日志处理完成，耗时: {(log_process_time - lock_release_time):.4f}秒 - {datetime.now().isoformat()}")
                    
                    # 锁外执行广播和文件保存操作
                    if updated_info_for_broadcast:
                        await debounced_broadcast_autox_clients_status()
                        await save_autox_clients_to_file()
                        
                    # 记录整个状态更新处理完成时间
                    status_update_end_time = time.time()
                    logger.debug(f"状态更新处理完成，总耗时: {(status_update_end_time - status_update_start_time):.4f}秒 - {datetime.now().isoformat()}")
 
                elif message_type == "pong":
                    # 收到客户端的 pong 回复，更新最后活动时间和 pong 时间
                    with autox_clients_lock:
                        if websocket in active_autox_clients:
                            active_autox_clients[websocket]['last_seen'] = now_utc().isoformat()
                            active_autox_clients[websocket]['last_pong_time'] = now_utc().isoformat() # 更新 pong 时间
                            if client_id_local and client_id_local in persistent_autox_clients_data:
                                persistent_autox_clients_data[client_id_local]['last_seen'] = now_utc().isoformat()
                                # persistent_autox_clients_data 不存储 last_pong_time，只存储最后活动时间
                else:
                    print(f"收到来自AutoX客户端 {client_id_local or '未知'} 的未知消息类型: {message_type}")
                    await websocket.send_json({"type": "error", "message": f"不支持的消息类型: {message_type}"})

            except json.JSONDecodeError:
                print(f"Received invalid JSON from AutoX client {client_id_local or 'unknown'}.")
                await websocket.send_json({"type": "error", "message": "Invalid JSON format."})
            except Exception as e:
                print(f"Error processing AutoX client text message ({client_id_local or 'unknown'}): {e}\n{traceback.format_exc()}")
                await websocket.send_json({"type": "error", "message": f"Error processing message: {str(e)}"})

            # Handle other message types if needed (e.g., binary)

            except WebSocketDisconnect:
                print(f"AutoX客户端 {client_id_local or getattr(websocket, 'client', 'N/A')} 断开连接。")
                break # Exit the loop on disconnect
            except Exception as e:
                print(f"AutoX WebSocket端点错误 ({client_id_local or getattr(websocket, 'client', 'N/A')}): {e}\n{traceback.format_exc()}")
                # Optionally send an error back before breaking
                try:
                    await websocket.send_json({"type": "error", "message": f"Server error: {str(e)}"})
                except: pass # Ignore errors sending error message
                break # Exit the loop on other exceptions
    finally:
       logger.info(f"进入 autox_websocket_endpoint 的 finally 块 (Client: {client_id_local or f'{websocket.client.host}:{websocket.client.port}'})")
       autox_manager.disconnect(websocket)
       disconnected_client_id = None
       with autox_clients_lock:
           client_info_at_disconnect = active_autox_clients.pop(websocket, None)
           if client_info_at_disconnect:
               disconnected_client_id = client_info_at_disconnect.get('client_id')
               logger.info(f"AutoX客户端 {disconnected_client_id or '未知'} 已从活动列表移除。")
               
               if disconnected_client_id and disconnected_client_id in persistent_autox_clients_data:
                   # 当客户端断开连接时，我们将其在持久化存储中的状态标记为'offline'
                   # 这样UI可以明确显示其离线，并且派单逻辑也不会选择它。
                   # last_signal_id 在这里通常不清除，以便查看它离线前处理的最后一个信号。
                   persistent_autox_clients_data[disconnected_client_id]['status'] = 'offline'
                   persistent_autox_clients_data[disconnected_client_id]['last_seen'] = now_utc().isoformat()
           else:
               logger.warning(f"在 finally 块中，未在 active_autox_clients 中找到当前 websocket (Client: {client_id_local or 'N/A'})")

       
       if disconnected_client_id:
           await debounced_broadcast_autox_clients_status()
           await save_autox_clients_to_file()
       logger.info(f"autox_websocket_endpoint 的 finally 块执行完毕 (Client: {client_id_local or 'N/A'})")
 
 
  
 # --- WebSocket 端点 for AutoX Status (/ws/autox_status) (逻辑不变) ---
@app.websocket("/ws/autox_status")
async def autox_status_websocket_endpoint(websocket: WebSocket):
    await autox_status_manager.connect(websocket)
    try:
        await debounced_broadcast_autox_clients_status() # 连接成功后立即发送当前客户端列表
        while True:
            await websocket.receive_text() # 保持连接，可以处理ping/pong
    except WebSocketDisconnect:
        print(f"AutoX状态前端客户端 {getattr(websocket, 'client', 'N/A')} 断开连接。")
    except Exception as e:
        print(f"AutoX状态WebSocket端点错误 ({getattr(websocket, 'client', 'N/A')}): {e}\n{traceback.format_exc()}")
    finally:
        autox_status_manager.disconnect(websocket)


# --- 防抖控制变量 for broadcast_autox_clients_status ---
_debounce_task_autox_status: Optional[asyncio.Task] = None
DEBOUNCE_DELAY_AUTOX_STATUS: float = 0.3  # 300毫秒

async def _execute_broadcast_after_delay():
    """实际执行广播的辅助函数，在延迟后调用。"""
    try:
        await asyncio.sleep(DEBOUNCE_DELAY_AUTOX_STATUS)
        # 再次检查任务是否在 sleep 期间被取消
        if not asyncio.current_task().cancelled():
            logger.debug(f"Debounce delay for broadcast_autox_clients_status complete, executing now.")
            await broadcast_autox_clients_status()
        else:
            logger.debug(f"Debounce task for broadcast_autox_clients_status was cancelled during delay.")
    except asyncio.CancelledError:
        logger.debug(f"Debounce execution task for broadcast_autox_clients_status cancelled.")
    except Exception as e:
        logger.error(f"Error during debounced execution of broadcast_autox_clients_status: {e}", exc_info=True)


async def debounced_broadcast_autox_clients_status():
    """
    防抖版本的 broadcast_autox_clients_status。
    如果在延迟期间被再次调用，则重置计时器。
    """
    global _debounce_task_autox_status
    
    if _debounce_task_autox_status and not _debounce_task_autox_status.done():
        logger.debug("Debouncing broadcast_autox_clients_status: cancelling previous task.")
        _debounce_task_autox_status.cancel()
        try:
            await _debounce_task_autox_status  # 等待任务实际完成取消
        except asyncio.CancelledError:
            logger.debug("Previous debounce task for broadcast_autox_clients_status successfully cancelled.")
        except Exception as e:
            # 这个异常不应该发生，因为我们期望CancelledError
            logger.error(f"Unexpected error waiting for previous debounce task cancellation: {e}", exc_info=True)

    logger.debug(f"Scheduling new debounced broadcast_autox_clients_status in {DEBOUNCE_DELAY_AUTOX_STATUS}s.")
    _debounce_task_autox_status = asyncio.create_task(_execute_broadcast_after_delay())

# --- 辅助函数：广播AutoX客户端状态 (现在从 persistent_autox_clients_data 读取) ---
async def broadcast_autox_clients_status():
    """广播当前AutoX客户端列表到所有连接的AutoX状态前端WebSocket。
    优化：将数据准备和广播分离，避免在锁内进行耗时IO操作
    """
    # 记录开始时间，用于性能监控
    start_time = time.time()
    logger.debug(f"开始准备AutoX客户端状态数据 - {datetime.now().isoformat()}")
    
    # 第一步：在锁内准备所有需要的数据
    clients_data_to_broadcast = []
    persistent_copy = {}
    active_client_ids = set()
    
    with autox_clients_lock: # 保护对客户端数据的读取，但尽量减少锁内操作
        # 创建副本进行迭代和处理
        persistent_copy = {cid: cinfo.copy() for cid, cinfo in persistent_autox_clients_data.items()}
        
        # 收集活跃客户端ID
        for ws, active_info in active_autox_clients.items():
            if active_info.get('client_id'):
                active_client_ids.add(active_info['client_id'])
    
    # 记录锁释放时间
    lock_release_time = time.time()
    logger.debug(f"锁内数据准备完成，耗时: {(lock_release_time - start_time):.4f}秒 - {datetime.now().isoformat()}")
    
    # 第二步：锁外处理数据
    for client_id, client_info_dict in persistent_copy.items():
        # 确保 Pydantic 模型用于最终序列化，以处理 datetime 等类型
        try:
            # 如果客户端在 active_client_ids 中，可以认为其在线
            # 状态已经在 client_info_dict 中（由 register 和 status_update维护）
            # 这里的 is_online 仅用于额外展示，实际状态在 client_info_dict['status']
            is_online = client_id in active_client_ids
            client_info_dict_enriched = client_info_dict.copy()
            client_info_dict_enriched['is_explicitly_online'] = is_online # 添加一个明确的在线标志
            
            # 使用 Pydantic 模型序列化
            client_model_instance = AutoXClientInfo(**client_info_dict_enriched)
            clients_data_to_broadcast.append(client_model_instance.model_dump(mode='json'))
        except Exception as e_val:
            logger.error(f"序列化AutoX客户端 {client_id} 信息时出错: {e_val}")
    
    # 记录数据处理完成时间
    data_process_time = time.time()
    logger.debug(f"数据处理完成，耗时: {(data_process_time - lock_release_time):.4f}秒 - {datetime.now().isoformat()}")
    
    # 第三步：执行广播（锁外操作）
    payload = {"type": "autox_clients_update", "data": clients_data_to_broadcast}
    logger.debug(f"开始广播AutoX客户端状态 - {datetime.now().isoformat()}")
    await autox_status_manager.broadcast_json(payload)
    
    # 记录广播完成时间
    broadcast_end_time = time.time()
    logger.debug(f"广播完成，耗时: {(broadcast_end_time - data_process_time):.4f}秒，总耗时: {(broadcast_end_time - start_time):.4f}秒 - {datetime.now().isoformat()}")
    # print(f"已广播AutoX客户端状态更新到 {len(autox_status_manager.active_connections)} 个前端连接。")


# --- HTML 页面路由 (保持不变) ---
@app.get("/", response_class=HTMLResponse)
async def read_index():
    try:
        with open("frontend/index.html", "r", encoding="utf-8") as f: return HTMLResponse(content=f.read())
    except FileNotFoundError: raise HTTPException(status_code=404, detail="index.html not found")

@app.get("/live", response_class=HTMLResponse)
async def read_live_test():
    try:
        with open("frontend/live-test.html", "r", encoding="utf-8") as f: return HTMLResponse(content=f.read())
    except FileNotFoundError: raise HTTPException(status_code=404, detail="live-test.html not found")

@app.get("/autox", response_class=HTMLResponse)
async def read_autox_manager():
    try:
        with open("frontend/autox-manager.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<html><head><title>AutoX Manager</title></head><body><h1>AutoX Manager</h1><p>管理页面HTML文件 (autox-manager.html) 未找到。</p></body></html>", status_code=404)

@app.get("/optimization", response_class=HTMLResponse)
async def read_optimization():
    try:
        with open("frontend/optimization.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="optimization.html not found")


# --- API 端点 ---
@app.get("/api/symbols", response_model=List[str])
async def get_symbols_endpoint(): # 基本不变，依赖的binance_client方法是同步的
    try:
        # 这些调用是同步的，如果币安API响应慢，这里会阻塞
        # 理想情况下，BinanceClient 也应提供异步方法
        # 为简单起见，暂时保持同步，如果成为瓶颈再优化
        hot_symbols = await asyncio.to_thread(binance_client.get_hot_symbols_by_volume, top_n=50)
        all_symbols = await asyncio.to_thread(binance_client.get_available_symbols)
        
        combined_symbols = []; seen_symbols = set()
        if hot_symbols:
            for s in hot_symbols:
                if s not in seen_symbols: combined_symbols.append(s); seen_symbols.add(s)
        if all_symbols:
            for s in sorted(all_symbols): 
                 if s not in seen_symbols: combined_symbols.append(s); seen_symbols.add(s)
        return combined_symbols if combined_symbols else ["BTCUSDT", "ETHUSDT"]
    except Exception as e:
        print(f"获取交易对列表失败: {e}")
        return ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT"] # 返回默认列表

@app.get("/api/prediction-strategies", response_model=List[Dict[str, Any]])
async def get_prediction_strategies_endpoint(): # 基本不变，get_available_strategies 是同步的但快
    try: return [{'id': s['id'], 'name': s['name'], 'description': s['description'], 'parameters': s['parameters']} for s in get_available_strategies()]
    except Exception as e: raise HTTPException(status_code=500, detail=f"获取预测策略失败: {str(e)}")

@app.get("/api/investment-strategies", response_model=List[Dict[str, Any]])
async def get_investment_strategies_endpoint(): # 基本不变
    try: return [{'id': s['id'], 'name': s['name'], 'description': s['description'], 'parameters': s['parameters']} for s in get_available_investment_strategies()]
    except Exception as e: raise HTTPException(status_code=500, detail=f"获取投资策略失败: {str(e)}")

# --- 策略参数优化相关API ---

class OptimizationRequest(BaseModel):
    """优化请求模型"""
    symbol: str = Field(..., description="交易对")
    interval: str = Field(..., description="K线周期")
    start_date: str = Field(..., description="开始日期")
    end_date: str = Field(..., description="结束日期")
    strategy_id: str = Field(..., description="策略ID")
    strategy_params_ranges: Dict[str, Dict[str, Any]] = Field(..., description="策略参数范围")
    event_period: str = Field(default="10m", description="事件周期")
    investment_strategy_id: str = Field(default="fixed", description="投资策略ID")
    investment_strategy_params: Optional[Dict[str, Any]] = Field(default=None, description="投资策略参数")
    exclude_time_ranges: Optional[List[Dict[str, str]]] = Field(default=None, description="排除时间段")
    include_time_ranges: Optional[List[Dict[str, str]]] = Field(default=None, description="包含时间段")
    exclude_weekdays: Optional[List[int]] = Field(default=None, description="排除星期")
    max_combinations: int = Field(default=10000, description="最大组合数")
    min_trades: int = Field(default=10, description="最小交易次数")
    evaluation_weights: Optional[Dict[str, float]] = Field(default=None, description="评估权重")

@app.post("/api/optimization/start")
async def start_optimization(request: OptimizationRequest):
    """启动策略参数优化"""
    try:
        loop = asyncio.get_running_loop()
        engine = get_optimization_engine(main_loop=loop)

        # 转换请求为配置字典
        config = {
            'symbol': request.symbol,
            'interval': request.interval,
            'start_date': request.start_date,
            'end_date': request.end_date,
            'strategy_id': request.strategy_id,
            'strategy_params_ranges': request.strategy_params_ranges,
            'event_period': request.event_period,
            'investment_strategy_id': request.investment_strategy_id,
            'investment_strategy_params': request.investment_strategy_params or {'amount': 20.0},
            'exclude_time_ranges': request.exclude_time_ranges or [],
            'include_time_ranges': request.include_time_ranges or [],
            'exclude_weekdays': request.exclude_weekdays or [],
            'max_combinations': request.max_combinations,
            'min_trades': request.min_trades,
            'evaluation_weights': request.evaluation_weights
        }

        # Pass the send_update method of the optimization_manager as a callback
        optimization_id = engine.optimize_strategy(config, optimization_manager.send_update)

        return {
            'status': 'success',
            'optimization_id': optimization_id,
            'message': '优化任务已启动'
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"启动优化失败: {str(e)}")

@app.get("/api/optimization/progress/{optimization_id}")
async def get_optimization_progress(optimization_id: str):
    """获取优化进度"""
    try:
        engine = get_optimization_engine()
        progress = engine.get_optimization_progress(optimization_id)

        if progress is None:
            raise HTTPException(status_code=404, detail="优化任务不存在")

        return progress

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取进度失败: {str(e)}")

@app.post("/api/optimization/stop/{optimization_id}")
async def stop_optimization(optimization_id: str):
    """停止优化"""
    try:
        engine = get_optimization_engine()
        success = engine.stop_optimization(optimization_id)

        if success:
            return {'status': 'success', 'message': '优化已停止'}
        else:
            return {'status': 'error', 'message': '停止优化失败'}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"停止优化失败: {str(e)}")

@app.get("/api/optimization/results/{optimization_id}")
async def get_optimization_results(optimization_id: str, limit: Optional[int] = Query(None, description="结果数量限制")):
    """获取优化结果"""
    try:
        engine = get_optimization_engine()
        results = engine.get_optimization_results(optimization_id, limit)

        if 'error' in results:
            raise HTTPException(status_code=404, detail=results['error'])

        return results

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取结果失败: {str(e)}")

@app.get("/api/strategies/{strategy_id}/parameter_ranges")
async def get_strategy_parameter_ranges(strategy_id: str):
    """获取策略参数范围"""
    try:
        engine = get_optimization_engine()
        ranges = engine.get_strategy_parameter_ranges(strategy_id)

        if 'error' in ranges:
            raise HTTPException(status_code=404, detail=ranges['error'])

        return ranges

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取参数范围失败: {str(e)}")

@app.get("/api/strategies/{strategy_id}/parameter_presets")
async def get_strategy_parameter_presets(strategy_id: str):
    """获取策略参数预设"""
    try:
        engine = get_optimization_engine()
        presets = engine.get_parameter_presets(strategy_id)

        if 'error' in presets:
            raise HTTPException(status_code=404, detail=presets['error'])

        return presets

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取参数预设失败: {str(e)}")

@app.post("/api/optimization/export/{optimization_id}")
async def export_optimization_results(optimization_id: str, format: str = Query("csv", description="导出格式")):
    """导出优化结果"""
    try:
        engine = get_optimization_engine()

        # 生成文件路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optimization_results_{optimization_id}_{timestamp}.{format}"
        file_path = os.path.join("logs", filename)

        success = engine.export_optimization_results(optimization_id, file_path, format)

        if success:
            return {
                'status': 'success',
                'file_path': file_path,
                'message': '结果导出成功'
            }
        else:
            return {
                'status': 'error',
                'message': '结果导出失败'
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"导出结果失败: {str(e)}")

@app.get("/api/optimization/current")
async def get_current_optimization():
    """获取当前正在运行的优化任务"""
    try:
        engine = get_optimization_engine()
        current_task = await engine.get_current_optimization()

        if current_task:
            return {
                'status': 'success',
                'data': current_task
            }
        else:
            return {
                'status': 'success',
                'data': None,
                'message': '当前没有运行中的优化任务'
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取当前任务失败: {str(e)}")

@app.get("/api/optimization/history")
async def get_optimization_history(limit: int = Query(10, description="记录数量限制")):
    """获取优化历史记录"""
    try:
        engine = get_optimization_engine()
        history = await engine.get_optimization_history(limit)

        return {
            'status': 'success',
            'data': history
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取历史记录失败: {str(e)}")

@app.delete("/api/optimization/record/{record_id}")
async def delete_optimization_record(record_id: str):
    """删除优化记录"""
    try:
        engine = get_optimization_engine()
        success = await engine.delete_optimization_record(record_id)

        if success:
            return {
                'status': 'success',
                'message': '记录删除成功'
            }
        else:
            return {
                'status': 'error',
                'message': '记录删除失败'
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除记录失败: {str(e)}")

@app.delete("/api/optimization/{optimization_id}")
async def cleanup_optimization(optimization_id: str):
    """清理优化数据"""
    try:
        engine = get_optimization_engine()
        engine.cleanup_optimization(optimization_id)

        return {
            'status': 'success',
            'message': '优化数据已清理'
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清理数据失败: {str(e)}")

@app.post("/api/backtest") # 修改：get_historical_klines 和 Backtester.run() 移至线程池
async def run_backtest_endpoint(request: BacktestRequest):
    global strategy_parameters_config, active_backtest_tasks, backtest_cancellation_flags # 读取是线程安全的

    # 注册任务
    task_id = request.task_id or f"backtest_{int(time.time() * 1000)}"
    active_backtest_tasks[task_id] = {
        "status": "running",
        "start_time": now_utc(),
        "symbol": request.symbol,
        "interval": request.interval
    }
    backtest_cancellation_flags[task_id] = False

    try:
        start_utc = to_utc(request.start_time); end_utc = to_utc(request.end_time)
        if start_utc >= end_utc or end_utc > now_utc() or start_utc > now_utc(): 
            raise HTTPException(status_code=400, detail="回测时间范围无效。")

        # --- 修改：将 get_historical_klines 移至线程池 ---
        def _fetch_data_in_thread():
            """在单个线程中获取K线和指数价格数据，以避免多次启动线程的开销。"""
            start_ms = int(start_utc.timestamp() * 1000)
            end_ms = int(end_utc.timestamp() * 1000)
            
            kline_data = binance_client.get_historical_klines(
                request.symbol, request.interval, start_ms, end_ms
            )
            index_price_data = binance_client.get_index_price_klines(
                request.symbol, request.interval, start_ms, end_ms
            )
            return kline_data, index_price_data

        df_klines, df_index_price = await asyncio.to_thread(_fetch_data_in_thread)
        # --- 结束修改 ---

        if df_klines.empty: raise HTTPException(status_code=404, detail="未找到指定范围的K线数据。")
        
        pred_id = request.prediction_strategy_id
        # 策略参数获取逻辑 (与原版类似)
        final_pred_params = request.prediction_strategy_params
        if final_pred_params is None:
            final_pred_params = strategy_parameters_config.get("prediction_strategies", {}).get(pred_id, {})
        if not final_pred_params: 
            pred_def_info = next((s for s in get_available_strategies() if s['id'] == pred_id), None)
            if pred_def_info and 'parameters' in pred_def_info: 
                final_pred_params = {p['name']: p['default'] for p in pred_def_info['parameters']}
        
        pred_strategy_definition = next((s for s in get_available_strategies() if s['id'] == pred_id), None)
        if not pred_strategy_definition: raise HTTPException(status_code=404, detail=f"未找到预测策略ID: {pred_id}")
        
        # 实例化策略对象（同步，但通常很快）
        prediction_instance = pred_strategy_definition['class'](params=final_pred_params or {})
        
        inv_id = request.investment.investment_strategy_id
        # 投资策略参数获取逻辑 (与原版类似)
        final_inv_params = request.investment.investment_strategy_specific_params
        if final_inv_params is None:
            final_inv_params = strategy_parameters_config.get("investment_strategies", {}).get(inv_id, {})
        if not final_inv_params:
            inv_def_info = next((s for s in get_available_investment_strategies() if s['id'] == inv_id), None)
            if inv_def_info and 'parameters' in inv_def_info: 
                final_inv_params = {p['name']: p['default'] for p in inv_def_info['parameters'] if not p.get('readonly')}

        # investment_args_dict = request.investment.model_dump(exclude={'investment_strategy_id', 'investment_strategy_specific_params'})

        # --- 修改：将 Backtester 实例化和 run() 移至线程池 ---
        def _run_backtest_in_thread():
            """在单独的线程中运行回测以避免阻塞事件循环。"""
            backtester = Backtester(
                df=df_klines.copy(),
                df_index_price=df_index_price.copy(),
                strategy=prediction_instance,
                symbol=request.symbol,
                interval=request.interval,
                event_period=request.event_period,
                confidence_threshold=request.confidence_threshold,
                investment_strategy_id=inv_id,
                investment_strategy_params=final_inv_params or {},
                initial_balance=request.investment.initial_balance,
                profit_rate_pct=request.investment.profit_rate_pct,
                loss_rate_pct=request.investment.loss_rate_pct,
                min_investment_amount=request.investment.min_investment_amount,
                max_investment_amount=request.investment.max_investment_amount,
                min_trade_interval_minutes=request.investment.min_trade_interval_minutes,
                task_id=task_id,  # 传递任务ID
                cancellation_flags=backtest_cancellation_flags  # 传递取消标志字典
            )
            return backtester.run()

        results_data = await asyncio.to_thread(_run_backtest_in_thread)
        # --- 结束修改 ---

        # 标记任务完成
        active_backtest_tasks[task_id]["status"] = "completed"
        active_backtest_tasks[task_id]["end_time"] = now_utc()

        # 结果处理（同步，但通常很快）
        for pred_item_data in results_data.get('predictions', []):
            for time_key_str in ['signal_time', 'end_time_expected', 'end_time_actual']:
                if time_key_str in pred_item_data and isinstance(pred_item_data[time_key_str], datetime):
                    pred_item_data[time_key_str] = format_for_display(pred_item_data[time_key_str])
        return results_data
    except HTTPException as http_exc:
        # 标记任务失败
        if task_id in active_backtest_tasks:
            active_backtest_tasks[task_id]["status"] = "failed"
            active_backtest_tasks[task_id]["end_time"] = now_utc()
            active_backtest_tasks[task_id]["error"] = str(http_exc.detail)
        raise http_exc
    except Exception as exc:
        # 标记任务失败
        if task_id in active_backtest_tasks:
            active_backtest_tasks[task_id]["status"] = "failed"
            active_backtest_tasks[task_id]["end_time"] = now_utc()
            active_backtest_tasks[task_id]["error"] = str(exc)
        error_detail_msg = f"回测过程中发生错误: {str(exc)}"; print(f"{error_detail_msg}\n{traceback.format_exc()}"); raise HTTPException(status_code=500, detail=error_detail_msg)
    finally:
        # 清理取消标志
        if task_id in backtest_cancellation_flags:
            del backtest_cancellation_flags[task_id]

@app.post("/api/backtest/cancel")
async def cancel_backtest_endpoint(request: CancelBacktestRequest):
    global active_backtest_tasks, backtest_cancellation_flags

    task_id = request.task_id
    if task_id not in active_backtest_tasks:
        raise HTTPException(status_code=404, detail=f"回测任务 {task_id} 不存在")

    task_info = active_backtest_tasks[task_id]
    if task_info["status"] != "running":
        raise HTTPException(status_code=400, detail=f"回测任务 {task_id} 当前状态为 {task_info['status']}，无法取消")

    # 设置取消标志
    backtest_cancellation_flags[task_id] = True
    task_info["status"] = "cancelling"
    task_info["cancel_time"] = now_utc()

    return {"message": f"回测任务 {task_id} 取消请求已发送", "task_id": task_id}

@app.get("/api/live-signals") # 修改：确保线程安全地读取 live_signals
async def get_live_signals_http_endpoint():
    with live_signals_lock: # 保护对 live_signals 的读取
        # 返回副本，避免外部修改影响全局状态
        return [s.copy() for s in live_signals] 

@app.get("/api/debug-signal-data")
async def debug_signal_data(symbol: str = "BTCUSDT", interval: str = "1m", strategy_id: str = "simple_rsi"):
    """
    调试API：对比实时信号数据获取和回测数据获取的差异
    """
    try:
        current_time_utc = datetime.now(timezone.utc)

        # 1. 模拟实时信号的数据获取方式
        end_time_for_klines = current_time_utc.replace(second=0, microsecond=0)
        if interval == '5m':
            end_time_for_klines = current_time_utc.replace(minute=(current_time_utc.minute // 5) * 5, second=0, microsecond=0)
        elif interval == '15m':
            end_time_for_klines = current_time_utc.replace(minute=(current_time_utc.minute // 15) * 15, second=0, microsecond=0)
        elif interval == '1h':
            end_time_for_klines = current_time_utc.replace(minute=0, second=0, microsecond=0)

        end_time_ms = int(end_time_for_klines.timestamp() * 1000)

        live_data = await asyncio.to_thread(
            binance_client.get_historical_klines,
            symbol, interval, None, end_time_ms, 100
        )

        # 2. 模拟回测的数据获取方式（获取相同时间范围的完整数据）
        if not live_data.empty:
            start_time_for_backtest = live_data.index[0]
            end_time_for_backtest = live_data.index[-1]
            start_ms = int(start_time_for_backtest.timestamp() * 1000)
            end_ms = int(end_time_for_backtest.timestamp() * 1000)

            backtest_data = await asyncio.to_thread(
                binance_client.get_historical_klines,
                symbol, interval, start_ms, end_ms
            )
        else:
            backtest_data = pd.DataFrame()

        # 3. 对比数据
        comparison = {
            "current_time": current_time_utc.isoformat(),
            "end_time_used": end_time_for_klines.isoformat(),
            "live_data": {
                "count": len(live_data),
                "first_time": live_data.index[0].isoformat() if not live_data.empty else None,
                "last_time": live_data.index[-1].isoformat() if not live_data.empty else None,
                "last_3_closes": live_data['close'].tail(3).tolist() if not live_data.empty else []
            },
            "backtest_data": {
                "count": len(backtest_data),
                "first_time": backtest_data.index[0].isoformat() if not backtest_data.empty else None,
                "last_time": backtest_data.index[-1].isoformat() if not backtest_data.empty else None,
                "last_3_closes": backtest_data['close'].tail(3).tolist() if not backtest_data.empty else []
            },
            "data_consistency": {
                "same_count": len(live_data) == len(backtest_data),
                "same_last_time": (live_data.index[-1] == backtest_data.index[-1]) if not live_data.empty and not backtest_data.empty else False
            }
        }

        return comparison

    except Exception as e:
        return {"error": f"Debug failed: {str(e)}"}

@app.get("/api/test-signal") # 修改：保存操作改为异步
async def generate_test_signal():
    current_time = now_utc(); signal_time = current_time; end_time = current_time + timedelta(minutes=10)
    symbol = random.choice(["BTCUSDT", "ETHUSDT", "BNBUSDT"])

    try:
        # get_latest_price 是同步的，如果币安API慢，这里会阻塞
        # 理想情况也应改为异步
        price_val = await asyncio.to_thread(binance_client.get_latest_price, symbol)
    except:
        price_val = random.uniform(100, 70000)

@app.get("/api/test-price-update/{signal_id}")
async def test_price_update(signal_id: str):
    """测试价格更新功能"""
    try:
        # 查找信号
        signal_found = None
        with live_signals_lock:
            for signal in live_signals:
                if signal.get('id') == signal_id:
                    signal_found = signal
                    break

        if not signal_found:
            return {"error": f"Signal {signal_id} not found"}

        # 手动触发价格更新
        signal_time_str = signal_found.get('signal_time')
        if signal_time_str:
            signal_time_dt = parse_frontend_datetime(signal_time_str)
            if not signal_time_dt.tzinfo:
                signal_time_dt = signal_time_dt.replace(tzinfo=timezone.utc)
            else:
                signal_time_dt = signal_time_dt.astimezone(timezone.utc)

            # 创建异步任务来更新价格
            asyncio.create_task(update_signal_entry_price(
                signal_id,
                signal_found.get('symbol'),
                signal_time_dt,
                signal_found.get('interval')
            ))

            return {"message": f"Price update task created for signal {signal_id}"}
        else:
            return {"error": "Signal time not found"}

    except Exception as e:
        return {"error": f"Error: {str(e)}"}

    pred_id_test = "simple_rsi"; 
    # 读取 strategy_parameters_config 是安全的，因为它通常只在启动时修改
    pred_params_test = strategy_parameters_config.get("prediction_strategies", {}).get(pred_id_test, {"rsi_period": 14, "rsi_overbought": 70, "rsi_oversold": 30})
    
    test_signal_data = {
        'id': f"TEST_{symbol}_{pred_id_test}_{int(time.time())}_{random.randint(100,999)}", 'symbol': symbol, 'interval': "1m", 'prediction_strategy_id': pred_id_test, 'prediction_strategy_params': pred_params_test,
        'signal_time': format_for_display(signal_time),
        'signal': random.choice([1, -1]), 'confidence': random.uniform(60, 95), 'signal_price': price_val, 'event_period': "10m",
        'expected_end_time': format_for_display(end_time), 'investment_amount': 20.0, 'profit_rate_pct': 80.0, 'loss_rate_pct': 100.0,
        'actual_end_price': None, 'price_change_pct': None, 'result': None, 'pnl_pct': None, 'actual_profit_loss_amount': None, 'verified': False, 'verify_time': None,
        'origin_config_id': 'test_signal_broadcast_all', 'autox_triggered_info': None
    }
    
    with live_signals_lock: # 保护对 live_signals 的写入
        live_signals.append(test_signal_data)
    
    await save_live_signals_async() # 调用已修改的异步保存函数
    
    # 广播操作已经是异步的
    await manager.broadcast_json( {"type": "new_signal", "data": test_signal_data}, filter_func=lambda conn: True )
    return {"status": "success", "message": "测试信号已生成并广播给所有连接的客户端", "signal": test_signal_data}

@app.get("/api/load_all_strategy_parameters", response_model=Dict[str, Any])
async def load_all_strategy_parameters_endpoint(): # 基本不变，读取全局变量是安全的
    global strategy_parameters_config; 
    # 返回副本以避免外部修改
    return {
        "prediction_strategies": strategy_parameters_config.get("prediction_strategies", {}).copy(),
        "investment_strategies": strategy_parameters_config.get("investment_strategies", {}).copy()
    }

@app.post("/api/save_strategy_parameter_set") # 修改：保存操作改为异步
async def save_strategy_parameter_set_endpoint(param_set: StrategyParameterSet):
    global strategy_parameters_config
    try:
        # 修改 strategy_parameters_config 是同步的，但很快
        if param_set.strategy_type == "prediction": 
            strategy_parameters_config["prediction_strategies"][param_set.strategy_id] = param_set.params
        elif param_set.strategy_type == "investment": 
            strategy_parameters_config["investment_strategies"][param_set.strategy_id] = param_set.params
        else: raise HTTPException(status_code=400, detail="无效的 strategy_type。必须是 'prediction' 或 'investment'。")
        
        await save_strategy_parameters_to_file() # 调用已修改的异步保存函数
        return {"status": "success", "message": "策略参数已保存"}
    except HTTPException as http_exc: raise http_exc
    except Exception as e: print(f"保存策略参数失败: {e}\n{traceback.format_exc()}"); raise HTTPException(status_code=500, detail=f"保存策略参数失败: {str(e)}")

# --- AutoX管理相关的API端点 ---
@app.get("/api/autox/trade_logs", response_model=List[AutoXTradeLogEntry])
async def get_autox_trade_logs_endpoint(limit: int = Query(50, ge=1, le=MAX_AUTOX_LOG_ENTRIES)):
    with autox_trade_logs_lock: # 保护对 autox_trade_logs 的读取
        # 先转换为 AutoXTradeLogEntry 对象列表，再序列化
        log_objects = [AutoXTradeLogEntry(**log_dict) for log_dict in autox_trade_logs]
        # 在锁外进行排序和切片可能更好，如果日志非常多
        # 但如果排序很快，放在锁内也可以接受，这里假设它很快
        sorted_logs = sorted(log_objects, key=lambda x: x.timestamp, reverse=True)
        # model_dump 是同步的，但对于单个对象通常很快
        return [log.model_dump(mode='json') for log in sorted_logs[:limit]]


@app.post("/api/autox/clients/{client_id}/send_test_command", response_model=Dict[str, Any])
async def send_test_command_to_autox_client(client_id: str, command_type: str = Query("test_echo", description="测试指令类型")):
    target_ws: Optional[WebSocket] = None
    with autox_clients_lock: # 保护对 active_autox_clients 的读取
        for ws, info_dict in active_autox_clients.items(): # 迭代副本以防修改
            if info_dict.get("client_id") == client_id:
                target_ws = ws
                break
    
    if not target_ws:
        raise HTTPException(status_code=404, detail=f"未找到 Client ID 为 {client_id} 的活动AutoX客户端。")

    test_payload = {"message": f"这是一个来自服务器的测试指令 ({command_type})", "timestamp": format_for_display(now_utc())}
    command_to_send = {"type": command_type, "payload": test_payload}
    
    try:
        await _send_autox_command(target_ws, command_to_send) # _send_autox_command 已经是异步的
        return {"status": "success", "message": f"测试指令 '{command_type}' 已发送给客户端 {client_id}。"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"发送测试指令给客户端 {client_id} 失败: {str(e)}")
    

@app.post("/api/autox/clients/{client_id}/trigger_trade_command", response_model=Dict[str, Any])
async def trigger_trade_command_for_autox_client(client_id: str, trade_details: TriggerAutoXTradePayload):
    target_ws: Optional[WebSocket] = None
    client_info_snapshot_dict: Optional[Dict[str, Any]] = None 
    
    with autox_clients_lock: # 保护对 active_autox_clients 和 persistent_autox_clients_data 的修改
        # 首先找到对应的 WebSocket
        for ws_iter, info_dict_iter in active_autox_clients.items():
            if info_dict_iter.get("client_id") == client_id:
                target_ws = ws_iter
                client_info_snapshot_dict = info_dict_iter # 这是 active_autox_clients 中的字典引用
                break
        
        if not target_ws or not client_info_snapshot_dict:
            raise HTTPException(status_code=404, detail=f"未找到 Client ID 为 {client_id} 的活动AutoX客户端。")

        if client_info_snapshot_dict.get("status") != "idle":
            raise HTTPException(status_code=409, detail=f"客户端 {client_id} 当前不处于 idle 状态 (当前状态: {client_info_snapshot_dict.get('status')})，无法发送新指令。")
        
        # 更新状态
        signal_id_for_status = trade_details.signal_id or f"test_trigger_{uuid.uuid4().hex[:8]}"
        client_info_snapshot_dict['status'] = 'processing_trade' 
        client_info_snapshot_dict['last_signal_id'] = signal_id_for_status
        
        # 同步更新 persistent_autox_clients_data
        if client_id in persistent_autox_clients_data:
            persistent_autox_clients_data[client_id]['status'] = 'processing_trade'
            persistent_autox_clients_data[client_id]['last_signal_id'] = signal_id_for_status
    
    # 如果成功更新了状态，则发送指令并保存
    if target_ws: # 再次检查，以防万一
        signal_id_to_use = trade_details.signal_id or f"test_trigger_{uuid.uuid4().hex[:8]}"
        command_payload = {
            "signal_id": signal_id_to_use, "symbol": trade_details.symbol,
            "direction": trade_details.direction, "amount": str(trade_details.amount), # AutoX需要字符串金额
            "timestamp": format_for_display(now_utc())
        }
        command_to_send = {"type": "execute_trade", "payload": command_payload}
        
        try:
            await _send_autox_command(target_ws, command_to_send) # 异步发送
            # 日志记录 (同步，但通常很快)
            log_entry_data_trigger = {
                "client_id": client_id, "signal_id": signal_id_to_use,
                "command_type": "test_triggered_execute_trade", "command_payload": command_payload,
                "status": "test_command_sent_to_client",
                "details": f"测试触发的 execute_trade 指令已发送给客户端 {client_id}。",
            }
            with autox_trade_logs_lock: # 保护写入
                autox_trade_logs.append(AutoXTradeLogEntry(**log_entry_data_trigger).model_dump(mode='json'))
                if len(autox_trade_logs) > MAX_AUTOX_LOG_ENTRIES: autox_trade_logs.pop(0)

            await save_autox_clients_to_file() # 异步保存状态更改
            await debounced_broadcast_autox_clients_status() # 广播状态更新

            return {"status": "success", "message": f"'execute_trade' 指令已作为测试发送给客户端 {client_id}。", "sent_command": command_to_send}
        except Exception as e:
            # 如果发送失败，尝试回滚状态
            with autox_clients_lock:
                if client_id in persistent_autox_clients_data:
                    persistent_autox_clients_data[client_id]['status'] = 'idle'
                    persistent_autox_clients_data[client_id].pop('last_signal_id', None)
                if target_ws in active_autox_clients: # 检查 ws 是否仍然是键
                     active_autox_clients[target_ws]['status'] = 'idle'
                     active_autox_clients[target_ws].pop('last_signal_id', None)

            await save_autox_clients_to_file() # 保存回滚后的状态
            await debounced_broadcast_autox_clients_status()
            raise HTTPException(status_code=500, detail=f"发送 'execute_trade' 指令给客户端 {client_id} 失败: {str(e)}")
    else: # 理论上不会到这里，因为前面已经检查过
        raise HTTPException(status_code=404, detail="未能定位到目标客户端的WebSocket连接。")


@app.post("/api/autox/clients/{client_id}/notes", response_model=AutoXClientInfo)
async def update_client_notes_endpoint(client_id: str, notes_payload: ClientNotesPayload):
    """更新指定AutoX客户端的备注。"""
    updated_client_info_dict: Optional[Dict[str, Any]] = None
    found_client = False

    with autox_clients_lock: # 保护对 persistent_autox_clients_data 和 active_autox_clients 的修改
        if client_id in persistent_autox_clients_data:
            persistent_autox_clients_data[client_id]["notes"] = notes_payload.notes
            persistent_autox_clients_data[client_id]["last_seen"] = now_utc().isoformat() # 更新last_seen
            updated_client_info_dict = persistent_autox_clients_data[client_id].copy()
            found_client = True

            # 如果客户端也在线，同时更新 active_autox_clients 中的信息
            for ws, active_info in active_autox_clients.items():
                if active_info.get("client_id") == client_id:
                    active_info["notes"] = notes_payload.notes
                    active_info["last_seen"] = now_utc().isoformat()
                    # 不需要再从 active_info 更新 updated_client_info_dict，因为 persistent 的已经更新了
                    break 
        
    if not found_client:
        raise HTTPException(status_code=404, detail=f"未找到 Client ID 为 {client_id} 的AutoX客户端记录。")
    
    print(f"客户端 {client_id} 的备注已更新为: '{notes_payload.notes}'")
    await save_autox_clients_to_file() # 异步保存
    await debounced_broadcast_autox_clients_status() # 广播状态更新（可能包含备注变化）

    # 返回 Pydantic 模型确保响应格式正确
    return AutoXClientInfo(**updated_client_info_dict) if updated_client_info_dict else None


@app.delete("/api/autox/clients/{client_id}", response_model=Dict[str, Any])
async def delete_autox_client_endpoint(client_id: str):
    """
    从持久化存储中删除一个AutoX客户端记录，并断开其活动的WebSocket连接。
    """
    global persistent_autox_clients_data, active_autox_clients
    
    ws_to_close: Optional[WebSocket] = None
    client_found_and_deleted = False

    with autox_clients_lock:
        # 1. 从持久化数据中移除
        if client_id in persistent_autox_clients_data:
            del persistent_autox_clients_data[client_id]
            client_found_and_deleted = True
            logger.info(f"已从持久化存储中删除客户端: {client_id}")

        # 2. 查找并标记要关闭的活动WebSocket连接
        for ws, active_info in active_autox_clients.items():
            if active_info.get("client_id") == client_id:
                ws_to_close = ws
                break # 找到后即可退出循环

    if not client_found_and_deleted:
        raise HTTPException(status_code=404, detail=f"未在持久化存储中找到 Client ID 为 {client_id} 的客户端记录。")

    # 3. 在锁外执行可能耗时的操作 (关闭连接)
    if ws_to_close:
        logger.info(f"准备断开已删除客户端 {client_id} 的活动WebSocket连接。")
        try:
            # 从管理器中断开连接，这将阻止未来的广播
            autox_manager.disconnect(ws_to_close)
            # 从活动客户端字典中移除
            with autox_clients_lock:
                active_autox_clients.pop(ws_to_close, None)
            # 尝试优雅地关闭连接
            await ws_to_close.close(code=1000, reason="客户端记录已被管理员删除")
            logger.info(f"已成功关闭客户端 {client_id} 的WebSocket连接。")
        except Exception as e:
            logger.error(f"关闭客户端 {client_id} 的WebSocket连接时出错: {e}", exc_info=True)
            # 即使关闭失败，我们仍然继续执行，因为记录已经被删除了

    # 4. 保存更改并广播更新
    await save_autox_clients_to_file()
    await debounced_broadcast_autox_clients_status()

    return {"status": "success", "message": f"客户端 {client_id} 已成功删除。"}


@app.post("/api/live-signals/delete-batch", response_model=Dict[str, Any])
async def delete_live_signals_batch(request: DeleteSignalsRequest):
    """异步批量删除实时信号，并在删除前调整相关配置的余额和盈亏。"""
    global live_signals, running_live_test_configs, active_live_test_config_id, websocket_to_config_id_map, global_running_balance, global_running_balance_lock
    # logger 实例应该在模块级别可用
    
    deleted_count = 0
    
    if not request.signal_ids:
        raise HTTPException(status_code=400, detail="signal_ids 列表不能为空。")

    ids_to_delete_set = set(request.signal_ids)
    
    signals_to_keep = []
    deleted_signals_info = [] # 存储被删除信号的完整信息以进行后续处理
    
    # 步骤1: 识别要删除的信号并收集其信息
    with live_signals_lock:
        current_live_signals_copy = [s.copy() for s in live_signals]
        
    for signal_in_copy in current_live_signals_copy:
        if signal_in_copy.get('id') in ids_to_delete_set:
            deleted_signals_info.append(signal_in_copy)

    if not deleted_signals_info:
        not_found_count_initial = len(request.signal_ids)
        return {"status": "warning", "message": f"请求删除的 {not_found_count_initial} 个信号均未找到。", "deleted_count": 0, "not_found_count": not_found_count_initial}

    # 步骤2: 处理余额和利润损失调整
    configs_to_update_broadcast: Dict[str, Dict[str, Any]] = {}

    for signal_data in deleted_signals_info:
        config_id = signal_data.get('origin_config_id')
        if not config_id:
            logger.warning(f"Signal {signal_data.get('id')} missing origin_config_id for deletion balance adjustment.")
            continue

        investment_amount = signal_data.get('investment_amount', 0.0)
        if not isinstance(investment_amount, (int, float)):
            try:
                investment_amount = float(investment_amount)
            except (ValueError, TypeError):
                logger.warning(f"Signal {signal_data.get('id')} has invalid investment_amount '{investment_amount}', using 0.")
                investment_amount = 0.0
        
        actual_profit_loss = signal_data.get('actual_profit_loss_amount')
        if actual_profit_loss is not None and not isinstance(actual_profit_loss, (int, float)):
             try:
                actual_profit_loss = float(actual_profit_loss)
             except (ValueError, TypeError):
                logger.warning(f"Signal {signal_data.get('id')} has invalid actual_profit_loss_amount '{actual_profit_loss}', treating as None.")
                actual_profit_loss = None

        is_verified_and_has_result = signal_data.get('verified', False) and signal_data.get('result') is not None

        with running_live_test_configs_lock:
            if config_id in running_live_test_configs:
                config_entry = running_live_test_configs[config_id]
                original_balance = config_entry.get('current_balance', 0.0)
                original_pnl = config_entry.get('total_profit_loss_amount', 0.0)
                balance_change_for_this_signal_net = 0.0

                if is_verified_and_has_result and actual_profit_loss is not None:
                    logger.info(f"Processing validated signal {signal_data.get('id')} for deletion (Config: {config_id}). Investment: {investment_amount}, PnL: {actual_profit_loss}.")
                    
                    config_entry['total_profit_loss_amount'] -= actual_profit_loss
                    config_entry['current_balance'] -= actual_profit_loss
                    balance_change_for_this_signal_net -= actual_profit_loss
                    
                    config_entry['current_balance'] += investment_amount
                    balance_change_for_this_signal_net += investment_amount
                    
                    logger.info(f"Config {config_id} balance update (validated signal {signal_data.get('id')} deleted). Original Balance: {original_balance:.2f}, Original Total PnL: {original_pnl:.2f}. New Balance: {config_entry['current_balance']:.2f}, New Total PnL: {config_entry['total_profit_loss_amount']:.2f}. Net balance change from this signal: {balance_change_for_this_signal_net:.2f}")

                elif not is_verified_and_has_result and investment_amount > 0: # Unverified but investment was deducted
                    logger.info(f"Processing unvalidated signal {signal_data.get('id')} for deletion (Config: {config_id}). Refunding investment: {investment_amount}.")
                    
                    config_entry['current_balance'] += investment_amount
                    balance_change_for_this_signal_net += investment_amount

                    logger.info(f"Config {config_id} balance update (unvalidated signal {signal_data.get('id')} deleted). Original Balance: {original_balance:.2f}. New Balance: {config_entry['current_balance']:.2f}. Net balance change from this signal: {balance_change_for_this_signal_net:.2f}")
                else:
                    logger.info(f"Signal {signal_data.get('id')} (Config: {config_id}) requires no balance adjustment on deletion. Verified: {is_verified_and_has_result}, PnL: {actual_profit_loss}, Investment: {investment_amount}")

                if balance_change_for_this_signal_net != 0 or (is_verified_and_has_result and actual_profit_loss is not None):
                    if config_id not in configs_to_update_broadcast:
                        configs_to_update_broadcast[config_id] = {
                            "new_balance": config_entry['current_balance'],
                            "total_profit_loss_amount": config_entry['total_profit_loss_amount'],
                            "accumulated_balance_change_from_delete": 0.0
                        }
                    configs_to_update_broadcast[config_id]["new_balance"] = config_entry['current_balance']
                    configs_to_update_broadcast[config_id]["total_profit_loss_amount"] = config_entry['total_profit_loss_amount']
                    configs_to_update_broadcast[config_id]["accumulated_balance_change_from_delete"] += balance_change_for_this_signal_net
            else:
                logger.warning(f"Config ID {config_id} not found when trying to adjust balance for signal {signal_data.get('id')}.")
    
    current_active_config_id_local = None
    with active_live_test_config_lock:
        current_active_config_id_local = active_live_test_config_id
    
    if current_active_config_id_local and current_active_config_id_local in configs_to_update_broadcast:
        logger.info(f"Active config {current_active_config_id_local} balance/PnL updated due to signal deletion, saving.")
        await save_active_test_config()

    for signal_item in current_live_signals_copy:
        if signal_item.get('id') not in ids_to_delete_set:
            signals_to_keep.append(signal_item)
        else:
            deleted_count +=1

    not_found_count = len(request.signal_ids) - deleted_count
    if not_found_count < 0: not_found_count = 0

    if deleted_count > 0:
        with live_signals_lock:
            live_signals = signals_to_keep
        
        await save_live_signals_async()
            
        for config_id_to_bc, update_data in configs_to_update_broadcast.items():
            balance_payload = {
                "type": "config_specific_balance_update",
                "data": {
                    "config_id": config_id_to_bc,
                    "new_balance": round(update_data["new_balance"], 2),
                    "last_pnl_amount": round(update_data["accumulated_balance_change_from_delete"], 2),
                    "total_profit_loss_amount": round(update_data["total_profit_loss_amount"], 2)
                }
            }
            logger.info(f"Broadcasting balance update for config {config_id_to_bc} due to signal deletion: {balance_payload['data']}")
            await manager.broadcast_json(
                balance_payload,
                filter_func=lambda c: websocket_to_config_id_map.get(c) == config_id_to_bc
            )

        stats_payload_data_for_broadcast = []
        with live_signals_lock:
            stats_payload_data_for_broadcast = [s.copy() for s in live_signals]

        verified_list_global = [s for s in stats_payload_data_for_broadcast if s.get('verified')]
        total_verified_global = len(verified_list_global)
        total_correct_global = sum(1 for s in verified_list_global if s.get('result'))
        
        current_total_actual_profit_loss_amount_global = 0.0
        if verified_list_global:
            valid_pnl_amounts = [s.get('actual_profit_loss_amount', 0.0) for s in verified_list_global if s.get('actual_profit_loss_amount') is not None]
            if valid_pnl_amounts:
                 current_total_actual_profit_loss_amount_global = sum(valid_pnl_amounts)

        current_total_pnl_pct_sum_global = 0.0
        if verified_list_global:
            valid_pnl_pcts = [s.get('pnl_pct', 0.0) for s in verified_list_global if s.get('pnl_pct') is not None]
            if valid_pnl_pcts:
                current_total_pnl_pct_sum_global = sum(valid_pnl_pcts)
        
        average_pnl_pct_global = current_total_pnl_pct_sum_global / total_verified_global if total_verified_global > 0 else 0
        
        current_global_balance_val = 0.0
        with global_running_balance_lock:
             current_global_balance_val = global_running_balance

        stats_payload = {
            "total_signals": len(stats_payload_data_for_broadcast),
            "total_verified": total_verified_global,
            "total_correct": total_correct_global,
            "win_rate": round(total_correct_global / total_verified_global * 100 if total_verified_global > 0 else 0, 2),
            "total_pnl_pct": round(current_total_pnl_pct_sum_global, 2),
            "average_pnl_pct": round(average_pnl_pct_global, 2),
            "total_profit_amount": round(current_total_actual_profit_loss_amount_global, 2),
            "current_balance": round(current_global_balance_val, 2)
        }
        await manager.broadcast_json({"type": "stats_update", "data": stats_payload})
        await manager.broadcast_json({"type": "signals_deleted_notification", "data": {"deleted_ids": list(ids_to_delete_set), "message": f"部分信号已删除。"}})

        return {"status": "success", "message": f"成功删除 {deleted_count} 个信号。" + (f" {not_found_count} 个请求的信号未找到。" if not_found_count > 0 else ""), "deleted_count": deleted_count, "not_found_count": not_found_count}
    
    elif not_found_count > 0 :
         return {"status": "warning", "message": f"请求删除的 {len(request.signal_ids)} 个信号均未找到。", "deleted_count": 0, "not_found_count": not_found_count}
    else:
        return {"status": "info", "message": "没有信号被删除（可能请求的ID列表为空或所有ID均未找到）。", "deleted_count": 0, "not_found_count": 0}
 
@app.get("/api/test-signal-enhanced", tags=["Testing"]) # 可以用新名字或修改原有的
async def generate_enhanced_test_signal(
    target_config_id: Optional[str] = Query(None, description="如果提供，则信号关联此config_id并尝试基于此配置触发AutoX"),
    symbol: Optional[str] = Query(None, description="指定交易对，默认随机"),
    direction: Optional[int] = Query(None, description="信号方向 (1 for up, -1 for down), 默认随机"),
    confidence: Optional[float] = Query(None, description="信号置信度, 默认随机 (60-95)"),
    amount: Optional[float] = Query(None, description="投资金额, 默认20.0"),
    event_period: Optional[str] = Query("10m", description="事件周期, e.g., 10m, 30m"),
    trigger_autox_now: bool = Query(False, description="是否立即尝试触发AutoX逻辑")
):
    global running_live_test_configs, live_signals, strategy_parameters_config, active_autox_clients, binance_client # 确保这些全局变量可用

    current_time = now_utc()
    signal_time_dt = current_time
    
    # 确定事件周期和结束时间
    event_period_minutes_map = {'10m': 10, '30m': 30, '1h': 60, '1d': 1440}
    actual_event_period_minutes = event_period_minutes_map.get(event_period, 10)
    expected_end_time_dt = signal_time_dt + timedelta(minutes=actual_event_period_minutes)

    final_symbol = symbol or random.choice(["BTCUSDT", "ETHUSDT", "BNBUSDT"])
    final_direction = direction if direction in [1, -1] else random.choice([1, -1])
    final_confidence = confidence if confidence is not None and 0 <= confidence <= 100 else random.uniform(60, 95)
    final_amount = amount if amount is not None and amount > 0 else 20.0

    try:
        price_val = binance_client.get_latest_price(final_symbol)
    except Exception:
        price_val = random.uniform(100, 70000) # Fallback

    # 默认的预测策略（可以不那么重要，因为我们是手动触发）
    pred_id_test = "manual_test_signal"
    pred_params_test = {}

    # origin_config_id 逻辑
    actual_origin_config_id = 'test_signal_broadcast_all' # 默认
    live_test_config_data_for_autox = None # 用于AutoX触发的配置数据
    autox_enabled_for_this_signal = False

    if target_config_id:
        with running_live_test_configs_lock:
            if target_config_id in running_live_test_configs:
                actual_origin_config_id = target_config_id # 将信号与特定配置关联
                # 获取此配置的副本以检查 autox_enabled 状态
                live_test_config_data_for_autox = running_live_test_configs[target_config_id].copy()
                live_test_config_data_for_autox['_config_id'] = target_config_id # 确保_config_id存在
                
                # 从配置中获取autox_enabled状态和投资设置
                autox_enabled_for_this_signal = live_test_config_data_for_autox.get('autox_enabled', False)
                
                # 如果需要，可以基于live_test_config_data_for_autox中的investment_settings重新计算final_amount
                # 为了简化，我们这里仍然使用传入的final_amount或默认值
                # 但如果要精确模拟，需要复制handle_kline_data中的投资计算逻辑
                inv_settings_from_config = live_test_config_data_for_autox.get("investment_settings")
                if inv_settings_from_config:
                    try:
                        live_inv_model = InvestmentStrategySettings(**inv_settings_from_config)
                        # 此处可以加入更复杂的金额计算，如果amount参数未提供
                        if amount is None: # 如果外部没有指定amount，则尝试根据策略计算
                             # ... (这里需要完整的投资策略计算逻辑，如我之前在 simulate_signal_for_autox 中写的)
                             # 为了这个示例的简洁性，我们假设如果amount是None，则使用配置中的基础amount
                            calculated_amount = live_inv_model.amount # 简化
                            final_amount = max(live_inv_model.minAmount, min(live_inv_model.maxAmount, calculated_amount))
                        else: # 如果外部指定了amount，则优先使用，但仍需通过min/maxAmount校验
                            final_amount = max(live_inv_model.minAmount, min(live_inv_model.maxAmount, final_amount))

                    except Exception as e_inv:
                        print(f"增强测试信号：从配置计算投资金额时出错: {e_inv}")
                        # 保持使用之前的 final_amount
            else:
                print(f"警告：提供的 target_config_id '{target_config_id}' 未在运行中的配置中找到。信号将不会与任何特定配置关联以进行AutoX。")


    test_signal_data = {
        'id': f"ENH_TEST_{final_symbol}_{pred_id_test}_{int(time.time())}_{random.randint(100,999)}",
        'symbol': final_symbol,
        'interval': "1m", # 对于测试信号，这个可能不那么重要
        'prediction_strategy_id': pred_id_test,
        'prediction_strategy_params': pred_params_test,
        'signal_time': format_for_display(signal_time_dt),
        'signal': final_direction,
        'confidence': round(final_confidence, 2),
        'signal_price': round(price_val, 4),
        'event_period': event_period,
        'expected_end_time': format_for_display(expected_end_time_dt),
        'investment_amount': round(final_amount, 2),
        'profit_rate_pct': 80.0, # 可以从配置或参数获取
        'loss_rate_pct': 100.0,  # 可以从配置或参数获取
        'verified': False,
        'origin_config_id': actual_origin_config_id,
        'autox_triggered_info': None
    }

    autox_attempt_log = {"triggered_by_api_flag": trigger_autox_now, "config_autox_enabled": autox_enabled_for_this_signal}

    if trigger_autox_now and autox_enabled_for_this_signal and live_test_config_data_for_autox:
        # 只有当API明确要求触发，并且关联的配置也启用了AutoX时，才尝试
        # 这里的逻辑与 handle_kline_data 和我之前写的 simulate_signal_for_autox 中的 AutoX 触发部分相同
        AUTOX_GLOBAL_ENABLED = True # 假设全局启用

        if AUTOX_GLOBAL_ENABLED:
            target_autox_client_ws = None
            autox_attempt_log["status"] = "looking_for_client"
            with autox_clients_lock:
                for ws_client, client_info in active_autox_clients.items():
                    if client_info.get('status') == 'idle' and \
                       test_signal_data['symbol'] in client_info.get('supported_symbols', []):
                        target_autox_client_ws = ws_client
                        client_info['status'] = 'processing_trade'
                        client_info['last_signal_id'] = test_signal_data['id']
                        autox_attempt_log["client_found"] = client_info.get('client_id')
                        break
            
            if target_autox_client_ws:
                trade_command_payload = {
                    "signal_id": test_signal_data['id'],
                    "symbol": test_signal_data['symbol'],
                    "direction": "up" if test_signal_data['signal'] == 1 else "down",
                    "amount": str(test_signal_data['investment_amount']),
                    "timestamp": test_signal_data['signal_time']
                }
                command_to_send = {"type": "execute_trade", "payload": trade_command_payload}
                try:
                    await _send_autox_command(target_autox_client_ws, command_to_send)
                    client_id_for_signal = active_autox_clients.get(target_autox_client_ws, {}).get('client_id')
                    test_signal_data['autox_triggered_info'] = {
                        "client_id": client_id_for_signal,
                        "sent_at": format_for_display(now_utc()),
                        "status": "command_sent"
                    }
                    autox_attempt_log["status"] = "command_sent"
                    autox_attempt_log["client_id"] = client_id_for_signal
                except Exception as e_autox_send_test:
                    autox_attempt_log["status"] = "send_command_error"
                    autox_attempt_log["error"] = str(e_autox_send_test)
                    print(f"增强测试信号：发送AutoX指令失败: {e_autox_send_test}")
                    with autox_clients_lock: # 重置客户端状态
                        if target_autox_client_ws in active_autox_clients:
                            active_autox_clients[target_autox_client_ws]['status'] = 'idle'
                            active_autox_clients[target_autox_client_ws].pop('last_signal_id', None)
            else:
                test_signal_data['autox_triggered_info'] = {"status": "no_available_client_for_broadcast"}
                autox_attempt_log["status"] = "no_available_client"
        else: # AUTOX_GLOBAL_ENABLED is False
             autox_attempt_log["status"] = "autox_globally_disabled"
    
    elif trigger_autox_now and not autox_enabled_for_this_signal:
        autox_attempt_log["status"] = "not_triggered_config_autox_disabled"
        print(f"增强测试信号：API请求触发AutoX，但目标配置 {target_config_id} 未启用AutoX。")
    elif trigger_autox_now and not live_test_config_data_for_autox:
         autox_attempt_log["status"] = "not_triggered_no_valid_config_for_autox"
         print(f"增强测试信号：API请求触发AutoX，但未找到有效的 target_config_id 或其未启用AutoX。")


    with live_signals_lock: live_signals.append(test_signal_data)
    await save_live_signals_async()

    # 广播逻辑
    if actual_origin_config_id == 'test_signal_broadcast_all':
        # 如果没有目标配置，或目标配置无效，则广播给所有UI
        await manager.broadcast_json( {"type": "new_signal", "data": test_signal_data}, filter_func=lambda conn: True )
    else:
        # 如果有关联的配置，则只广播给监听该配置的UI
        await manager.broadcast_json(
            {"type": "new_signal", "data": test_signal_data},
            filter_func=lambda c: websocket_to_config_id_map.get(c) == actual_origin_config_id
        )

    return {
        "status": "success",
        "message": "增强测试信号已生成，并根据参数尝试了AutoX触发。",
        "signal": test_signal_data,
        "autox_attempt_details": autox_attempt_log
    }

# --- 统一的优雅关闭机制 ---
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("服务正在关闭，开始执行优雅的清理操作...")
    shutdown_event_async.set() # 1. 通知所有异步后台任务停止

    # 2. 取消 broadcast_autox_clients_status 的防抖任务 (如果存在)
    global _debounce_task_autox_status
    if _debounce_task_autox_status and not _debounce_task_autox_status.done():
        logger.info("正在取消 broadcast_autox_clients_status 的防抖任务...")
        _debounce_task_autox_status.cancel()
        try:
            await _debounce_task_autox_status
        except asyncio.CancelledError:
            logger.info("broadcast_autox_clients_status 的防抖任务已成功取消。")
        except Exception as e:
            logger.error(f"等待取消的 broadcast_autox_clients_status 防抖任务时发生错误: {e}", exc_info=True)

    # 新增：向活动的 AutoX 客户端发送关闭通知
    logger.info("准备向活动的 AutoX 客户端发送关闭通知...")
    shutdown_notification_tasks = []
    notification_message = {
        "type": "server_shutting_down",
        "payload": {
            "message": "服务器正在关闭，请在约60秒后尝试重连。",
            "reconnect_delay_seconds": 60
        }
    }

    active_ws_to_notify_with_ids = []
    # 使用 active_autox_clients 来获取 WebSocket 对象和 client_id
    # 需要 autox_clients_lock 来确保线程安全
    with autox_clients_lock:
        for ws_client, client_info in active_autox_clients.items():
            # client_info 是一个字典，例如 {'client_id': 'some_id', ...}
            client_id_for_log = client_info.get('client_id', '未知')
            active_ws_to_notify_with_ids.append({"ws": ws_client, "id": client_id_for_log})

    if active_ws_to_notify_with_ids:
        logger.info(f"将向 {len(active_ws_to_notify_with_ids)} 个 AutoX 客户端发送关闭通知。")
        for client_data in active_ws_to_notify_with_ids:
            ws_client_conn = client_data["ws"]
            client_id_for_log = client_data["id"]
            
            async def send_notification_wrapper(ws, msg, cid):
                try:
                    # 直接使用 ws.send_json() 并设置超时
                    await asyncio.wait_for(ws.send_json(msg), timeout=3.0) # 3秒超时
                    logger.info(f"已向 AutoX 客户端 {cid} 发送关闭通知。")
                except asyncio.TimeoutError:
                    logger.warning(f"向 AutoX 客户端 {cid} 发送关闭通知超时。")
                except WebSocketDisconnect: # WebSocketDisconnect 已在文件顶部导入
                    logger.warning(f"向 AutoX 客户端 {cid} 发送关闭通知时发现连接已断开。")
                except Exception as e_send_notify:
                    logger.error(f"向 AutoX 客户端 {cid} 发送关闭通知失败: {e_send_notify}")

            shutdown_notification_tasks.append(send_notification_wrapper(ws_client_conn, notification_message, client_id_for_log))
    
        if shutdown_notification_tasks:
            # return_exceptions=True 确保一个任务的失败不会中止其他任务
            results = await asyncio.gather(*shutdown_notification_tasks, return_exceptions=True)
            successful_sends = 0
            failed_sends = 0
            for res_idx, res_item in enumerate(results):
                cid_log = active_ws_to_notify_with_ids[res_idx]["id"]
                if isinstance(res_item, Exception):
                    failed_sends +=1
                    # 错误已经在 wrapper 中记录，这里可以记录一个聚合信息
                    logger.debug(f"发送关闭通知给 {cid_log} 的任务返回异常: {res_item}")
                else:
                    successful_sends +=1
            logger.info(f"所有 AutoX 客户端关闭通知发送尝试完成。成功: {successful_sends}, 失败: {failed_sends} (总计: {len(results)})")
    else:
        logger.info("没有活动的 AutoX 客户端需要发送关闭通知。")
    # --- 结束新增 ---

    # 3. 停止策略优化引擎
    try:
        logger.info("正在停止策略优化引擎...")
        engine = get_optimization_engine()
        # 停止所有正在运行的优化任务
        engine.stop_all_optimizations()
        logger.info("策略优化引擎已停止。")
    except Exception as e:
        logger.error(f"停止策略优化引擎时发生错误: {e}", exc_info=True)

    # 4. 停止币安WebSocket连接 (BinanceClient.stop_all_websockets 是同步的)
    # 这个操作应该相对较快，或者其内部有自己的超时和线程管理
    try:
        logger.info("正在停止所有币安WebSocket连接...")
        binance_client.stop_all_websockets()
        logger.info("所有币安WebSocket连接已停止。")
    except Exception as e:
        logger.error(f"停止币安WebSocket连接时发生错误: {e}", exc_info=True)

    # 5. 并发保存所有持久化数据
    logger.info("正在并发保存所有持久化数据...")
    save_data_tasks = [
        save_live_signals_async(),
        save_strategy_parameters_to_file(),
        save_autox_clients_to_file(),
        save_active_test_config(),
        save_autox_trade_logs_async()
    ]
    save_results = await asyncio.gather(*save_data_tasks, return_exceptions=True)
    for i, res in enumerate(save_results):
        task_name = getattr(save_data_tasks[i], '__name__', f"保存任务_{i}")
        if isinstance(res, Exception):
            logger.error(f"{task_name} 失败: {res}", exc_info=res)
        else:
            logger.info(f"{task_name} 完成。")
    logger.info("所有持久化数据保存尝试完成。")

    # 6. 并发关闭所有客户端WebSocket连接 (UI, AutoX, Status)
    logger.info("正在并发关闭所有活动的客户端WebSocket连接...")
    close_client_ws_tasks = []
    
    # 为每个管理器收集关闭任务
    for ws_mgr_name, ws_mgr_instance in [("AutoX客户端", autox_manager), ("UI客户端", manager), ("状态监控客户端", autox_status_manager)]:
        active_ws_list = list(ws_mgr_instance.active_connections) # 创建副本进行迭代
        if active_ws_list:
            logger.info(f"准备关闭 {len(active_ws_list)} 个 {ws_mgr_name} WebSocket连接...")
            for ws_client_conn in active_ws_list:
                close_client_ws_tasks.append(ws_client_conn.close(code=1000, reason="服务器正在关闭"))
        else:
            logger.info(f"没有活动的 {ws_mgr_name} WebSocket连接需要关闭。")

    if close_client_ws_tasks:
        close_ws_results = await asyncio.gather(*close_client_ws_tasks, return_exceptions=True)
        for i, res in enumerate(close_ws_results):
            if isinstance(res, Exception):
                logger.error(f"关闭客户端WebSocket连接任务 {i} 失败: {res}", exc_info=res)
        logger.info("所有客户端WebSocket连接关闭尝试完成。")
    else:
        logger.info("没有需要关闭的客户端WebSocket连接。")
    
    # 等待后台任务完成 (给它们一点时间响应 shutdown_event_async)
    # Uvicorn 的 graceful_shutdown 也会等待后台任务，但这里可以显式等待一小段时间
    # 确保在Uvicorn强制终止前，我们的任务有机会清理
    logger.info("等待后台任务响应关闭信号 (最多等待3秒)...")
    await asyncio.sleep(3) # 稍微增加等待时间，确保后台任务有足够时间清理

    # 7. 强制清理任何剩余的asyncio任务
    try:
        logger.info("检查并清理剩余的asyncio任务...")
        pending_tasks = [task for task in asyncio.all_tasks() if not task.done()]
        if pending_tasks:
            logger.warning(f"发现 {len(pending_tasks)} 个未完成的asyncio任务，尝试取消...")
            for task in pending_tasks:
                if not task.done():
                    task.cancel()

            # 等待任务取消完成
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)
            logger.info("剩余asyncio任务已清理完成。")
        else:
            logger.info("没有发现剩余的asyncio任务。")
    except Exception as e:
        logger.error(f"清理剩余asyncio任务时发生错误: {e}", exc_info=True)

    logger.info("服务关闭清理操作全部完成。")

if __name__ == "__main__":
    # 创建启动任务列表
    startup_tasks = [
        # 核心后台任务
        process_kline_queue(),
        background_signal_verifier(),
        
        # 加载持久化数据
        load_strategy_parameters_from_file(),
        load_live_signals_async(),
        load_autox_clients_from_file(),
        load_active_test_config() # 加载活动测试配置，实现会话恢复
    ]
    
    # 使用asyncio.gather启动所有任务
    loop = asyncio.get_event_loop()
    for task in startup_tasks:
        loop.create_task(task)
    
    # 启动服务器
    # 从环境变量读取主机和端口，如果不存在则使用默认值
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SERVER_PORT", "8000")) # 端口通常是整数
    # 设置超时参数，确保服务能够正常关闭
    # 减少超时时间以加快关闭速度
    uvicorn.run(app, host=host, port=port, timeout_keep_alive=15, timeout_graceful_shutdown=15)

# --- END OF FILE main.py ---