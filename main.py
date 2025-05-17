# --- START OF FILE main.py ---

# 导入必要的库
import os
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Depends, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
import uvicorn
import numpy as np
import json
import time
import asyncio # 确保导入 asyncio
import threading
from queue import Queue, Empty
import random
import traceback
import uuid # 唯一ID生成

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
STRATEGY_PARAMS_FILE = "config/strategy_parameters.json" # 策略参数配置文件路径
AUTOX_CLIENTS_FILE = "config/autox_clients_data.json" # AutoX 客户端数据文件路径

# 用于存储从文件加载的持久化AutoX客户端数据
# 这个字典将以 client_id 为键，存储客户端的注册信息和最新状态
persistent_autox_clients_data: Dict[str, Dict[str, Any]] = {}


# 初始化 FastAPI 应用
app = FastAPI(
    title="币安事件合约交易信号机器人", # 应用标题
    description="基于技术指标的币安事件合约交易信号生成和回测系统", # 应用描述
    version="1.4.2" # 应用版本
)

# --- WebSocket 连接管理器 (保持不变) ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    async def connect(self, websocket: WebSocket):
        await websocket.accept(); self.active_connections.append(websocket)
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections: self.active_connections.remove(websocket)
    async def broadcast_json(self, data: dict, filter_func=None):
        active_connections_copy = list(self.active_connections) # 迭代副本以允许在广播时断开连接
        for connection in active_connections_copy:
            if filter_func is None or filter_func(connection):
                try: await connection.send_json(data)
                except WebSocketDisconnect: self.disconnect(connection)
                except Exception as e: print(f"广播到 {getattr(connection, 'client', 'N/A')} 失败: {e}")

# --- Pydantic 模型定义 (保持不变) ---
class InvestmentStrategySettings(BaseModel):
    amount: float = Field(20.0, description="基础投资金额或固定金额")
    strategy_id: str = Field("fixed", description="投资策略ID")
    minAmount: float = Field(5.0, description="最小投资金额")
    maxAmount: float = Field(250.0, description="最大投资金额")
    percentageOfBalance: Optional[float] = Field(None, description="账户百分比策略的百分比值 (1-100)")
    profitRate: float = Field(80.0, description="事件合约获胜收益率 (%)")
    lossRate: float = Field(100.0, description="事件合约失败损失率 (%)")
    simulatedBalance: Optional[float] = Field(None, description="模拟账户总资金 (用于百分比投资策略计算)")


class BacktestInvestmentSettings(BaseModel):
    initial_balance: float = Field(1000.0, description="回测初始模拟资金")
    investment_strategy_id: str = Field("fixed", description="投资策略ID")
    investment_strategy_specific_params: Optional[Dict[str, Any]] = Field(None, description="选定投资策略的特定参数")
    min_investment_amount: float = Field(5.0, description="单次最小投资额")
    max_investment_amount: float = Field(250.0, description="单次最大投资额")
    profit_rate_pct: float = Field(80.0, description="事件合约盈利百分比 (%)")
    loss_rate_pct: float = Field(100.0, description="事件合约亏损百分比 (%)")

class BacktestRequest(BaseModel):
    symbol: str = Field(..., description="交易对")
    interval: str = Field(..., description="K线周期")
    start_time: datetime = Field(..., description="回测开始时间 (中国时区)")
    end_time: datetime = Field(..., description="回测结束时间 (中国时区)")
    prediction_strategy_id: str = Field(..., description="预测策略ID")
    prediction_strategy_params: Optional[Dict[str, Any]] = Field(None, description="预测策略参数")
    event_period: str = Field(..., description="事件合约周期")
    confidence_threshold: float = Field(0, description="预测置信度阈值, 0-100")
    investment: BacktestInvestmentSettings = Field(..., description="投资设置")

class SymbolInfo(BaseModel):
    symbol: str; base_asset: str; quote_asset: str
class StrategyParameterSet(BaseModel):
    strategy_type: str = Field(..., description="策略类型: 'prediction' or 'investment'")
    strategy_id: str = Field(..., description="策略ID")
    params: Dict[str, Any] = Field(..., description="策略参数")

class AutoXClientInfo(BaseModel):
    client_id: str
    status: str = "idle" 
    supported_symbols: List[str] = ["BTCUSDT"]
    last_seen: Optional[datetime] = None
    connected_at: datetime = Field(default_factory=now_utc)
    notes: Optional[str] = Field(None, description="管理员为客户端添加的备注")

class AutoXTradeLogEntry(BaseModel):
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
    symbol: str = "ETHUSDT"
    direction: str = Query(..., pattern="^(up|down)$") 
    amount: str = "5" 
    signal_id: Optional[str] = None

class ClientNotesPayload(BaseModel):
    notes: Optional[str] = Field(None, max_length=255)

class DeleteSignalsRequest(BaseModel):
    signal_ids: List[str]

# --- CORS, StaticFiles, BinanceClient ---
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
app.mount("/templates", StaticFiles(directory="templates"), name="templates") # Add this line to serve templates directory
binance_client = BinanceClient()

# --- WebSocket 连接管理器 ---
manager = ConnectionManager()
autox_manager = ConnectionManager()
autox_status_manager = ConnectionManager()


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

# --- 新增：用于执行阻塞文件操作的辅助函数 ---
def _blocking_save_json_to_file(file_path: str, data_to_save: Any):
    """
    一个同步的辅助函数，用于将数据保存到JSON文件。
    它会创建临时文件并进行原子替换，以确保数据完整性。
    这个函数应该通过 asyncio.to_thread 来调用。
    """
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

        print(f"成功从 {ACTIVE_TEST_CONFIG_FILE} 加载活动测试配置 (ID: {active_config_id})。") # 修改日志
        # 可以选择在这里打印加载的配置数据，但可能比较冗长
        # print(f"加载的配置数据: {config_data}")

    else:
        print(f"{ACTIVE_TEST_CONFIG_FILE} 未找到活动测试配置或配置不完整。") # 修改日志

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
    while True:
        try:
            # signals_queue.get_nowait() 是同步的，但通常很快
            # 如果队列为空，它会立即抛出 Empty 异常，然后 asyncio.sleep(0.01) 释放控制权
            kline_data = signals_queue.get_nowait() 
            await handle_kline_data(kline_data) # handle_kline_data 现在内部会有异步操作
        except Empty:
            await asyncio.sleep(0.01) # 队列为空时短暂休眠，避免CPU空转
        except Exception as e:
            print(f"处理K线队列时出错: {e}\n{traceback.format_exc()}");
            await asyncio.sleep(1) # 出错时稍长休眠

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
            try:
                end_time_utc = parse_frontend_datetime(signal_copy_to_verify['expected_end_time'])
                if not end_time_utc.tzinfo: end_time_utc = end_time_utc.replace(tzinfo=timezone.utc)
                else: end_time_utc = end_time_utc.astimezone(timezone.utc)

                kline_df = await asyncio.to_thread( # IO密集型操作，放入线程池
                    binance_client.get_historical_klines,
                    signal_copy_to_verify['symbol'], '1m', 
                    start_time=int(end_time_utc.timestamp() * 1000), limit=1
                )
                actual_price = None
                if not kline_df.empty:
                    actual_price = float(kline_df.iloc[0]['close'])
                else: 
                    print(f"验证 {signal_copy_to_verify['id']}: 未找到 {end_time_utc.isoformat()} 的1m K线，尝试获取最新价...")
                    try:
                        actual_price = await asyncio.to_thread(binance_client.get_latest_price, signal_copy_to_verify['symbol'])
                    except Exception as e_latest:
                         print(f"获取 {signal_copy_to_verify['symbol']} 最新价格失败: {e_latest}")

                if actual_price is None:
                    print(f"无法获取 {signal_copy_to_verify['symbol']} 的结束价格进行验证。信号ID: {signal_copy_to_verify['id']}")
                    with live_signals_lock:
                        for i, sig_live in enumerate(live_signals):
                            if sig_live.get('id') == signal_copy_to_verify['id'] and not sig_live.get('verified'):
                                live_signals[i].update({'verified': True, 'result': False, 'actual_end_price': -3, 'verify_notes': '无法获取结束价', 'verify_time': format_for_display(current_time_utc)})
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
                    'verified': True, 'verify_time': format_for_display(current_time_utc)
                }

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
    except Exception as e:
        print(f"向AutoX客户端发送指令失败: {e}")


async def handle_kline_data(kline_data: dict):
    global live_signals, strategy_parameters_config, running_live_test_configs, active_autox_clients, binance_client
    try:
        kline_symbol = kline_data.get('symbol')
        kline_interval = kline_data.get('interval')
        is_kline_closed = kline_data.get('is_kline_closed', False)
        if not (kline_symbol and kline_interval and is_kline_closed):
            return

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

            df_klines = await asyncio.to_thread( # IO密集型操作，放入线程池
                binance_client.get_historical_klines,
                live_test_config_data['symbol'],
                live_test_config_data['interval'],
                None, None, 100 
            )
            
            if df_klines.empty:
                print(f"Config ID {live_test_config_data['_config_id']}: 获取历史K线数据为空，跳过。")
                continue

            pred_strat_info = next((s for s in get_available_strategies() if s['id'] == pred_strat_id), None)
            if not pred_strat_info:
                print(f"Config ID {live_test_config_data['_config_id']}: 未找到预测策略 {pred_strat_id}，跳过。")
                continue
            
            signal_df = pred_strat_info['class'](params=final_pred_params).generate_signals(df_klines.copy()) # 策略计算通常是CPU密集，但如果涉及IO则需注意
            
            if signal_df.empty or 'signal' not in signal_df.columns or 'confidence' not in signal_df.columns:
                continue 
            
            latest_sig_data = signal_df.iloc[-1]
            sig_val = int(latest_sig_data.get('signal', 0))
            conf_val = float(latest_sig_data.get('confidence', 0))
            current_confidence_threshold = live_test_config_data.get('confidence_threshold', 0)

            if sig_val != 0 and conf_val >= current_confidence_threshold:
                event_period_minutes = {'10m': 10, '30m': 30, '1h': 60, '1d': 1440}.get(
                    live_test_config_data.get("event_period", "10m"), 10
                )
                
                sig_time_dt = now_utc() 
                exp_end_time_dt = sig_time_dt + timedelta(minutes=event_period_minutes)
                sig_price = float(latest_sig_data['close']) 
                
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
                            inv_instance = inv_strat_def_cfg['class'](params=strategy_specific_params_for_instance)
                            
                            # --- MODIFICATION FOR CURRENT BALANCE ---
                            # 使用 live_test_config_data 中存储的 current_balance (这是副本中的)
                            current_balance_for_calc = live_test_config_data.get('current_balance')
                            
                            # 如果 current_balance 由于某种原因没有在 live_test_config_data 中
                            # (理论上应该有，因为上面在创建副本时已确保其存在)
                            # 则回退到 investment_settings 中的 simulatedBalance (初始值)
                            if current_balance_for_calc is None:
                                current_balance_for_calc = live_inv_model.simulatedBalance
                            
                            # 最终回退，主要针对百分比策略，确保它有余额可用
                            if current_balance_for_calc is None:
                                if inv_strat_id_cfg == 'percentage_of_balance':
                                    print(f"警告 (Config ID: {live_test_config_data['_config_id']}): 百分比投资策略缺少 current_balance 和 simulatedBalance，将使用默认1000。")
                                    current_balance_for_calc = 1000.0 
                                else: 
                                    # 其他策略可能不直接依赖此余额，或使用 live_inv_model.amount
                                    current_balance_for_calc = live_inv_model.amount 
                            # --- END MODIFICATION FOR CURRENT BALANCE ---
                            
                            inv_amount = inv_instance.calculate_investment( 
                                current_balance=current_balance_for_calc, 
                                previous_trade_result=None, 
                                base_investment_from_settings=live_inv_model.amount # 传递Pydantic模型中的基础投资额
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
                    'autox_triggered_info': []
                }

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
                
                await manager.broadcast_json(
                    {"type": "new_signal", "data": new_live_signal},
                    filter_func=lambda c: websocket_to_config_id_map.get(c) == new_live_signal['origin_config_id']
                )
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
    await load_live_signals_async()
    await load_strategy_parameters_from_file()
    await load_autox_clients_from_file()
    await load_active_test_config() # 新增：加载活动测试配置

    # 创建后台任务
    asyncio.create_task(process_kline_queue())
    asyncio.create_task(background_signal_verifier())
    print("应用启动完成。后台任务已启动。")

@app.on_event("shutdown")
async def shutdown_event(): # 基本不变
    print("应用关闭，停止币安WebSocket连接...")
    binance_client.stop_all_websockets() 
    print("币安WebSocket连接已停止。")

# --- WebSocket 端点 for Web UI (/ws/live-test) (逻辑微调，确保使用锁和副本) ---
@app.websocket("/ws/live-test")
async def websocket_endpoint(websocket: WebSocket):
    global active_live_test_config_id
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
            # 发送活动配置信息给新连接的客户端
            await websocket.send_json({
                "type": "active_config_notification",
                "data": {
                    "config_id": current_active_config_id,
                    "config": current_active_config,
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
                    # 确保恢复的配置有 current_balance
                    if 'current_balance' not in restored_config_data:
                        sim_bal = restored_config_data.get('investment_settings', {}).get('simulatedBalance')
                        restored_config_data['current_balance'] = sim_bal if sim_bal is not None else 1000.0
                        print(f"恢复会话 {client_config_id}: current_balance 不存在，已从simulatedBalance或默认值设置。")


                    websocket_to_config_id_map[websocket] = client_config_id
                    # 发送给客户端的 restored_config_data 将包含 current_balance
                    await websocket.send_json({"type": "session_restored", "data": {"config_id": client_config_id, "config_details": restored_config_data}})
                else:
                    await websocket.send_json({"type": "session_not_found", "data": {"config_id": client_config_id}})
            
            elif message_type == 'set_runtime_config':
                config_payload_data = data.get('data', {})
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

                existing_config_id = websocket_to_config_id_map.pop(websocket, None)
                if existing_config_id: 
                    # 如果当前WebSocket已经关联到一个配置ID，先移除这个关联
                    # 但不删除配置，因为可能其他设备还在使用
                    pass
                
                # 检查是否已有活动的测试配置
                current_active_config_id = None
                with active_live_test_config_lock:
                    current_active_config_id = active_live_test_config_id
                
                if current_active_config_id:
                    # 已有活动配置，使用现有配置ID
                    new_config_id = current_active_config_id
                    
                    # 获取现有配置数据
                    existing_config = None
                    with running_live_test_configs_lock:
                        if current_active_config_id in running_live_test_configs:
                            existing_config = running_live_test_configs[current_active_config_id]
                    
                    if existing_config:
                        # 发送会话恢复通知
                        await websocket.send_json({
                            "type": "session_recovered",
                            "data": {
                                "config_id": new_config_id,
                                "message": "已恢复现有测试会话",
                                "config": existing_config
                            }
                        })
                        
                        # 将当前WebSocket关联到现有配置ID
                        websocket_to_config_id_map[websocket] = new_config_id
                        return
                else:
                    # 没有活动配置，创建新配置ID
                    new_config_id = uuid.uuid4().hex
                    
                    # 设置为全局活动配置ID
                    with active_live_test_config_lock:
                        active_live_test_config_id = new_config_id
                
                new_symbol = config_payload_data['symbol']; new_interval = config_payload_data['interval']
                
                # current_balance 使用 investment_settings 中的 simulatedBalance 初始化
                # 如果 simulatedBalance 未提供，则使用默认值 (例如 1000.0)
                initial_simulated_balance = config_payload_data["investment_settings"].get("simulatedBalance")
                initial_current_balance = initial_simulated_balance if initial_simulated_balance is not None else 1000.0

                full_config_to_store = { 
                    "symbol": new_symbol, "interval": new_interval,
                    "prediction_strategy_id": config_payload_data["prediction_strategy_id"],
                    "prediction_strategy_params": config_payload_data.get("prediction_strategy_params"),
                    "confidence_threshold": config_payload_data["confidence_threshold"],
                    "event_period": config_payload_data["event_period"],
                    "investment_settings": config_payload_data["investment_settings"], 
                    "autox_enabled": config_payload_data.get("autox_enabled", True),
                    "current_balance": initial_current_balance, # 初始化会话余额
                    "total_profit_loss_amount": 0.0, # 新增：初始化总盈亏额
                    "created_at": format_for_display(now_utc()) # 记录创建时间
                }

                try:
                    if new_symbol != 'all' and new_interval != 'all': 
                        await start_kline_websocket_if_needed(new_symbol, new_interval)
                    
                    with running_live_test_configs_lock: 
                        running_live_test_configs[new_config_id] = full_config_to_store
                    
                    websocket_to_config_id_map[websocket] = new_config_id
                    
                    # 保存活动配置到文件
                    await save_active_test_config()
                    # 发送给客户端的 full_config_to_store 将包含 current_balance
                    await websocket.send_json({"type": "config_set_confirmation", "data": {"success": True, "message": "运行时配置已应用。", "config_id": new_config_id, "applied_config": full_config_to_store}})
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
    await autox_manager.connect(websocket)
    client_id_local: Optional[str] = None # 用于在 finally 中记录
    
    try:
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type")
            payload = data.get("payload", {})

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
                            print(f"AutoX客户端 {client_id_local} 重复连接 (不同WebSocket)，准备关闭旧连接。")
                            old_ws_to_close = ws_iter
                            break
                    
                    if old_ws_to_close:
                        try:
                            await old_ws_to_close.close(code=1000, reason="New connection for this client_id")
                        except Exception: pass 
                        autox_manager.disconnect(old_ws_to_close) 
                        active_autox_clients.pop(old_ws_to_close, None) 

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
                    active_autox_clients[websocket] = client_info_to_store_dict 
                    persistent_autox_clients_data[client_id_local] = client_info_to_store_dict 

                print(f"AutoX客户端已注册/更新: ID={client_id_local}, 支持交易对={supported_symbols_list}, 备注='{client_info_to_store_dict.get('notes', '') if client_info_to_store_dict else ''}'")
                await websocket.send_json({"type": "registered", "message": "客户端注册成功。", "client_info": client_info_to_store_dict})
                
                await broadcast_autox_clients_status() 
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
                    # "trade_execution_succeeded" # 如果JS将来会报告明确的成功，也应加进来
                }

                with autox_clients_lock:
                    if websocket not in active_autox_clients:
                        print(f"警告: 收到来自未知/已断开 WebSocket (client_id登记为: {client_id_local}) 的状态更新。忽略。")
                        continue # 忽略来自不再活跃的websocket连接的消息
                    
                    current_client_info_active = active_autox_clients[websocket]
                    current_client_info_persistent = persistent_autox_clients_data.get(client_id_local)

                    # 默认情况下，客户端的下一个状态是它自己报告的状态
                    next_client_status_to_set = client_reported_status

                    # 如果报告的状态是终端状态，则将客户端的最终状态设置为 'idle'
                    if client_reported_status in TERMINAL_AND_RESET_TO_IDLE_STATUSES:
                        next_client_status_to_set = "idle"
                        print(f"AutoX客户端 {client_id_local} 报告状态 '{client_reported_status}', 将其重置为 'idle'。")
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

                # 日志记录部分
                log_payload = { 
                    "client_id": client_id_local, "signal_id": payload.get("signal_id"),
                    "command_type": "status_from_client", "command_payload": payload,
                    "status": client_reported_status, # 日志中记录JS上报的原始状态
                    "details": payload.get("details"), "error_message": payload.get("error_message"),
                }
                with autox_trade_logs_lock: 
                    autox_trade_logs.append(AutoXTradeLogEntry(**log_payload).model_dump(mode='json'))
                    if len(autox_trade_logs) > MAX_AUTOX_LOG_ENTRIES: autox_trade_logs.pop(0)

                final_set_status = updated_info_for_broadcast.get('status') if updated_info_for_broadcast else 'N/A'
                print(f"收到AutoX客户端 {client_id_local} 状态更新: '{client_reported_status}' (原始), Signal ID: {payload.get('signal_id')}. "
                      f"客户端最终状态设置为: '{final_set_status}'")
                
                if updated_info_for_broadcast:
                    await broadcast_autox_clients_status() 
                    await save_autox_clients_to_file() 

            elif message_type == "pong": 
                with autox_clients_lock: 
                     if websocket in active_autox_clients:
                        active_autox_clients[websocket]['last_seen'] = now_utc().isoformat()
                        if client_id_local and client_id_local in persistent_autox_clients_data:
                            persistent_autox_clients_data[client_id_local]['last_seen'] = now_utc().isoformat()
            else:
                print(f"收到来自AutoX客户端 {client_id_local or '未知'} 的未知消息类型: {message_type}")
                await websocket.send_json({"type": "error", "message": f"不支持的消息类型: {message_type}"})

    except WebSocketDisconnect:
        print(f"AutoX客户端 {client_id_local or getattr(websocket, 'client', 'N/A')} 断开连接。")
    except Exception as e:
        print(f"AutoX WebSocket端点错误 ({client_id_local or getattr(websocket, 'client', 'N/A')}): {e}\n{traceback.format_exc()}")
    finally:
        autox_manager.disconnect(websocket) 
        disconnected_client_id = None
        with autox_clients_lock: 
            client_info_at_disconnect = active_autox_clients.pop(websocket, None) 
            if client_info_at_disconnect:
                disconnected_client_id = client_info_at_disconnect.get('client_id')
                print(f"AutoX客户端 {disconnected_client_id or '未知'} 已从活动列表移除。")
                
                if disconnected_client_id and disconnected_client_id in persistent_autox_clients_data:
                    # 当客户端断开连接时，我们将其在持久化存储中的状态标记为'offline'
                    # 这样UI可以明确显示其离线，并且派单逻辑也不会选择它。
                    # last_signal_id 在这里通常不清除，以便查看它离线前处理的最后一个信号。
                    persistent_autox_clients_data[disconnected_client_id]['status'] = 'offline' 
                    persistent_autox_clients_data[disconnected_client_id]['last_seen'] = now_utc().isoformat() 
        
        if disconnected_client_id: 
            await broadcast_autox_clients_status() 
            await save_autox_clients_to_file()


# --- WebSocket 端点 for AutoX Status (/ws/autox_status) (逻辑不变) ---
@app.websocket("/ws/autox_status")
async def autox_status_websocket_endpoint(websocket: WebSocket):
    await autox_status_manager.connect(websocket)
    try:
        await broadcast_autox_clients_status() # 连接成功后立即发送当前客户端列表
        while True:
            await websocket.receive_text() # 保持连接，可以处理ping/pong
    except WebSocketDisconnect:
        print(f"AutoX状态前端客户端 {getattr(websocket, 'client', 'N/A')} 断开连接。")
    except Exception as e:
        print(f"AutoX状态WebSocket端点错误 ({getattr(websocket, 'client', 'N/A')}): {e}\n{traceback.format_exc()}")
    finally:
        autox_status_manager.disconnect(websocket)


# --- 辅助函数：广播AutoX客户端状态 (现在从 persistent_autox_clients_data 读取) ---
async def broadcast_autox_clients_status():
    """广播当前AutoX客户端列表到所有连接的AutoX状态前端WebSocket。"""
    clients_data_to_broadcast = []
    with autox_clients_lock: # 保护对 persistent_autox_clients_data 的读取
        # 创建副本进行迭代和处理
        persistent_copy = {cid: cinfo.copy() for cid, cinfo in persistent_autox_clients_data.items()}

    active_client_ids = set()
    with autox_clients_lock: # 保护对 active_autox_clients 的读取
        for ws, active_info in active_autox_clients.items():
            if active_info.get('client_id'):
                active_client_ids.add(active_info['client_id'])

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
            print(f"序列化AutoX客户端 {client_id} 信息时出错: {e_val}")


    payload = {"type": "autox_clients_update", "data": clients_data_to_broadcast}
    await autox_status_manager.broadcast_json(payload)
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

@app.post("/api/backtest") # 修改：get_historical_klines 和 Backtester.run() 移至线程池
async def run_backtest_endpoint(request: BacktestRequest):
    global strategy_parameters_config # 读取是线程安全的
    try:
        start_utc = to_utc(request.start_time); end_utc = to_utc(request.end_time)
        if start_utc >= end_utc or end_utc > now_utc() or start_utc > now_utc(): 
            raise HTTPException(status_code=400, detail="回测时间范围无效。")

        # --- 修改：将 get_historical_klines 移至线程池 ---
        df_klines = await asyncio.to_thread(
            binance_client.get_historical_klines,
            request.symbol,
            request.interval,
            int(start_utc.timestamp() * 1000),
            int(end_utc.timestamp() * 1000)
        )
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

        investment_args_dict = request.investment.model_dump(exclude={'investment_strategy_id', 'investment_strategy_specific_params'})

        # --- 修改：将 Backtester 实例化和 run() 移至线程池 ---
        # 需要一个辅助函数来执行这部分，因为它涉及多个步骤
        def _run_backtest_blocking(df, strategy_instance, event_period, confidence_threshold, inv_strategy_id, inv_strategy_params, **inv_args_dict):
            backtester_instance = Backtester(
                df=df, 
                strategy=strategy_instance, 
                event_period=event_period, 
                confidence_threshold=confidence_threshold, 
                investment_strategy_id=inv_strategy_id, 
                investment_strategy_params=inv_strategy_params or {}, 
                **inv_args_dict
            )
            return backtester_instance.run()

        results_data = await asyncio.to_thread(
            _run_backtest_blocking,
            df_klines.copy(), # 传递DataFrame副本
            prediction_instance, # 策略实例已创建
            request.event_period,
            request.confidence_threshold,
            inv_id,
            final_inv_params or {},
            **investment_args_dict
        )
        # --- 结束修改 ---

        # 结果处理（同步，但通常很快）
        for pred_item_data in results_data.get('predictions', []):
            for time_key_str in ['signal_time', 'end_time_expected', 'end_time_actual']:
                if time_key_str in pred_item_data and isinstance(pred_item_data[time_key_str], datetime):
                    pred_item_data[time_key_str] = format_for_display(pred_item_data[time_key_str]) 
        return results_data
    except HTTPException as http_exc: raise http_exc
    except Exception as exc:
        error_detail_msg = f"回测过程中发生错误: {str(exc)}"; print(f"{error_detail_msg}\n{traceback.format_exc()}"); raise HTTPException(status_code=500, detail=error_detail_msg)

@app.get("/api/live-signals") # 修改：确保线程安全地读取 live_signals
async def get_live_signals_http_endpoint():
    with live_signals_lock: # 保护对 live_signals 的读取
        # 返回副本，避免外部修改影响全局状态
        return [s.copy() for s in live_signals] 

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

    pred_id_test = "simple_rsi"; 
    # 读取 strategy_parameters_config 是安全的，因为它通常只在启动时修改
    pred_params_test = strategy_parameters_config.get("prediction_strategies", {}).get(pred_id_test, {"rsi_period": 14, "rsi_overbought": 70, "rsi_oversold": 30})
    
    test_signal_data = {
        'id': f"TEST_{symbol}_{pred_id_test}_{int(time.time())}_{random.randint(100,999)}", 'symbol': symbol, 'interval': "1m", 'prediction_strategy_id': pred_id_test, 'prediction_strategy_params': pred_params_test,
        'signal_time': format_for_display(signal_time), 'signal': random.choice([1, -1]), 'confidence': random.uniform(60, 95), 'signal_price': price_val, 'event_period': "10m",
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
    except Exception as e: # 通常 _send_autox_command 内部会处理错误
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
            await broadcast_autox_clients_status() # 广播状态更新

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
            await broadcast_autox_clients_status()
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
    await broadcast_autox_clients_status() # 广播状态更新（可能包含备注变化）

    # 返回 Pydantic 模型确保响应格式正确
    return AutoXClientInfo(**updated_client_info_dict) if updated_client_info_dict else None


@app.post("/api/live-signals/delete-batch", response_model=Dict[str, Any])
async def delete_live_signals_batch(request: DeleteSignalsRequest):
    """异步批量删除实时信号。"""
    global live_signals
    deleted_count = 0
    
    if not request.signal_ids:
        raise HTTPException(status_code=400, detail="signal_ids 列表不能为空。")

    ids_to_delete_set = set(request.signal_ids)
    
    # 在锁外准备新的列表，减少锁的持有时间
    signals_to_keep = []
    current_signals_copy = []
    with live_signals_lock: # 获取锁以安全地复制数据
        current_signals_copy = [s.copy() for s in live_signals]

    for signal in current_signals_copy:
        if signal.get('id') in ids_to_delete_set:
            deleted_count += 1
        else:
            signals_to_keep.append(signal)
    
    not_found_count = len(request.signal_ids) - deleted_count
    if not_found_count < 0: not_found_count = 0 

    if deleted_count > 0:
        with live_signals_lock: # 获取锁以更新全局 live_signals
            live_signals = signals_to_keep
        
        await save_live_signals_async() # 调用已修改的异步保存函数
            
        # 准备广播统计数据
        # 再次获取锁和副本以确保统计数据的准确性
        stats_payload_data_for_broadcast = []
        with live_signals_lock:
            stats_payload_data_for_broadcast = [s.copy() for s in live_signals]

        verified_list_global = [s for s in stats_payload_data_for_broadcast if s.get('verified')]
        total_verified_global = len(verified_list_global)
        total_correct_global = sum(1 for s in verified_list_global if s.get('result'))
        total_actual_profit_loss_amount_global = sum(s.get('actual_profit_loss_amount', 0.0) for s in verified_list_global if s.get('actual_profit_loss_amount') is not None)
        total_pnl_pct_sum_global = sum(s.get('pnl_pct', 0.0) for s in verified_list_global if s.get('pnl_pct') is not None)
        average_pnl_pct_global = total_pnl_pct_sum_global / total_verified_global if total_verified_global > 0 else 0
        stats_payload = {
            "total_signals": len(stats_payload_data_for_broadcast), 
            "total_verified": total_verified_global,
            "total_correct": total_correct_global,
            "win_rate": round(total_correct_global / total_verified_global * 100 if total_verified_global > 0 else 0, 2),
            "total_pnl_pct": round(total_pnl_pct_sum_global, 2),
            "average_pnl_pct": round(average_pnl_pct_global, 2),
            "total_profit_amount": round(total_actual_profit_loss_amount_global, 2)
        }
        await manager.broadcast_json({"type": "stats_update", "data": stats_payload})
        await manager.broadcast_json({"type": "signals_deleted_notification", "data": {"deleted_ids": list(ids_to_delete_set), "message": f"部分信号已删除。"}})

        return {"status": "success", "message": f"成功删除 {deleted_count} 个信号。" + (f" {not_found_count} 个请求的信号未找到。" if not_found_count > 0 else ""), "deleted_count": deleted_count, "not_found_count": not_found_count}
    
    elif not_found_count > 0 : # 没有删除，但有未找到的
         return {"status": "warning", "message": f"请求删除的 {len(request.signal_ids)} 个信号均未找到。", "deleted_count": 0, "not_found_count": not_found_count}
    else: # deleted_count == 0 and not_found_count == 0 (例如空列表请求，已被前面拦截)
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
                        "status": "command_sent_via_test_api"
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
                test_signal_data['autox_triggered_info'] = {"status": "no_available_client_for_test_api"}
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
    uvicorn.run(app, host="0.0.0.0", port=8000)

# --- END OF FILE main.py ---