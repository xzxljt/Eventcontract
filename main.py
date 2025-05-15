# --- START OF FILE main.py ---

import os
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Depends, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
import uvicorn
import numpy as np
import json
import time
import asyncio
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

# --- 全局变量 ---
strategy_parameters_config: Dict[str, Any] = {
    "prediction_strategies": {},
    "investment_strategies": {}
}
STRATEGY_PARAMS_FILE = "config/strategy_parameters.json"
AUTOX_CLIENTS_FILE = "config/autox_clients_data.json" # AutoX 客户端数据文件

app = FastAPI(
    title="币安事件合约交易信号机器人",
    description="基于技术指标的币安事件合约交易信号生成和回测系统",
    version="1.4.1" # 版本更新: AutoX 客户端备注初步支持
)

# --- WebSocket 连接管理器 ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    async def connect(self, websocket: WebSocket):
        await websocket.accept(); self.active_connections.append(websocket)
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections: self.active_connections.remove(websocket)
    async def broadcast_json(self, data: dict, filter_func=None):
        active_connections_copy = list(self.active_connections)
        for connection in active_connections_copy:
            if filter_func is None or filter_func(connection):
                try: await connection.send_json(data)
                except WebSocketDisconnect: self.disconnect(connection)
                except Exception as e: print(f"广播到 {getattr(connection, 'client', 'N/A')} 失败: {e}")
                
# --- Pydantic 模型定义 ---
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

# AutoX 相关Pydantic模型
class AutoXClientInfo(BaseModel):
    client_id: str
    status: str = "idle" # idle, processing_trade, error
    supported_symbols: List[str] = ["BTCUSDT"]
    last_seen: Optional[datetime] = None
    connected_at: datetime = Field(default_factory=now_utc)
    notes: Optional[str] = Field(None, description="管理员为客户端添加的备注") # 新增备注字段

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

class ClientNotesPayload(BaseModel): # 新增用于更新备注的Payload模型
    notes: Optional[str] = Field(None, max_length=255)


# --- CORS, StaticFiles, BinanceClient ---
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
binance_client = BinanceClient()

# --- WebSocket 连接管理器 ---
manager = ConnectionManager()
autox_manager = ConnectionManager()
autox_status_manager = ConnectionManager() # 新增用于AutoX状态广播的连接管理器


# --- 实时信号与队列 ---
live_signals = []
LIVE_SIGNALS_FILE = "live_signals.json"
live_signals_lock = threading.Lock()
signals_queue = Queue()

# --- 后台持续运行的实时测试核心状态 ---
active_kline_streams: Dict[str, int] = {}
active_kline_streams_lock = threading.Lock()
running_live_test_configs: Dict[str, Dict[str, Any]] = {}
running_live_test_configs_lock = threading.Lock()
websocket_to_config_id_map: Dict[WebSocket, str] = {}

# --- AutoX.js 控制相关全局变量 ---
active_autox_clients: Dict[WebSocket, Dict[str, Any]] = {} # Value is AutoXClientInfo.model_dump()
autox_clients_lock = threading.Lock()
autox_trade_logs: List[Dict[str, Any]] = []
autox_trade_logs_lock = threading.Lock()
MAX_AUTOX_LOG_ENTRIES = 200

# --- 辅助函数：确保数据是JSON可序列化的 ---
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

# --- 持久化相关函数 ---
async def load_autox_clients_from_file():
    """从文件加载 AutoX 客户端数据。"""
    global active_autox_clients
    default_clients = {}
    try:
        clients_dir = os.path.dirname(AUTOX_CLIENTS_FILE)
        if clients_dir and not os.path.exists(clients_dir):
            try: os.makedirs(clients_dir); print(f"AutoX客户端文件目录 {clients_dir} 已创建。")
            except OSError as e: print(f"创建AutoX客户端文件目录 {clients_dir} 失败: {e}。")

        if os.path.exists(AUTOX_CLIENTS_FILE):
            with open(AUTOX_CLIENTS_FILE, "r", encoding="utf-8") as f: content = f.read()
            if content.strip():
                try:
                    loaded_clients_data = json.loads(content)
                    # 验证加载的数据结构，并只保留有效的客户端信息
                    valid_clients = {}
                    for client_id, client_info in loaded_clients_data.items():
                        try:
                            # 使用 Pydantic 模型验证并清理数据
                            client_model = AutoXClientInfo(**client_info)
                            valid_clients[client_id] = client_model.model_dump()
                        except Exception as e_val:
                            print(f"加载AutoX客户端数据时验证失败 (ID: {client_id}): {e_val}. 跳过此客户端。")
                            # 可选：记录无效的客户端数据
                            # with open("invalid_autox_clients.log", "a", encoding="utf-8") as log_f:
                            #     log_f.write(f"Invalid client data for ID {client_id}: {client_info} - Error: {e_val}\n")

                    with autox_clients_lock:
                         # 注意：这里加载的是离线信息，不包含WebSocket连接对象
                         # 在客户端实际连接时，会根据client_id匹配并更新active_autox_clients
                         # 这里只是加载历史信息，以便在前端展示离线客户端或保留备注等信息
                         # 实际的 active_autox_clients 字典仍然以 WebSocket 对象为键
                         # 我们需要一个单独的结构来存储持久化的客户端信息，或者在连接时合并
                         # 为了简化，我们暂时只加载数据，并在连接时查找匹配的持久化信息
                         # 更完善的方案是维护一个持久化客户端列表，并在连接/断开时更新
                         # 暂时将加载的数据存储在一个新的全局变量中，或者在需要时查找
                         # 考虑到当前的 active_autox_clients 结构，直接加载到它不合适
                         # 我们需要修改 active_autox_clients 的结构或引入新的持久化存储
                         # 鉴于当前 active_autox_clients 以 WebSocket 为键，我们不能直接加载到它
                         # 让我们创建一个新的全局变量来存储持久化的客户端信息
                         global persistent_autox_clients_data
                         persistent_autox_clients_data = valid_clients
                         print(f"AutoX客户端数据已从 {AUTOX_CLIENTS_FILE} 加载。加载了 {len(valid_clients)} 个有效客户端。")

                except json.JSONDecodeError:
                    print(f"错误: {AUTOX_CLIENTS_FILE} 包含无效JSON。将使用默认配置。")
                    # 不覆盖无效文件，保留原始数据以便调试
            else:
                print(f"{AUTOX_CLIENTS_FILE} 为空，使用默认配置。")
        else:
            print(f"{AUTOX_CLIENTS_FILE} 未找到，将创建。使用默认配置。")
            # 文件不存在，不需要创建，保存时会自动创建目录和文件
    except Exception as e:
        print(f"加载AutoX客户端数据时发生其他错误: {e}\n{traceback.format_exc()}。将使用默认配置。")
        # 发生错误时，不加载任何数据，使用默认空配置

async def save_autox_clients_to_file():
    """将当前活动和持久化的 AutoX 客户端数据保存到文件。"""
    global active_autox_clients, persistent_autox_clients_data
    temp_file_path = AUTOX_CLIENTS_FILE + ".tmp"
    try:
        clients_dir = os.path.dirname(AUTOX_CLIENTS_FILE)
        if clients_dir and not os.path.exists(clients_dir): os.makedirs(clients_dir)

        # 合并当前活动客户端信息和持久化客户端信息
        # 活动客户端信息优先，因为它包含最新状态和last_seen时间
        all_clients_to_save = {}
        # 从持久化数据开始，保留离线客户端信息
        if persistent_autox_clients_data:
             all_clients_to_save.update(persistent_autox_clients_data)

        # 添加或更新活动客户端信息
        with autox_clients_lock:
            for ws, info_dict in active_autox_clients.items():
                 client_id = info_dict.get('client_id')
                 if client_id:
                     # 使用 Pydantic 模型确保数据结构正确
                     try:
                         client_model = AutoXClientInfo(**info_dict)
                         all_clients_to_save[client_id] = client_model.model_dump()
                     except Exception as e_val:
                         print(f"保存AutoX客户端数据时验证失败 (ID: {client_id}): {e_val}. 跳过此客户端。")


        serializable_clients_data = ensure_json_serializable(all_clients_to_save)

        with open(temp_file_path, "w", encoding="utf-8") as f:
            json.dump(serializable_clients_data, f, indent=4)
            f.flush(); os.fsync(f.fileno()) # 确保数据写入磁盘
        os.replace(temp_file_path, AUTOX_CLIENTS_FILE)
        print(f"AutoX客户端数据已保存到文件 {AUTOX_CLIENTS_FILE}。")
    except Exception as e:
        print(f"保存AutoX客户端数据到文件 {AUTOX_CLIENTS_FILE} 失败: {e}\n{traceback.format_exc()}")
        if os.path.exists(temp_file_path):
            try: os.remove(temp_file_path)
            except Exception as rm_err: print(f"清理临时文件 {temp_file_path} 失败: {rm_err}")


async def load_strategy_parameters_from_file():
    global strategy_parameters_config
    default_config = {"prediction_strategies": {}, "investment_strategies": {}}
    try:
        params_dir = os.path.dirname(STRATEGY_PARAMS_FILE)
        if params_dir and not os.path.exists(params_dir):
            try: os.makedirs(params_dir); print(f"参数文件目录 {params_dir} 已创建。")
            except OSError as e: print(f"创建参数文件目录 {params_dir} 失败: {e}。")

        if os.path.exists(STRATEGY_PARAMS_FILE):
            with open(STRATEGY_PARAMS_FILE, "r", encoding="utf-8") as f: content = f.read()
            if content.strip():
                try:
                    loaded_params = json.loads(content)
                    strategy_parameters_config["prediction_strategies"] = ensure_json_serializable(loaded_params.get("prediction_strategies", {}))
                    strategy_parameters_config["investment_strategies"] = ensure_json_serializable(loaded_params.get("investment_strategies", {}))
                    print(f"策略参数已从 {STRATEGY_PARAMS_FILE} 加载。")
                except json.JSONDecodeError:
                    strategy_parameters_config = default_config
                    print(f"错误: {STRATEGY_PARAMS_FILE} 包含无效JSON。将使用默认配置并尝试覆盖。")
                    try:
                        with open(STRATEGY_PARAMS_FILE, "w", encoding="utf-8") as wf_err: json.dump(default_config, wf_err, indent=4)
                        print(f"已用默认配置覆盖无效的 {STRATEGY_PARAMS_FILE}。")
                    except Exception as e_write_inv: print(f"覆盖无效JSON文件失败: {e_write_inv}")
            else:
                strategy_parameters_config = default_config
                print(f"{STRATEGY_PARAMS_FILE} 为空，使用默认配置并初始化文件。")
                try:
                    with open(STRATEGY_PARAMS_FILE, "w", encoding="utf-8") as wf: json.dump(default_config, wf, indent=4)
                    print(f"已将默认配置写入空的 {STRATEGY_PARAMS_FILE}。")
                except Exception as e_write_emp: print(f"写入默认配置到空文件失败: {e_write_emp}")
        else:
            strategy_parameters_config = default_config
            print(f"{STRATEGY_PARAMS_FILE} 未找到，将创建。使用默认配置。")
            try:
                with open(STRATEGY_PARAMS_FILE, "w", encoding="utf-8") as f: json.dump(default_config, f, indent=4)
                print(f"已创建并初始化 {STRATEGY_PARAMS_FILE}。")
            except Exception as e_create: print(f"创建 {STRATEGY_PARAMS_FILE} 失败: {e_create}")
    except Exception as e:
        strategy_parameters_config = default_config
        print(f"加载策略参数时发生其他错误: {e}\n{traceback.format_exc()}。将使用默认配置。")

async def save_strategy_parameters_to_file():
    global strategy_parameters_config
    temp_file_path = STRATEGY_PARAMS_FILE + ".tmp"
    try:
        params_dir = os.path.dirname(STRATEGY_PARAMS_FILE)
        if params_dir and not os.path.exists(params_dir): os.makedirs(params_dir)
        serializable_config = ensure_json_serializable(strategy_parameters_config)
        with open(temp_file_path, "w", encoding="utf-8") as f:
            json.dump(serializable_config, f, indent=4)
            f.flush(); os.fsync(f.fileno())
        os.replace(temp_file_path, STRATEGY_PARAMS_FILE)
    except Exception as e:
        print(f"保存策略参数到文件 {STRATEGY_PARAMS_FILE} 失败: {e}\n{traceback.format_exc()}")
        if os.path.exists(temp_file_path):
            try: os.remove(temp_file_path)
            except Exception as rm_err: print(f"清理临时文件 {temp_file_path} 失败: {rm_err}")

# --- 核心业务逻辑函数 ---
async def load_live_signals_async():
    global live_signals
    with live_signals_lock:
        try:
            if os.path.exists(LIVE_SIGNALS_FILE):
                with open(LIVE_SIGNALS_FILE, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip(): live_signals = json.loads(content)
                    else: live_signals = []
            else: live_signals = []
        except json.JSONDecodeError: print(f"加载历史信号失败 (JSON解析错误): {LIVE_SIGNALS_FILE}"); live_signals = []
        except Exception as e: print(f"加载历史信号失败 ({LIVE_SIGNALS_FILE}): {str(e)}"); live_signals = []
    if live_signals: print(f"已加载 {len(live_signals)} 个历史实时信号从 {LIVE_SIGNALS_FILE}")

async def save_live_signals_async():
    with live_signals_lock:
        try:
            with open(LIVE_SIGNALS_FILE, 'w', encoding='utf-8') as f:
                json.dump(live_signals, f, default=str, indent=4)
        except Exception as e: print(f"保存实时信号失败 ({LIVE_SIGNALS_FILE}): {str(e)}")

async def process_kline_queue():
    while True:
        try:
            kline_data = signals_queue.get_nowait()
            await handle_kline_data(kline_data)
        except Empty: await asyncio.sleep(0.01)
        except Exception as e:
            print(f"处理K线队列时出错: {e}\n{traceback.format_exc()}"); await asyncio.sleep(1)

async def background_signal_verifier():
    global live_signals
    while True:
        await asyncio.sleep(60) # 每分钟检查一次
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

                kline_df = binance_client.get_historical_klines(
                    signal_copy_to_verify['symbol'], 
                    '1m', 
                    start_time=int(end_time_utc.timestamp() * 1000), 
                    limit=1
                )
                
                actual_price = None
                if not kline_df.empty:
                    actual_price = float(kline_df.iloc[0]['close'])
                else: 
                    print(f"验证 {signal_copy_to_verify['id']}: 未找到 {end_time_utc.isoformat()} 的1m K线，尝试获取最新价...")
                    try:
                        actual_price = binance_client.get_latest_price(signal_copy_to_verify['symbol'])
                    except Exception as e_latest:
                         print(f"获取 {signal_copy_to_verify['symbol']} 最新价格失败: {e_latest}")

                if actual_price is None:
                    print(f"无法获取 {signal_copy_to_verify['symbol']} 的结束价格进行验证。信号ID: {signal_copy_to_verify['id']}")
                    continue

                signal_price_val = signal_copy_to_verify['signal_price']
                if signal_price_val is None: print(f"信号 {signal_copy_to_verify['id']} 缺少 signal_price。"); continue

                change_pct = ((actual_price - signal_price_val) / signal_price_val) * 100 if signal_price_val != 0 else 0
                correct = (signal_copy_to_verify['signal'] == 1 and actual_price > signal_price_val) or \
                          (signal_copy_to_verify['signal'] == -1 and actual_price < signal_price_val)
                pnl = change_pct if signal_copy_to_verify['signal'] == 1 else -change_pct 

                inv_amt = signal_copy_to_verify.get('investment_amount', 0)
                profit_r = signal_copy_to_verify.get('profit_rate_pct', 80.0) / 100.0
                loss_r = signal_copy_to_verify.get('loss_rate_pct', 100.0) / 100.0
                actual_pnl_amt = inv_amt * profit_r if correct else -(inv_amt * loss_r) if inv_amt > 0 else 0.0

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

        with live_signals_lock:
            verified_list_global = [s for s in live_signals if s.get('verified')]
            total_verified_global = len(verified_list_global)
            total_correct_global = sum(1 for s in verified_list_global if s.get('result'))
            total_actual_profit_loss_amount_global = sum(s.get('actual_profit_loss_amount', 0.0) for s in verified_list_global if s.get('actual_profit_loss_amount') is not None)
            total_pnl_pct_sum_global = sum(s.get('pnl_pct', 0.0) for s in verified_list_global if s.get('pnl_pct') is not None)
            average_pnl_pct_global = total_pnl_pct_sum_global / total_verified_global if total_verified_global > 0 else 0
            stats_payload = {
                "total_signals": len(live_signals), "total_verified": total_verified_global,
                "total_correct": total_correct_global,
                "win_rate": round(total_correct_global / total_verified_global * 100 if total_verified_global > 0 else 0, 2),
                "total_pnl_pct": round(total_pnl_pct_sum_global, 2), 
                "average_pnl_pct": round(average_pnl_pct_global, 2), 
                "total_profit_amount": round(total_actual_profit_loss_amount_global, 2)
            }
        await manager.broadcast_json({"type": "stats_update", "data": stats_payload})

async def _send_autox_command(client_ws: WebSocket, command: Dict[str, Any]):
    try:
        await client_ws.send_json(command)
        print(f"已向AutoX客户端 {active_autox_clients.get(client_ws, {}).get('client_id', '未知')} 发送指令: {command.get('type')}")
        log_entry_data = {
            "client_id": active_autox_clients.get(client_ws, {}).get('client_id', '未知'),
            "signal_id": command.get("payload", {}).get("signal_id"),
            "command_type": command.get("type"),
            "command_payload": command.get("payload"),
            "status": "command_sent_to_client",
            "details": f"指令已发送给客户端。",
        }
        with autox_trade_logs_lock:
            autox_trade_logs.append(AutoXTradeLogEntry(**log_entry_data).model_dump())
            if len(autox_trade_logs) > MAX_AUTOX_LOG_ENTRIES:
                autox_trade_logs.pop(0)
    except Exception as e:
        print(f"向AutoX客户端发送指令失败: {e}")

async def handle_kline_data(kline_data: dict):
    global live_signals, strategy_parameters_config, running_live_test_configs, active_autox_clients
    try:
        kline_symbol = kline_data.get('symbol')
        kline_interval = kline_data.get('interval')
        is_kline_closed = kline_data.get('is_kline_closed', False)
        if not (kline_symbol and kline_interval and is_kline_closed):
            return

        active_test_configs_for_this_kline = []
        with running_live_test_configs_lock:
            for config_id, config_content in running_live_test_configs.items():
                if config_content.get('symbol') == kline_symbol and \
                   config_content.get('interval') == kline_interval:
                    config_content_copy = config_content.copy()
                    config_content_copy['_should_autox_trigger'] = config_content.get('autox_enabled', True)
                    active_test_configs_for_this_kline.append({**config_content_copy, '_config_id': config_id})

        if not active_test_configs_for_this_kline:
            return

        for live_test_config_data in active_test_configs_for_this_kline:
            pred_strat_id = live_test_config_data['prediction_strategy_id']
            pred_params = live_test_config_data.get('prediction_strategy_params')
            if pred_params is None: 
                pred_params = strategy_parameters_config.get("prediction_strategies", {}).get(pred_strat_id, {})
                if not pred_params: 
                    pred_def = next((s for s in get_available_strategies() if s['id'] == pred_strat_id), None)
                    if pred_def and 'parameters' in pred_def:
                        pred_params = {p['name']: p['default'] for p in pred_def['parameters']}
            
            hist_df = binance_client.get_historical_klines(live_test_config_data['symbol'], live_test_config_data['interval'], limit=200)
            if hist_df.empty: continue

            pred_strat_info = next((s for s in get_available_strategies() if s['id'] == pred_strat_id), None)
            if not pred_strat_info: continue
            
            signal_df = pred_strat_info['class'](params=pred_params).generate_signals(hist_df.copy())
            if signal_df.empty or 'signal' not in signal_df.columns or 'confidence' not in signal_df.columns: continue
            
            latest_sig_data = signal_df.iloc[-1]
            sig_val = int(latest_sig_data.get('signal', 0))
            conf_val = float(latest_sig_data.get('confidence', 0))
            current_confidence_threshold = live_test_config_data.get('confidence_threshold', 0)

            if sig_val != 0 and conf_val >= current_confidence_threshold:
                event_period_minutes = {'10m': 10, '30m': 30, '1h': 60, '1d': 1440}.get(live_test_config_data.get("event_period", "10m"), 10)
                sig_time_dt = datetime.fromtimestamp(kline_data.get('kline_start_time', time.time() * 1000) / 1000, tz=timezone.utc)
                exp_end_time_dt = sig_time_dt + timedelta(minutes=event_period_minutes)
                sig_price = float(latest_sig_data['close'])
                
                inv_amount = 20.0; profit_pct = 80.0; loss_pct = 100.0
                inv_settings_from_config = live_test_config_data.get("investment_settings")

                if inv_settings_from_config:
                    try:
                        live_inv_model = InvestmentStrategySettings(**inv_settings_from_config)
                        profit_pct = live_inv_model.profitRate; loss_pct = live_inv_model.lossRate
                        inv_strat_id_cfg = live_inv_model.strategy_id
                        inv_specific_params_cfg = inv_settings_from_config.get('investment_strategy_specific_params', {})
                        final_inv_calc_params_from_global = strategy_parameters_config.get("investment_strategies", {}).get(inv_strat_id_cfg, {})
                        inv_strat_def_cfg = next((s for s in get_available_investment_strategies() if s['id'] == inv_strat_id_cfg), None)
                        default_inv_params_from_def = {}
                        if inv_strat_def_cfg and 'parameters' in inv_strat_def_cfg:
                            default_inv_params_from_def = { p['name']: p['default'] for p in inv_strat_def_cfg['parameters'] if not p.get('readonly') and p.get('name') not in ['amount', 'minAmount', 'maxAmount'] }
                        final_inv_calc_params = {**default_inv_params_from_def, **final_inv_calc_params_from_global, **inv_specific_params_cfg}
                        if inv_strat_id_cfg == 'fixed' and 'amount' not in final_inv_calc_params:
                             final_inv_calc_params['amount'] = live_inv_model.amount
                        if inv_strat_def_cfg:
                            inv_instance = inv_strat_def_cfg['class'](params=final_inv_calc_params)
                            current_balance_for_calc = live_inv_model.amount 
                            if inv_strat_id_cfg == 'percentage_of_balance' and live_inv_model.simulatedBalance is not None:
                                current_balance_for_calc = live_inv_model.simulatedBalance
                            inv_amount = inv_instance.calculate_investment( current_balance=current_balance_for_calc, previous_trade_result=None, base_investment_from_settings=live_inv_model.amount )
                            inv_amount = max(live_inv_model.minAmount, min(live_inv_model.maxAmount, inv_amount))
                        else:
                            inv_amount = max(live_inv_model.minAmount, min(live_inv_model.maxAmount, live_inv_model.amount))
                    except Exception as e: print(f"实时信号投资金额计算错误 (Config ID: {live_test_config_data['_config_id']}): {e}\n{traceback.format_exc()}")

                signal_id_str = f"{live_test_config_data['symbol']}_{live_test_config_data['interval']}_{pred_strat_id}_{int(sig_time_dt.timestamp())}_{random.randint(100,999)}"
                new_live_signal = {
                    'id': signal_id_str,
                    'symbol': live_test_config_data['symbol'], 'interval': live_test_config_data['interval'], 
                    'prediction_strategy_id': pred_strat_id, 'prediction_strategy_params': pred_params, 
                    'signal_time': format_for_display(sig_time_dt), 'signal': sig_val, 'confidence': conf_val, 
                    'signal_price': sig_price, 'event_period': live_test_config_data.get("event_period", "10m"),
                    'expected_end_time': format_for_display(exp_end_time_dt),
                    'investment_amount': round(inv_amount, 2),
                    'profit_rate_pct': profit_pct, 'loss_rate_pct': loss_pct,
                    'verified': False, 'origin_config_id': live_test_config_data['_config_id'],
                    'autox_triggered_info': None
                }
                
                should_trigger_autox = live_test_config_data.get('_should_autox_trigger', False)
                AUTOX_GLOBAL_ENABLED = True 

                if should_trigger_autox and AUTOX_GLOBAL_ENABLED:
                    target_autox_client_ws = None
                    with autox_clients_lock:
                        for ws_client, client_info in active_autox_clients.items():
                            if client_info.get('status') == 'idle' and \
                               new_live_signal['symbol'] in client_info.get('supported_symbols', []):
                                target_autox_client_ws = ws_client
                                client_info['status'] = 'processing_trade' 
                                client_info['last_signal_id'] = new_live_signal['id']
                                break
                    
                    if target_autox_client_ws:
                        trade_command_payload = {
                            "signal_id": new_live_signal['id'],
                            "symbol": new_live_signal['symbol'],
                            "direction": "up" if new_live_signal['signal'] == 1 else "down",
                            "amount": str(new_live_signal['investment_amount']),
                            "timestamp": new_live_signal['signal_time']
                        }
                        command_to_send = {"type": "execute_trade", "payload": trade_command_payload}
                        await _send_autox_command(target_autox_client_ws, command_to_send)
                        new_live_signal['autox_triggered_info'] = {
                            "client_id": active_autox_clients.get(target_autox_client_ws, {}).get('client_id'),
                            "sent_at": format_for_display(now_utc()),
                            "status": "command_sent"
                        }
                    else:
                        print(f"信号 {new_live_signal['id']} 符合AutoX触发条件，但未找到合适的空闲AutoX客户端。")
                        new_live_signal['autox_triggered_info'] = {"status": "no_available_client"}

                with live_signals_lock: live_signals.append(new_live_signal)
                await save_live_signals_async()
                
                print(f"有效交易信号 (Config ID: {live_test_config_data['_config_id']}): ID={new_live_signal['id']}, 交易对={new_live_signal['symbol']}_{new_live_signal['interval']}, 方向={'上涨' if new_live_signal['signal'] == 1 else '下跌'}, 投资额={new_live_signal['investment_amount']:.2f}, AutoX: {new_live_signal['autox_triggered_info']}")
                
                await manager.broadcast_json(
                    {"type": "new_signal", "data": new_live_signal},
                    filter_func=lambda c: websocket_to_config_id_map.get(c) == new_live_signal['origin_config_id']
                )
    except Exception as e: print(f"处理K线数据时发生严重错误: {e}\n{traceback.format_exc()}")


def kline_callback_wrapper(kline_data):
    try: signals_queue.put_nowait(kline_data)
    except Exception as e: print(f"kline_callback_wrapper 中发生错误: {e}")

# --- K线流管理辅助函数 ---
async def start_kline_websocket_if_needed(symbol: str, interval: str):
    stream_key = f"{symbol}_{interval}"
    with active_kline_streams_lock:
        current_refs = active_kline_streams.get(stream_key, 0)
        if current_refs == 0:
            try:
                binance_client.start_kline_websocket(symbol, interval, kline_callback_wrapper)
                print(f"已为 {symbol} {interval} 启动K线流 (首次需要)。")
            except Exception as e:
                print(f"为 {symbol} {interval} 启动K线流失败: {e}")
                raise
        active_kline_streams[stream_key] = current_refs + 1

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
        if new_refs == 0:
            try:
                binance_client.stop_kline_websocket(symbol, interval)
                print(f"已停止 {symbol} {interval} 的K线流 (不再需要)。")
            except Exception as e:
                print(f"停止 {symbol} {interval} 的K线流失败: {e}")
            if stream_key in active_kline_streams:
                 del active_kline_streams[stream_key]

# --- FastAPI 事件和API端点 ---
@app.on_event("startup")
async def startup_event():
    await load_live_signals_async()
    await load_strategy_parameters_from_file()
    await load_autox_clients_from_file() # 加载 AutoX 客户端数据
    asyncio.create_task(process_kline_queue())
    asyncio.create_task(background_signal_verifier())
    print("应用启动完成。后台测试配置为空，K线流未启动。")

@app.on_event("shutdown")
async def shutdown_event():
    print("应用关闭，停止币安WebSocket连接...")
    binance_client.stop_all_websockets() 
    print("币安WebSocket连接已停止。")

# --- WebSocket 端点 for Web UI (/ws/live-test) ---
@app.websocket("/ws/live-test")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        with live_signals_lock:
            initial_signals_to_send = sorted( [s.copy() for s in live_signals], key=lambda s: s.get('signal_time', ''), reverse=True )[:50]
        await websocket.send_json({"type": "initial_signals", "data": initial_signals_to_send})

        with live_signals_lock: # 发送初始全局统计
            verified_list_global_init = [s for s in live_signals if s.get('verified')]
            total_verified_global_init = len(verified_list_global_init)
            total_correct_global_init = sum(1 for s in verified_list_global_init if s.get('result'))
            total_actual_profit_loss_amount_global_init = sum(s.get('actual_profit_loss_amount', 0.0) for s in verified_list_global_init if s.get('actual_profit_loss_amount') is not None)
            stats_payload_init = {
                "total_signals": len(live_signals), "total_verified": total_verified_global_init,
                "total_correct": total_correct_global_init,
                "win_rate": round(total_correct_global_init / total_verified_global_init * 100 if total_verified_global_init > 0 else 0, 2),
                "total_pnl_pct": round(sum(s.get('pnl_pct', 0.0) for s in verified_list_global_init if s.get('pnl_pct') is not None), 2),
                "average_pnl_pct": round(sum(s.get('pnl_pct', 0.0) for s in verified_list_global_init if s.get('pnl_pct') is not None) / total_verified_global_init if total_verified_global_init > 0 else 0, 2),
                "total_profit_amount": round(total_actual_profit_loss_amount_global_init, 2)
            }
        await websocket.send_json({"type": "initial_stats", "data": stats_payload_init})
        
        while True: 
            data = await websocket.receive_json()
            message_type = data.get('type')

            if message_type == 'restore_session':
                client_config_id = data.get('data', {}).get('config_id')
                restored_config_data = None
                if client_config_id:
                    with running_live_test_configs_lock: restored_config_data = running_live_test_configs.get(client_config_id)
                if restored_config_data:
                    websocket_to_config_id_map[websocket] = client_config_id
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
                    validated_investment_settings = InvestmentStrategySettings(**config_payload_data["investment_settings"]).model_dump()
                    config_payload_data["investment_settings"] = validated_investment_settings
                except Exception as e_val:
                     await websocket.send_json({"type": "error", "data": {"message": f"配置数据验证失败: {str(e_val)}"}})
                     continue

                existing_config_id = websocket_to_config_id_map.pop(websocket, None)
                if existing_config_id: 
                    with running_live_test_configs_lock: old_config_content = running_live_test_configs.pop(existing_config_id, None)
                    if old_config_content: await stop_kline_websocket_if_not_needed(old_config_content['symbol'], old_config_content['interval'])
                
                new_config_id = uuid.uuid4().hex
                new_symbol = config_payload_data['symbol']; new_interval = config_payload_data['interval']
                try: 
                    if new_symbol != 'all' and new_interval != 'all': await start_kline_websocket_if_needed(new_symbol, new_interval)
                    full_config_to_store = { 
                        "symbol": new_symbol, "interval": new_interval,
                        "prediction_strategy_id": config_payload_data["prediction_strategy_id"],
                        "prediction_strategy_params": config_payload_data.get("prediction_strategy_params"),
                        "confidence_threshold": config_payload_data["confidence_threshold"],
                        "event_period": config_payload_data["event_period"],
                        "investment_settings": config_payload_data["investment_settings"],
                        "autox_enabled": config_payload_data.get("autox_enabled", True)
                    }
                    with running_live_test_configs_lock: running_live_test_configs[new_config_id] = full_config_to_store
                    websocket_to_config_id_map[websocket] = new_config_id
                    await websocket.send_json({"type": "config_set_confirmation", "data": {"success": True, "message": "运行时配置已应用。", "config_id": new_config_id, "applied_config": full_config_to_store}})
                except Exception as e_start_stream: 
                    await websocket.send_json({"type": "error", "data": {"message": f"应用配置时启动K线流失败: {str(e_start_stream)}"}})
                    with running_live_test_configs_lock: running_live_test_configs.pop(new_config_id, None)
                    websocket_to_config_id_map.pop(websocket, None)
            
            elif message_type == 'stop_current_test':
                config_id_to_stop = websocket_to_config_id_map.pop(websocket, None)
                stopped_config_content = None
                if config_id_to_stop:
                    with running_live_test_configs_lock: stopped_config_content = running_live_test_configs.pop(config_id_to_stop, None)
                if stopped_config_content:
                    await stop_kline_websocket_if_not_needed(stopped_config_content['symbol'], stopped_config_content['interval'])
                    await websocket.send_json({"type": "test_stopped_confirmation", "data": {"success": True, "message": "当前测试配置已停止。", "stopped_config_id": config_id_to_stop}})
                else:
                    await websocket.send_json({"type": "error", "data": {"message": "未找到活动的测试配置来停止。"}})
            await asyncio.sleep(0.1)
    except WebSocketDisconnect: print(f"客户端 {websocket.client} 断开连接。")
    except Exception as e: print(f"WebSocket端点错误 ({getattr(websocket, 'client', 'N/A')}): {e}\n{traceback.format_exc()}")
    finally:
        manager.disconnect(websocket)
        config_id_to_clean = websocket_to_config_id_map.pop(websocket, None)
        # 注释掉下面这部分，因为不应该在UI断开时自动停止测试，除非显式停止
        # if config_id_to_clean:
        #     with running_live_test_configs_lock:
        #         config_to_stop_on_disconnect = running_live_test_configs.pop(config_id_to_clean, None)
        #     if config_to_stop_on_disconnect:
        #         print(f"客户端断开，停止其关联的测试配置: {config_id_to_clean}")
        #         await stop_kline_websocket_if_not_needed(
        #             config_to_stop_on_disconnect['symbol'], 
        #             config_to_stop_on_disconnect['interval']
        #         )


# --- WebSocket 端点 for AutoX Clients (/ws/autox_control) ---
@app.websocket("/ws/autox_control")
async def autox_websocket_endpoint(websocket: WebSocket):
    await autox_manager.connect(websocket)
    client_id_local = None
    try:
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type")
            payload = data.get("payload", {})

            if message_type == "register":
                client_id = payload.get("client_id")
                supported_symbols_list = payload.get("supported_symbols", ["BTCUSDT"])
                client_notes_from_payload = payload.get("notes") # 客户端可以在注册时提供备注

                if not client_id:
                    await websocket.send_json({"type": "error", "message": "注册失败：client_id 不能为空。"})
                    await websocket.close(code=1008)
                    return

                client_id_local = client_id

                with autox_clients_lock:
                    existing_client_info = None
                    for ws, info in list(active_autox_clients.items()):
                        if info.get('client_id') == client_id:
                            print(f"AutoX客户端 {client_id} 重复连接，断开旧连接并保留信息。")
                            existing_client_info = info.copy() # 保留旧信息
                            try:
                                await ws.close(code=1000, reason="New connection for this client_id")
                            except Exception: pass # 旧连接可能已失效
                            autox_manager.disconnect(ws)
                            active_autox_clients.pop(ws, None)
                            break

                    # 创建或更新客户端信息
                    if existing_client_info:
                        # 如果是重新连接，优先使用已存在的备注，除非新连接提供了备注
                        final_notes = client_notes_from_payload if client_notes_from_payload is not None else existing_client_info.get('notes')
                        client_info_model = AutoXClientInfo(
                            client_id=client_id,
                            status=existing_client_info.get('status', 'idle'), # 保留之前的状态或默认为idle
                            supported_symbols=supported_symbols_list,
                            last_seen=now_utc(),
                            connected_at=existing_client_info.get('connected_at', now_utc()), # 保留初次连接时间
                            notes=final_notes
                        )
                    else:
                        client_info_model = AutoXClientInfo(
                            client_id=client_id,
                            supported_symbols=supported_symbols_list,
                            last_seen=now_utc(),
                            notes=client_notes_from_payload # 新客户端，使用提供的备注
                        )
                    active_autox_clients[websocket] = client_info_model.model_dump()

                print(f"AutoX客户端已注册/更新: ID={client_id}, 支持交易对={supported_symbols_list}, 备注='{client_info_model.notes or ''}'")
                await websocket.send_json({"type": "registered", "message": "客户端注册成功。", "client_info": client_info_model.model_dump(mode='json')})

                # 广播更新后的客户端列表到前端状态WebSocket
                await broadcast_autox_clients_status()
                await save_autox_clients_to_file() # 保存 AutoX 客户端数据

            elif message_type == "status_update":
                if not client_id_local:
                    await websocket.send_json({"type": "error", "message": "未注册的客户端不能发送状态更新。"})
                    continue

                with autox_clients_lock:
                    if websocket in active_autox_clients:
                        active_autox_clients[websocket]['status'] = payload.get("status", active_autox_clients[websocket].get('status', 'idle'))
                        active_autox_clients[websocket]['last_seen'] = now_utc().isoformat() # 确保是 isoformat
                        if payload.get("status") in ["trade_ready_for_confirmation", "trade_execution_failed", "manual_confirmation_pending", "idle"]: # 增加idle
                             active_autox_clients[websocket]['status'] = "idle"
                             active_autox_clients[websocket].pop('last_signal_id', None)

                log_entry_data = {
                    "client_id": client_id_local, "signal_id": payload.get("signal_id"),
                    "command_type": "status_from_client", "command_payload": payload,
                    "status": payload.get("status", "unknown_client_status"),
                    "details": payload.get("details"), "error_message": payload.get("error_message"),
                }
                with autox_trade_logs_lock:
                    autox_trade_logs.append(AutoXTradeLogEntry(**log_entry_data).model_dump())
                    if len(autox_trade_logs) > MAX_AUTOX_LOG_ENTRIES: autox_trade_logs.pop(0)

                print(f"收到AutoX客户端 {client_id_local} 状态更新: {payload.get('status')}, Signal ID: {payload.get('signal_id')}")
                # 广播更新后的客户端列表到前端状态WebSocket
                await broadcast_autox_clients_status()


                await save_autox_clients_to_file() # 保存 AutoX 客户端数据

            elif message_type == "pong":
                with autox_clients_lock:
                     if websocket in active_autox_clients:
                        active_autox_clients[websocket]['last_seen'] = now_utc().isoformat()

            else:
                print(f"收到来自AutoX客户端 {client_id_local or '未知'} 的未知消息类型: {message_type}")
                await websocket.send_json({"type": "error", "message": f"不支持的消息类型: {message_type}"})

    except WebSocketDisconnect:
        print(f"AutoX客户端 {client_id_local or getattr(websocket, 'client', 'N/A')} 断开连接。")
        # 当客户端断开时，我们可以选择保留其信息（例如备注），但标记为离线
        # 或者完全移除。这里我们选择保留信息，但可以考虑增加一个 "offline_at"字段或修改 status.
        # 当前 active_autox_clients 是以 websocket 对象为 key，所以断开后自然会移除。
        # 如果需要持久化客户端列表（即使离线也显示），则需要不同的存储方式。
    except Exception as e:
        print(f"AutoX WebSocket端点错误 ({client_id_local or getattr(websocket, 'client', 'N/A')}): {e}\n{traceback.format_exc()}")
    finally:
        autox_manager.disconnect(websocket)
        with autox_clients_lock:
            client_info_at_disconnect = active_autox_clients.pop(websocket, None) # 从活动连接中移除
            if client_info_at_disconnect:
                print(f"AutoX客户端 {client_info_at_disconnect.get('client_id', '未知')} 已从活动列表移除。")
                # 如果需要保留离线客户端信息，可以在这里将其信息转移到另一个列表或数据库
                # 例如: offline_autox_clients[client_info_at_disconnect['client_id']] = client_info_at_disconnect
                # 并在 GET /api/autox/clients 中合并在线和离线客户端信息
        # 客户端断开连接后，广播更新后的客户端列表到前端状态WebSocket
        await broadcast_autox_clients_status()
        await save_autox_clients_to_file() # 保存 AutoX 客户端数据


# --- WebSocket 端点 for AutoX Status (/ws/autox_status) ---
@app.websocket("/ws/autox_status")
async def autox_status_websocket_endpoint(websocket: WebSocket):
    await autox_status_manager.connect(websocket)
    try:
        # 连接成功后立即发送当前客户端列表
        await broadcast_autox_clients_status()
        while True:
            # 这个端点主要用于发送，但为了保持连接，可以接收消息（例如ping/pong或心跳）
            # 目前我们不需要处理任何特定消息，可以简单地等待或处理连接断开
            await websocket.receive_text() # 或者 receive_json() 如果前端会发送消息
    except WebSocketDisconnect:
        print(f"AutoX状态前端客户端 {getattr(websocket, 'client', 'N/A')} 断开连接。")
    except Exception as e:
        print(f"AutoX状态WebSocket端点错误 ({getattr(websocket, 'client', 'N/A')}): {e}\n{traceback.format_exc()}")
    finally:
        autox_status_manager.disconnect(websocket)


# --- 辅助函数：广播AutoX客户端状态 ---
async def broadcast_autox_clients_status():
    """广播当前AutoX客户端列表到所有连接的AutoX状态前端WebSocket。"""
    with autox_clients_lock:
        clients_data = []
        for ws, info_dict in active_autox_clients.items():
            client_model = AutoXClientInfo(**info_dict)
            clients_data.append(client_model.model_dump(mode='json'))

    payload = {"type": "autox_clients_update", "data": clients_data}
    await autox_status_manager.broadcast_json(payload)
    print(f"已广播AutoX客户端状态更新到 {len(autox_status_manager.active_connections)} 个前端连接。")


# --- HTML 页面路由 ---
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
async def get_symbols_endpoint():
    try:
        hot_symbols = binance_client.get_hot_symbols_by_volume(top_n=50) 
        all_symbols = binance_client.get_available_symbols() 
        combined_symbols = []; seen_symbols = set()
        if hot_symbols:
            for s in hot_symbols:
                if s not in seen_symbols: combined_symbols.append(s); seen_symbols.add(s)
        if all_symbols:
            for s in sorted(all_symbols): 
                 if s not in seen_symbols: combined_symbols.append(s); seen_symbols.add(s)
        return combined_symbols if combined_symbols else ["BTCUSDT", "ETHUSDT"]
    except Exception:
        return ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT"]

@app.get("/api/prediction-strategies", response_model=List[Dict[str, Any]])
async def get_prediction_strategies_endpoint():
    try: return [{'id': s['id'], 'name': s['name'], 'description': s['description'], 'parameters': s['parameters']} for s in get_available_strategies()]
    except Exception as e: raise HTTPException(status_code=500, detail=f"获取预测策略失败: {str(e)}")

@app.get("/api/investment-strategies", response_model=List[Dict[str, Any]])
async def get_investment_strategies_endpoint():
    try: return [{'id': s['id'], 'name': s['name'], 'description': s['description'], 'parameters': s['parameters']} for s in get_available_investment_strategies()]
    except Exception as e: raise HTTPException(status_code=500, detail=f"获取投资策略失败: {str(e)}")

@app.post("/api/backtest")
async def run_backtest_endpoint(request: BacktestRequest):
    global strategy_parameters_config
    try:
        start_utc = to_utc(request.start_time); end_utc = to_utc(request.end_time)
        if start_utc >= end_utc or end_utc > now_utc() or start_utc > now_utc(): raise HTTPException(status_code=400, detail="回测时间范围无效。")
        df_klines = binance_client.get_historical_klines(request.symbol, request.interval, int(start_utc.timestamp() * 1000), int(end_utc.timestamp() * 1000))
        if df_klines.empty: raise HTTPException(status_code=404, detail="未找到指定范围的K线数据。")
        pred_id = request.prediction_strategy_id
        final_pred_params = request.prediction_strategy_params if request.prediction_strategy_params is not None else strategy_parameters_config.get("prediction_strategies", {}).get(pred_id, {})
        if not final_pred_params: 
            pred_def_info = next((s for s in get_available_strategies() if s['id'] == pred_id), None)
            if pred_def_info and 'parameters' in pred_def_info: final_pred_params = {p['name']: p['default'] for p in pred_def_info['parameters']}
        pred_strategy_definition = next((s for s in get_available_strategies() if s['id'] == pred_id), None)
        if not pred_strategy_definition: raise HTTPException(status_code=404, detail=f"未找到预测策略ID: {pred_id}")
        prediction_instance = pred_strategy_definition['class'](params=final_pred_params)
        inv_id = request.investment.investment_strategy_id
        final_inv_params = request.investment.investment_strategy_specific_params if request.investment.investment_strategy_specific_params is not None else strategy_parameters_config.get("investment_strategies", {}).get(inv_id, {})
        if not final_inv_params: 
            inv_def_info = next((s for s in get_available_investment_strategies() if s['id'] == inv_id), None)
            if inv_def_info and 'parameters' in inv_def_info: final_inv_params = {p['name']: p['default'] for p in inv_def_info['parameters'] if not p.get('readonly')}
        investment_args = request.investment.model_dump(exclude={'investment_strategy_id', 'investment_strategy_specific_params'})
        backtester_instance = Backtester( df=df_klines, strategy=prediction_instance, event_period=request.event_period, confidence_threshold=request.confidence_threshold, investment_strategy_id=inv_id, investment_strategy_params=final_inv_params, **investment_args )
        results_data = backtester_instance.run()
        for pred_item_data in results_data.get('predictions', []):
            for time_key_str in ['signal_time', 'end_time_expected', 'end_time_actual']:
                if time_key_str in pred_item_data and isinstance(pred_item_data[time_key_str], datetime):
                    pred_item_data[time_key_str] = format_for_display(pred_item_data[time_key_str]) 
        return results_data
    except HTTPException as http_exc: raise http_exc
    except Exception as exc:
        error_detail_msg = f"回测过程中发生错误: {str(exc)}"; print(f"{error_detail_msg}\n{traceback.format_exc()}"); raise HTTPException(status_code=500, detail=error_detail_msg)

@app.get("/api/live-signals")
async def get_live_signals_http_endpoint():
    with live_signals_lock: return [s.copy() for s in live_signals]

@app.get("/api/test-signal")
async def generate_test_signal():
    current_time = now_utc(); signal_time = current_time; end_time = current_time + timedelta(minutes=10)
    symbol = random.choice(["BTCUSDT", "ETHUSDT", "BNBUSDT"])
    try: price_val = binance_client.get_latest_price(symbol)
    except: price_val = random.uniform(100, 70000)
    pred_id_test = "simple_rsi"; pred_params_test = strategy_parameters_config.get("prediction_strategies", {}).get(pred_id_test, {"rsi_period": 14, "rsi_overbought": 70, "rsi_oversold": 30})
    test_signal_data = {
        'id': f"TEST_{symbol}_{pred_id_test}_{int(time.time())}_{random.randint(100,999)}", 'symbol': symbol, 'interval': "1m", 'prediction_strategy_id': pred_id_test, 'prediction_strategy_params': pred_params_test,
        'signal_time': format_for_display(signal_time), 'signal': random.choice([1, -1]), 'confidence': random.uniform(60, 95), 'signal_price': price_val, 'event_period': "10m",
        'expected_end_time': format_for_display(end_time), 'investment_amount': 20.0, 'profit_rate_pct': 80.0, 'loss_rate_pct': 100.0,
        'actual_end_price': None, 'price_change_pct': None, 'result': None, 'pnl_pct': None, 'actual_profit_loss_amount': None, 'verified': False, 'verify_time': None,
        'origin_config_id': 'test_signal_broadcast_all', 'autox_triggered_info': None
    }
    with live_signals_lock: live_signals.append(test_signal_data)
    await save_live_signals_async()
    await manager.broadcast_json( {"type": "new_signal", "data": test_signal_data}, filter_func=lambda conn: True )
    return {"status": "success", "message": "测试信号已生成并广播给所有连接的客户端", "signal": test_signal_data}

@app.get("/api/load_all_strategy_parameters", response_model=Dict[str, Any])
async def load_all_strategy_parameters_endpoint():
    global strategy_parameters_config; return strategy_parameters_config

@app.post("/api/save_strategy_parameter_set")
async def save_strategy_parameter_set_endpoint(param_set: StrategyParameterSet):
    global strategy_parameters_config
    try:
        if param_set.strategy_type == "prediction": strategy_parameters_config["prediction_strategies"][param_set.strategy_id] = param_set.params
        elif param_set.strategy_type == "investment": strategy_parameters_config["investment_strategies"][param_set.strategy_id] = param_set.params
        else: raise HTTPException(status_code=400, detail="无效的 strategy_type。必须是 'prediction' 或 'investment'。")
        await save_strategy_parameters_to_file()
        return {"status": "success", "message": "策略参数已保存"}
    except HTTPException as http_exc: raise http_exc
    except Exception as e: print(f"保存策略参数失败: {e}\n{traceback.format_exc()}"); raise HTTPException(status_code=500, detail=f"保存策略参数失败: {str(e)}")

# --- AutoX管理相关的API端点 ---
@app.get("/api/autox/trade_logs", response_model=List[AutoXTradeLogEntry]) # 使用Pydantic模型
async def get_autox_trade_logs_endpoint(limit: int = Query(50, ge=1, le=MAX_AUTOX_LOG_ENTRIES)):
    with autox_trade_logs_lock:
        # 先转换为 AutoXTradeLogEntry 对象列表，再序列化，确保时间等字段正确处理
        log_objects = [AutoXTradeLogEntry(**log_dict) for log_dict in autox_trade_logs]
        sorted_logs = sorted(log_objects, key=lambda x: x.timestamp, reverse=True)
        return [log.model_dump(mode='json') for log in sorted_logs[:limit]]

@app.post("/api/autox/clients/{client_id}/send_test_command", response_model=Dict[str, Any])
async def send_test_command_to_autox_client(client_id: str, command_type: str = Query("test_echo", description="测试指令类型")):
    target_ws = None
    with autox_clients_lock:
        for ws, info in active_autox_clients.items():
            if info.get("client_id") == client_id:
                target_ws = ws
                break
    
    if not target_ws:
        raise HTTPException(status_code=404, detail=f"未找到 Client ID 为 {client_id} 的活动AutoX客户端。")

    test_payload = {"message": f"这是一个来自服务器的测试指令 ({command_type})", "timestamp": format_for_display(now_utc())}
    command_to_send = {"type": command_type, "payload": test_payload}
    
    try:
        await _send_autox_command(target_ws, command_to_send)
        return {"status": "success", "message": f"测试指令 '{command_type}' 已发送给客户端 {client_id}。"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"发送测试指令给客户端 {client_id} 失败: {str(e)}")
    
@app.post("/api/autox/clients/{client_id}/trigger_trade_command", response_model=Dict[str, Any])
async def trigger_trade_command_for_autox_client(client_id: str, trade_details: TriggerAutoXTradePayload):
    target_ws = None
    client_info_snapshot_dict = None 
    with autox_clients_lock:
        for ws, info_dict in active_autox_clients.items():
            if info_dict.get("client_id") == client_id:
                if info_dict.get("status") != "idle":
                    raise HTTPException(status_code=409, detail=f"客户端 {client_id} 当前不处于 idle 状态 (当前状态: {info_dict.get('status')})，无法发送新指令。")
                target_ws = ws
                client_info_snapshot_dict = info_dict 
                info_dict['status'] = 'processing_trade' 
                info_dict['last_signal_id'] = trade_details.signal_id or f"test_trigger_{uuid.uuid4().hex[:8]}"
                break
    
    if not target_ws:
        raise HTTPException(status_code=404, detail=f"未找到 Client ID 为 {client_id} 的活动AutoX客户端。")

    signal_id_to_use = trade_details.signal_id or f"test_trigger_{uuid.uuid4().hex[:8]}"
    command_payload = {
        "signal_id": signal_id_to_use, "symbol": trade_details.symbol,
        "direction": trade_details.direction, "amount": trade_details.amount,
        "timestamp": format_for_display(now_utc())
    }
    command_to_send = {"type": "execute_trade", "payload": command_payload}
    
    try:
        await _send_autox_command(target_ws, command_to_send)
        log_entry_data_trigger = {
            "client_id": client_id, "signal_id": signal_id_to_use,
            "command_type": "test_triggered_execute_trade", "command_payload": command_payload,
            "status": "test_command_sent_to_client",
            "details": f"测试触发的 execute_trade 指令已发送给客户端 {client_id}。",
        }
        with autox_trade_logs_lock:
            autox_trade_logs.append(AutoXTradeLogEntry(**log_entry_data_trigger).model_dump())
            if len(autox_trade_logs) > MAX_AUTOX_LOG_ENTRIES: autox_trade_logs.pop(0)

        return {"status": "success", "message": f"'execute_trade' 指令已作为测试发送给客户端 {client_id}。", "sent_command": command_to_send}
    except Exception as e:
        if client_info_snapshot_dict: 
             with autox_clients_lock: # 确保线程安全地修改回状态
                # 再次查找以防 target_ws 对应的条目已变化或移除
                for ws_iter, info_iter_dict in active_autox_clients.items():
                    if ws_iter == target_ws and info_iter_dict.get('client_id') == client_id:
                        info_iter_dict['status'] = 'idle'
                        info_iter_dict.pop('last_signal_id', None)
                        break
        raise HTTPException(status_code=500, detail=f"发送 'execute_trade' 指令给客户端 {client_id} 失败: {str(e)}")

@app.post("/api/autox/clients/{client_id}/notes", response_model=AutoXClientInfo)
async def update_client_notes_endpoint(client_id: str, notes_payload: ClientNotesPayload):
    """更新指定AutoX客户端的备注。"""
    updated_client_info = None
    with autox_clients_lock:
        for ws, client_info_dict in active_autox_clients.items():
            if client_info_dict.get("client_id") == client_id:
                client_info_dict["notes"] = notes_payload.notes
                # 由于 active_autox_clients 的值是字典，这里直接更新了原字典
                # 为了返回 AutoXClientInfo 模型，我们基于更新后的字典创建一个新模型实例
                updated_client_info = AutoXClientInfo(**client_info_dict).model_dump(mode='json')
                break
    
    if not updated_client_info:
        raise HTTPException(status_code=404, detail=f"未找到 Client ID 为 {client_id} 的活动AutoX客户端。")
    
    print(f"客户端 {client_id} 的备注已更新为: '{notes_payload.notes}'")
    await save_autox_clients_to_file() # 保存 AutoX 客户端数据
    return updated_client_info
 
 
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# --- END OF FILE main.py ---