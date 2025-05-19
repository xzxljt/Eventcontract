# --- START OF FILE binance_client.py ---

import os
import ssl # 添加ssl模块导入，用于WebSocket连接配置
from dotenv import load_dotenv
load_dotenv(override=True) # 强制覆盖已存在的同名环境变量
use_proxy_str = os.environ.get('USE_PROXY', 'false').lower()
should_use_proxy = use_proxy_str == 'true'

if should_use_proxy:
    proxy_https_url = os.environ.get('PROXY_HTTPS_URL')
    proxy_http_url = os.environ.get('PROXY_HTTP_URL')

    if proxy_https_url:
        print(f"从 .env 设置代理: HTTPS_PROXY={proxy_https_url}")
        os.environ['HTTPS_PROXY'] = proxy_https_url # 实际设置给系统环境，供后续库使用
    if proxy_http_url:
        print(f"从 .env 设置代理: HTTP_PROXY={proxy_http_url}")
        os.environ['HTTP_PROXY'] = proxy_http_url   # 实际设置给系统环境
else:
    print("根据 .env 配置，不使用代理。")

import time # 用于时间相关操作，例如时间戳和延迟
import hashlib # 用于生成哈希，如签名
import hmac # 用于生成基于哈希的消息认证码
import requests # 用于发送HTTP请求
import pandas as pd # 用于数据处理，特别是DataFrame
from datetime import datetime, timezone, timedelta # 用于处理日期和时间
from typing import Optional, List, Dict, Any # Python类型提示
from urllib.parse import urlencode # 用于将字典编码为URL查询字符串
import json # 用于处理JSON数据
# import hashlib # 重复导入，移除
import logging # 导入logging模块
import threading # 新增
import websocket # 新增 (确保 websocket-client 已安装)
from functools import partial # 新增

# 配置日志记录
logger = logging.getLogger(__name__)

class BinanceClient:
    BASE_URL = "https://api.binance.com" # 现货 API 基础URL
    CACHE_DIR = ".kline_cache" # K线数据缓存目录
    FAPI_BASE_URL = "https://fapi.binance.com" # U本位合约 API 基础URL
    
    # K线数据节流日志间隔 (秒)
    _kline_log_interval_sec = 30 # 默认为30秒

    # K线时间间隔对应的毫秒数
    INTERVAL_MS = {
        '1m': 60000, '3m': 180000, '5m': 300000, '15m': 900000,
        '30m': 1800000, '1h': 3600000, '2h': 7200000, '4h': 14400000,
        '6h': 21600000, '8h': 28800000, '12h': 43200000, '1d': 86400000,
        '3d': 259200000, '1w': 604800000, '1M': 2592000000 # 1个月约等于30天
    }
    MAX_KLINE_LIMIT = 1000 # 币安API单次请求K线数据的最大数量限制

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        # 优先使用传入的API密钥，否则尝试从环境变量加载
        self.api_key = api_key or os.environ.get('BINANCE_API_KEY')
        # 确保缓存目录存在
        if not os.path.exists(self.CACHE_DIR):
            os.makedirs(self.CACHE_DIR)
            print(f"创建K线缓存目录: {self.CACHE_DIR}")

        self.api_secret = api_secret or os.environ.get('BINANCE_API_SECRET')
        self.session = requests.Session() # 使用Session可以保持TCP连接，提高效率
        if self.api_key: # 只有在提供了API Key时才设置请求头
            self.session.headers.update({'X-MBX-APIKEY': self.api_key})
        
        # 用于管理WebSocket连接
        self.ws_connections: Dict[str, Any] = {} # 存储活动的WebSocket连接对象
        self.ws_threads: Dict[str, threading.Thread] = {} # 存储运行WebSocket的线程对象, 类型提示
        # 用于标记是否是主动停止的WebSocket连接，避免意外断线重连
        self._intentional_stops: Dict[str, bool] = {} # 键为 ws_key
        # 用于记录上次打印K线日志的时间
        self._last_kline_log_time: Dict[str, float] = {} # 键为 ws_key, 值为时间戳 (秒)
        
        # WebSocket 重连配置和状态
        # 从环境变量读取WebSocket重连设置，如果不存在则使用默认值
        self._max_reconnect_attempts = int(os.getenv("WS_RECONNECT_ATTEMPTS", "10")) # 最大重连尝试次数
        self._reconnect_delay_base = int(os.getenv("WS_RECONNECT_INTERVAL", "5")) # 基础重连延迟 (秒)
        self._reconnect_attempts: Dict[str, int] = {} # 存储每个连接的重连尝试次数
        
        # WebSocket 连接监控和超时设置
        self._ws_monitor_interval = int(os.getenv("WS_MONITOR_INTERVAL", "30")) # WebSocket监控间隔（秒）
        self._ws_connection_timeout = int(os.getenv("WS_CONNECTION_TIMEOUT", "60")) # WebSocket连接超时（秒）
        self._ws_shutdown_timeout = int(os.getenv("WS_SHUTDOWN_TIMEOUT", "30")) # WebSocket关闭超时（秒）
        self._ws_last_activity: Dict[str, float] = {} # 存储每个连接的最后活动时间
        self._ws_monitor_thread = None # WebSocket监控线程
        self._ws_monitor_running = False # WebSocket监控线程运行标志
        
        self.ws_management_lock = threading.RLock() # 使用可重入锁
        
        # 启动WebSocket连接监控线程
        self._start_ws_monitor()

    def _sign_request(self, params: dict) -> str:
        """为需要签名的请求生成签名"""
        if not self.api_secret:
            raise ValueError("未设置API secret，无法对请求进行签名。")
        query_string = urlencode(params) # 将参数字典编码为查询字符串
        # 使用HMAC SHA256算法生成签名
        signature = hmac.new(self.api_secret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
        return signature

    def _get_cache_key(self, symbol: str, interval: str, start_time: Optional[int], end_time: Optional[int]) -> str:
        """根据参数生成缓存键 (使用哈希)"""
        key_str = f"{symbol}_{interval}_{start_time}_{end_time}"
        return hashlib.sha256(key_str.encode('utf-8')).hexdigest()

    def _get_cache_path(self, key: str) -> str:
        """根据缓存键获取缓存文件路径"""
        return os.path.join(self.CACHE_DIR, f"{key}.json")

    def _load_cache(self, key: str) -> Optional[List[List[Any]]]:
        """从缓存文件加载数据"""
        cache_path = self._get_cache_path(key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                # print(f"从缓存加载数据: {cache_path}") # 根据需要减少日志输出
                return data
            except (IOError, json.JSONDecodeError) as e:
                print(f"加载缓存文件失败 {cache_path}: {e}")
                # 如果加载失败，删除可能损坏的缓存文件
                try:
                    os.remove(cache_path)
                    print(f"已删除损坏的缓存文件: {cache_path}")
                except OSError as remove_err:
                    print(f"删除损坏的缓存文件失败 {cache_path}: {remove_err}")
        return None

    def _save_cache(self, key: str, data: List[List[Any]]):
        """将数据保存到缓存文件"""
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f)
            # print(f"数据已保存到缓存: {cache_path}") # 根据需要减少日志输出
        except IOError as e:
            print(f"保存数据到缓存文件失败 {cache_path}: {e}")

    def _request(self, method: str, url_path: str, params: Optional[dict] = None, signed: bool = False, is_fapi: bool = False):
        """
        发送HTTP请求到币安API的通用方法。
        """
        base = self.FAPI_BASE_URL if is_fapi else self.BASE_URL 
        full_url = f"{base}{url_path}"
        
        if signed:
            if not self.api_key or not self.api_secret:
                 raise ValueError("API key 和 secret 必须为签名请求设置。")
            if params is None: params = {}
            params['timestamp'] = int(time.time() * 1000) 
            params['signature'] = self._sign_request(params) 

        try:
            # logger.debug(f"发送 {method.upper()} 请求到 {full_url}，参数: {params}")
            # logger.debug(f"requests session 代理配置: {self.session.proxies}")
            if method.upper() == 'GET':
                response = self.session.get(full_url, params=params, timeout=(10, 30)) 
            elif method.upper() == 'POST':
                # 修正POST请求参数传递方式
                response = self.session.post(full_url, params=params if signed else None, data=None if signed else params, timeout=(10, 30))
            else:
                raise ValueError(f"不支持的HTTP方法: {method}")
            
            response.raise_for_status() 
            # logger.debug(f"请求 {full_url} 成功，状态码: {response.status_code}") 
            return response.json() 
        except requests.exceptions.HTTPError as http_err:
            error_details = ""
            try:
                error_data = response.json()
                if 'msg' in error_data: 
                    error_details = f" (币安错误信息: {error_data['msg']}, 错误码: {error_data.get('code')})"
            except ValueError: 
                error_details = f" (原始响应内容: {response.text})"
            except Exception: 
                pass
            logger.error(f"HTTP请求错误: {http_err} - URL: {response.url}{error_details}") 
            raise 
        except requests.exceptions.RequestException as req_err: 
            logger.error(f"请求 {full_url} 时发生网络或连接错误: {req_err}，参数: {params}") 
            raise 
        except Exception as e:
            logger.error(f"请求 {full_url} 时发生未知错误: {e}，参数: {params}", exc_info=True) 
            raise

    def get_server_time(self) -> int:
        """获取币安服务器时间戳 (毫秒)"""
        data = self._request('GET', '/api/v3/time') 
        return data['serverTime']

    def get_exchange_info(self, symbol: Optional[str] = None) -> dict:
        """获取交易所交易规则和交易对信息"""
        params = {}
        if symbol:
            params['symbol'] = symbol.upper()
        return self._request('GET', '/fapi/v1/exchangeInfo', params=params, is_fapi=True)

    def get_available_symbols(self, quote_asset: str = "USDT") -> List[str]:
        """获取所有可用的U本位永续合约交易对列表"""
        try:
            exchange_info = self.get_exchange_info()
            symbols = [
                s['symbol'] for s in exchange_info['symbols']
                if s['quoteAsset'] == quote_asset.upper() and \
                   s['contractType'] == 'PERPETUAL' and \
                   s['status'] == 'TRADING' 
            ]
            return sorted(symbols)
        except Exception as e:
            print(f"获取可用交易对列表失败: {e}。将返回一个默认列表。")
            return ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "DOGEUSDT", "DOTUSDT", "SOLUSDT"]

    def get_24hr_ticker_statistics(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取24小时交易量统计信息 (U本位合约)"""
        params = {}
        if symbol:
            params['symbol'] = symbol.upper()
        return self._request('GET', '/fapi/v1/ticker/24hr', params=params, is_fapi=True) # 确保使用fapi

    def get_hot_symbols_by_volume(self, quote_asset: str = "USDT", top_n: int = 50) -> List[str]:
        """
        获取热度较高的U本位永续合约交易对列表，基于24小时交易量排序。
        """
        try:
            all_usdt_perpetuals = self.get_available_symbols(quote_asset=quote_asset)
            if not all_usdt_perpetuals:
                print("未获取到任何可用U本位永续合约交易对。")
                return []

            all_tickers_fapi = self.get_24hr_ticker_statistics() # 获取所有U本位合约的24hr统计
            
            hot_symbols_data = []
            for ticker in all_tickers_fapi: # 迭代从fapi获取的ticker
                if ticker['symbol'] in all_usdt_perpetuals: # 确保是我们关心的USDT计价永续合约
                    try:
                        volume = float(ticker.get('quoteVolume', 0))
                        hot_symbols_data.append({'symbol': ticker['symbol'], 'volume': volume})
                    except (ValueError, TypeError):
                        # print(f"警告: 交易对 {ticker['symbol']} 的交易量数据无效: {ticker.get('quoteVolume')}")
                        continue 
            
            hot_symbols_data.sort(key=lambda x: x['volume'], reverse=True)
            
            top_symbols = [item['symbol'] for item in hot_symbols_data[:top_n]]
            
            # print(f"已获取并筛选出前 {top_n} 个热门U本位永续合约 (基于24小时交易量)。")
            return top_symbols

        except Exception as e:
            print(f"获取热门交易对失败: {e}。将返回一个默认列表。")
            return ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "DOGEUSDT", "DOTUSDT", "SOLUSDT"]


    def get_historical_klines(self, symbol: str, interval: str,
                              start_time: Optional[int] = None, 
                              end_time: Optional[int] = None,   
                              limit: Optional[int] = None) -> pd.DataFrame:
        all_klines_data = []
        
        if limit is not None and start_time is None:
            params = {
                'symbol': symbol.upper(), 'interval': interval,
                'limit': min(limit, self.MAX_KLINE_LIMIT) 
            }
            if end_time: params['endTime'] = end_time
            logger.info(f"获取最新的 {params['limit']} 条K线数据 (不缓存): 交易对={symbol}, 周期={interval}")
            data = self._request('GET', '/fapi/v1/klines', params=params, is_fapi=True)
            all_klines_data.extend(data)
        
        elif start_time is not None:
            cache_key = self._get_cache_key(symbol, interval, start_time, end_time)
            cached_data = self._load_cache(cache_key)

            if cached_data:
                # print(f"从缓存加载K线数据: 交易对={symbol}, 周期={interval}, 时间范围={start_time}-{end_time}")
                all_klines_data = cached_data
            else:
                current_start_time_ms = start_time
                interval_duration_ms = self.INTERVAL_MS.get(interval)
                if not interval_duration_ms:
                    raise ValueError(f"不支持的K线周期: {interval}")

                # start_dt_utc = datetime.fromtimestamp(start_time/1000, tz=timezone.utc)
                # end_dt_utc_str = datetime.fromtimestamp(end_time/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC') if end_time else "现在"
                # print(f"缓存未命中，开始分页获取K线数据: 交易对={symbol}, 周期={interval}, 从 {start_dt_utc.strftime('%Y-%m-%d %H:%M:%S UTC')} 到 {end_dt_utc_str}")
                
                fetch_iteration = 0 
                while True:
                    fetch_iteration += 1
                    current_request_limit = self.MAX_KLINE_LIMIT
                    
                    params_segment = {
                        'symbol': symbol.upper(), 'interval': interval,
                        'startTime': current_start_time_ms, 'limit': current_request_limit
                    }
                    if end_time: params_segment['endTime'] = end_time
                    
                    # current_start_dt_str = datetime.fromtimestamp(current_start_time_ms/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                    # print(f"  分页请求 #{fetch_iteration}: 开始时间={current_start_dt_str} UTC, 限制条数={current_request_limit}")

                    retry_attempts = 0; max_retries = 3; data_segment = None
                    while retry_attempts < max_retries:
                        try:
                            data_segment = self._request('GET', '/fapi/v1/klines', params=params_segment, is_fapi=True)
                            break 
                        except requests.exceptions.RequestException as req_err:
                            retry_attempts += 1
                            delay = 1 * (2 ** retry_attempts) # Exponential backoff, starting milder
                            logger.warning(f"分页获取K线时网络错误 (尝试 {retry_attempts}/{max_retries}, {symbol} {interval}): {req_err}. 等待 {delay}s.")
                            time.sleep(delay)
                        # HTTPError and other unexpected errors will be re-raised by _request and stop pagination here
                    
                    if not data_segment: # Max retries failed or API returned empty for valid reason
                        # print(f"  分页请求 #{fetch_iteration}: 未获取到数据或达到最大重试次数。")
                        break
                    
                    all_klines_data.extend(data_segment)
                    
                    last_kline_open_time_ms = int(data_segment[-1][0])
                    current_start_time_ms = last_kline_open_time_ms + interval_duration_ms
                    
                    if (end_time and current_start_time_ms > end_time) or \
                       (limit is not None and len(all_klines_data) >= limit) or \
                       (len(data_segment) < current_request_limit):
                        # print(f"  分页请求 #{fetch_iteration}: 达到结束条件。")
                        break
                    
                    time.sleep(0.15) # Reduced delay, be mindful of API rate limits

                if limit is not None and len(all_klines_data) > limit: # Ensure limit is strictly adhered to
                    all_klines_data = all_klines_data[:limit]

                if all_klines_data:
                    self._save_cache(cache_key, all_klines_data)
        else: 
            raise ValueError("get_historical_klines: 必须提供 start_time (用于分页获取指定时间范围数据) 或 limit (用于获取最新的N条数据)。")

        if not all_klines_data: 
            # print(f"未能获取交易对 {symbol} 在周期 {interval} 的K线数据。")
            return pd.DataFrame() 

        df = pd.DataFrame(all_klines_data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])

        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True) 
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True) 
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce') 
        df['number_of_trades'] = df['number_of_trades'].astype(int) 
        
        df.drop_duplicates(subset=['open_time'], keep='first', inplace=True)
        df.set_index('open_time', inplace=True) 
        df.sort_index(inplace=True) 

        # min_time_str = df.index.min().strftime('%Y-%m-%d %H:%M:%S UTC') if not df.empty else "N/A"
        # max_time_str = df.index.max().strftime('%Y-%m-%d %H:%M:%S UTC') if not df.empty else "N/A"
        # print(f"K线数据获取完成: {symbol} {interval}, 共 {len(df)} 条, 时间范围从 {min_time_str} 到 {max_time_str}")
        return df

    def get_latest_price(self, symbol: str) -> float:
        """获取指定U本位永续合约的最新价格"""
        params = {'symbol': symbol.upper()}
        data = self._request('GET', '/fapi/v1/ticker/price', params=params, is_fapi=True)
        return float(data['price'])

    # --- WebSocket 相关方法 ---
    def _get_ws_url(self, stream_name: str) -> str:
        """获取U本位合约的WebSocket基础URL"""
        return f"wss://fstream.binance.com/ws/{stream_name}" 

    def start_kline_websocket(self, symbol: str, interval: str, user_callback: callable):
        stream_name = f"{symbol.lower()}@kline_{interval}"
        ws_key = f"{symbol.lower()}_{interval}"

        with self.ws_management_lock: # Acquire lock
            if ws_key in self.ws_connections and \
               self.ws_connections[ws_key].sock and \
               self.ws_connections[ws_key].sock.connected:
                logger.info(f"交易对 {symbol} {interval} (key={ws_key}) 的K线WebSocket已经运行。实例ID: {id(self.ws_connections[ws_key])}")
                return

            # If an old instance exists (even if disconnected), try to clean it up.
            if ws_key in self.ws_connections:
                old_ws_app = self.ws_connections.pop(ws_key, None) # Remove from tracking
                self.ws_threads.pop(ws_key, None) # Remove its thread reference
                
                logger.warning(f"为 {ws_key} 启动新连接前，发现旧实例 {id(old_ws_app) if old_ws_app else 'N/A'}。尝试关闭...")
                if old_ws_app:
                    try:
                        # Mark temporarily to prevent its on_close from initiating a reconnect cycle for the OLD instance
                        self._intentional_stops[ws_key] = True 
                        old_ws_app.close()
                        logger.info(f"已请求关闭旧的 {ws_key} WebSocketApp 实例 {id(old_ws_app)}。")
                    except Exception as e_old_close:
                        logger.error(f"关闭旧的 {ws_key} WebSocketApp 实例 {id(old_ws_app)} 时出错: {e_old_close}")
            
            # Clear/reset states for this ws_key before starting a new connection
            self._reconnect_attempts.pop(ws_key, None)
            self._intentional_stops.pop(ws_key, None) # Ensure new connection doesn't carry a stale stop flag

            ws_url = self._get_ws_url(stream_name)
            logger.info(f"正在启动交易对 {symbol} {interval} (key={ws_key}) 的新K线WebSocket连接到 {ws_url}")

            # Create new WebSocketApp instance
            # Pass ws_key and other necessary args to callbacks using partial
            new_ws_app = websocket.WebSocketApp(ws_url,
                                      on_open=partial(self._on_ws_open, ws_key=ws_key),
                                      on_message=partial(self._on_ws_message, ws_key=ws_key, user_callback=user_callback, symbol_arg=symbol, interval_arg=interval),
                                      on_error=partial(self._on_ws_error, ws_key=ws_key, symbol_arg=symbol, interval_arg=interval),
                                      on_close=partial(self._on_ws_close, ws_key=ws_key, user_callback=user_callback, symbol_arg=symbol, interval_arg=interval),
                                      on_ping=partial(self._on_ws_ping, ws_key=ws_key),
                                      on_pong=partial(self._on_ws_pong, ws_key=ws_key))
            
            # 保存回调函数到WebSocketApp实例，以便监控线程可以访问
            new_ws_app.callback = user_callback
            
            self.ws_connections[ws_key] = new_ws_app # Store the new instance
            
            # 设置WebSocket连接超时和心跳参数
            ping_interval = 30  # 30秒发送一次ping
            ping_timeout = 10   # 10秒内没有收到pong则认为连接断开
            connection_timeout = 30  # 连接超时时间
            
            # Pass the new_ws_app to the lambda to ensure the correct app is run
            # 添加超时参数和心跳检测
            ws_thread = threading.Thread(
                target=lambda app=new_ws_app: app.run_forever(
                    reconnect=False,
                    ping_interval=ping_interval,
                    ping_timeout=ping_timeout,
                    http_proxy_host=os.environ.get('HTTP_PROXY_HOST'),
                    http_proxy_port=os.environ.get('HTTP_PROXY_PORT'),
                    sslopt={"cert_reqs": ssl.CERT_NONE} if should_use_proxy else None,
                    skip_utf8_validation=True
                ),
                daemon=True
            )
            self.ws_threads[ws_key] = ws_thread
            ws_thread.start()
            logger.info(f"交易对 {symbol} {interval} (key={ws_key}) 的K线WebSocket处理线程已启动。新实例ID: {id(new_ws_app)}，ping间隔: {ping_interval}秒，ping超时: {ping_timeout}秒")

    def _on_ws_open(self, ws_app_instance, ws_key: str):
        with self.ws_management_lock:
            # Check if this is the instance we are currently tracking for this ws_key
            if self.ws_connections.get(ws_key) == ws_app_instance:
                logger.info(f"WebSocket OPENED: key={ws_key}, instance_id={id(ws_app_instance)}, url={ws_app_instance.url}")
                self._reconnect_attempts[ws_key] = 0 # Reset reconnect attempts on successful open
                self._intentional_stops.pop(ws_key, None) # Clear any stale stop flag
                # 初始化或更新最后活动时间
                self._ws_last_activity[ws_key] = time.time()
            else:
                # This on_open is from a stale instance, not the one we just started (or an unexpected one)
                logger.warning(f"WebSocket OPENED (STALE): key={ws_key}, instance_id={id(ws_app_instance)}. 当前跟踪实例为 {id(self.ws_connections.get(ws_key)) if self.ws_connections.get(ws_key) else 'None'}. 关闭此过时连接。")
                try:
                    ws_app_instance.close() # Attempt to close the stale connection
                except Exception:
                    pass # Ignore errors on closing stale connection
                    
    def _on_ws_ping(self, ws_app_instance, message, ws_key: str):
        """处理WebSocket ping事件
        
        更新连接的最后活动时间，并记录ping事件
        """
        with self.ws_management_lock:
            if ws_key in self.ws_connections and self.ws_connections[ws_key] == ws_app_instance:
                current_time = time.time()
                self._ws_last_activity[ws_key] = current_time
                # 每10分钟记录一次ping事件，避免日志过多
                if not hasattr(self, '_last_ping_log') or current_time - getattr(self, '_last_ping_log', {}).get(ws_key, 0) > 600:
                    logger.debug(f"WebSocket PING: key={ws_key}, instance_id={id(ws_app_instance)}")
                    if not hasattr(self, '_last_ping_log'):
                        self._last_ping_log = {}
                    self._last_ping_log[ws_key] = current_time
            else:
                # 如果是过时的实例发送的ping，记录警告并尝试关闭
                logger.warning(f"收到来自过时WebSocket实例的PING: key={ws_key}, instance_id={id(ws_app_instance)}")
                try:
                    ws_app_instance.close()
                except Exception:
                    pass
    
    def _on_ws_pong(self, ws_app_instance, message, ws_key: str):
        """处理WebSocket pong事件
        
        更新连接的最后活动时间，并记录pong事件
        """
        with self.ws_management_lock:
            if ws_key in self.ws_connections and self.ws_connections[ws_key] == ws_app_instance:
                current_time = time.time()
                self._ws_last_activity[ws_key] = current_time
                # 每10分钟记录一次pong事件，避免日志过多
                if not hasattr(self, '_last_pong_log') or current_time - getattr(self, '_last_pong_log', {}).get(ws_key, 0) > 600:
                    logger.debug(f"WebSocket PONG: key={ws_key}, instance_id={id(ws_app_instance)}")
                    if not hasattr(self, '_last_pong_log'):
                        self._last_pong_log = {}
                    self._last_pong_log[ws_key] = current_time
            else:
                # 如果是过时的实例发送的pong，记录警告
                logger.warning(f"收到来自过时WebSocket实例的PONG: key={ws_key}, instance_id={id(ws_app_instance)}")
                try:
                    ws_app_instance.close()
                except Exception:
                    pass

    def _on_ws_message(self, ws_app_instance, message_str: str, ws_key: str, user_callback: callable, symbol_arg: str, interval_arg: str):
        # Optional: For high-frequency messages, checking instance ID every time might add overhead.
        # Consider if this check is critical for on_message or if other mechanisms (like thread termination) suffice.
        # with self.ws_management_lock:
        #     if self.ws_connections.get(ws_key) != ws_app_instance:
        #         # logger.debug(f"WebSocket MESSAGE (STALE): key={ws_key}, instance_id={id(ws_app_instance)}. 忽略消息。")
        #         return

        # 更新最后活动时间
        with self.ws_management_lock:
            self._ws_last_activity[ws_key] = time.time()

        try:
            data = json.loads(message_str)
            if 'e' in data and data['e'] == 'kline':
                current_time = time.time()
                last_log_time = self._last_kline_log_time.get(ws_key, 0)
                kline_payload = data['k']
                
                # Log throttling
                if current_time - last_log_time >= self._kline_log_interval_sec:
                    logger.info(f"收到来自 {symbol_arg} {interval_arg} (key={ws_key}, inst={id(ws_app_instance)}) 的K线数据: O:{kline_payload['o']} H:{kline_payload['h']} L:{kline_payload['l']} C:{kline_payload['c']} Closed:{kline_payload['x']}")
                    self._last_kline_log_time[ws_key] = current_time
                
                processed_kline = {
                    'event_type': data['e'],
                    'event_time': datetime.fromtimestamp(data['E'] / 1000, tz=timezone.utc),
                    'symbol': data['s'],
                    'interval': kline_payload['i'],
                    'kline_start_time': kline_payload['t'],
                    'kline_close_time': kline_payload['T'],
                    'open': float(kline_payload['o']), 'high': float(kline_payload['h']),
                    'low': float(kline_payload['l']), 'close': float(kline_payload['c']),
                    'volume': float(kline_payload['v']),
                    'number_of_trades': kline_payload['n'],
                    'is_kline_closed': kline_payload['x'],
                    'quote_asset_volume': float(kline_payload['q']),
                    'taker_buy_base_asset_volume': float(kline_payload['V']),
                    'taker_buy_quote_asset_volume': float(kline_payload['Q'])
                }
                user_callback(processed_kline)
        except Exception as e:
            logger.error(f"处理来自 {symbol_arg} {interval_arg} (key={ws_key}, inst={id(ws_app_instance)}) 的WebSocket消息时出错: {e}\n原始消息: {message_str}", exc_info=False) # Set exc_info=False if stack traces are too verbose

    def _on_ws_error(self, ws_app_instance, error_msg, ws_key: str, symbol_arg: str, interval_arg: str):
        is_current_instance = False
        with self.ws_management_lock:
             if self.ws_connections.get(ws_key) == ws_app_instance:
                 is_current_instance = True
        
        log_level = logging.ERROR if is_current_instance else logging.WARNING
        # Show full exc_info for current instance errors, but not for stale ones to reduce noise
        logger.log(log_level, f"WebSocket ERROR{' (CURRENT)' if is_current_instance else ' (STALE)'}: key={ws_key}, instance_id={id(ws_app_instance)}, symbol={symbol_arg}, interval={interval_arg}, error: {error_msg}", exc_info=is_current_instance)
        # Error often (but not always) leads to on_close. Reconnection logic is primarily in on_close.

    def _on_ws_close(self, ws_app_instance, close_status_code, close_msg, ws_key: str, user_callback: callable, symbol_arg: str, interval_arg: str):
        """处理WebSocket连接关闭事件
        
        此方法负责处理连接关闭时的资源清理和重连逻辑
        
        Args:
            ws_app_instance: WebSocketApp实例
            close_status_code: 关闭状态码
            close_msg: 关闭消息
            ws_key: WebSocket连接键
            user_callback: 用户回调函数
            symbol_arg: 交易对
            interval_arg: K线周期
        """
        is_intentional_stop_local = False
        was_current_active_instance = False # 是否是当前正在跟踪的实例

        with self.ws_management_lock:
            logger.info(f"WebSocket CLOSED: key={ws_key}, instance_id={id(ws_app_instance)}, code={close_status_code}, reason='{close_msg}'")
            
            # 检查关闭的实例是否是当前正在跟踪的实例
            if self.ws_connections.get(ws_key) == ws_app_instance:
                was_current_active_instance = True
                logger.info(f"关闭的实例 {id(ws_app_instance)} 是当前 {ws_key} 的活跃实例。从跟踪中移除。")
                # 从跟踪中移除
                self.ws_connections.pop(ws_key, None) 
                self.ws_threads.pop(ws_key, None)     
                # 更新最后活动时间，确保监控线程不会尝试重置已关闭的连接
                self._ws_last_activity.pop(ws_key, None)
            elif ws_key not in self.ws_connections and ws_key not in self.ws_threads:
                # 这意味着ws_key已经被清理，例如通过stop_kline_websocket
                logger.info(f"{ws_key} 的连接已被移除，此关闭事件 ({id(ws_app_instance)}) 可能对应已处理的实例。")
                # was_current_active_instance保持为False
            else: # ws_key存在于ws_connections中，但实例ID不匹配
                logger.warning(f"关闭的实例 {id(ws_app_instance)} (key={ws_key}) 不是当前跟踪的实例 {id(self.ws_connections.get(ws_key)) if self.ws_connections.get(ws_key) else 'None'}。不从此回调触发重连。")
                # 尝试关闭过时的socket，防止资源泄漏
                try:
                    if hasattr(ws_app_instance, 'sock') and ws_app_instance.sock:
                        ws_app_instance.sock.close()
                except Exception:
                    pass
                return # 不为过时实例的关闭事件触发重连

            # 在确定是否是当前实例后，检查是否是主动停止
            is_intentional_stop_local = self._intentional_stops.pop(ws_key, False) # 获取并移除标志

        # 如果是主动停止，清理资源并返回
        if is_intentional_stop_local:
            logger.info(f"WebSocket {ws_key} (inst={id(ws_app_instance)}) 是主动停止，不进行重连。")
            with self.ws_management_lock: # 确保这些状态也被清理
                self._reconnect_attempts.pop(ws_key, None)
                self._last_kline_log_time.pop(ws_key, None)
                # 确保所有相关资源都被清理
                if hasattr(self, '_last_ping_log'):
                    self._last_ping_log.pop(ws_key, None)
                if hasattr(self, '_last_pong_log'):
                    self._last_pong_log.pop(ws_key, None)
            return
        
        # 只有当它是当前活跃实例且不是主动停止时，才尝试重连
        if was_current_active_instance: 
            logger.info(f"WebSocket {ws_key} (inst={id(ws_app_instance)}) 非主动关闭，准备尝试重连。")
            # 传递原始的交易对、周期和用户回调函数进行重连尝试
            self._reconnect_websocket(symbol_arg, interval_arg, user_callback)
        # 如果was_current_active_instance为False，意味着关闭的实例是过时的，
        # 或者ws_key已经被清理（例如通过显式停止）。
        # 在这些情况下，我们已经返回或决定不重连。

    def _reconnect_websocket(self, symbol: str, interval: str, user_callback: callable): # Renamed arg for clarity
        ws_key = f"{symbol.lower()}_{interval}"
        
        current_attempt = 0
        with self.ws_management_lock:
            # CRITICAL PRE-CHECK: Ensure the connection for this key is indeed gone from tracking.
            # If _on_ws_close was for the current instance, it should have removed it.
            if ws_key in self.ws_connections:
                # This state should ideally not be reached if _on_ws_close works as intended.
                logger.error(f"CRITICAL_RECONNECT_PRECHECK_FAIL: key={ws_key} 仍存在于 ws_connections (inst_id={id(self.ws_connections[ws_key])})！本不应发生。终止重连尝试以防问题扩大。")
                return

            current_attempt = self._reconnect_attempts.get(ws_key, 0) + 1
            self._reconnect_attempts[ws_key] = current_attempt

            if current_attempt > self._max_reconnect_attempts:
                logger.warning(f"交易对 {symbol} {interval} (key={ws_key}) 已达到最大重连尝试次数 ({self._max_reconnect_attempts})，停止重连。")
                # Clean up states associated with this ws_key as we're giving up
                self._reconnect_attempts.pop(ws_key, None)
                self._last_kline_log_time.pop(ws_key, None)
                # ws_connections and ws_threads for this key should already be None if _on_ws_close worked
                return
        
        delay = self._reconnect_delay_base * (2 ** (current_attempt - 1))
        logger.info(f"交易对 {symbol} {interval} (key={ws_key}) 将在 {delay} 秒后尝试第 {current_attempt} 次重连...")
        
        # Perform sleep outside the lock to avoid holding it for long
        time.sleep(delay)

        logger.info(f"正在尝试重新连接交易对 {symbol} {interval} (key={ws_key}, 第 {current_attempt} 次)...")
        try:
            # start_kline_websocket will acquire the lock internally
            self.start_kline_websocket(symbol, interval, user_callback)
        except Exception as e_reconnect_start:
            logger.error(f"交易对 {symbol} {interval} (key={ws_key}) 在尝试启动重连 (第 {current_attempt} 次) 时失败: {e_reconnect_start}", exc_info=True)
            # Consider if a failed start attempt should schedule another _reconnect_websocket,
            # or rely on a potential subsequent error/close if the library retries internally.
            # For now, if start_kline_websocket fails catastrophically, this path ends.

    def stop_kline_websocket(self, symbol: str, interval: str):
        ws_key = f"{symbol.lower()}_{interval}"
        with self.ws_management_lock: # Acquire lock for modifying shared state
            logger.info(f"请求停止交易对 {symbol} {interval} (key={ws_key}) 的K线WebSocket...")
            self._intentional_stops[ws_key] = True # Mark that this stop is user-initiated

            ws_app_to_stop = self.ws_connections.pop(ws_key, None) # Get and remove from tracking
            self.ws_threads.pop(ws_key, None)   # Also remove its thread reference

            if ws_app_to_stop:
                logger.info(f"找到活动的WebSocket实例 {id(ws_app_to_stop)} for {ws_key}。正在发送关闭请求。")
                try:
                    ws_app_to_stop.close() # This will trigger its on_close
                                           # which should see _intentional_stops and not reconnect
                except Exception as e_stop_close:
                     logger.error(f"尝试关闭WebSocket实例 {id(ws_app_to_stop)} for {ws_key} 时出错: {e_stop_close}")
                logger.info(f"已发送停止请求给交易对 {symbol} {interval} (key={ws_key}, inst={id(ws_app_to_stop)}) 的K线WebSocket。")
            else:
                logger.warning(f"未找到活动的K线WebSocket连接以停止: key={ws_key}。可能已被关闭或从未启动。")
            
            # Clean up other related states for this ws_key
            self._reconnect_attempts.pop(ws_key, None)
            self._last_kline_log_time.pop(ws_key, None)
            
    def reset_kline_websocket(self, symbol: str, interval: str, user_callback: callable):
        """重置指定交易对的K线WebSocket连接，用于解决连接卡死的情况。
        
        此方法会先停止现有连接，然后重新建立连接。
        """
        ws_key = f"{symbol.lower()}_{interval}"
        logger.warning(f"正在重置交易对 {symbol} {interval} (key={ws_key}) 的K线WebSocket连接...")
        
        # 先停止现有连接
        self.stop_kline_websocket(symbol, interval)
        
        # 等待一小段时间确保连接完全关闭
        time.sleep(2)
        
        # 重新建立连接
        try:
            self.start_kline_websocket(symbol, interval, user_callback)
            logger.info(f"交易对 {symbol} {interval} (key={ws_key}) 的K线WebSocket连接已重置")
            return True
        except Exception as e:
            logger.error(f"重置交易对 {symbol} {interval} (key={ws_key}) 的K线WebSocket连接失败: {e}")
            return False
            
    def _start_ws_monitor(self):
        """启动WebSocket连接监控线程
        
        此线程定期检查所有WebSocket连接的状态，重置超时连接，并确保连接活跃
        """
        if self._ws_monitor_running:
            logger.debug("WebSocket监控线程已在运行")
            return
            
        def monitor_websockets():
            self._ws_monitor_running = True
            logger.info(f"WebSocket连接监控线程已启动，监控间隔: {self._ws_monitor_interval}秒，连接超时: {self._ws_connection_timeout}秒")
            
            while self._ws_monitor_running:
                try:
                    with self.ws_management_lock:
                        current_time = time.time()
                        # 检查所有WebSocket连接
                        for ws_key, last_activity in list(self._ws_last_activity.items()):
                            # 跳过已经停止的连接
                            if ws_key not in self.ws_connections or self._intentional_stops.get(ws_key, False):
                                # 清理不再需要的活动记录
                                if ws_key not in self.ws_connections:
                                    self._ws_last_activity.pop(ws_key, None)
                                continue
                                
                            # 获取WebSocket实例
                            ws_app = self.ws_connections.get(ws_key)
                            if not ws_app:
                                continue
                                
                            # 检查连接是否超时
                            inactive_time = current_time - last_activity
                            if inactive_time > self._ws_connection_timeout:
                                logger.warning(f"WebSocket连接 {ws_key} 超时 ({inactive_time:.1f}秒)，尝试重置")
                                try:
                                    # 检查连接状态
                                    is_connected = ws_app.sock and ws_app.sock.connected
                                    logger.info(f"WebSocket {ws_key} 连接状态: {'已连接' if is_connected else '未连接'}, 实例ID: {id(ws_app)}")
                                    
                                    # 解析ws_key获取symbol和interval
                                    if '_' in ws_key:
                                        symbol, interval = ws_key.split('_', 1)
                                        # 获取回调函数
                                        if hasattr(ws_app, 'callback'):
                                            # 重置连接
                                            self.reset_kline_websocket(symbol, interval, ws_app.callback)
                                        else:
                                            logger.error(f"WebSocket {ws_key} 实例没有callback属性，无法重置")
                                            # 尝试强制关闭并移除
                                            try:
                                                # 标记为主动停止，防止重连
                                                self._intentional_stops[ws_key] = True
                                                ws_app.close()
                                            except Exception as e_close:
                                                logger.error(f"关闭WebSocket {ws_key} 失败: {e_close}")
                                                # 尝试强制关闭socket
                                                try:
                                                    if ws_app.sock:
                                                        ws_app.sock.close()
                                                except Exception:
                                                    pass
                                            # 从跟踪中移除
                                            self.ws_connections.pop(ws_key, None)
                                            self.ws_threads.pop(ws_key, None)
                                            self._ws_last_activity.pop(ws_key, None)
                                    else:
                                        logger.error(f"无法解析WebSocket键 {ws_key}")
                                except Exception as e:
                                    logger.error(f"重置超时WebSocket连接 {ws_key} 失败: {e}", exc_info=True)
                                    # 尝试强制关闭并移除
                                    try:
                                        self._intentional_stops[ws_key] = True
                                        if ws_app.sock:
                                            ws_app.sock.close()
                                        self.ws_connections.pop(ws_key, None)
                                        self.ws_threads.pop(ws_key, None)
                                        self._ws_last_activity.pop(ws_key, None)
                                    except Exception as e_force_close:
                                        logger.error(f"强制关闭WebSocket {ws_key} 失败: {e_force_close}")
                            elif inactive_time > self._ws_connection_timeout * 0.7:
                                # 如果接近超时（超过70%阈值），尝试发送ping来保活
                                logger.info(f"WebSocket连接 {ws_key} 接近超时 ({inactive_time:.1f}秒)，尝试发送ping保活")
                                try:
                                    if ws_app.sock and ws_app.sock.connected:
                                        ws_app.sock.ping()
                                        # 更新最后活动时间，避免频繁ping
                                        self._ws_last_activity[ws_key] = current_time - (self._ws_connection_timeout * 0.5)
                                    else:
                                        logger.warning(f"WebSocket {ws_key} 的socket不可用或未连接，无法发送ping")
                                        # 如果socket已断开但WebSocketApp仍在跟踪中，尝试重置
                                        if '_' in ws_key and hasattr(ws_app, 'callback'):
                                            symbol, interval = ws_key.split('_', 1)
                                            self.reset_kline_websocket(symbol, interval, ws_app.callback)
                                except Exception as e:
                                    logger.warning(f"向WebSocket {ws_key} 发送ping失败: {e}")
                                    # 如果ping失败，可能连接已经断开，尝试重置
                                    if '_' in ws_key and hasattr(ws_app, 'callback'):
                                        try:
                                            symbol, interval = ws_key.split('_', 1)
                                            self.reset_kline_websocket(symbol, interval, ws_app.callback)
                                        except Exception as e_reset:
                                            logger.error(f"重置WebSocket {ws_key} 失败: {e_reset}")
                except Exception as e:
                    logger.error(f"WebSocket监控线程发生错误: {e}", exc_info=True)
                    
                # 休眠指定时间
                time.sleep(self._ws_monitor_interval)
                
            logger.info("WebSocket连接监控线程已停止")
        
        # 创建并启动监控线程
        self._ws_monitor_thread = threading.Thread(target=monitor_websockets, daemon=True, name="WS-Monitor")
        self._ws_monitor_thread.start()
        
    def _stop_ws_monitor(self):
        """停止WebSocket连接监控线程
        
        此方法会尝试优雅地停止监控线程，并在超时后返回
        """
        if not self._ws_monitor_running:
            logger.debug("WebSocket连接监控线程未运行，无需停止")
            return True
            
        logger.info("正在停止WebSocket连接监控线程...")
        self._ws_monitor_running = False
        
        # 等待线程结束
        if self._ws_monitor_thread and self._ws_monitor_thread.is_alive():
            # 设置更长的超时时间，确保线程有足够时间完成当前迭代并退出
            monitor_stop_timeout = 10  # 10秒超时
            logger.debug(f"等待监控线程结束，超时时间: {monitor_stop_timeout}秒")
            
            self._ws_monitor_thread.join(timeout=monitor_stop_timeout)
            if self._ws_monitor_thread.is_alive():
                logger.warning(f"WebSocket连接监控线程未能在{monitor_stop_timeout}秒内停止，这可能导致资源泄漏")
                return False
        
        logger.info("WebSocket连接监控线程已成功停止")
        return True
    
    def stop_all_websockets(self):
        """停止所有WebSocket连接
        
        此方法会尝试优雅地关闭所有活动的WebSocket连接，并在必要时强制关闭
        以确保服务能够正常退出，防止进程卡死
        
        添加了锁超时机制，避免在服务关闭时可能发生的死锁问题
        """
        logger.warning("正在停止所有WebSocket连接...")
        
        # 首先停止监控线程
        self._stop_ws_monitor()
        
        # 第一阶段：尝试优雅关闭所有连接
        # 添加锁获取超时，避免死锁
        lock_timeout = 5.0  # 5秒锁超时
        lock_acquired = False
        
        try:
            # 尝试获取锁，但设置超时以避免死锁
            lock_acquired = self.ws_management_lock.acquire(timeout=lock_timeout)
            
            if not lock_acquired:
                logger.error(f"无法在 {lock_timeout} 秒内获取ws_management_lock，可能存在死锁。将尝试不使用锁继续关闭操作。")
                # 创建当前连接的副本，以便在没有锁的情况下继续
                ws_keys = list(self.ws_connections.keys()) if hasattr(self, 'ws_connections') else []
                # 标记所有连接为主动停止，即使没有锁也尝试设置
                for ws_key in ws_keys:
                    self._intentional_stops[ws_key] = True
            else:
                # 正常获取锁的情况
                # 复制键列表，避免在迭代过程中修改字典
                ws_keys = list(self.ws_connections.keys())
                logger.info(f"需要关闭 {len(ws_keys)} 个WebSocket连接")
                
                # 标记所有连接为主动停止，防止重连
                for ws_key in ws_keys:
                    self._intentional_stops[ws_key] = True
        finally:
            # 如果成功获取了锁，释放它
            if lock_acquired:
                self.ws_management_lock.release()
        
        # 尝试优雅关闭每个连接
        for ws_key in ws_keys:
            try:
                # 解析ws_key获取symbol和interval
                if '_' in ws_key:
                    symbol, interval = ws_key.split('_', 1)
                    logger.info(f"正在停止 {symbol} {interval} 的WebSocket连接...")
                    # 修改：直接调用stop_kline_websocket可能会导致死锁，因为它内部也使用了锁
                    # 我们尝试使用超时机制获取锁
                    try:
                        lock_acquired = self.ws_management_lock.acquire(timeout=lock_timeout)
                        if lock_acquired:
                            try:
                                # 在锁内执行关键操作
                                self._intentional_stops[ws_key] = True
                                ws_app_to_stop = self.ws_connections.pop(ws_key, None)
                                self.ws_threads.pop(ws_key, None)
                                
                                if ws_app_to_stop:
                                    try:
                                        ws_app_to_stop.close()
                                    except Exception as e_close:
                                        logger.error(f"尝试关闭WebSocket实例 {id(ws_app_to_stop)} for {ws_key} 时出错: {e_close}")
                            finally:
                                self.ws_management_lock.release()
                        else:
                            # 如果无法获取锁，尝试直接关闭连接
                            logger.warning(f"无法获取锁来停止 {ws_key}，尝试直接关闭连接")
                            ws_app = self.ws_connections.get(ws_key)
                            if ws_app:
                                try:
                                    ws_app.close()
                                except Exception as e:
                                    logger.error(f"直接关闭WebSocket连接 {ws_key} 失败: {e}")
                    except Exception as e_lock:
                        logger.error(f"获取锁或关闭WebSocket {ws_key} 时出错: {e_lock}")
                else:
                    logger.warning(f"无法解析WebSocket键 {ws_key}，尝试直接关闭")
                    # 直接关闭连接
                    try:
                        lock_acquired = self.ws_management_lock.acquire(timeout=lock_timeout)
                        if lock_acquired:
                            try:
                                if ws_key in self.ws_connections:
                                    ws_app = self.ws_connections[ws_key]
                                    logger.info(f"正在关闭WebSocket连接 {ws_key} (实例ID: {id(ws_app)})")
                                    try:
                                        ws_app.close()
                                    except Exception as e:
                                        logger.error(f"关闭WebSocket连接 {ws_key} 失败: {e}")
                                    # 无论是否成功关闭，都从跟踪中移除
                                    self.ws_connections.pop(ws_key, None)
                            finally:
                                self.ws_management_lock.release()
                        else:
                            # 如果无法获取锁，尝试直接关闭连接
                            logger.warning(f"无法获取锁来停止 {ws_key}，尝试直接关闭连接")
                            ws_app = self.ws_connections.get(ws_key)
                            if ws_app:
                                try:
                                    ws_app.close()
                                except Exception as e:
                                    logger.error(f"直接关闭WebSocket连接 {ws_key} 失败: {e}")
                    except Exception as e_lock:
                        logger.error(f"获取锁或关闭WebSocket {ws_key} 时出错: {e_lock}")
            except Exception as e:
                logger.error(f"停止WebSocket连接 {ws_key} 时发生错误: {e}")
        
        # 第二阶段：等待线程结束，并在必要时强制关闭
        shutdown_start = time.time()
        force_close_timeout = self._ws_shutdown_timeout * 0.7  # 70%的时间用于等待优雅关闭
        force_close_deadline = shutdown_start + force_close_timeout
        final_deadline = shutdown_start + self._ws_shutdown_timeout
        
        logger.info(f"等待WebSocket线程结束，优雅关闭超时: {force_close_timeout:.1f}秒，最终超时: {self._ws_shutdown_timeout}秒")
        
        # 等待线程结束，直到优雅关闭超时
        while time.time() < force_close_deadline:
            remaining_threads = []
            try:
                # 尝试获取锁，但设置超时
                lock_acquired = self.ws_management_lock.acquire(timeout=1.0)  # 较短的超时，因为这是循环
                if lock_acquired:
                    try:
                        remaining_threads = [(k, t) for k, t in self.ws_threads.items() if t.is_alive()]
                        if not remaining_threads:
                            logger.info("所有WebSocket线程已正常结束")
                            break
                        logger.debug(f"仍有 {len(remaining_threads)} 个WebSocket线程在运行...")
                    finally:
                        self.ws_management_lock.release()
                else:
                    # 如果无法获取锁，假设仍有线程在运行
                    logger.warning("无法获取锁来检查剩余线程，假设仍有线程在运行")
            except Exception as e:
                logger.error(f"检查剩余线程时出错: {e}")
            time.sleep(0.5)
        
        # 如果仍有线程在运行，尝试强制关闭
        remaining_threads = []
        try:
            # 尝试获取锁，但设置超时
            lock_acquired = self.ws_management_lock.acquire(timeout=lock_timeout)
            if lock_acquired:
                try:
                    remaining_threads = [(k, t) for k, t in self.ws_threads.items() if t.is_alive()]
                    if remaining_threads:
                        logger.warning(f"有 {len(remaining_threads)} 个WebSocket线程在 {force_close_timeout:.1f} 秒内未能正常结束，尝试强制关闭")
                        
                        # 强制关闭所有剩余的WebSocket连接
                        for ws_key, thread in remaining_threads:
                            try:
                                ws_app = self.ws_connections.get(ws_key)
                                if ws_app and ws_app.sock:
                                    logger.warning(f"强制关闭WebSocket连接 {ws_key} 的socket")
                                    try:
                                        ws_app.sock.close()
                                    except Exception as e:
                                        logger.error(f"强制关闭WebSocket {ws_key} 的socket失败: {e}")
                            except Exception as e:
                                logger.error(f"强制关闭WebSocket {ws_key} 时发生错误: {e}")
                finally:
                    self.ws_management_lock.release()
            else:
                # 如果无法获取锁，尝试直接关闭所有连接
                logger.error("无法获取锁来强制关闭连接，尝试直接关闭所有连接")
                # 尝试直接访问连接字典，即使没有锁保护
                for ws_key, ws_app in list(self.ws_connections.items()) if hasattr(self, 'ws_connections') else []:
                    try:
                        if ws_app and hasattr(ws_app, 'sock') and ws_app.sock:
                            logger.warning(f"直接强制关闭WebSocket连接 {ws_key} 的socket")
                            ws_app.sock.close()
                    except Exception as e:
                        logger.error(f"直接强制关闭WebSocket {ws_key} 失败: {e}")
        except Exception as e:
            logger.error(f"强制关闭WebSocket连接时出错: {e}")
        
        # 最后等待所有线程结束或达到最终超时
        while time.time() < final_deadline:
            remaining_threads = []
            try:
                # 尝试获取锁，但设置超时
                lock_acquired = self.ws_management_lock.acquire(timeout=1.0)  # 较短的超时
                if lock_acquired:
                    try:
                        remaining_threads = [(k, t) for k, t in self.ws_threads.items() if t.is_alive()]
                        if not remaining_threads:
                            logger.info("所有WebSocket线程已结束")
                            break
                        logger.warning(f"仍有 {len(remaining_threads)} 个WebSocket线程在强制关闭后仍在运行...")
                    finally:
                        self.ws_management_lock.release()
                else:
                    # 如果无法获取锁，假设仍有线程在运行
                    logger.warning("无法获取锁来检查剩余线程，假设仍有线程在运行")
            except Exception as e:
                logger.error(f"检查剩余线程时出错: {e}")
            time.sleep(0.5)
        
        # 最终清理
        try:
            # 尝试获取锁，但设置超时
            lock_acquired = self.ws_management_lock.acquire(timeout=lock_timeout)
            if lock_acquired:
                try:
                    remaining_threads = [(k, t) for k, t in self.ws_threads.items() if t.is_alive()]
                    if remaining_threads:
                        logger.error(f"有 {len(remaining_threads)} 个WebSocket线程在 {self._ws_shutdown_timeout} 秒内未能结束，服务可能无法正常退出")
                        for ws_key, thread in remaining_threads:
                            logger.error(f"未能结束的线程: {ws_key}, 线程ID: {thread.ident}")
                    
                    # 清理所有连接和线程记录
                    self.ws_connections.clear()
                    self.ws_threads.clear()
                    self._intentional_stops.clear()
                    self._ws_last_activity.clear()
                    self._reconnect_attempts.clear()
                    self._last_kline_log_time.clear()
                    
                    # 清理其他可能的循环引用
                    if hasattr(self, '_last_ping_log'):
                        self._last_ping_log.clear()
                    if hasattr(self, '_last_pong_log'):
                        self._last_pong_log.clear()
                finally:
                    self.ws_management_lock.release()
            else:
                # 如果无法获取锁，尝试直接清理
                logger.error("无法获取锁来清理资源，尝试直接清理")
                # 尝试直接清理资源，即使没有锁保护
                if hasattr(self, 'ws_connections'): self.ws_connections.clear()
                if hasattr(self, 'ws_threads'): self.ws_threads.clear()
                if hasattr(self, '_intentional_stops'): self._intentional_stops.clear()
                if hasattr(self, '_ws_last_activity'): self._ws_last_activity.clear()
                if hasattr(self, '_reconnect_attempts'): self._reconnect_attempts.clear()
                if hasattr(self, '_last_kline_log_time'): self._last_kline_log_time.clear()
                if hasattr(self, '_last_ping_log'): self._last_ping_log.clear()
                if hasattr(self, '_last_pong_log'): self._last_pong_log.clear()
        except Exception as e:
            logger.error(f"清理WebSocket资源时出错: {e}")
            
        logger.warning("所有WebSocket连接资源已清理完毕")
        return True

    def reset_kline_websocket(self, symbol: str, interval: str, user_callback: callable):
        """重置指定交易对和周期的K线WebSocket连接
        
        此方法会先停止现有连接，然后重新启动一个新连接
        """
        ws_key = f"{symbol.lower()}_{interval}"
        logger.info(f"正在重置交易对 {symbol} {interval} (key={ws_key}) 的K线WebSocket连接")
        
        try:
            # 先停止现有连接
            self.stop_kline_websocket(symbol, interval)
            
            # 等待一小段时间确保连接完全关闭
            time.sleep(1)
            
            # 启动新连接
            self.start_kline_websocket(symbol, interval, user_callback)
            logger.info(f"交易对 {symbol} {interval} (key={ws_key}) 的K线WebSocket连接已重置")
        except Exception as e:
            logger.error(f"重置交易对 {symbol} {interval} (key={ws_key}) 的K线WebSocket连接失败: {e}", exc_info=True)
            # 尝试再次启动连接
            try:
                self.start_kline_websocket(symbol, interval, user_callback)
            except Exception as e2:
                logger.error(f"重试启动交易对 {symbol} {interval} (key={ws_key}) 的K线WebSocket连接失败: {e2}", exc_info=True)
    
    # 原始的stop_all_websockets方法已被上面的增强版本替代
            # Iterate over a copy of keys because stop_kline_websocket modifies the dictionary
            for ws_key in list(self.ws_connections.keys()): 
                try:
                    parts = ws_key.split('_', 1)
                    if len(parts) == 2:
                        symbol, interval = parts[0].upper(), parts[1]
                        # stop_kline_websocket will acquire the lock again, which is fine with RLock
                        self.stop_kline_websocket(symbol, interval) 
                    else:
                        logger.warning(f"无法从键 '{ws_key}' 中解析交易对和周期，跳过停止。")
                except Exception as e:
                    logger.error(f"停止WebSocket (键: {ws_key}) 时发生错误: {e}", exc_info=True)
            
            # 强制关闭任何剩余的连接
            remaining_connections = list(self.ws_connections.items())
            if remaining_connections:
                logger.warning(f"在正常停止后，仍有 {len(remaining_connections)} 个WebSocket连接未关闭。尝试强制关闭...")
                for ws_key, ws_app in remaining_connections:
                    try:
                        # 标记为主动停止，防止重连
                        self._intentional_stops[ws_key] = True
                        # 强制关闭连接
                        if hasattr(ws_app, 'sock') and ws_app.sock:
                            ws_app.sock.close()
                        # 从跟踪中移除
                        self.ws_connections.pop(ws_key, None)
                        self.ws_threads.pop(ws_key, None)
                        logger.info(f"已强制关闭WebSocket连接: {ws_key}")
                    except Exception as e:
                        logger.error(f"强制关闭 {ws_key} 的WebSocket连接失败: {e}")
            
            # 清理所有相关状态
            self._reconnect_attempts.clear()
            self._last_kline_log_time.clear()
            self._intentional_stops.clear()
            
            # 最终检查
            if self.ws_connections:
                logger.error(f"在所有清理尝试后，仍有 {len(self.ws_connections)} 个WebSocket连接未关闭。")
            else:
                logger.info("所有WebSocket连接已成功停止和清理。")
        
        logger.info(f"已发送停止请求给所有活动的WebSocket连接。")


    def test_websocket_connection(self, symbol: str = "BTCUSDT", interval: str = "1m"):
        """
        测试WebSocket连接是否成功建立，并打印收到的第一条消息。
        """
        ws_key = f"{symbol.lower()}_{interval}"
        test_passed = False
        first_message_received = threading.Event() 
        received_message_data = None 

        def test_callback(kline_data_dict):
            nonlocal received_message_data 
            if not first_message_received.is_set():
                logger.info(f"TEST_CALLBACK: 收到第一条WebSocket消息 for {ws_key}: Close={kline_data_dict.get('close')}")
                received_message_data = kline_data_dict
                first_message_received.set() 

        logger.info(f"正在测试交易对 {symbol} {interval} (key={ws_key}) 的WebSocket连接...")

        try:
            self.start_kline_websocket(symbol, interval, test_callback)

            time.sleep(2) # Give some time for connection to establish
            
            if first_message_received.wait(timeout=15): # Increased timeout for potentially slow networks/first connect
                 logger.info(f"TEST: 成功收到WebSocket消息 for {ws_key}.")
                 test_passed = True
            else:
                 logger.error(f"TEST: 在规定时间内未收到WebSocket消息 for {ws_key}.")
                 test_passed = False


            with self.ws_management_lock: # Check connection status under lock
                if ws_key in self.ws_connections:
                    ws_app = self.ws_connections[ws_key]
                    if ws_app.sock and ws_app.sock.connected:
                        logger.info(f"TEST: WebSocket连接状态良好 for {ws_key} (inst={id(ws_app)}).")
                    else:
                        logger.warning(f"TEST: WebSocket连接Socket未连接 for {ws_key} (inst={id(ws_app) if ws_app else 'N/A'}).")
                elif test_passed: # If message was received but ws_key is not in connections, it means it connected then closed quickly
                    logger.warning(f"TEST: 收到消息 for {ws_key} 但连接实例已不在 ws_connections 字典中。可能已快速关闭。")
                else:
                     logger.warning(f"TEST: WebSocket连接实例不存在 for {ws_key} in ws_connections dict 且未收到消息。")


        except Exception as e:
            logger.error(f"测试交易对 {symbol} {interval} (key={ws_key}) 的WebSocket连接时发生异常: {e}", exc_info=True)
            test_passed = False

        finally:
            logger.info(f"TEST: 清理WebSocket连接 for {ws_key}...")
            self.stop_kline_websocket(symbol, interval) # This will handle lock internally
            # Allow some time for threads to close, especially if a reconnect was in progress
            # Max typical reconnect delay could be self._reconnect_delay_base * (2**(self._max_reconnect_attempts-1))
            # but a few seconds should be enough for graceful shutdown.
            time.sleep(self._reconnect_delay_base + 2) # Wait a bit longer than base reconnect delay plus some buffer
            logger.info(f"交易对 {symbol} {interval} (key={ws_key}) 的WebSocket连接测试清理完成。")

        assert test_passed, f"WebSocket连接测试失败 for {ws_key}: 未收到消息或发生错误。"


if __name__ == '__main__':
    # 如果需要深度调试WebSocket帧（PING/PONG等），取消下一行注释
    # 这会产生非常多的日志输出。
    # websocket.enableTrace(True) 
    
    client = BinanceClient()

    try:
        server_time_dt = datetime.fromtimestamp(client.get_server_time() / 1000, tz=timezone.utc)
        print(f"\n币安服务器时间: {server_time_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")

        available_symbols_list = client.get_available_symbols()
        print(f"获取到 {len(available_symbols_list)} 个U本位永续合约交易对。")
        if available_symbols_list: print(f"  前5个: {available_symbols_list[:5]}")
        
        hot_symbols = client.get_hot_symbols_by_volume(top_n=5)
        print(f"热门U本位合约 (前5): {hot_symbols}")


        print(f"\n--- 测试短时段1分钟K线分页获取 (BTCUSDT) ---")
        now_utc_main = datetime.now(timezone.utc)
        start_time_short_1m_main = int((now_utc_main - timedelta(minutes=10)).timestamp() * 1000) # Get last 10 mins
        end_time_short_1m_main = int(now_utc_main.timestamp() * 1000)

        df_1m_short_data_main = client.get_historical_klines(
            symbol="BTCUSDT",
            interval="1m",
            start_time=start_time_short_1m_main,
            end_time=end_time_short_1m_main
        )
        if not df_1m_short_data_main.empty:
            print(f"成功获取 {len(df_1m_short_data_main)} 条1分钟K线 (短时段 BTCUSDT)。")
            print(f"数据时间范围: {df_1m_short_data_main.index.min()} 至 {df_1m_short_data_main.index.max()}")
        else:
            print("未能获取到短时段1分钟K线数据 (BTCUSDT)。")
        
        print(f"\n--- 测试获取最近N条K线 (ETHUSDT) ---")
        df_latest_klines_main = client.get_historical_klines(symbol="ETHUSDT", interval="15m", limit=5)
        if not df_latest_klines_main.empty:
            print(f"成功获取最新的 {len(df_latest_klines_main)} 条15分钟ETH K线。")
            print(df_latest_klines_main.tail())
        else:
            print("未能获取最新的K线数据 (ETHUSDT)。")

        print(f"\n获取 BTCUSDT 最新价格: {client.get_latest_price('BTCUSDT')}")


        # --- WebSocket 连接测试 ---
        print("\n--- 开始WebSocket连接测试 ---")
        client.test_websocket_connection("BTCUSDT", "1m") # Test BTC
        print("\nSUCCESS: BTCUSDT 1m WebSocket连接测试通过。")
        
        print("\n--- 测试ETHUSDT 1m的启动、停止、重启 ---")
        def dummy_callback_eth_main(data): 
            # logger.info(f"ETH_MAIN_CALLBACK (Close): {data.get('close')} (Closed: {data.get('is_kline_closed')})")
            pass # Keep it silent for this test run unless debugging specific callbacks
        
        client.start_kline_websocket("ETHUSDT", "1m", dummy_callback_eth_main)
        logger.info("ETHUSDT 1m 已启动，等待几秒...")
        time.sleep(7) # Let it run and receive some messages
        
        client.stop_kline_websocket("ETHUSDT", "1m")
        logger.info("ETHUSDT 1m 已请求停止。等待几秒后重启...")
        time.sleep(4) # Wait for it to fully stop and cleanup
        
        client.start_kline_websocket("ETHUSDT", "1m", dummy_callback_eth_main) # Restart
        logger.info("ETHUSDT 1m 已重新启动。再等待几秒...")
        time.sleep(7)
        
        print("SUCCESS: ETHUSDT 1m 启动/停止/重启测试序列完成。")

    except AssertionError as e_assert: # Catch assertion errors from test_websocket_connection
        print(f"\nERROR in WebSocket Tests: {e_assert}")
    except Exception as main_exception:
        logger.error(f"主程序发生错误: {main_exception}", exc_info=True)
    finally:
        print("\n--- 主测试流程执行完毕。确保所有WebSocket都已停止。 ---")
        client.stop_all_websockets() # Ensure all are stopped
        print("已请求停止所有活动的WebSocket连接。")
        # Give threads a moment to gracefully exit, especially if reconnects were happening.
        # This needs to be longer if max_reconnect_attempts and delay are high.
        time.sleep(3) 
        print("退出主测试程序。")

# --- END OF FILE binance_client.py ---