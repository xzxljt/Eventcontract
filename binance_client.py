import os
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
import hashlib # 用于生成哈希，如签名
import logging # 导入logging模块

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BinanceClient:
    BASE_URL = "https://api.binance.com" # 现货 API 基础URL
    CACHE_DIR = ".kline_cache" # K线数据缓存目录
    FAPI_BASE_URL = "https://fapi.binance.com" # U本位合约 API 基础URL

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
        self.ws_threads: Dict[str, Any] = {} # 存储运行WebSocket的线程对象
        
        # WebSocket 重连配置和状态
        self._max_reconnect_attempts = 10 # 最大重连尝试次数
        self._reconnect_delay_base = 5 # 基础重连延迟 (秒)
        self._reconnect_attempts: Dict[str, int] = {} # 存储每个连接的重连尝试次数

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
                    print(f"从缓存加载数据: {cache_path}")
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
            print(f"数据已保存到缓存: {cache_path}")
        except IOError as e:
            print(f"保存数据到缓存文件失败 {cache_path}: {e}")

    def _request(self, method: str, url_path: str, params: Optional[dict] = None, signed: bool = False, is_fapi: bool = False):
        """
        发送HTTP请求到币安API的通用方法。

        参数:
            method (str): HTTP方法 (GET, POST等).
            url_path (str): API的路径 (例如 /api/v3/time).
            params (Optional[dict]): 请求参数.
            signed (bool): 此请求是否需要签名.
            is_fapi (bool): 此请求是否是针对U本位合约API (fapi).
        """
        base = self.FAPI_BASE_URL if is_fapi else self.BASE_URL # 根据is_fapi选择基础URL
        full_url = f"{base}{url_path}"
        
        if signed:
            if not self.api_key or not self.api_secret:
                 raise ValueError("API key 和 secret 必须为签名请求设置。")
            if params is None: params = {}
            params['timestamp'] = int(time.time() * 1000) # 添加时间戳参数
            params['signature'] = self._sign_request(params) # 生成并添加签名

        try:
            logger.debug(f"发送 {method.upper()} 请求到 {full_url}，参数: {params}") # 添加日志
            logger.debug(f"发送 {method.upper()} 请求到 {full_url}，参数: {params}") # 添加日志
            logger.debug(f"requests session 代理配置: {self.session.proxies}") # 添加代理配置日志
            if method.upper() == 'GET':
                response = self.session.get(full_url, params=params, timeout=(10, 30)) # 设置连接和读取超时
            elif method.upper() == 'POST':
                # 对于POST请求，如果需要签名，参数通常放在查询字符串中；如果不需要签名，则放在请求体中
                if signed:
                    response = self.session.post(full_url, params=params, timeout=(10, 30)) # 设置连接和读取超时
                else:
                    response = self.session.post(full_url, data=params, timeout=(10, 30)) # 设置连接和读取超时
            else:
                raise ValueError(f"不支持的HTTP方法: {method}")
            
            response.raise_for_status() # 如果HTTP状态码表示错误 (4xx或5xx), 则抛出HTTPError异常
            logger.debug(f"请求 {full_url} 成功，状态码: {response.status_code}") # 添加成功日志
            return response.json() # 解析响应的JSON数据
        except requests.exceptions.HTTPError as http_err:
            # 尝试从响应中获取更详细的币安错误信息
            error_details = ""
            try:
                error_data = response.json()
                if 'msg' in error_data: # 币安通常在msg字段中提供错误描述
                    error_details = f" (币安错误信息: {error_data['msg']}, 错误码: {error_data.get('code')})"
            except ValueError: # 如果响应不是有效的JSON
                error_details = f" (原始响应内容: {response.text})"
            except Exception: # 其他可能的解析错误
                pass
            logger.error(f"HTTP请求错误: {http_err} - URL: {response.url}{error_details}") # 使用logger.error
            raise # 重新抛出原始的HTTPError，或者可以包装成自定义异常
        except requests.exceptions.RequestException as req_err: # 捕获更广泛的请求异常
            logger.error(f"请求 {full_url} 时发生网络或连接错误: {req_err}，参数: {params}") # 记录详细错误和参数
            raise # 重新抛出异常
        except Exception as e:
            logger.error(f"请求 {full_url} 时发生未知错误: {e}，参数: {params}", exc_info=True) # 记录未知错误和参数，包含堆栈信息
            raise

    def get_server_time(self) -> int:
        """获取币安服务器时间戳 (毫秒)"""
        data = self._request('GET', '/api/v3/time') # 现货API路径，但通常通用
        return data['serverTime']

    def get_exchange_info(self, symbol: Optional[str] = None) -> dict:
        """获取交易所交易规则和交易对信息"""
        params = {}
        if symbol:
            params['symbol'] = symbol.upper()
        # U本位合约使用 /fapi/v1/exchangeInfo
        return self._request('GET', '/fapi/v1/exchangeInfo', params=params, is_fapi=True)

    def get_available_symbols(self, quote_asset: str = "USDT") -> List[str]:
        """获取所有可用的U本位永续合约交易对列表"""
        try:
            exchange_info = self.get_exchange_info()
            symbols = [
                s['symbol'] for s in exchange_info['symbols']
                if s['quoteAsset'] == quote_asset.upper() and \
                   s['contractType'] == 'PERPETUAL' and \
                   s['status'] == 'TRADING' # 只选择交易中的永续合约
            ]
            return sorted(symbols)
        except Exception as e:
            print(f"获取可用交易对列表失败: {e}。将返回一个默认列表。")
            # 在API调用失败时返回一个常用的交易对列表作为备选
            return ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "DOGEUSDT", "DOTUSDT", "SOLUSDT"]

    def get_24hr_ticker_statistics(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取24小时交易量统计信息"""
        params = {}
        if symbol:
            params['symbol'] = symbol.upper()
        # 现货和U本位合约都有类似的接口，这里使用现货接口，因为它通常包含更多交易对
        # 如果需要U本位合约的24小时交易量，可以使用 /fapi/v1/ticker/24hr
        return self._request('GET', '/api/v3/ticker/24hr', params=params, is_fapi=False) # 使用现货API

    def get_hot_symbols_by_volume(self, quote_asset: str = "USDT", top_n: int = 50) -> List[str]:
        """
        获取热度较高的U本位永续合约交易对列表，基于24小时交易量排序。
        
        参数:
            quote_asset (str): 计价资产，默认为 "USDT"。
            top_n (int): 返回交易量最高的交易对数量。
        """
        try:
            # 1. 获取所有U本位永续合约交易对
            all_symbols = self.get_available_symbols(quote_asset=quote_asset)
            
            if not all_symbols:
                print("未获取到任何可用交易对。")
                return []

            # 2. 获取所有交易对的24小时交易量统计
            # 注意：/api/v3/ticker/24hr 默认返回所有交易对，但可能不限于U本位合约
            # 更好的做法是获取所有统计后，再根据 all_symbols 列表进行筛选
            all_tickers = self.get_24hr_ticker_statistics()
            
            # 将统计数据转换为字典，方便查找
            ticker_dict = {t['symbol']: t for t in all_tickers}
            
            # 3. 筛选出U本位永续合约的统计数据，并按交易量排序
            hot_symbols_data = []
            for symbol in all_symbols:
                if symbol in ticker_dict:
                    ticker = ticker_dict[symbol]
                    try:
                        # 使用 quoteVolume (计价资产交易量) 作为热度指标
                        volume = float(ticker.get('quoteVolume', 0))
                        hot_symbols_data.append({'symbol': symbol, 'volume': volume})
                    except (ValueError, TypeError):
                        print(f"警告: 交易对 {symbol} 的交易量数据无效: {ticker.get('quoteVolume')}")
                        continue # 跳过无效数据
            
            # 按交易量降序排序
            hot_symbols_data.sort(key=lambda x: x['volume'], reverse=True)
            
            # 4. 提取前 top_n 个交易对的symbol
            top_symbols = [item['symbol'] for item in hot_symbols_data[:top_n]]
            
            print(f"已获取并筛选出前 {top_n} 个热门交易对 (基于24小时交易量)。")
            return top_symbols

        except Exception as e:
            print(f"获取热门交易对失败: {e}。将返回一个默认列表。")
            # 在API调用失败时返回一个常用的交易对列表作为备选
            return ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "DOGEUSDT", "DOTUSDT", "SOLUSDT"]


    def get_historical_klines(self, symbol: str, interval: str,
                              start_time: Optional[int] = None, # UTC时间戳 (毫秒)
                              end_time: Optional[int] = None,   # UTC时间戳 (毫秒)
                              limit: Optional[int] = None) -> pd.DataFrame:
        """
        获取历史K线数据，自动处理分页以获取指定时间范围内的所有数据。
        如果只提供 limit 而没有 start_time，则获取最新的N条K线。
        如果提供了 start_time，则会分页获取从 start_time 到 end_time (如果提供) 的所有数据，
        此时 limit 参数(如果提供)作为获取总条数的上限。
        """
        all_klines_data = [] # 用于存储所有获取到的K线数据段
        
        # 情况1: 获取最新的N条K线 (limit提供, start_time未提供) - 不缓存最新数据
        if limit is not None and start_time is None:
            params = {
                'symbol': symbol.upper(),
                'interval': interval,
                'limit': min(limit, self.MAX_KLINE_LIMIT) # 遵守API单次最大限制
            }
            if end_time: # 虽然不常用，但允许在获取最新数据时指定一个结束点
                params['endTime'] = end_time
            print(f"获取最新的 {params['limit']} 条K线数据 (不缓存): 交易对={symbol}, 周期={interval}")
            data = self._request('GET', '/fapi/v1/klines', params=params, is_fapi=True)
            all_klines_data.extend(data)
        
        # 情况2: 分页获取指定时间范围的K线数据 (start_time提供) - 尝试使用缓存
        elif start_time is not None:
            cache_key = self._get_cache_key(symbol, interval, start_time, end_time)
            cached_data = self._load_cache(cache_key)

            if cached_data:
                print(f"从缓存加载K线数据: 交易对={symbol}, 周期={interval}, 时间范围={start_time}-{end_time}")
                all_klines_data = cached_data
            else:
                current_start_time_ms = start_time
                interval_duration_ms = self.INTERVAL_MS.get(interval)
                if not interval_duration_ms:
                    raise ValueError(f"不支持的K线周期: {interval}")

                start_dt_utc = datetime.fromtimestamp(start_time/1000, tz=timezone.utc)
                end_dt_utc_str = datetime.fromtimestamp(end_time/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC') if end_time else "现在"
                print(f"缓存未命中，开始分页获取K线数据: 交易对={symbol}, 周期={interval}, 从 {start_dt_utc.strftime('%Y-%m-%d %H:%M:%S UTC')} 到 {end_dt_utc_str}")
                
                fetch_iteration = 0 # 请求次数计数
                while True:
                    fetch_iteration += 1
                    current_request_limit = self.MAX_KLINE_LIMIT # 每次请求都尝试获取最大允许条数
                    
                    params_segment = {
                        'symbol': symbol.upper(),
                        'interval': interval,
                        'startTime': current_start_time_ms,
                        'limit': current_request_limit
                    }
                    if end_time: # 如果有总的结束时间，则将其传递给API
                        params_segment['endTime'] = end_time
                    
                    current_start_dt_str = datetime.fromtimestamp(current_start_time_ms/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                    print(f"  分页请求 #{fetch_iteration}: 开始时间={current_start_dt_str} UTC, 限制条数={current_request_limit}")

                    retry_attempts = 0
                    max_retries = 5 # 设置最大重试次数
                    data_segment = None # 初始化 data_segment

                    while retry_attempts < max_retries:
                        try:
                            data_segment = self._request('GET', '/fapi/v1/klines', params=params_segment, is_fapi=True)
                            break # 如果成功，跳出重试循环
                        except requests.exceptions.RequestException as req_err:
                            retry_attempts += 1
                            delay = 2 ** retry_attempts # 指数退避延迟
                            print(f"  分页请求 #{fetch_iteration} 时发生网络错误: {req_err}。尝试第 {retry_attempts}/{max_retries} 次重试，等待 {delay} 秒。")
                            time.sleep(delay)
                        except requests.exceptions.HTTPError as http_err:
                            # HTTPError 仍然直接抛出，因为这通常不是暂时性网络问题
                            print(f"  分页请求 #{fetch_iteration} 时发生HTTP错误: {http_err}。")
                            raise
                        except Exception as e:
                            # 其他未知错误仍然终止
                            print(f"  分页请求 #{fetch_iteration} 时发生未知错误: {e}。终止获取。")
                            # 可以在这里记录更详细的错误信息
                            import traceback
                            traceback.print_exc()
                            data_segment = None # 确保 data_segment 为 None 以便后续检查
                            break # 终止重试循环并继续外层循环，外层循环会检查 data_segment 是否为空

                    if retry_attempts == max_retries and data_segment is None:
                         print(f"  分页请求 #{fetch_iteration} 达到最大重试次数，获取失败。")
                         # data_segment 已经是 None

                    if not data_segment:
                        print(f"  分页请求 #{fetch_iteration}: 未获取到数据，已到达数据末尾或指定时间范围无数据。")
                        break
                    
                    all_klines_data.extend(data_segment)
                    
                    last_kline_open_time_ms = int(data_segment[-1][0])
                    current_start_time_ms = last_kline_open_time_ms + interval_duration_ms
                    
                    if end_time and current_start_time_ms > end_time:
                        print(f"  分页请求 #{fetch_iteration}: 下一个开始时间已超过总结束时间 {end_dt_utc_str}。")
                        break
                    
                    if limit is not None and len(all_klines_data) >= limit:
                        print(f"  分页请求 #{fetch_iteration}: 已达到请求的总K线条数上限 {limit}。")
                        all_klines_data = all_klines_data[:limit]
                        break
                    
                    if len(data_segment) < current_request_limit:
                        print(f"  分页请求 #{fetch_iteration}: 获取到的K线条数 ({len(data_segment)}) 少于请求的限制 ({current_request_limit})，可能已是最后的数据段。")
                        break
                    
                    time.sleep(0.25)

                # 如果成功获取到数据，则保存到缓存
                if all_klines_data:
                    self._save_cache(cache_key, all_klines_data)

        else: # 如果既没有提供 start_time 也没有提供 limit，则这是一个不明确的请求
            raise ValueError("get_historical_klines: 必须提供 start_time (用于分页获取指定时间范围数据) 或 limit (用于获取最新的N条数据)。")

        if not all_klines_data: # 如果最终没有获取到任何数据
            print(f"未能获取交易对 {symbol} 在周期 {interval} 的K线数据。")
            return pd.DataFrame() # 返回空的DataFrame

        # 将所有获取到的K线数据转换为Pandas DataFrame
        df = pd.DataFrame(all_klines_data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])

        # 进行数据类型转换
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True) # 开盘时间转换为UTC的datetime对象
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True) # 收盘时间
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce') # 数值列转换为数字类型，无法转换的变为NaN
        df['number_of_trades'] = df['number_of_trades'].astype(int) # 成交笔数转换为整数
        
        # API有时可能返回少量重复数据，尤其是在分页边界，这里基于开盘时间去重
        df.drop_duplicates(subset=['open_time'], keep='first', inplace=True)
        
        df.set_index('open_time', inplace=True) # 将开盘时间设为索引
        df.sort_index(inplace=True) # 按时间索引排序

        min_time_str = df.index.min().strftime('%Y-%m-%d %H:%M:%S UTC') if not df.empty else "N/A"
        max_time_str = df.index.max().strftime('%Y-%m-%d %H:%M:%S UTC') if not df.empty else "N/A"
        print(f"K线数据获取完成: 共获取到 {len(df)} 条不重复记录, 时间范围从 {min_time_str} 到 {max_time_str}")
        return df

    def get_latest_price(self, symbol: str) -> float:
        """获取指定U本位永续合约的最新价格"""
        params = {'symbol': symbol.upper()}
        # U本位合约使用 /fapi/v1/ticker/price
        data = self._request('GET', '/fapi/v1/ticker/price', params=params, is_fapi=True)
        return float(data['price'])

    # --- WebSocket 相关方法 ---
    def _get_ws_url(self, stream_name: str) -> str:
        """获取U本位合约的WebSocket基础URL"""
        return f"wss://fstream.binance.com/ws/{stream_name}" # 注意是 fstream

    def start_kline_websocket(self, symbol: str, interval: str, callback: callable):
        """启动指定交易对和周期的K线WebSocket数据流"""
        import websocket # 需要安装 websocket-client 库
        import threading # 用于在后台线程运行WebSocket
        import json # 用于解析JSON消息

        stream_name = f"{symbol.lower()}@kline_{interval}" # WebSocket流名称格式
        ws_key = f"{symbol.lower()}_{interval}" # 用于内部管理连接的唯一键

        # 检查是否已存在此流的连接
        # 注意：WebSocketApp对象没有标准的 is_connected() 方法，需要根据实际情况检查
        # 例如，可以检查 ws.sock 是否存在且连接，或者依赖 on_close 来清理状态
        if ws_key in self.ws_connections and self.ws_connections[ws_key].sock and self.ws_connections[ws_key].sock.connected:
            logger.info(f"交易对 {symbol} 周期 {interval} 的K线WebSocket已经运行。")
            return

        ws_url = self._get_ws_url(stream_name)
        logger.info(f"正在启动交易对 {symbol} 周期 {interval} 的K线WebSocket连接到 {ws_url}")
        
        # WebSocket消息处理回调函数
        def on_message(ws_app, message_str):
            try:
                data = json.loads(message_str)
                logger.debug(f"收到来自 {symbol} {interval} 的WebSocket消息 (事件类型: {data.get('e')})") # 记录收到的消息类型，避免记录大量数据
                if 'e' in data and data['e'] == 'kline': # 确认是K线事件
                    kline_payload = data['k']
                    # 格式化K线数据以便回调函数使用
                    processed_kline = {
                        'event_type': data['e'],
                        'event_time': datetime.fromtimestamp(data['E'] / 1000, tz=timezone.utc), # 事件时间
                        'symbol': data['s'], # 交易对
                        'interval': kline_payload['i'], # K线周期
                        'kline_start_time': kline_payload['t'], # K线开盘时间 (毫秒时间戳)
                        'kline_close_time': kline_payload['T'], # K线收盘时间 (毫秒时间戳)
                        'open': float(kline_payload['o']), 'high': float(kline_payload['h']),
                        'low': float(kline_payload['l']), 'close': float(kline_payload['c']),
                        'volume': float(kline_payload['v']), # 成交量
                        'number_of_trades': kline_payload['n'], # 成交笔数
                        'is_kline_closed': kline_payload['x'], # 此K线是否已收盘 (True/False)
                        'quote_asset_volume': float(kline_payload['q']), # 成交额
                        'taker_buy_base_asset_volume': float(kline_payload['V']), # 主动买入的交易量
                        'taker_buy_quote_asset_volume': float(kline_payload['Q']) # 主动买入的成交额
                    }
                    callback(processed_kline) # 调用外部传入的回调函数
            except Exception as e:
                logger.error(f"处理来自 {symbol} {interval} 的WebSocket消息时出错: {e}", exc_info=True) # 记录完整的错误信息和堆栈跟踪
                logger.debug(f"原始消息: {message_str}") # 记录原始消息，用于调试

        # WebSocket错误处理回调
        def on_error(ws_app, error_msg):
            logger.error(f"交易对 {symbol} 周期 {interval} 的WebSocket发生错误: {error_msg}", exc_info=True) # 记录完整的错误信息和堆栈跟踪
            # error_msg 可能是异常对象或字符串，尝试打印详细信息
            if not isinstance(error_msg, Exception):
                 logger.error(f"错误详情: {error_msg}") # 如果不是异常对象，打印错误信息字符串
            
            # 尝试重连
            self._reconnect_websocket(symbol, interval, callback)


        # WebSocket关闭处理回调
        def on_close(ws_app, close_status_code, close_msg):
            logger.info(f"交易对 {symbol} 周期 {interval} 的WebSocket已关闭。状态码: {close_status_code}, 原因: {close_msg}")
            
            # 清理此连接的记录
            ws_key_local = f"{symbol.lower()}_{interval}" # 使用本地变量避免闭包问题
            if ws_key_local in self.ws_connections:
                 # 注意：这里不立即删除，因为 _reconnect_websocket 可能需要访问这些信息
                 # 清理将在重连成功或达到最大尝试次数后进行
                 pass # 暂时不在这里删除连接和线程记录
            
            # 尝试重连
            self._reconnect_websocket(symbol, interval, callback)


        # WebSocket成功打开回调
        def on_open(ws_app):
            logger.info(f"交易对 {symbol} 周期 {interval} 的K线WebSocket连接成功打开。URL: {ws_app.url}")
            # 连接成功后，重置重连尝试次数
            ws_key_local = f"{symbol.lower()}_{interval}"
            self._reconnect_attempts[ws_key_local] = 0
            logger.debug(f"交易对 {symbol} 周期 {interval} 的WebSocket重连尝试次数已重置。")


        # 创建WebSocketApp实例
        ws = websocket.WebSocketApp(ws_url,
                                  on_open=on_open,
                                  on_message=on_message,
                                  on_error=on_error,
                                  on_close=on_close)
        
        self.ws_connections[ws_key] = ws # 存储WebSocketApp实例
        
        # 创建并启动一个新的守护线程来运行WebSocket，避免阻塞主程序
        # 注意：run_forever 会阻塞，所以必须在单独的线程中运行
        ws_thread = threading.Thread(target=lambda: ws.run_forever(reconnect=False), daemon=True) # 设置 reconnect=False，我们自己处理重连
        self.ws_threads[ws_key] = ws_thread
        ws_thread.start()
        print(f"交易对 {symbol} 周期 {interval} 的K线WebSocket处理线程已启动。")

    def _reconnect_websocket(self, symbol: str, interval: str, callback: callable):
        """尝试重新连接WebSocket"""
        ws_key = f"{symbol.lower()}_{interval}"
        attempt = self._reconnect_attempts.get(ws_key, 0) + 1
        self._reconnect_attempts[ws_key] = attempt

        if attempt > self._max_reconnect_attempts:
            logger.warning(f"交易对 {symbol} 周期 {interval} 的WebSocket已达到最大重连尝试次数 ({self._max_reconnect_attempts})，停止重连。")
            # 清理连接和线程记录
            if ws_key in self.ws_connections: del self.ws_connections[ws_key]
            if ws_key in self.ws_threads: del self.ws_threads[ws_key]
            return

        # 指数退避延迟
        delay = self._reconnect_delay_base * (2 ** (attempt - 1))
        logger.info(f"交易对 {symbol} 周期 {interval} 的WebSocket将在 {delay} 秒后尝试第 {attempt} 次重连...")
        time.sleep(delay)

        logger.info(f"正在尝试重新连接交易对 {symbol} 周期 {interval} 的WebSocket...")
        # 重新启动WebSocket连接
        # 注意：这里需要重新创建并启动一个新的WebSocketApp实例和线程
        # 因为旧的连接可能已经损坏或关闭
        try:
            # 确保在重新启动前清理旧的连接和线程（如果它们仍然存在）
            if ws_key in self.ws_connections and self.ws_connections[ws_key].sock:
                 try:
                      self.ws_connections[ws_key].close()
                 except Exception:
                      pass # 忽略关闭旧连接时的错误
                 del self.ws_connections[ws_key]

            if ws_key in self.ws_threads and self.ws_threads[ws_key].is_alive():
                 # 线程可能需要一些时间来结束，这里不强制终止，依赖daemon=True
                 pass # 暂时不在这里等待线程结束

            # 重新调用 start_kline_websocket 来建立新的连接
            # 注意：start_kline_websocket 内部会检查是否已存在连接，这里需要确保旧的连接已被清理
            # 或者修改 start_kline_websocket 逻辑以允许强制重新创建
            # 为了简单起见，我们依赖上面的清理，并假设 start_kline_websocket 会创建新的
            self.start_kline_websocket(symbol, interval, callback)
            logger.info(f"交易对 {symbol} 周期 {interval} 的WebSocket第 {attempt} 次重连尝试已启动。")

        except Exception as e:
            logger.error(f"交易对 {symbol} 周期 {interval} 的WebSocket第 {attempt} 次重连尝试失败: {e}", exc_info=True) # 记录完整的错误信息和堆栈跟踪
            # 如果重连失败，继续尝试下一次重连
            self._reconnect_websocket(symbol, interval, callback)


    def stop_kline_websocket(self, symbol: str, interval: str):
        """停止指定交易对和周期的K线WebSocket"""
        ws_key = f"{symbol.lower()}_{interval}"
        if ws_key in self.ws_connections:
            logger.info(f"正在停止交易对 {symbol} 周期 {interval} 的K线WebSocket...")
            ws = self.ws_connections[ws_key]
            ws.close() # 关闭WebSocket连接，这将触发on_close回调进行清理
            # on_close回调中会删除 self.ws_connections 和 self.ws_threads 中的对应项
            logger.info(f"已发送停止请求给交易对 {symbol} 周期 {interval} 的K线WebSocket。")
        else:
            logger.warning(f"未找到活动的K线WebSocket连接以停止: 交易对={symbol}, 周期={interval}。")

    def stop_all_websockets(self):
        """停止所有当前活动的WebSocket连接"""
        num_connections = len(self.ws_connections)
        if num_connections == 0:
            logger.info("没有活动的WebSocket连接需要停止。")
            return
            
        logger.info(f"正在停止所有 {num_connections} 个活动的WebSocket连接...")
        # 迭代键的副本，因为stop_kline_websocket会修改字典
        for ws_key in list(self.ws_connections.keys()):
            try:
                # 从ws_key中解析出symbol和interval有点麻烦，如果key格式固定可以做
                # 假设ws_key就是 "symbol_interval"格式
                parts = ws_key.split('_', 1)
                if len(parts) == 2:
                    self.stop_kline_websocket(parts[0].upper(), parts[1])
                else:
                    logger.warning(f"无法从键 '{ws_key}' 中解析交易对和周期，跳过停止。")
            except Exception as e:
                logger.error(f"停止WebSocket (键: {ws_key}) 时发生错误: {e}", exc_info=True)
        
        # 等待所有线程结束（可选，用于确保清理）
        # for thread_key, thread_obj in list(self.ws_threads.items()):
        #     if thread_obj.is_alive():
        #         thread_obj.join(timeout=3) # 等待3秒
        #         if thread_obj.is_alive():
        #             print(f"警告: WebSocket线程 {thread_key} 在请求停止后仍未结束。")
        
        print(f"已发送停止请求给所有活动的WebSocket连接。")


    def test_websocket_connection(self, symbol: str = "BTCUSDT", interval: str = "1m"):
        """
        测试WebSocket连接是否成功建立，并打印收到的第一条消息。
        尝试启动一个K线WebSocket连接，等待片刻，然后检查连接状态并捕获消息。
        """
        import time
        import threading
        import websocket # 确保 websocket-client 已安装
        import json # 用于解析JSON消息

        ws_key = f"{symbol.lower()}_{interval}"
        test_passed = False
        first_message_received = threading.Event() # 用于信号量，表示是否收到第一条消息
        received_message_data = None # 用于存储收到的第一条消息数据

        # 修改后的回调函数，用于捕获第一条消息
        def test_callback(kline_data_dict):
            nonlocal received_message_data # 允许修改外部函数的变量
            if not first_message_received.is_set():
                logger.info(f"收到第一条WebSocket消息: {kline_data_dict}")
                received_message_data = kline_data_dict
                first_message_received.set() # 设置事件，表示已收到第一条消息

        logger.info(f"正在测试交易对 {symbol} 周期 {interval} 的WebSocket连接...")

        try:
            # 启动WebSocket连接，使用修改后的回调函数
            self.start_kline_websocket(symbol, interval, test_callback)

            # 等待连接建立并接收第一条消息，给足够的时间
            # 等待连接成功打开 (on_open 会记录日志)
            time.sleep(2) # 给连接建立一些时间
            
            # 等待第一条消息到达，最多等待10秒
            if first_message_received.wait(timeout=10):
                 logger.info("成功收到WebSocket消息。")
                 test_passed = True
            else:
                 logger.error("在规定时间内未收到WebSocket消息。")
                 test_passed = False


            # 检查连接状态 (可选，但可以提供更多信息)
            if ws_key in self.ws_connections:
                ws_app = self.ws_connections[ws_key]
                if ws_app.sock and ws_app.sock.connected:
                    logger.info(f"交易对 {symbol} 周期 {interval} 的WebSocket连接状态良好。")
                else:
                    logger.warning(f"交易对 {symbol} 周期 {interval} 的WebSocket连接Socket未连接。")
            else:
                logger.warning(f"交易对 {symbol} 周期 {interval} 的WebSocket连接实例不存在。")


        except Exception as e:
            logger.error(f"测试交易对 {symbol} 周期 {interval} 的WebSocket连接时发生异常: {e}", exc_info=True)
            test_passed = False

        finally:
            # 清理：停止WebSocket连接
            self.stop_kline_websocket(symbol, interval)
            # 给一些时间让线程结束
            time.sleep(2)
            logger.info(f"交易对 {symbol} 周期 {interval} 的WebSocket连接测试清理完成。")

        # 断言连接是否成功 (基于是否收到第一条消息)
        assert test_passed, f"WebSocket连接测试失败: 未收到消息或发生错误。交易对={symbol}, 周期={interval}"


if __name__ == '__main__':
    client = BinanceClient()

    # 添加WebSocket连接测试
    try:
        client.test_websocket_connection("BTCUSDT", "1m")
        print("\nWebSocket连接测试通过。")
    except AssertionError as e:
        print(f"\nWebSocket连接测试失败: {e}")
    except Exception as e:
        print(f"\nWebSocket连接测试过程中发生错误: {e}")


    try:
        server_time_dt = datetime.fromtimestamp(client.get_server_time() / 1000, tz=timezone.utc)
        print(f"\n币安服务器时间: {server_time_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")

        # available_symbols_list = client.get_available_symbols()
        # print(f"获取到 {len(available_symbols_list)} 个U本位永续合约交易对。")
        # if available_symbols_list: print(f"  前5个: {available_symbols_list[:5]}")

        print(f"\n--- 测试短时段1分钟K线分页获取 (预计1-2次API调用) ---")
        now_utc = datetime.now(timezone.utc)
        start_time_short_1m = int((now_utc - timedelta(hours=3)).timestamp() * 1000)
        end_time_short_1m = int((now_utc - timedelta(hours=1)).timestamp() * 1000)

        df_1m_short_data = client.get_historical_klines(
            symbol="BTCUSDT",
            interval="1m",
            start_time=start_time_short_1m,
            end_time=end_time_short_1m
        )
        if not df_1m_short_data.empty:
            print(f"成功获取 {len(df_1m_short_data)} 条1分钟K线 (短时段)。")
            print(f"数据时间范围: {df_1m_short_data.index.min()} 至 {df_1m_short_data.index.max()}")
        else:
            print("未能获取到短时段1分钟K线数据。")

        print(f"\n--- 测试长时段1分钟K线分页获取 (预计多次API调用) ---")
        # 预期获取1天3小时 = 27小时 = 1620分钟的1分钟K线数据，约需要 1620/1000 + 1 = 2到3次API调用
        start_time_long_1m = int((now_utc - timedelta(days=1, hours=5)).timestamp() * 1000)
        end_time_long_1m = int((now_utc - timedelta(hours=2)).timestamp() * 1000)

        df_1m_long_data = client.get_historical_klines(
            symbol="BTCUSDT",
            interval="1m",
            start_time=start_time_long_1m,
            end_time=end_time_long_1m
        )
        if not df_1m_long_data.empty:
            print(f"成功获取 {len(df_1m_long_data)} 条1分钟K线 (长时段)。")
            print(f"数据时间范围: {df_1m_long_data.index.min()} 至 {df_1m_long_data.index.max()}")
            # 简单检查数据连续性
            time_diffs = df_1m_long_data.index.to_series().diff().dropna()
            expected_interval_td = pd.Timedelta(minutes=1)
            # 找出时间间隔大于预期1.5倍的K线（允许一些小的网络延迟或API时间戳不完美）
            gaps = time_diffs[time_diffs > expected_interval_td * 1.5]
            if not gaps.empty:
                print(f"警告: 在1分钟K线数据中发现潜在的时间跳空点:\n{gaps}")
            else:
                print("1分钟K线数据看起来是连续的。")
        else:
            print("未能获取到长时段1分钟K线数据。")

        # print(f"\n--- 测试获取最近N条K线 ---")
        # df_latest_klines = client.get_historical_klines(symbol="ETHUSDT", interval="15m", limit=20)
        # if not df_latest_klines.empty:
        #     print(f"成功获取最新的 {len(df_latest_klines)} 条15分钟ETH K线。")
        #     print(df_latest_klines.tail())
        # else:
        #     print("未能获取最新的K线数据。")

        # print(f"\n获取 {symbol.upper()} 最新价格: {client.get_latest_price('BTCUSDT')}")

    except Exception as main_exception:
        logger.error(f"主程序发生错误: {main_exception}", exc_info=True)