# --- START OF FILE gate_client.py ---

import os
import time
import json
import hashlib
import logging
import threading
from typing import Optional, List, Dict, Any

import requests
import pandas as pd

logger = logging.getLogger(__name__)


class GateClient:
    BASE_URL = "https://api.gateio.ws"
    FUTURES_PREFIX = "/api/v4/futures/usdt"
    CACHE_DIR = ".kline_cache"

    INTERVAL_SECONDS = {
        '1m': 60,
        '3m': 180,
        '5m': 300,
        '15m': 900,
        '30m': 1800,
        '1h': 3600,
        '2h': 7200,
        '4h': 14400,
        '6h': 21600,
        '8h': 28800,
        '12h': 43200,
        '1d': 86400,
        '3d': 259200,
        '1w': 604800,
        '1M': 2592000
    }
    MAX_KLINE_LIMIT = 1000

    def __init__(self):
        self.session = requests.Session()
        if not os.path.exists(self.CACHE_DIR):
            os.makedirs(self.CACHE_DIR)
            print(f"创建K线缓存目录: {self.CACHE_DIR}")

        self.ws_threads: Dict[str, threading.Thread] = {}
        self.ws_stop_flags: Dict[str, threading.Event] = {}
        self.ws_management_lock = threading.RLock()

    def _request(self, method: str, url_path: str, params: Optional[dict] = None):
        full_url = f"{self.BASE_URL}{url_path}"
        try:
            if method.upper() == 'GET':
                response = self.session.get(full_url, params=params, timeout=(10, 30))
            else:
                raise ValueError(f"不支持的HTTP方法: {method}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP请求错误: {http_err} - URL: {full_url}")
            raise
        except requests.exceptions.RequestException as req_err:
            logger.error(f"请求 {full_url} 时发生网络或连接错误: {req_err}，参数: {params}")
            raise
        except Exception as e:
            logger.error(f"请求 {full_url} 时发生未知错误: {e}，参数: {params}", exc_info=True)
            raise

    def _normalize_symbol(self, symbol: str) -> str:
        symbol_upper = symbol.upper()
        if "_" in symbol_upper:
            return symbol_upper
        if symbol_upper.endswith("USDT") and len(symbol_upper) > 4:
            return f"{symbol_upper[:-4]}_USDT"
        return symbol_upper

    def _get_cache_key(self, symbol: str, interval: str, start_time: Optional[int], end_time: Optional[int]) -> str:
        key_str = f"gate_{symbol}_{interval}_{start_time}_{end_time}"
        return hashlib.sha256(key_str.encode('utf-8')).hexdigest()

    def _get_cache_path(self, key: str) -> str:
        return os.path.join(self.CACHE_DIR, f"{key}.json")

    def _load_cache(self, key: str) -> Optional[List[Any]]:
        cache_path = self._get_cache_path(key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (IOError, json.JSONDecodeError) as e:
                print(f"加载缓存文件失败 {cache_path}: {e}")
                try:
                    os.remove(cache_path)
                except OSError:
                    pass
        return None

    def _save_cache(self, key: str, data: List[Any]):
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f)
        except IOError as e:
            print(f"保存数据到缓存文件失败 {cache_path}: {e}")

    def _parse_candle_row(self, row: Any) -> Optional[Dict[str, Any]]:
        try:
            if isinstance(row, dict):
                timestamp = row.get('t') or row.get('timestamp') or row.get('time')
                open_price = row.get('o') or row.get('open')
                high_price = row.get('h') or row.get('high')
                low_price = row.get('l') or row.get('low')
                close_price = row.get('c') or row.get('close')
                volume = row.get('v') or row.get('volume') or row.get('base_volume') or row.get('amount')
            else:
                if len(row) < 6:
                    return None
                timestamp = row[0]
                volume = row[1]
                close_price = row[2]
                high_price = row[3]
                low_price = row[4]
                open_price = row[5]

            return {
                'open_time': int(float(timestamp)) * 1000,
                'open': float(open_price),
                'high': float(high_price),
                'low': float(low_price),
                'close': float(close_price),
                'volume': float(volume) if volume is not None else 0.0
            }
        except Exception:
            return None

    def _candles_to_dataframe(self, candles: List[Any], interval: str) -> pd.DataFrame:
        interval_seconds = self.INTERVAL_SECONDS.get(interval)
        if not interval_seconds:
            raise ValueError(f"不支持的K线周期: {interval}")

        rows = []
        for row in candles:
            parsed = self._parse_candle_row(row)
            if parsed:
                close_time_ms = parsed['open_time'] + interval_seconds * 1000
                parsed['close_time'] = close_time_ms
                rows.append(parsed)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)
        df.set_index('open_time', inplace=True)
        df.sort_index(inplace=True)
        return df

    def get_available_symbols(self, quote_asset: str = "USDT") -> List[str]:
        try:
            data = self._request('GET', f"{self.FUTURES_PREFIX}/contracts")
            symbols = []
            for item in data:
                name = item.get('name') or item.get('contract') or item.get('id')
                if not name:
                    continue
                if quote_asset.upper() not in name.upper():
                    continue
                symbols.append(name.upper())
            return sorted(set(symbols))
        except Exception as e:
            print(f"获取可用交易对列表失败: {e}。将返回一个默认列表。")
            return ["BTC_USDT", "ETH_USDT", "BNB_USDT", "ADA_USDT", "XRP_USDT", "DOGE_USDT", "DOT_USDT", "SOL_USDT"]

    def get_hot_symbols_by_volume(self, quote_asset: str = "USDT", top_n: int = 50) -> List[str]:
        try:
            data = self._request('GET', f"{self.FUTURES_PREFIX}/tickers")
            hot_symbols = []
            for item in data:
                symbol = item.get('contract') or item.get('name')
                if not symbol or quote_asset.upper() not in symbol.upper():
                    continue
                volume = item.get('volume_24h') or item.get('volume') or item.get('quote_volume') or 0
                try:
                    hot_symbols.append({'symbol': symbol.upper(), 'volume': float(volume)})
                except (ValueError, TypeError):
                    continue

            hot_symbols.sort(key=lambda x: x['volume'], reverse=True)
            return [item['symbol'] for item in hot_symbols[:top_n]]
        except Exception as e:
            print(f"获取热门交易对失败: {e}。将返回一个默认列表。")
            return ["BTC_USDT", "ETH_USDT", "BNB_USDT", "ADA_USDT", "XRP_USDT", "DOGE_USDT", "DOT_USDT", "SOL_USDT"]

    def get_historical_klines(self, symbol: str, interval: str,
                              start_time: Optional[int] = None,
                              end_time: Optional[int] = None,
                              limit: Optional[int] = None) -> pd.DataFrame:
        normalized_symbol = self._normalize_symbol(symbol)

        if limit is not None and start_time is None:
            params = {
                'contract': normalized_symbol,
                'interval': interval,
                'limit': min(limit, self.MAX_KLINE_LIMIT)
            }
            if end_time:
                params['to'] = int(end_time / 1000)
            data = self._request('GET', f"{self.FUTURES_PREFIX}/candlesticks", params=params)
            return self._candles_to_dataframe(data, interval)

        if start_time is None:
            raise ValueError("get_historical_klines: 必须提供 start_time 或 limit。")

        cache_key = self._get_cache_key(normalized_symbol, interval, start_time, end_time)
        cached_data = self._load_cache(cache_key)
        if cached_data:
            return self._candles_to_dataframe(cached_data, interval)

        all_data: List[Any] = []
        current_start = int(start_time / 1000)
        final_end = int(end_time / 1000) if end_time else int(time.time())
        interval_seconds = self.INTERVAL_SECONDS.get(interval)
        if not interval_seconds:
            raise ValueError(f"不支持的K线周期: {interval}")

        while current_start <= final_end:
            request_end = min(
                final_end,
                current_start + interval_seconds * (self.MAX_KLINE_LIMIT - 1)
            )
            params = {
                'contract': normalized_symbol,
                'interval': interval,
                'from': current_start,
                'to': request_end,
                'limit': self.MAX_KLINE_LIMIT
            }
            data_segment = self._request('GET', f"{self.FUTURES_PREFIX}/candlesticks", params=params)
            if not data_segment:
                break
            all_data.extend(data_segment)

            last_row = self._parse_candle_row(data_segment[-1])
            if not last_row:
                break
            last_open_time = int(last_row['open_time'] / 1000)
            next_start = last_open_time + interval_seconds
            if next_start <= current_start:
                break
            current_start = next_start

            if len(data_segment) < self.MAX_KLINE_LIMIT:
                break
            time.sleep(0.15)

        if all_data:
            self._save_cache(cache_key, all_data)
        return self._candles_to_dataframe(all_data, interval)

    def get_latest_price(self, symbol: str) -> float:
        normalized_symbol = self._normalize_symbol(symbol)
        params = {'contract': normalized_symbol}
        data = self._request('GET', f"{self.FUTURES_PREFIX}/tickers", params=params)
        if isinstance(data, list) and data:
            ticker = data[0]
        else:
            ticker = data
        price = ticker.get('last') or ticker.get('mark_price') or ticker.get('index_price') or 0
        return float(price)

    def get_index_price_klines(self, symbol: str, interval: str,
                               start_time: Optional[int] = None,
                               end_time: Optional[int] = None,
                               limit: Optional[int] = None) -> pd.DataFrame:
        logger.info("Gate 暂无指数K线专用接口，使用合约K线替代。")
        return self.get_historical_klines(symbol, interval, start_time, end_time, limit)

    def start_kline_websocket(self, symbol: str, interval: str, user_callback: callable):
        normalized_symbol = self._normalize_symbol(symbol)
        ws_key = f"{normalized_symbol}::{interval}"

        with self.ws_management_lock:
            if ws_key in self.ws_threads and self.ws_threads[ws_key].is_alive():
                logger.info(f"交易对 {normalized_symbol} {interval} 的K线轮询已运行。")
                return

            stop_event = threading.Event()
            self.ws_stop_flags[ws_key] = stop_event

            def poll():
                last_open_time = None
                while not stop_event.is_set():
                    try:
                        df = self.get_historical_klines(normalized_symbol, interval, limit=2)
                        if df.empty:
                            time.sleep(1)
                            continue
                        latest_row = df.iloc[-1]
                        open_time = df.index[-1]
                        if last_open_time is None or open_time > last_open_time:
                            interval_seconds = self.INTERVAL_SECONDS.get(interval, 60)
                            open_timestamp = open_time.to_pydatetime().timestamp()
                            kline_data = {
                                'symbol': normalized_symbol,
                                'interval': interval,
                                'is_kline_closed': True,
                                'kline_start_time': int(open_timestamp * 1000),
                                'kline_close_time': int((open_timestamp + interval_seconds) * 1000),
                                'open': float(latest_row.get('open', 0)),
                                'high': float(latest_row.get('high', 0)),
                                'low': float(latest_row.get('low', 0)),
                                'close': float(latest_row.get('close', 0)),
                                'volume': float(latest_row.get('volume', 0)),
                                'quote_asset_volume': 0,
                                'number_of_trades': 0,
                                'taker_buy_base_asset_volume': 0,
                                'taker_buy_quote_asset_volume': 0
                            }
                            user_callback(kline_data)
                            last_open_time = open_time
                    except Exception as e:
                        logger.error(f"Gate K线轮询出错: {e}", exc_info=True)
                    time.sleep(1)

            thread = threading.Thread(target=poll, daemon=True)
            self.ws_threads[ws_key] = thread
            thread.start()

    def stop_kline_websocket(self, symbol: str, interval: str):
        normalized_symbol = self._normalize_symbol(symbol)
        ws_key = f"{normalized_symbol}::{interval}"
        with self.ws_management_lock:
            stop_event = self.ws_stop_flags.pop(ws_key, None)
            thread = self.ws_threads.pop(ws_key, None)
            if stop_event:
                stop_event.set()
            if thread and thread.is_alive():
                thread.join(timeout=5)

    def stop_all_websockets(self):
        with self.ws_management_lock:
            keys = list(self.ws_threads.keys())
        for key in keys:
            symbol, interval = key.split("::", 1)
            self.stop_kline_websocket(symbol, interval)

# --- END OF FILE gate_client.py ---
