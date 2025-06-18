import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
import logging # 导入logging模块
logger = logging.getLogger(__name__)

class Strategy:
    """策略基类，所有具体策略都应继承此类"""
    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or {}
        self.name = "基础策略"
        self.data_checked = False
        self.min_history_periods = 1 # 基类默认至少需要1条

    def _ensure_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.data_checked: # 只检查一次
            df_copy = df.copy() # 操作副本
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df_copy.columns:
                    if df_copy[col].dtype == 'object':
                        try:
                            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                        except ValueError:
                            print(f"警告: 列 {col} 无法转换为数值类型。")
                    elif not pd.api.types.is_numeric_dtype(df_copy[col]):
                         print(f"警告: 列 {col} 不是数值类型 (类型为 {df_copy[col].dtype})。")
            if 'close' not in df_copy.columns:
                raise ValueError("DataFrame 中必须包含 'close' 列。")
            if not pd.api.types.is_numeric_dtype(df_copy['close']):
                 raise ValueError("'close' 列必须是数值类型。")
            self.data_checked = True
            return df_copy
        return df # 如果已检查，直接返回原df，避免不必要的复制

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        在完整的DataFrame上一次性计算所有需要的技术指标。
        子策略应重写此方法。
        """
        logger.info(f"({self.name}) 基类 calculate_all_indicators 调用，检查数据类型。")
        df_checked = self._ensure_data_types(df)
        # 子类将在这里添加具体的指标计算
        return df_checked

    def generate_signals_from_indicators_on_window(self, df_window_with_indicators: pd.DataFrame) -> pd.DataFrame:
        """
        在给定的、已经包含预计算指标的窗口数据上，为窗口的最后一行生成信号。
        子策略应重写此方法。
        返回的DataFrame应与输入df_window_with_indicators具有相同索引和列，
        并增加了'signal'和'confidence'列，信号主要设置在最后一行。
        """
        # print(f"({self.name}) 基类 generate_signals_from_indicators_on_window 调用。")
        df_out = df_window_with_indicators.copy()
        if 'signal' not in df_out.columns: df_out['signal'] = 0
        if 'confidence' not in df_out.columns: df_out['confidence'] = 0
        # 子类会在这里基于窗口内的指标为最后一行判断信号
        return df_out

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标准的信号生成方法，供实测或非优化回测使用。
        它会计算指标，并为输入DataFrame的每一行（从有足够历史开始）生成信号。
        子策略应重写此方法。
        """
        print(f"({self.name}) 基类 generate_signals 调用 (通常用于实测或完整信号生成)。")
        # 1. 计算指标 (对整个传入的df，通常是实测时的100条窗口)
        df_with_indicators = self.calculate_all_indicators(df.copy())
        
        # 2. 初始化信号列
        df_signaled = df_with_indicators.copy()
        if 'signal' not in df_signaled.columns: df_signaled['signal'] = 0
        if 'confidence' not in df_signaled.columns: df_signaled['confidence'] = 0
        
        # 3. 子类在这里实现对df_signaled每一行的信号判断逻辑
        # (这是各个策略原有的 generate_signals 内部的循环判断部分)
        return df_signaled

# --- SimpleRSIStrategy ---
class SimpleRSIStrategy(Strategy):
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {'rsi_period': 14, 'rsi_overbought': 70, 'rsi_oversold': 30}
        if params: default_params.update(params)
        super().__init__(default_params)
        self.name = "简单RSI策略"
        self.min_history_periods = self.params['rsi_period'] + 1 # RSI需要 دوره + diff
        # print(f"初始化 {self.name}, 参数: {self.params}")

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = super().calculate_all_indicators(df) # 调用基类方法，可能进行数据类型检查

            # 确保 'close' 列存在且为数值类型
            if 'close' not in df.columns or not pd.api.types.is_numeric_dtype(df['close']):
                logger.error(f"({self.name}) 'close' 列不存在或不是数值类型。无法计算RSI。RSI将填充为50。")
                df['rsi'] = 50.0
                return df

            close_prices = df['close'].copy() # 使用副本以避免修改原始DataFrame中的 'close' 列

            # 1. 数据清洗: 处理 close_prices 中的 inf 值，将其替换为 NaN
            #    NaN 值由后续的 pandas 操作自行处理或传播
            original_inf_count = np.isinf(close_prices).sum()
            if original_inf_count > 0:
                logger.warning(f"({self.name}) 'close' 列在RSI计算前包含 {original_inf_count} 个 inf 值，将替换为 NaN。")
                close_prices = close_prices.replace([np.inf, -np.inf], np.nan)

            # 检查在处理 inf 后，或原本 'close' 列是否所有值都为 NaN
            if close_prices.isnull().all():
                logger.warning(f"({self.name}) 'close' 列所有值均为 NaN (可能在替换 inf 后)。RSI 将被设置为 50。")
                df['rsi'] = 50.0
                return df
            
            # 获取 RSI 周期参数，提供默认值以防参数缺失 (尽管 __init__ 中应已设置)
            rsi_period = self.params.get('rsi_period', 14)

            # 2. 计算价格变化 (delta)
            delta = close_prices.diff() # delta 的第一个值是 NaN

            # 3. 分离涨跌
            # delta > 0 或 delta < 0 对于 NaN 会是 False，所以 .where 的 other 参数 (0.0) 会被使用
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0) # loss 定义为正值

            # 4. 计算平均涨跌幅 (Exponential Moving Average)
            # min_periods=rsi_period 确保在有足够数据之前结果为 NaN
            avg_gain = gain.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
            avg_loss = loss.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
            
            # 5. 防御性处理 avg_gain 和 avg_loss 中的 inf 值 (理论上不应出现)
            avg_gain = avg_gain.replace([np.inf, -np.inf], np.nan)
            avg_loss = avg_loss.replace([np.inf, -np.inf], np.nan)

            # 6. 计算 RS (Relative Strength)
            #    处理 avg_loss 中的 0 (替换为 NaN 以避免除以零错误)
            #    avg_loss 本身也可能包含 NaN (来自 ewm 的初期或输入数据)
            avg_loss_safe = avg_loss.replace(0, np.nan)
            
            rs = avg_gain / avg_loss_safe
            
            # 7. 计算 RSI = 100 - (100 / (1 + RS))
            #    首先，确保 RS 本身不含 inf (已通过 avg_gain/avg_loss 清洗完成)
            #    然后，处理 1 + RS 可能为 0 的情况 (即 RS = -1)
            #    理论上 RS >= 0，但为安全起见，将导致除以零的分母替换为 NaN
            rs = rs.replace([np.inf, -np.inf], np.nan) # 再次确保 rs 没有 inf
            denominator = 1 + rs
            denominator_safe = denominator.replace(0, np.nan) # 若 1+rs=0, 则替换为 NaN
            
            rsi_values = 100 - (100 / denominator_safe)
            
            # 8. 最终填充 RSI 中的所有 NaN 值 (来自计算过程或原始数据问题)
            #    使用中性值 50.0 进行填充
            #    也替换掉任何可能意外产生的 inf 值
            rsi_values = rsi_values.replace([np.inf, -np.inf], np.nan)
            df['rsi'] = rsi_values.fillna(50.0)
            
            # logger.info(f"({self.name}) RSI calculation successful for period {rsi_period}.")

        except Exception as e:
            logger.error(f"({self.name}) 在 calculate_all_indicators 中计算RSI时发生严重错误: {e}", exc_info=True)
            # 发生未知错误时，确保 'rsi' 列存在并填充默认值
            df['rsi'] = 50.0 # 覆盖或创建 'rsi' 列

        return df

    def generate_signals_from_indicators_on_window(self, df_window_with_indicators: pd.DataFrame) -> pd.DataFrame:
        df_out = df_window_with_indicators.copy()
        if 'signal' not in df_out.columns: df_out['signal'] = 0
        if 'confidence' not in df_out.columns: df_out['confidence'] = 0

        if len(df_out) < 2 or 'rsi' not in df_out.columns:
            return df_out

        # 判断窗口的最后一行
        curr_idx = -1
        prev_idx = -2
        
        if pd.isna(df_out.iloc[curr_idx]['rsi']) or pd.isna(df_out.iloc[prev_idx]['rsi']):
            return df_out

        curr_rsi = df_out.iloc[curr_idx]['rsi']
        prev_rsi = df_out.iloc[prev_idx]['rsi']
        signal = 0
        confidence_score = 0

        if curr_rsi < self.params['rsi_oversold'] and prev_rsi >= self.params['rsi_oversold']:
            signal = 1
            rsi_distance = self.params['rsi_oversold'] - curr_rsi
            confidence_score = 50 + min(50, rsi_distance * 3)
        elif curr_rsi > self.params['rsi_overbought'] and prev_rsi <= self.params['rsi_overbought']:
            signal = -1
            rsi_distance = curr_rsi - self.params['rsi_overbought']
            confidence_score = 50 + min(50, rsi_distance * 3)
        
        df_out.iloc[curr_idx, df_out.columns.get_loc('signal')] = signal
        df_out.iloc[curr_idx, df_out.columns.get_loc('confidence')] = int(max(0, min(100, confidence_score)))
        return df_out

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df_with_indicators = self.calculate_all_indicators(df.copy())
        df_signaled = df_with_indicators.copy()
        if 'signal' not in df_signaled.columns: df_signaled['signal'] = 0
        if 'confidence' not in df_signaled.columns: df_signaled['confidence'] = 0
        
        if 'rsi' not in df_signaled.columns: # 如果没有RSI列，则不产生任何信号
            print(f"({self.name}) 'rsi' 列未在 df_with_indicators 中找到。")
            return df_signaled

        # print(f"({self.name}) 兼容模式 generate_signals, 行数: {len(df_signaled)}")
        for i in range(self.params['rsi_period'], len(df_signaled)): # Start from where RSI is likely valid
            curr_rsi = df_signaled.iloc[i]['rsi']
            prev_rsi = df_signaled.iloc[i-1]['rsi']
            signal = 0
            confidence_score = 0

            # 添加详细日志输出
            # print(f"({self.name}) 检查信号: 索引={i}, 当前RSI={curr_rsi:.2f}, 前一RSI={prev_rsi:.2f}, 超买={self.params['rsi_overbought']}, 超卖={self.params['rsi_oversold']}")

            if pd.isna(curr_rsi) or pd.isna(prev_rsi):
                print(f"({self.name}) 索引={i}: RSI为NaN，跳过信号检查。")
                continue
            
            if curr_rsi < self.params['rsi_oversold'] and prev_rsi >= self.params['rsi_oversold']:
                signal = 1
                rsi_distance = self.params['rsi_oversold'] - curr_rsi
                confidence_score = 50 + min(50, rsi_distance * 3)
                # print(f"({self.name}) 索引={i}: 触发上涨信号! RSI={curr_rsi:.2f}, 置信度={confidence_score:.2f}")
            elif curr_rsi > self.params['rsi_overbought'] and prev_rsi <= self.params['rsi_overbought']:
                signal = -1
                rsi_distance = curr_rsi - self.params['rsi_overbought']
                confidence_score = 50 + min(50, rsi_distance * 3)
                # print(f"({self.name}) 索引={i}: 触发下跌信号! RSI={curr_rsi:.2f}, 置信度={confidence_score:.2f}")
            # else:
                # print(f"({self.name}) 索引={i}: 未触发信号。")
            
            df_signaled.iloc[i, df_signaled.columns.get_loc('signal')] = signal
            df_signaled.iloc[i, df_signaled.columns.get_loc('confidence')] = int(max(0, min(100, confidence_score)))
        return df_signaled

# --- MACDStrategy ---
class MACDStrategy(Strategy):
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
        if params: default_params.update(params)
        super().__init__(default_params)
        self.name = "MACD策略"
        self.min_history_periods = self.params['slow_period'] + self.params['signal_period'] # 粗略估计
        # print(f"初始化 {self.name}, 参数: {self.params}")

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().calculate_all_indicators(df)
        # print(f"({self.name}) 全局计算MACD指标, 行数: {len(df)}")
        fast = df['close'].ewm(span=self.params['fast_period'], adjust=False).mean()
        slow = df['close'].ewm(span=self.params['slow_period'], adjust=False).mean()
        df['macd'] = fast - slow
        df['macd_signal_line'] = df['macd'].ewm(span=self.params['signal_period'], adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal_line']
        return df

    def generate_signals_from_indicators_on_window(self, df_window_with_indicators: pd.DataFrame) -> pd.DataFrame:
        df_out = df_window_with_indicators.copy()
        if 'signal' not in df_out.columns: df_out['signal'] = 0
        if 'confidence' not in df_out.columns: df_out['confidence'] = 0

        if len(df_out) < 2 or not all(col in df_out.columns for col in ['macd', 'macd_signal_line', 'close']):
            return df_out

        curr_idx = -1
        prev_idx = -2

        if pd.isna(df_out.iloc[curr_idx]['macd']) or pd.isna(df_out.iloc[curr_idx]['macd_signal_line']) or \
           pd.isna(df_out.iloc[prev_idx]['macd']) or pd.isna(df_out.iloc[prev_idx]['macd_signal_line']):
            return df_out
            
        curr_macd = df_out.iloc[curr_idx]['macd']
        curr_sig = df_out.iloc[curr_idx]['macd_signal_line']
        prev_macd = df_out.iloc[prev_idx]['macd']
        prev_sig = df_out.iloc[prev_idx]['macd_signal_line']
        curr_close = df_out.iloc[curr_idx]['close']

        signal = 0
        confidence_score = 0
        price_scale = curr_close * 0.00005 
        if price_scale == 0: price_scale = 0.00001 # Avoid division by zero for price_scale

        if curr_macd > curr_sig and prev_macd <= prev_sig: # Golden cross
            signal = 1
            macd_diff = curr_macd - curr_sig
            confidence_score = min(100, 50 + abs(macd_diff / price_scale) * 25)
        elif curr_macd < curr_sig and prev_macd >= prev_sig: # Death cross
            signal = -1
            macd_diff = curr_sig - curr_macd
            confidence_score = min(100, 50 + abs(macd_diff / price_scale) * 25)
        
        df_out.iloc[curr_idx, df_out.columns.get_loc('signal')] = signal
        df_out.iloc[curr_idx, df_out.columns.get_loc('confidence')] = int(max(0, min(100, confidence_score)))
        return df_out

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df_with_indicators = self.calculate_all_indicators(df.copy())
        df_signaled = df_with_indicators.copy()
        if 'signal' not in df_signaled.columns: df_signaled['signal'] = 0
        if 'confidence' not in df_signaled.columns: df_signaled['confidence'] = 0

        if not all(col in df_signaled.columns for col in ['macd', 'macd_signal_line', 'close']):
            print(f"({self.name}) MACD相关列未找到。")
            return df_signaled
            
        # print(f"({self.name}) 兼容模式 generate_signals, 行数: {len(df_signaled)}")
        for i in range(1, len(df_signaled)):
            if pd.isna(df_signaled.iloc[i]['macd']) or pd.isna(df_signaled.iloc[i]['macd_signal_line']) or \
               pd.isna(df_signaled.iloc[i-1]['macd']) or pd.isna(df_signaled.iloc[i-1]['macd_signal_line']):
                continue
            
            curr_macd = df_signaled.iloc[i]['macd']
            curr_sig = df_signaled.iloc[i]['macd_signal_line']
            prev_macd = df_signaled.iloc[i-1]['macd']
            prev_sig = df_signaled.iloc[i-1]['macd_signal_line']
            curr_close = df_signaled.iloc[i]['close']
            signal = 0
            confidence_score = 0
            price_scale = curr_close * 0.00005 
            if price_scale == 0: price_scale = 0.00001

            if curr_macd > curr_sig and prev_macd <= prev_sig:
                signal = 1
                macd_diff = curr_macd - curr_sig
                confidence_score = min(100, 50 + abs(macd_diff / price_scale) * 25)
            elif curr_macd < curr_sig and prev_macd >= prev_sig:
                signal = -1
                macd_diff = curr_sig - curr_macd
                confidence_score = min(100, 50 + abs(macd_diff / price_scale) * 25)
            
            df_signaled.iloc[i, df_signaled.columns.get_loc('signal')] = signal
            df_signaled.iloc[i, df_signaled.columns.get_loc('confidence')] = int(max(0, min(100, confidence_score)))
        return df_signaled

# --- RSISMA_Strategy ---
class RSISMA_Strategy(Strategy):
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {'rsi_period': 14, 'rsi_overbought': 70, 'rsi_oversold': 30, 'sma_period': 30}
        if params: default_params.update(params)
        super().__init__(default_params)
        self.name = "RSI+SMA策略"
        self.min_history_periods = max(self.params['rsi_period'] + 1, self.params['sma_period'])
        # print(f"初始化 {self.name}, 参数: {self.params}")

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().calculate_all_indicators(df)
        # print(f"({self.name}) 全局计算RSI和SMA指标, 行数: {len(df)}")
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(com=self.params['rsi_period'] - 1, min_periods=self.params['rsi_period']).mean()
        avg_loss = loss.ewm(com=self.params['rsi_period'] - 1, min_periods=self.params['rsi_period']).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'].fillna(50, inplace=True)
        # SMA
        df['sma'] = df['close'].rolling(window=self.params['sma_period']).mean()
        return df

    def generate_signals_from_indicators_on_window(self, df_window_with_indicators: pd.DataFrame) -> pd.DataFrame:
        df_out = df_window_with_indicators.copy()
        if 'signal' not in df_out.columns: df_out['signal'] = 0
        if 'confidence' not in df_out.columns: df_out['confidence'] = 0

        if len(df_out) < 2 or not all(col in df_out.columns for col in ['rsi', 'sma', 'close']):
            return df_out
            
        curr_idx = -1
        prev_idx = -2

        if pd.isna(df_out.iloc[curr_idx]['rsi']) or pd.isna(df_out.iloc[curr_idx]['sma']) or \
           pd.isna(df_out.iloc[prev_idx]['rsi']): # SMA on prev_idx not strictly needed for this logic
            return df_out

        curr_rsi = df_out.iloc[curr_idx]['rsi']
        prev_rsi = df_out.iloc[prev_idx]['rsi']
        curr_close = df_out.iloc[curr_idx]['close']
        curr_sma = df_out.iloc[curr_idx]['sma']
        signal = 0
        confidence_score = 0
        
        price_div = curr_close if curr_close != 0 else 0.0001

        if curr_rsi < self.params['rsi_oversold'] and prev_rsi >= self.params['rsi_oversold'] and curr_close < curr_sma:
            signal = 1
            rsi_conf = min(50, (self.params['rsi_oversold'] - curr_rsi) * 2)
            sma_conf = min(50, ((curr_sma - curr_close) / price_div * 100) * 5)
            confidence_score = min(100, 40 + rsi_conf * 0.6 + sma_conf * 0.4)
        elif curr_rsi > self.params['rsi_overbought'] and prev_rsi <= self.params['rsi_overbought'] and curr_close > curr_sma:
            signal = -1
            rsi_conf = min(50, (curr_rsi - self.params['rsi_overbought']) * 2)
            sma_conf = min(50, ((curr_close - curr_sma) / price_div * 100) * 5)
            confidence_score = min(100, 40 + rsi_conf * 0.6 + sma_conf * 0.4)
            
        df_out.iloc[curr_idx, df_out.columns.get_loc('signal')] = signal
        df_out.iloc[curr_idx, df_out.columns.get_loc('confidence')] = int(max(0, min(100, confidence_score)))
        return df_out

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df_with_indicators = self.calculate_all_indicators(df.copy())
        df_signaled = df_with_indicators.copy()
        if 'signal' not in df_signaled.columns: df_signaled['signal'] = 0
        if 'confidence' not in df_signaled.columns: df_signaled['confidence'] = 0
        
        if not all(col in df_signaled.columns for col in ['rsi', 'sma', 'close']):
            print(f"({self.name}) RSI/SMA相关列未找到。")
            return df_signaled

        # print(f"({self.name}) 兼容模式 generate_signals, 行数: {len(df_signaled)}")
        for i in range(1, len(df_signaled)):
            if pd.isna(df_signaled.iloc[i]['rsi']) or pd.isna(df_signaled.iloc[i]['sma']) or \
               pd.isna(df_signaled.iloc[i-1]['rsi']):
                continue

            curr_rsi = df_signaled.iloc[i]['rsi']
            prev_rsi = df_signaled.iloc[i-1]['rsi']
            curr_close = df_signaled.iloc[i]['close']
            curr_sma = df_signaled.iloc[i]['sma']
            signal = 0
            confidence_score = 0
            price_div = curr_close if curr_close != 0 else 0.0001

            if curr_rsi < self.params['rsi_oversold'] and prev_rsi >= self.params['rsi_oversold'] and curr_close < curr_sma:
                signal = 1
                rsi_conf = min(50, (self.params['rsi_oversold'] - curr_rsi) * 2)
                sma_conf = min(50, ((curr_sma - curr_close) / price_div * 100) * 5)
                confidence_score = min(100, 40 + rsi_conf * 0.6 + sma_conf * 0.4)
            elif curr_rsi > self.params['rsi_overbought'] and prev_rsi <= self.params['rsi_overbought'] and curr_close > curr_sma:
                signal = -1
                rsi_conf = min(50, (curr_rsi - self.params['rsi_overbought']) * 2)
                sma_conf = min(50, ((curr_close - curr_sma) / price_div * 100) * 5)
                confidence_score = min(100, 40 + rsi_conf * 0.6 + sma_conf * 0.4)
            
            df_signaled.iloc[i, df_signaled.columns.get_loc('signal')] = signal
            df_signaled.iloc[i, df_signaled.columns.get_loc('confidence')] = int(max(0, min(100, confidence_score)))
        return df_signaled


# --- MovingAverageCrossStrategy ---
class MovingAverageCrossStrategy(Strategy):
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {'short_period': 9, 'long_period': 21, 'ma_type': 'EMA'}
        if params: default_params.update(params)
        super().__init__(default_params)
        self.name = f"{self.params['ma_type']}均线交叉策略"
        self.min_history_periods = self.params['long_period']
        # print(f"初始化 {self.name}, 参数: {self.params}")

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().calculate_all_indicators(df)
        # print(f"({self.name}) 全局计算均线指标, 行数: {len(df)}")
        if self.params['ma_type'] == 'SMA':
            df['short_ma'] = df['close'].rolling(window=self.params['short_period']).mean()
            df['long_ma'] = df['close'].rolling(window=self.params['long_period']).mean()
        elif self.params['ma_type'] == 'EMA':
            df['short_ma'] = df['close'].ewm(span=self.params['short_period'], adjust=False).mean()
            df['long_ma'] = df['close'].ewm(span=self.params['long_period'], adjust=False).mean()
        else:
            raise ValueError("ma_type 参数必须是 'SMA' 或 'EMA'")
        return df

    def generate_signals_from_indicators_on_window(self, df_window_with_indicators: pd.DataFrame) -> pd.DataFrame:
        df_out = df_window_with_indicators.copy()
        if 'signal' not in df_out.columns: df_out['signal'] = 0
        if 'confidence' not in df_out.columns: df_out['confidence'] = 0

        if len(df_out) < 2 or not all(col in df_out.columns for col in ['short_ma', 'long_ma']):
            return df_out
            
        curr_idx = -1
        prev_idx = -2

        if pd.isna(df_out.iloc[curr_idx]['short_ma']) or pd.isna(df_out.iloc[curr_idx]['long_ma']) or \
           pd.isna(df_out.iloc[prev_idx]['short_ma']) or pd.isna(df_out.iloc[prev_idx]['long_ma']):
            return df_out

        curr_short = df_out.iloc[curr_idx]['short_ma']
        curr_long = df_out.iloc[curr_idx]['long_ma']
        prev_short = df_out.iloc[prev_idx]['short_ma']
        prev_long = df_out.iloc[prev_idx]['long_ma']
        signal = 0
        confidence_score = 0
        long_div = curr_long if curr_long != 0 else 0.0001

        if curr_short > curr_long and prev_short <= prev_long: # Golden cross
            signal = 1
            confidence_score = 60 + min(40, ((curr_short - curr_long) / long_div * 100) * 2)
        elif curr_short < curr_long and prev_short >= prev_long: # Death cross
            signal = -1
            confidence_score = 60 + min(40, ((curr_long - curr_short) / long_div * 100) * 2)
            
        df_out.iloc[curr_idx, df_out.columns.get_loc('signal')] = signal
        df_out.iloc[curr_idx, df_out.columns.get_loc('confidence')] = int(max(0, min(100, confidence_score)))
        return df_out

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df_with_indicators = self.calculate_all_indicators(df.copy())
        df_signaled = df_with_indicators.copy()
        if 'signal' not in df_signaled.columns: df_signaled['signal'] = 0
        if 'confidence' not in df_signaled.columns: df_signaled['confidence'] = 0

        if not all(col in df_signaled.columns for col in ['short_ma', 'long_ma']):
            print(f"({self.name}) 均线列未找到。")
            return df_signaled
            
        # print(f"({self.name}) 兼容模式 generate_signals, 行数: {len(df_signaled)}")
        for i in range(1, len(df_signaled)):
            if pd.isna(df_signaled.iloc[i]['short_ma']) or pd.isna(df_signaled.iloc[i]['long_ma']) or \
               pd.isna(df_signaled.iloc[i-1]['short_ma']) or pd.isna(df_signaled.iloc[i-1]['long_ma']):
                continue

            curr_short = df_signaled.iloc[i]['short_ma']
            curr_long = df_signaled.iloc[i]['long_ma']
            prev_short = df_signaled.iloc[i-1]['short_ma']
            prev_long = df_signaled.iloc[i-1]['long_ma']
            signal = 0
            confidence_score = 0
            long_div = curr_long if curr_long != 0 else 0.0001

            if curr_short > curr_long and prev_short <= prev_long:
                signal = 1
                confidence_score = 60 + min(40, ((curr_short - curr_long) / long_div * 100) * 2)
            elif curr_short < curr_long and prev_short >= prev_long:
                signal = -1
                confidence_score = 60 + min(40, ((curr_long - curr_short) / long_div * 100) * 2)
            
            df_signaled.iloc[i, df_signaled.columns.get_loc('signal')] = signal
            df_signaled.iloc[i, df_signaled.columns.get_loc('confidence')] = int(max(0, min(100, confidence_score)))
        return df_signaled

# --- BollingerBandsReversalStrategy ---
class BollingerBandsReversalStrategy(Strategy):
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {'bb_period': 20, 'bb_std_dev': 2.0, 'reversal_confirmation_candles': 1}
        if params: default_params.update(params)
        super().__init__(default_params)
        self.name = "布林带均值回归策略"
        self.min_history_periods = self.params['bb_period'] + self.params['reversal_confirmation_candles']
        # print(f"初始化 {self.name}, 参数: {self.params}")

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().calculate_all_indicators(df)
        # print(f"({self.name}) 全局计算布林带指标, 行数: {len(df)}")
        df['bb_middle'] = df['close'].rolling(window=self.params['bb_period']).mean()
        df['bb_std'] = df['close'].rolling(window=self.params['bb_period']).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * self.params['bb_std_dev'])
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * self.params['bb_std_dev'])
        return df

    def generate_signals_from_indicators_on_window(self, df_window_with_indicators: pd.DataFrame) -> pd.DataFrame:
        df_out = df_window_with_indicators.copy()
        if 'signal' not in df_out.columns: df_out['signal'] = 0
        if 'confidence' not in df_out.columns: df_out['confidence'] = 0

        conf_candles = self.params['reversal_confirmation_candles']
        # 窗口需要至少 conf_candles + 1 (当前K线) + 1 (前一根K线用于比较) 的长度
        if len(df_out) < conf_candles + 2 or not all(col in df_out.columns for col in ['bb_lower', 'bb_upper', 'close', 'bb_std']):
            return df_out

        # 判断窗口的最后一行 (curr)
        # prev_touch_idx 是当前K线之前 conf_candles 的那根K线
        # prev_confirm_idx 是当前K线的前一根K线
        curr_idx = -1
        prev_confirm_idx = -2 
        prev_touch_idx = -(conf_candles + 1) # Index for the candle that might have touched/crossed the band

        # Ensure indices are valid
        if abs(prev_touch_idx) > len(df_out) -1 : return df_out


        if df_out.iloc[curr_idx][['bb_lower', 'bb_upper', 'close', 'bb_std']].isna().any() or \
           df_out.iloc[prev_confirm_idx][['close', 'bb_lower', 'bb_upper']].isna().any() or \
           df_out.iloc[prev_touch_idx][['close', 'bb_lower', 'bb_upper']].isna().any():
            return df_out
            
        curr_close = df_out.iloc[curr_idx]['close']
        curr_bb_lower = df_out.iloc[curr_idx]['bb_lower']
        curr_bb_upper = df_out.iloc[curr_idx]['bb_upper']
        curr_bb_std = df_out.iloc[curr_idx]['bb_std']

        prev_confirm_close = df_out.iloc[prev_confirm_idx]['close']
        prev_confirm_bb_lower = df_out.iloc[prev_confirm_idx]['bb_lower'] # For strict confirmation
        prev_confirm_bb_upper = df_out.iloc[prev_confirm_idx]['bb_upper'] # For strict confirmation

        prev_touch_close = df_out.iloc[prev_touch_idx]['close']
        prev_touch_bb_lower = df_out.iloc[prev_touch_idx]['bb_lower']
        prev_touch_bb_upper = df_out.iloc[prev_touch_idx]['bb_upper']
        
        signal = 0
        confidence_score = 0
        std_div = curr_bb_std if curr_bb_std != 0 else 0.0001

        # Bullish: Previous candle(s) touched/below lower, current candle closes above lower
        if prev_touch_close < prev_touch_bb_lower and \
           prev_confirm_close <= prev_confirm_bb_lower and \
           curr_close > curr_bb_lower:
            signal = 1
            distance_from_band_std = (curr_close - curr_bb_lower) / std_div
            confidence_score = 50 + min(50, distance_from_band_std * 15)
            
        # Bearish: Previous candle(s) touched/above upper, current candle closes below upper
        elif prev_touch_close > prev_touch_bb_upper and \
             prev_confirm_close >= prev_confirm_bb_upper and \
             curr_close < curr_bb_upper:
            signal = -1
            distance_from_band_std = (curr_bb_upper - curr_close) / std_div
            confidence_score = 50 + min(50, distance_from_band_std * 15)
            
        df_out.iloc[curr_idx, df_out.columns.get_loc('signal')] = signal
        df_out.iloc[curr_idx, df_out.columns.get_loc('confidence')] = int(max(0, min(100, confidence_score)))
        return df_out

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df_with_indicators = self.calculate_all_indicators(df.copy())
        df_signaled = df_with_indicators.copy()
        if 'signal' not in df_signaled.columns: df_signaled['signal'] = 0
        if 'confidence' not in df_signaled.columns: df_signaled['confidence'] = 0
        
        if not all(col in df_signaled.columns for col in ['bb_lower', 'bb_upper', 'close', 'bb_std']):
            print(f"({self.name}) 布林带相关列未找到。")
            return df_signaled

        # print(f"({self.name}) 兼容模式 generate_signals, 行数: {len(df_signaled)}")
        conf_candles = self.params['reversal_confirmation_candles']
        for i in range(conf_candles + 1, len(df_signaled)): # Need at least conf_candles history + current
            # Check NaNs for current (i), previous confirmation (i-1), and touch candle (i - conf_candles)
            if df_signaled.iloc[i][['bb_lower', 'bb_upper', 'close', 'bb_std']].isna().any() or \
               df_signaled.iloc[i-1][['close', 'bb_lower', 'bb_upper']].isna().any() or \
               df_signaled.iloc[i - conf_candles][['close', 'bb_lower', 'bb_upper']].isna().any():
                continue

            curr_close = df_signaled.iloc[i]['close']
            curr_bb_lower = df_signaled.iloc[i]['bb_lower']
            curr_bb_upper = df_signaled.iloc[i]['bb_upper']
            curr_bb_std = df_signaled.iloc[i]['bb_std']

            prev_confirm_close = df_signaled.iloc[i-1]['close'] # The candle right before current
            prev_confirm_bb_lower = df_signaled.iloc[i-1]['bb_lower']
            prev_confirm_bb_upper = df_signaled.iloc[i-1]['bb_upper']

            # The candle that should have touched or crossed the band
            prev_touch_candle_idx = i - conf_candles 
            prev_touch_close = df_signaled.iloc[prev_touch_candle_idx]['close']
            prev_touch_bb_lower = df_signaled.iloc[prev_touch_candle_idx]['bb_lower']
            prev_touch_bb_upper = df_signaled.iloc[prev_touch_candle_idx]['bb_upper']

            signal = 0
            confidence_score = 0
            std_div = curr_bb_std if curr_bb_std != 0 else 0.0001

            # Bullish signal logic:
            # 1. `prev_touch_candle` was below its lower band.
            # 2. The candle right before current (`prev_confirm_close`) was still at or below its lower band (confirming pressure).
            # 3. Current candle (`curr_close`) closes back above its lower band.
            if prev_touch_close < prev_touch_bb_lower and \
               prev_confirm_close <= prev_confirm_bb_lower and \
               curr_close > curr_bb_lower:
                signal = 1
                distance_from_band_std = (curr_close - curr_bb_lower) / std_div
                confidence_score = 50 + min(50, distance_from_band_std * 15)

            # Bearish signal logic: (Symmetrical)
            elif prev_touch_close > prev_touch_bb_upper and \
                 prev_confirm_close >= prev_confirm_bb_upper and \
                 curr_close < curr_bb_upper:
                signal = -1
                distance_from_band_std = (curr_bb_upper - curr_close) / std_div
                confidence_score = 50 + min(50, distance_from_band_std * 15)
            
            df_signaled.iloc[i, df_signaled.columns.get_loc('signal')] = signal
            df_signaled.iloc[i, df_signaled.columns.get_loc('confidence')] = int(max(0, min(100, confidence_score)))
        return df_signaled


# --- PriceBreakoutStrategy ---
class PriceBreakoutStrategy(Strategy):
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {'lookback_period': 20, 'volume_confirmation_factor': 1.2}
        if params: default_params.update(params)
        super().__init__(default_params)
        self.name = "价格突破策略"
        self.min_history_periods = self.params['lookback_period'] + 1
        # print(f"初始化 {self.name}, 参数: {self.params}")

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().calculate_all_indicators(df)
        # print(f"({self.name}) 全局计算突破高低点和成交量均值, 行数: {len(df)}")
        lookback = self.params["lookback_period"]
        df[f'high_{lookback}'] = df['high'].rolling(window=lookback, min_periods=1).max().shift(1)
        df[f'low_{lookback}'] = df['low'].rolling(window=lookback, min_periods=1).min().shift(1)
        if 'volume' in df.columns:
            df['volume_avg_lookback'] = df['volume'].rolling(window=lookback, min_periods=1).mean().shift(1)
        return df

    def generate_signals_from_indicators_on_window(self, df_window_with_indicators: pd.DataFrame) -> pd.DataFrame:
        df_out = df_window_with_indicators.copy()
        if 'signal' not in df_out.columns: df_out['signal'] = 0
        if 'confidence' not in df_out.columns: df_out['confidence'] = 0

        lookback = self.params["lookback_period"]
        high_col = f'high_{lookback}'
        low_col = f'low_{lookback}'
        vol_avg_col = 'volume_avg_lookback'
        
        # Window needs at least 1 row for current, and indicators depend on lookback_period
        # but indicators are pre-calculated. So, just need the columns.
        required_cols = [high_col, low_col, 'close']
        if 'volume' in df_out.columns: required_cols.extend(['volume', vol_avg_col])

        if not all(col in df_out.columns for col in required_cols):
            return df_out # Missing necessary indicator columns
            
        curr_idx = -1
        
        # Check NaNs for the current (last) row in the window
        if df_out.iloc[curr_idx][required_cols].isna().any():
            return df_out

        curr_close = df_out.iloc[curr_idx]['close']
        curr_high_lookback = df_out.iloc[curr_idx][high_col]
        curr_low_lookback = df_out.iloc[curr_idx][low_col]
        
        signal = 0
        confidence_score = 0
        volume_confirmed = True # Assume true if volume not used/available
        
        if 'volume' in df_out.columns and vol_avg_col in df_out.columns:
            curr_volume = df_out.iloc[curr_idx]['volume']
            curr_volume_avg = df_out.iloc[curr_idx][vol_avg_col]
            if pd.notna(curr_volume) and pd.notna(curr_volume_avg) and curr_volume_avg > 0:
                 volume_confirmed = curr_volume > curr_volume_avg * self.params['volume_confirmation_factor']
            else:
                 volume_confirmed = False # If volume data is problematic

        if curr_close > curr_high_lookback and volume_confirmed:
            signal = 1
            breakout_strength = (curr_close - curr_high_lookback) / (curr_high_lookback if curr_high_lookback !=0 else 0.0001) * 100
            confidence_score = 60 + min(40, breakout_strength * 5)
        elif curr_close < curr_low_lookback and volume_confirmed:
            signal = -1
            breakout_strength = (curr_low_lookback - curr_close) / (curr_low_lookback if curr_low_lookback !=0 else 0.0001) * 100
            confidence_score = 60 + min(40, breakout_strength * 5)
            
        df_out.iloc[curr_idx, df_out.columns.get_loc('signal')] = signal
        df_out.iloc[curr_idx, df_out.columns.get_loc('confidence')] = int(max(0, min(100, confidence_score)))
        return df_out

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df_with_indicators = self.calculate_all_indicators(df.copy())
        df_signaled = df_with_indicators.copy()
        if 'signal' not in df_signaled.columns: df_signaled['signal'] = 0
        if 'confidence' not in df_signaled.columns: df_signaled['confidence'] = 0

        lookback = self.params["lookback_period"]
        high_col = f'high_{lookback}'
        low_col = f'low_{lookback}'
        vol_avg_col = 'volume_avg_lookback'
        required_cols = [high_col, low_col, 'close']
        if 'volume' in df_signaled.columns: required_cols.extend(['volume', vol_avg_col])

        if not all(col in df_signaled.columns for col in required_cols):
            print(f"({self.name}) 突破所需列未找到。")
            return df_signaled

        # print(f"({self.name}) 兼容模式 generate_signals, 行数: {len(df_signaled)}")
        # Start from lookback_period because .shift(1) in indicator calculation means first few rows of indicators are NaN
        for i in range(lookback, len(df_signaled)): 
            if df_signaled.iloc[i][required_cols].isna().any():
                continue
            
            curr_close = df_signaled.iloc[i]['close']
            curr_high_lookback = df_signaled.iloc[i][high_col]
            curr_low_lookback = df_signaled.iloc[i][low_col]
            signal = 0
            confidence_score = 0
            volume_confirmed = True

            if 'volume' in df_signaled.columns and vol_avg_col in df_signaled.columns:
                curr_volume = df_signaled.iloc[i]['volume']
                curr_volume_avg = df_signaled.iloc[i][vol_avg_col]
                if pd.notna(curr_volume) and pd.notna(curr_volume_avg) and curr_volume_avg > 0:
                    volume_confirmed = curr_volume > curr_volume_avg * self.params['volume_confirmation_factor']
                else:
                    volume_confirmed = False

            if curr_close > curr_high_lookback and volume_confirmed:
                signal = 1
                breakout_strength = (curr_close - curr_high_lookback) / (curr_high_lookback if curr_high_lookback !=0 else 0.0001) * 100
                confidence_score = 60 + min(40, breakout_strength * 5)
            elif curr_close < curr_low_lookback and volume_confirmed:
                signal = -1
                breakout_strength = (curr_low_lookback - curr_close) / (curr_low_lookback if curr_low_lookback !=0 else 0.0001) * 100
                confidence_score = 60 + min(40, breakout_strength * 5)
            
            df_signaled.iloc[i, df_signaled.columns.get_loc('signal')] = signal
            df_signaled.iloc[i, df_signaled.columns.get_loc('confidence')] = int(max(0, min(100, confidence_score)))
        return df_signaled


# --- EnhancedCandlestickPatternStrategy ---
class EnhancedCandlestickPatternStrategy(Strategy):
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'engulfing_min_body_ratio': 0.6, 'engulfing_factor': 1.0,
            'pinbar_min_wick_body_ratio': 2.0, 'pinbar_max_body_range_ratio': 0.33,
        }
        if params: default_params.update(params)
        super().__init__(default_params)
        self.name = "增强K线形态策略"
        self.min_history_periods = 2 # Needs current and previous candle
        # print(f"初始化 {self.name}, 参数: {self.params}")

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # This strategy does not calculate storable indicators like RSI/SMA.
        # It directly uses O,H,L,C values.
        # print(f"({self.name}) 无需预计算存储指标。")
        return super().calculate_all_indicators(df) # Just for data type check

    def _is_bullish_pinbar(self, o, h, l, c):
        body = abs(c - o); range_val = h - l
        if range_val == 0: return False
        lower_wick = min(o, c) - l; upper_wick = h - max(o, c)
        return lower_wick >= body * self.params['pinbar_min_wick_body_ratio'] and \
               body <= range_val * self.params['pinbar_max_body_range_ratio'] and \
               upper_wick < body * 0.5 # Upper wick should be small

    def _is_bearish_pinbar(self, o, h, l, c):
        body = abs(c - o); range_val = h - l
        if range_val == 0: return False
        lower_wick = min(o, c) - l; upper_wick = h - max(o, c)
        return upper_wick >= body * self.params['pinbar_min_wick_body_ratio'] and \
               body <= range_val * self.params['pinbar_max_body_range_ratio'] and \
               lower_wick < body * 0.5 # Lower wick should be small

    def _check_candle_pattern(self, curr_o, curr_h, curr_l, curr_c, prev_o, prev_h, prev_l, prev_c):
        signal = 0
        confidence_score = 0

        curr_body = abs(curr_c - curr_o)
        prev_body = abs(prev_c - prev_o)
        curr_range = curr_h - curr_l
        if curr_range == 0: curr_range = 0.00001 # Avoid division by zero

        # Bullish Engulfing
        if prev_c < prev_o and curr_c > curr_o and \
           curr_o < prev_c and curr_c > prev_o and \
           (curr_body / curr_range) >= self.params['engulfing_min_body_ratio'] and \
           curr_body > prev_body * self.params['engulfing_factor']:
            signal = 1
            confidence_score = 70 + min(30, (curr_body / (prev_body if prev_body > 0 else 0.0001) - self.params['engulfing_factor']) * 15)
        
        # Bearish Engulfing
        elif prev_c > prev_o and curr_c < curr_o and \
             curr_o > prev_c and curr_c < prev_o and \
             (curr_body / curr_range) >= self.params['engulfing_min_body_ratio'] and \
             curr_body > prev_body * self.params['engulfing_factor']:
            signal = -1
            confidence_score = 70 + min(30, (curr_body / (prev_body if prev_body > 0 else 0.0001) - self.params['engulfing_factor']) * 15)

        # Bullish Pinbar (check only if no engulfing signal)
        elif signal == 0 and self._is_bullish_pinbar(curr_o, curr_h, curr_l, curr_c):
            signal = 1
            confidence_score = 65
            lower_wick = min(curr_o, curr_c) - curr_l
            confidence_score += min(35, (lower_wick / (curr_body if curr_body > 0 else 0.0001) - self.params['pinbar_min_wick_body_ratio']) * 10 )
        
        # Bearish Pinbar (check only if no engulfing or bullish pinbar signal)
        elif signal == 0 and self._is_bearish_pinbar(curr_o, curr_h, curr_l, curr_c):
            signal = -1
            confidence_score = 65
            upper_wick = curr_h - max(curr_o, curr_c)
            confidence_score += min(35, (upper_wick / (curr_body if curr_body > 0 else 0.0001) - self.params['pinbar_min_wick_body_ratio']) * 10 )
        
        return signal, int(max(0, min(100, confidence_score)))

    def generate_signals_from_indicators_on_window(self, df_window_with_indicators: pd.DataFrame) -> pd.DataFrame:
        df_out = df_window_with_indicators.copy()
        if 'signal' not in df_out.columns: df_out['signal'] = 0
        if 'confidence' not in df_out.columns: df_out['confidence'] = 0

        if len(df_out) < 2 or not all(col in df_out.columns for col in ['open', 'high', 'low', 'close']):
            return df_out
            
        curr_idx = -1
        prev_idx = -2
        
        ohlc_cols = ['open', 'high', 'low', 'close']
        if df_out.iloc[curr_idx][ohlc_cols].isna().any() or df_out.iloc[prev_idx][ohlc_cols].isna().any():
            return df_out

        curr_o, curr_h, curr_l, curr_c = df_out.iloc[curr_idx][ohlc_cols]
        prev_o, prev_h, prev_l, prev_c = df_out.iloc[prev_idx][ohlc_cols]

        signal, confidence_score = self._check_candle_pattern(curr_o, curr_h, curr_l, curr_c, prev_o, prev_h, prev_l, prev_c)
            
        df_out.iloc[curr_idx, df_out.columns.get_loc('signal')] = signal
        df_out.iloc[curr_idx, df_out.columns.get_loc('confidence')] = confidence_score
        return df_out

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df_with_indicators = self.calculate_all_indicators(df.copy()) # Does nothing but check types
        df_signaled = df_with_indicators.copy()
        if 'signal' not in df_signaled.columns: df_signaled['signal'] = 0
        if 'confidence' not in df_signaled.columns: df_signaled['confidence'] = 0

        if not all(col in df_signaled.columns for col in ['open', 'high', 'low', 'close']):
            print(f"({self.name}) OHLC列未找到。")
            return df_signaled

        # print(f"({self.name}) 兼容模式 generate_signals, 行数: {len(df_signaled)}")
        for i in range(1, len(df_signaled)): 
            ohlc_cols = ['open', 'high', 'low', 'close']
            if df_signaled.iloc[i][ohlc_cols].isna().any() or df_signaled.iloc[i-1][ohlc_cols].isna().any():
                continue

            curr_o, curr_h, curr_l, curr_c = df_signaled.iloc[i][ohlc_cols]
            prev_o, prev_h, prev_l, prev_c = df_signaled.iloc[i-1][ohlc_cols]
            
            signal, confidence_score = self._check_candle_pattern(curr_o, curr_h, curr_l, curr_c, prev_o, prev_h, prev_l, prev_c)
            
            df_signaled.iloc[i, df_signaled.columns.get_loc('signal')] = signal
            df_signaled.iloc[i, df_signaled.columns.get_loc('confidence')] = confidence_score
        return df_signaled

# --- StochasticOscillatorStrategy ---
class StochasticOscillatorStrategy(Strategy):
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {'k_period': 14, 'd_period': 3, 'slowing': 3, 'overbought': 80, 'oversold': 20}
        if params: default_params.update(params)
        super().__init__(default_params)
        self.name = "随机指标策略"
        self.min_history_periods = self.params['k_period'] + self.params['slowing'] + self.params['d_period'] - 2 # Rough est.
        # print(f"初始化 {self.name}, 参数: {self.params}")

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().calculate_all_indicators(df)
        # print(f"({self.name}) 全局计算随机指标, 行数: {len(df)}")
        lowest_low = df['low'].rolling(window=self.params['k_period']).min()
        highest_high = df['high'].rolling(window=self.params['k_period']).max()
        df['fast_k'] = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low).replace(0, np.nan))
        df['fast_k'].fillna(50, inplace=True)
        df['slow_k'] = df['fast_k'].rolling(window=self.params['slowing']).mean()
        df['slow_d'] = df['slow_k'].rolling(window=self.params['d_period']).mean()
        return df

    def generate_signals_from_indicators_on_window(self, df_window_with_indicators: pd.DataFrame) -> pd.DataFrame:
        df_out = df_window_with_indicators.copy()
        if 'signal' not in df_out.columns: df_out['signal'] = 0
        if 'confidence' not in df_out.columns: df_out['confidence'] = 0

        if len(df_out) < 2 or not all(col in df_out.columns for col in ['slow_k', 'slow_d']):
            return df_out
            
        curr_idx = -1
        prev_idx = -2

        if pd.isna(df_out.iloc[curr_idx]['slow_k']) or pd.isna(df_out.iloc[curr_idx]['slow_d']) or \
           pd.isna(df_out.iloc[prev_idx]['slow_k']) or pd.isna(df_out.iloc[prev_idx]['slow_d']):
            return df_out

        curr_k, curr_d = df_out.iloc[curr_idx][['slow_k', 'slow_d']]
        prev_k, prev_d = df_out.iloc[prev_idx][['slow_k', 'slow_d']]
        signal = 0
        confidence_score = 0

        if prev_k < self.params['oversold'] and curr_k >= self.params['oversold'] and \
           prev_k <= prev_d and curr_k > curr_d: # Oversold exit & K crosses D upwards
            signal = 1
            confidence_score = 60 + min(40, (curr_k - self.params['oversold']) * 2 + (curr_k - curr_d) * 5)
        elif prev_k > self.params['overbought'] and curr_k <= self.params['overbought'] and \
             prev_k >= prev_d and curr_k < curr_d: # Overbought exit & K crosses D downwards
            signal = -1
            confidence_score = 60 + min(40, (self.params['overbought'] - curr_k) * 2 + (curr_d - curr_k) * 5)
            
        df_out.iloc[curr_idx, df_out.columns.get_loc('signal')] = signal
        df_out.iloc[curr_idx, df_out.columns.get_loc('confidence')] = int(max(0, min(100, confidence_score)))
        return df_out

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df_with_indicators = self.calculate_all_indicators(df.copy())
        df_signaled = df_with_indicators.copy()
        if 'signal' not in df_signaled.columns: df_signaled['signal'] = 0
        if 'confidence' not in df_signaled.columns: df_signaled['confidence'] = 0

        if not all(col in df_signaled.columns for col in ['slow_k', 'slow_d']):
            print(f"({self.name}) Stochastic列未找到。")
            return df_signaled

        # print(f"({self.name}) 兼容模式 generate_signals, 行数: {len(df_signaled)}")
        for i in range(1, len(df_signaled)):
            if pd.isna(df_signaled.iloc[i]['slow_k']) or pd.isna(df_signaled.iloc[i]['slow_d']) or \
               pd.isna(df_signaled.iloc[i-1]['slow_k']) or pd.isna(df_signaled.iloc[i-1]['slow_d']):
                continue

            curr_k, curr_d = df_signaled.iloc[i][['slow_k', 'slow_d']]
            prev_k, prev_d = df_signaled.iloc[i-1][['slow_k', 'slow_d']]
            signal = 0
            confidence_score = 0

            if prev_k < self.params['oversold'] and curr_k >= self.params['oversold'] and \
               prev_k <= prev_d and curr_k > curr_d:
                signal = 1
                confidence_score = 60 + min(40, (curr_k - self.params['oversold']) * 2 + (curr_k - curr_d) * 5)
            elif prev_k > self.params['overbought'] and curr_k <= self.params['overbought'] and \
                 prev_k >= prev_d and curr_k < curr_d:
                signal = -1
                confidence_score = 60 + min(40, (self.params['overbought'] - curr_k) * 2 + (curr_d - curr_k) * 5)
            
            df_signaled.iloc[i, df_signaled.columns.get_loc('signal')] = signal
            df_signaled.iloc[i, df_signaled.columns.get_loc('confidence')] = int(max(0, min(100, confidence_score)))
        return df_signaled

# --- MicroTrendMomentumStrategy ---
class MicroTrendMomentumStrategy(Strategy):
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'ema_period': 5, 'volume_sma_period': 5, 'min_volume_factor': 1.0,
            'rsi_filter_period': 14, 'rsi_extreme_ob': 85, 'rsi_extreme_os': 15,
        }
        if params: default_params.update(params)
        super().__init__(default_params)
        self.name = "微趋势动能策略"
        self.min_history_periods = max(self.params['ema_period'], self.params['volume_sma_period'], self.params['rsi_filter_period']+1)
        # print(f"初始化 {self.name}, 参数: {self.params}")

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().calculate_all_indicators(df)
        # print(f"({self.name}) 全局计算微趋势指标, 行数: {len(df)}")
        df['short_ema'] = df['close'].ewm(span=self.params['ema_period'], adjust=False).mean()
        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(window=self.params['volume_sma_period'], min_periods=1).mean()
        # RSI Filter
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0); loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(com=self.params['rsi_filter_period'] - 1, min_periods=self.params['rsi_filter_period']).mean()
        avg_loss = loss.ewm(com=self.params['rsi_filter_period'] - 1, min_periods=self.params['rsi_filter_period']).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['rsi_filter'] = 100 - (100 / (1 + rs))
        df['rsi_filter'].fillna(50, inplace=True)
        return df

    def generate_signals_from_indicators_on_window(self, df_window_with_indicators: pd.DataFrame) -> pd.DataFrame:
        df_out = df_window_with_indicators.copy()
        if 'signal' not in df_out.columns: df_out['signal'] = 0
        if 'confidence' not in df_out.columns: df_out['confidence'] = 0

        required_cols = ['short_ema', 'rsi_filter', 'close']
        if 'volume' in df_out.columns: required_cols.extend(['volume', 'volume_sma'])
        
        if len(df_out) < 2 or not all(col in df_out.columns for col in required_cols):
            return df_out
            
        curr_idx = -1
        prev_idx = -2

        # Check NaNs for current and previous row for EMA comparison, and current for others
        if pd.isna(df_out.iloc[curr_idx]['short_ema']) or pd.isna(df_out.iloc[prev_idx]['short_ema']) or \
           pd.isna(df_out.iloc[curr_idx]['rsi_filter']) or pd.isna(df_out.iloc[curr_idx]['close']):
            return df_out
        if 'volume' in df_out.columns and (pd.isna(df_out.iloc[curr_idx]['volume']) or pd.isna(df_out.iloc[curr_idx]['volume_sma'])):
            return df_out # If volume is used, it must be valid

        curr = df_out.iloc[curr_idx]
        prev = df_out.iloc[prev_idx]
        signal = 0
        confidence_score = 0

        ema_trending_up = curr['short_ema'] > prev['short_ema']
        ema_trending_down = curr['short_ema'] < prev['short_ema']
        volume_ok = True
        if 'volume' in df_out.columns:
            if pd.notna(curr['volume']) and pd.notna(curr['volume_sma']) and curr['volume_sma'] > 0:
                 volume_ok = curr['volume'] >= curr['volume_sma'] * self.params['min_volume_factor']
            else: volume_ok = False
        
        prev_ema_div = prev['short_ema'] if prev['short_ema'] !=0 else 0.0001
        curr_ema_div = curr['short_ema'] if curr['short_ema'] !=0 else 0.0001

        if curr['close'] > curr['short_ema'] and ema_trending_up and volume_ok and curr['rsi_filter'] < self.params['rsi_extreme_ob']:
            signal = 1
            ema_slope_conf = min(20, (curr['short_ema'] - prev['short_ema']) / prev_ema_div * 10000)
            price_dist_conf = min(20, (curr['close'] - curr['short_ema']) / curr_ema_div * 100 * 2)
            rsi_pos_conf = min(20, (self.params['rsi_extreme_ob'] - curr['rsi_filter']) / 2)
            confidence_score = 40 + ema_slope_conf + price_dist_conf + rsi_pos_conf
        elif curr['close'] < curr['short_ema'] and ema_trending_down and volume_ok and curr['rsi_filter'] > self.params['rsi_extreme_os']:
            signal = -1
            ema_slope_conf = min(20, (prev['short_ema'] - curr['short_ema']) / prev_ema_div * 10000)
            price_dist_conf = min(20, (curr['short_ema'] - curr['close']) / curr_ema_div * 100 * 2)
            rsi_pos_conf = min(20, (curr['rsi_filter'] - self.params['rsi_extreme_os']) / 2)
            confidence_score = 40 + ema_slope_conf + price_dist_conf + rsi_pos_conf
            
        df_out.iloc[curr_idx, df_out.columns.get_loc('signal')] = signal
        df_out.iloc[curr_idx, df_out.columns.get_loc('confidence')] = int(max(0, min(100, confidence_score)))
        return df_out

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df_with_indicators = self.calculate_all_indicators(df.copy())
        df_signaled = df_with_indicators.copy()
        if 'signal' not in df_signaled.columns: df_signaled['signal'] = 0
        if 'confidence' not in df_signaled.columns: df_signaled['confidence'] = 0

        required_cols = ['short_ema', 'rsi_filter', 'close']
        if 'volume' in df_signaled.columns: required_cols.extend(['volume', 'volume_sma'])

        if not all(col in df_signaled.columns for col in required_cols):
            print(f"({self.name}) 微趋势所需列未找到。")
            return df_signaled

        # print(f"({self.name}) 兼容模式 generate_signals, 行数: {len(df_signaled)}")
        for i in range(1, len(df_signaled)):
            if pd.isna(df_signaled.iloc[i]['short_ema']) or pd.isna(df_signaled.iloc[i-1]['short_ema']) or \
               pd.isna(df_signaled.iloc[i]['rsi_filter']) or pd.isna(df_signaled.iloc[i]['close']):
                continue
            if 'volume' in df_signaled.columns and (pd.isna(df_signaled.iloc[i]['volume']) or pd.isna(df_signaled.iloc[i]['volume_sma'])):
                continue

            curr = df_signaled.iloc[i]
            prev = df_signaled.iloc[i-1]
            signal = 0
            confidence_score = 0
            ema_trending_up = curr['short_ema'] > prev['short_ema']
            ema_trending_down = curr['short_ema'] < prev['short_ema']
            volume_ok = True
            if 'volume' in df_signaled.columns:
                if pd.notna(curr['volume']) and pd.notna(curr['volume_sma']) and curr['volume_sma'] > 0:
                     volume_ok = curr['volume'] >= curr['volume_sma'] * self.params['min_volume_factor']
                else: volume_ok = False
            
            prev_ema_div = prev['short_ema'] if prev['short_ema'] !=0 else 0.0001
            curr_ema_div = curr['short_ema'] if curr['short_ema'] !=0 else 0.0001

            if curr['close'] > curr['short_ema'] and ema_trending_up and volume_ok and curr['rsi_filter'] < self.params['rsi_extreme_ob']:
                signal = 1
                ema_slope_conf = min(20, (curr['short_ema'] - prev['short_ema']) / prev_ema_div * 10000)
                price_dist_conf = min(20, (curr['close'] - curr['short_ema']) / curr_ema_div * 100 * 2)
                rsi_pos_conf = min(20, (self.params['rsi_extreme_ob'] - curr['rsi_filter']) / 2)
                confidence_score = 40 + ema_slope_conf + price_dist_conf + rsi_pos_conf
            elif curr['close'] < curr['short_ema'] and ema_trending_down and volume_ok and curr['rsi_filter'] > self.params['rsi_extreme_os']:
                signal = -1
                ema_slope_conf = min(20, (prev['short_ema'] - curr['short_ema']) / prev_ema_div * 10000)
                price_dist_conf = min(20, (curr['short_ema'] - curr['close']) / curr_ema_div * 100 * 2)
                rsi_pos_conf = min(20, (curr['rsi_filter'] - self.params['rsi_extreme_os']) / 2)
                confidence_score = 40 + ema_slope_conf + price_dist_conf + rsi_pos_conf
                
            df_signaled.iloc[i, df_signaled.columns.get_loc('signal')] = signal
            df_signaled.iloc[i, df_signaled.columns.get_loc('confidence')] = int(max(0, min(100, confidence_score)))
        return df_signaled

# --- BBSqueezeBreakoutStrategy ---
class BBSqueezeBreakoutStrategy(Strategy):
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'bb_period': 20, 'bb_std_dev': 2.0, 'squeeze_threshold_pct': 0.5,
            'breakout_confirmation_candles': 1, 'min_breakout_body_pct': 0.3,
        }
        if params: default_params.update(params)
        super().__init__(default_params)
        self.name = "布林带挤压突破策略"
        self.min_history_periods = self.params['bb_period'] + self.params['breakout_confirmation_candles']
        # print(f"初始化 {self.name}, 参数: {self.params}")

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().calculate_all_indicators(df)
        # print(f"({self.name}) 全局计算BB挤压指标, 行数: {len(df)}")
        df['bb_middle'] = df['close'].rolling(window=self.params['bb_period']).mean()
        df['bb_std'] = df['close'].rolling(window=self.params['bb_period']).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * self.params['bb_std_dev'])
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * self.params['bb_std_dev'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'].replace(0, np.nan) * 100
        df['bb_width'].fillna(method='bfill', inplace=True) # Fill NaNs for bb_width
        return df

    def generate_signals_from_indicators_on_window(self, df_window_with_indicators: pd.DataFrame) -> pd.DataFrame:
        df_out = df_window_with_indicators.copy()
        if 'signal' not in df_out.columns: df_out['signal'] = 0
        if 'confidence' not in df_out.columns: df_out['confidence'] = 0

        confirm_candles = self.params['breakout_confirmation_candles']
        # Window needs current + confirm_candles history for squeeze check
        required_len = confirm_candles + 1 
        required_cols = ['bb_upper', 'bb_lower', 'bb_width', 'open', 'close', 'high', 'low']

        if len(df_out) < required_len or not all(col in df_out.columns for col in required_cols):
            return df_out
            
        curr_idx = -1 # Current breakout candle
        # Squeeze check is done on the candle `confirm_candles` before the current one
        squeeze_check_idx = -(confirm_candles + 1) 
        if abs(squeeze_check_idx) > len(df_out) -1 : return df_out # Not enough history for squeeze check


        # Check NaNs for current row and squeeze_check_row
        if df_out.iloc[curr_idx][required_cols].isna().any() or \
           df_out.iloc[squeeze_check_idx][['bb_upper', 'bb_lower', 'bb_width']].isna().any():
            return df_out

        curr_breakout_candle = df_out.iloc[curr_idx]
        squeeze_check_candle = df_out.iloc[squeeze_check_idx]
        
        is_squeezed = squeeze_check_candle['bb_width'] < self.params['squeeze_threshold_pct']
        signal = 0
        confidence_score = 0

        breakout_body = abs(curr_breakout_candle['close'] - curr_breakout_candle['open'])
        breakout_range = curr_breakout_candle['high'] - curr_breakout_candle['low']
        body_is_significant = (breakout_body / (breakout_range if breakout_range > 0 else 0.0001)) >= self.params['min_breakout_body_pct']
        
        bb_upper_div = squeeze_check_candle['bb_upper'] if squeeze_check_candle['bb_upper'] != 0 else 0.0001
        bb_lower_div = squeeze_check_candle['bb_lower'] if squeeze_check_candle['bb_lower'] != 0 else 0.0001


        if is_squeezed and body_is_significant:
            if curr_breakout_candle['close'] > squeeze_check_candle['bb_upper'] and curr_breakout_candle['close'] > curr_breakout_candle['open']:
                signal = 1
                squeeze_conf = min(30, (self.params['squeeze_threshold_pct'] - squeeze_check_candle['bb_width']) * 20)
                breakout_conf = min(40, ((curr_breakout_candle['close'] - squeeze_check_candle['bb_upper']) / bb_upper_div) * 100 * 5)
                confidence_score = 30 + squeeze_conf + breakout_conf
            elif curr_breakout_candle['close'] < squeeze_check_candle['bb_lower'] and curr_breakout_candle['close'] < curr_breakout_candle['open']:
                signal = -1
                squeeze_conf = min(30, (self.params['squeeze_threshold_pct'] - squeeze_check_candle['bb_width']) * 20)
                breakout_conf = min(40, ((squeeze_check_candle['bb_lower'] - curr_breakout_candle['close']) / bb_lower_div) * 100 * 5)
                confidence_score = 30 + squeeze_conf + breakout_conf
            
        df_out.iloc[curr_idx, df_out.columns.get_loc('signal')] = signal
        df_out.iloc[curr_idx, df_out.columns.get_loc('confidence')] = int(max(0, min(100, confidence_score)))
        return df_out

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df_with_indicators = self.calculate_all_indicators(df.copy())
        df_signaled = df_with_indicators.copy()
        if 'signal' not in df_signaled.columns: df_signaled['signal'] = 0
        if 'confidence' not in df_signaled.columns: df_signaled['confidence'] = 0

        required_cols = ['bb_upper', 'bb_lower', 'bb_width', 'open', 'close', 'high', 'low']
        if not all(col in df_signaled.columns for col in required_cols):
            print(f"({self.name}) BB挤压所需列未找到。")
            return df_signaled

        # print(f"({self.name}) 兼容模式 generate_signals, 行数: {len(df_signaled)}")
        confirm_candles = self.params['breakout_confirmation_candles']
        # Loop starts from where there's enough history for both current candle and squeeze_check_candle
        start_loop_idx = self.params['bb_period'] + confirm_candles # Ensure all BB calcs are valid + squeeze hist
        
        for i in range(start_loop_idx, len(df_signaled)):
            # Squeeze check index relative to current `i`
            squeeze_check_idx_loop = i - confirm_candles 

            if df_signaled.iloc[i][required_cols].isna().any() or \
               df_signaled.iloc[squeeze_check_idx_loop][['bb_upper', 'bb_lower', 'bb_width']].isna().any():
                continue

            curr_breakout_candle = df_signaled.iloc[i]
            squeeze_check_candle = df_signaled.iloc[squeeze_check_idx_loop]
            is_squeezed = squeeze_check_candle['bb_width'] < self.params['squeeze_threshold_pct']
            signal = 0
            confidence_score = 0
            breakout_body = abs(curr_breakout_candle['close'] - curr_breakout_candle['open'])
            breakout_range = curr_breakout_candle['high'] - curr_breakout_candle['low']
            body_is_significant = (breakout_body / (breakout_range if breakout_range > 0 else 0.0001)) >= self.params['min_breakout_body_pct']

            bb_upper_div = squeeze_check_candle['bb_upper'] if squeeze_check_candle['bb_upper'] != 0 else 0.0001
            bb_lower_div = squeeze_check_candle['bb_lower'] if squeeze_check_candle['bb_lower'] != 0 else 0.0001

            if is_squeezed and body_is_significant:
                if curr_breakout_candle['close'] > squeeze_check_candle['bb_upper'] and curr_breakout_candle['close'] > curr_breakout_candle['open']:
                    signal = 1
                    squeeze_conf = min(30, (self.params['squeeze_threshold_pct'] - squeeze_check_candle['bb_width']) * 20)
                    breakout_conf = min(40, ((curr_breakout_candle['close'] - squeeze_check_candle['bb_upper']) / bb_upper_div) * 100 * 5)
                    confidence_score = 30 + squeeze_conf + breakout_conf
                elif curr_breakout_candle['close'] < squeeze_check_candle['bb_lower'] and curr_breakout_candle['close'] < curr_breakout_candle['open']:
                    signal = -1
                    squeeze_conf = min(30, (self.params['squeeze_threshold_pct'] - squeeze_check_candle['bb_width']) * 20)
                    breakout_conf = min(40, ((squeeze_check_candle['bb_lower'] - curr_breakout_candle['close']) / bb_lower_div) * 100 * 5)
                    confidence_score = 30 + squeeze_conf + breakout_conf
            
            df_signaled.iloc[i, df_signaled.columns.get_loc('signal')] = signal
            df_signaled.iloc[i, df_signaled.columns.get_loc('confidence')] = int(max(0, min(100, confidence_score)))
        return df_signaled


# --- get_available_strategies (no change needed, but ensure it lists these classes) ---
def get_available_strategies() -> List[Dict[str, Any]]:
    """获取所有可用策略的列表"""
    strategies = [
        {
            'id': 'simple_rsi', 'name': '简单RSI策略', 'class': SimpleRSIStrategy,
            'description': '使用RSI指标判断超买超卖',
            'parameters': [
                {'name': 'rsi_period', 'type': 'int', 'default': 14, 'min': 5, 'max': 30, 'description': 'RSI周期'},
                {'name': 'rsi_overbought', 'type': 'int', 'default': 70, 'min': 1, 'max': 99, 'description': 'RSI超买阈值'},
                {'name': 'rsi_oversold', 'type': 'int', 'default': 30, 'min': 1, 'max': 99, 'description': 'RSI超卖阈值'},
            ]
        },
        {
            'id': 'rsi_sma', 'name': 'RSI+SMA策略', 'class': RSISMA_Strategy,
            'description': '结合RSI指标和SMA均线判断趋势',
            'parameters': [
                {'name': 'rsi_period', 'type': 'int', 'default': 14, 'min': 5, 'max': 30, 'description': 'RSI周期'},
                {'name': 'rsi_overbought', 'type': 'int', 'default': 70, 'min': 60, 'max': 90, 'description': 'RSI超买阈值'},
                {'name': 'rsi_oversold', 'type': 'int', 'default': 30, 'min': 10, 'max': 40, 'description': 'RSI超卖阈值'},
                {'name': 'sma_period', 'type': 'int', 'default': 30, 'min': 10, 'max': 100, 'description': 'SMA均线周期'},
            ]
        },
        {
            'id': 'macd', 'name': 'MACD策略', 'class': MACDStrategy,
            'description': '使用MACD金叉死叉判断趋势转变',
            'parameters': [
                {'name': 'fast_period', 'type': 'int', 'default': 12, 'min': 5, 'max': 20, 'description': '快线周期'},
                {'name': 'slow_period', 'type': 'int', 'default': 26, 'min': 15, 'max': 50, 'description': '慢线周期'},
                {'name': 'signal_period', 'type': 'int', 'default': 9, 'min': 5, 'max': 15, 'description': '信号线周期'},
            ]
        },
        {
            'id': 'ma_cross', 'name': '均线交叉策略', 'class': MovingAverageCrossStrategy,
            'description': '短期均线上穿/下穿长期均线产生信号',
            'parameters': [
                {'name': 'short_period', 'type': 'int', 'default': 9, 'min': 3, 'max': 30, 'description': '短期均线'},
                {'name': 'long_period', 'type': 'int', 'default': 21, 'min': 10, 'max': 60, 'description': '长期均线'},
                {'name': 'ma_type', 'type': 'select', 'default': 'EMA', 'options': ['SMA', 'EMA'], 'description': '均线类型'},
            ]
        },
        {
            'id': 'bb_reversal', 'name': '布林带均值回归策略', 'class': BollingerBandsReversalStrategy,
            'description': '价格触及布林带边界后反转回带内 (谨慎)',
            'parameters': [
                {'name': 'bb_period', 'type': 'int', 'default': 20, 'min': 10, 'max': 50, 'description': '布林带周期'},
                {'name': 'bb_std_dev', 'type': 'float', 'default': 2.0, 'min': 1.5, 'max': 3.0, 'step': 0.1, 'description': '布林带标准差倍数'},
                {'name': 'reversal_confirmation_candles', 'type': 'int', 'default': 1, 'min': 0, 'max': 3, 'description': '价格回到带内确认K线数'},
            ]
        },
        {
            'id': 'enhanced_candlestick', 'name': '增强K线形态策略', 'class': EnhancedCandlestickPatternStrategy,
            'description': '识别吞没形态和Pin Bar (锤子/射击之星)',
            'parameters': [
                {'name': 'engulfing_min_body_ratio', 'type': 'float', 'default': 0.6, 'min': 0.4, 'max': 0.9, 'step':0.05, 'description': '吞没K线实体最小占比'},
                {'name': 'engulfing_factor', 'type': 'float', 'default': 1.0, 'min': 0.8, 'max': 1.5, 'step':0.1, 'description': '被吞没K线与吞没K线实体比例'},
                {'name': 'pinbar_min_wick_body_ratio', 'type': 'float', 'default': 2.0, 'min': 1.5, 'max': 5.0, 'step':0.1, 'description': 'PinBar影线/实体最小比例'},
                {'name': 'pinbar_max_body_range_ratio', 'type': 'float', 'default': 0.33, 'min': 0.1, 'max': 0.5, 'step':0.01, 'description': 'PinBar实体/总长度最大比例'},
            ]
        },
        {
            'id': 'price_breakout', 'name': '价格突破策略', 'class': PriceBreakoutStrategy,
            'description': '价格突破N周期高点/低点产生信号',
            'parameters': [
                {'name': 'lookback_period', 'type': 'int', 'default': 20, 'min': 5, 'max': 60, 'description': '回顾周期N'},
                {'name': 'volume_confirmation_factor', 'type': 'float', 'default': 1.2, 'min': 1.0, 'max': 2.5, 'step':0.1, 'description': '成交量确认因子'},
            ]
        },
        {
            'id': 'stochastic_oscillator', 'name': '随机指标策略', 'class': StochasticOscillatorStrategy,
            'description': '使用随机指标的超买超卖及金叉死叉',
            'parameters': [
                {'name': 'k_period', 'type': 'int', 'default': 14, 'min': 5, 'max': 30, 'description': '%K周期'},
                {'name': 'd_period', 'type': 'int', 'default': 3, 'min': 2, 'max': 10, 'description': '%D周期'},
                {'name': 'slowing', 'type': 'int', 'default': 3, 'min': 1, 'max': 10, 'description': '%K平滑周期'},
                {'name': 'overbought', 'type': 'int', 'default': 80, 'min': 70, 'max': 95, 'description': '超买阈值'},
                {'name': 'oversold', 'type': 'int', 'default': 20, 'min': 5, 'max': 30, 'description': '超卖阈值'},
            ]
        },
        {
            'id': 'micro_trend_momentum', 'name': '微趋势动能策略(剥头皮)', 'class': MicroTrendMomentumStrategy,
            'description': '捕捉极短期EMA趋势和成交量确认',
            'parameters': [
                {'name': 'ema_period', 'type': 'int', 'default': 5, 'min': 2, 'max': 10, 'description': '极短期EMA周期'},
                {'name': 'volume_sma_period', 'type': 'int', 'default': 5, 'min': 3, 'max': 10, 'description': '成交量SMA周期'},
                {'name': 'min_volume_factor', 'type': 'float', 'default': 1.0, 'min': 0.5, 'max': 2.0, 'step': 0.1, 'description': '成交量最小因子'},
                {'name': 'rsi_filter_period', 'type': 'int', 'default': 14, 'min': 7, 'max': 21, 'description': 'RSI过滤周期'},
                {'name': 'rsi_extreme_ob', 'type': 'int', 'default': 85, 'min': 75, 'max': 95, 'description': 'RSI极度超买'},
                {'name': 'rsi_extreme_os', 'type': 'int', 'default': 15, 'min': 5, 'max': 25, 'description': 'RSI极度超卖'},
            ]
        },
        {
            'id': 'bb_squeeze_breakout', 'name': '布林带挤压突破(剥头皮)', 'class': BBSqueezeBreakoutStrategy,
            'description': '在布林带收窄后，等待价格突破方向',
            'parameters': [
                {'name': 'bb_period', 'type': 'int', 'default': 20, 'min': 10, 'max': 30, 'description': '布林带周期'},
                {'name': 'bb_std_dev', 'type': 'float', 'default': 2.0, 'min': 1.5, 'max': 2.5, 'step': 0.1, 'description': '标准差倍数'},
                {'name': 'squeeze_threshold_pct', 'type': 'float', 'default': 0.5, 'min': 0.2, 'max': 1.5, 'step': 0.1, 'description': '挤压阈值(带宽/中轨%)'},
                {'name': 'breakout_confirmation_candles', 'type': 'int', 'default': 1, 'min': 1, 'max': 3, 'description': '突破确认K线数'},
                {'name': 'min_breakout_body_pct', 'type': 'float', 'default': 0.3, 'min': 0.1, 'max': 0.7, 'step': 0.05, 'description': '突破K线最小实体占比'},
            ]
        }
    ]
    ids = [s['id'] for s in strategies]
    if len(ids) != len(set(ids)):
        raise ValueError("策略ID不唯一！请检查 get_available_strategies 函数。")
    return strategies