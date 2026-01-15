import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Dict, Any, List
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
        if 'confidence' not in df_out.columns: df_out['confidence'] = 0.0
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
        if 'confidence' not in df_out.columns: df_out['confidence'] = 0.0

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

# --- RsiDivergenceStrategy (Divergence Logic as per user request) ---
class RsiDivergenceStrategy(Strategy):
    """
    增强型RSI策略，实现 "RSI穿越警报 + 背离确认" 逻辑。
    """
    def __init__(self, params: Dict[str, Any] = None):
        # Adapt the user's requested __init__ to the base class structure
        default_params = {
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'divergence_lookback': 50
        }
        if params:
            default_params.update(params)
        
        super().__init__(default_params)
        self.name = "RSI背离策略"
        self.min_history_periods = self.params.get('rsi_period', 14) + 1

        # Initialize state variables as requested
        # 0: 正常, 1: 观察顶背离, -1: 观察底背离
        self.state = 0
        # 用于存储第一个峰/谷的价格和RSI值
        self.first_peak_price = None
        self.first_peak_rsi = None
        self.first_trough_price = None
        self.first_trough_rsi = None
        # 用于标记背离条件是否已满足
        self.divergence_confirmed = False
        # K线计数器
        self.bars_since_alert = 0
        # 优化模式标记
        self.is_optimizing = False

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates the necessary indicators (RSI)."""
        df = super().calculate_all_indicators(df)
        rsi_period = self.params.get('rsi_period', 14)
        rsi_col = f"RSI_{rsi_period}"
        
        try:
            if rsi_col not in df.columns:
                # 尝试使用pandas_ta计算RSI
                try:
                    df.ta.rsi(length=rsi_period, append=True)
                except Exception as e:
                    logger.warning(f"({self.name}) 使用pandas_ta计算RSI失败，使用手动计算: {e}")
                    # 手动计算RSI作为备选方案
                    close_prices = df['close']
                    delta = close_prices.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
                    rs = gain / loss.replace(0, 1e-10)
                    rsi_values = 100 - (100 / (1 + rs))
                    df[rsi_col] = rsi_values.fillna(50.0)
            
            # For simplicity in the logic, create a generic 'rsi' column
            if rsi_col in df.columns:
                df['rsi'] = df[rsi_col].fillna(50.0)
            else:
                df['rsi'] = 50.0  # Fallback
        except Exception as e:
            logger.error(f"({self.name}) 计算RSI时发生严重错误: {e}")
            df['rsi'] = 50.0  # 确保RSI列存在
        
        return df

    def next(self, data, prev_data):
        """
        This method implements the core state machine logic as requested by the user.
        It's designed to be called iteratively.
        """
        try:
            close = data['close']
            rsi = data['rsi']
            prev_rsi = prev_data['rsi']
            
            # 检查数据有效性
            if pd.isna(close) or pd.isna(rsi) or pd.isna(prev_rsi):
                return 0
            
            signal = 0
            rsi_overbought = self.params.get('rsi_overbought', 70)
            rsi_oversold = self.params.get('rsi_oversold', 30)

            # --- State Machine Logic from user request ---
            if self.state == 0:
                # RSI 向上穿越超买线，进入观察顶背离状态
                if rsi > rsi_overbought and prev_rsi <= rsi_overbought:
                    self.state = 1
                    self.first_peak_price = close
                    self.first_peak_rsi = rsi
                    self.divergence_confirmed = False
                    self.bars_since_alert = 0 # 重置计数器
                # RSI 向下穿越超卖线，进入观察底背离状态
                elif rsi < rsi_oversold and prev_rsi >= rsi_oversold:
                    self.state = -1
                    self.first_trough_price = close
                    self.first_trough_rsi = rsi
                    self.divergence_confirmed = False
                    self.bars_since_alert = 0 # 重置计数器
            
            elif self.state == 1:  # 观察顶背离
                self.bars_since_alert += 1 # 递增计数器
                
                # 检查是否超时
                if self.bars_since_alert > self.params.get('divergence_lookback', 50):
                    self.state = 0
                    return 0 # 超时则重置状态并跳出

                # 价格创出新高，但RSI没有，确认背离条件
                if self.first_peak_price is not None and self.first_peak_rsi is not None:
                    if close > self.first_peak_price and rsi < self.first_peak_rsi:
                        self.divergence_confirmed = True
                
                # 如果背离已确认，且RSI回落到超买线以下，产生卖出信号
                if self.divergence_confirmed and rsi < rsi_overbought and prev_rsi >= rsi_overbought:
                    signal = -1
                    self.state = 0  # 重置状态
                # 如果RSI回落到50中线，趋势可能结束，重置状态
                elif rsi < 50:
                    self.state = 0

            elif self.state == -1:  # 观察底背离
                self.bars_since_alert += 1 # 递增计数器

                # 检查是否超时
                if self.bars_since_alert > self.params.get('divergence_lookback', 50):
                    self.state = 0
                    return 0 # 超时则重置状态并跳出

                # 价格创出新低，但RSI没有，确认背离条件
                if self.first_trough_price is not None and self.first_trough_rsi is not None:
                    if close < self.first_trough_price and rsi > self.first_trough_rsi:
                        self.divergence_confirmed = True

                # 如果背离已确认，且RSI回升到超卖线以上，产生买入信号
                if self.divergence_confirmed and rsi > rsi_oversold and prev_rsi <= rsi_oversold:
                    signal = 1
                    self.state = 0  # 重置状态
                # 如果RSI回升到50中线，趋势可能结束，重置状态
                elif rsi > 50:
                    self.state = 0
        except Exception as e:
            logger.error(f"({self.name}) 状态机逻辑执行错误: {e}")
            # 出错时重置状态
            self.state = 0
            self.divergence_confirmed = False
            signal = 0
        
        return signal

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates trading signals for the entire dataframe by iterating through it.
        """
        df_with_indicators = self.calculate_all_indicators(df.copy())
        df_signaled = df_with_indicators.copy()
        if 'signal' not in df_signaled.columns: df_signaled['signal'] = 0
        if 'confidence' not in df_signaled.columns: df_signaled['confidence'] = 0

        if 'rsi' not in df_signaled.columns or 'close' not in df_signaled.columns:
            return df_signaled

        # Reset state for a full backtest run
        self.state = 0
        self.first_peak_price = None
        self.first_peak_rsi = None
        self.first_trough_price = None
        self.first_trough_rsi = None
        self.divergence_confirmed = False
        self.bars_since_alert = 0

        # Loop through the data to apply the stateful logic
        for i in range(1, len(df_signaled)):
            signal = self.next(df_signaled.iloc[i], df_signaled.iloc[i-1])
            if signal != 0:
                df_signaled.iloc[i, df_signaled.columns.get_loc('signal')] = signal
                df_signaled.iloc[i, df_signaled.columns.get_loc('confidence')] = 100

        return df_signaled
    
    def generate_signals_from_indicators_on_window(self, df_window_with_indicators: pd.DataFrame) -> pd.DataFrame:
        """
        This method is required by the framework. For live trading, where state is
        maintained across calls, this would work by processing the last point of the window.
        For optimization, we need to ensure state is properly managed.
        """
        df_signaled = df_window_with_indicators.copy()
        if 'signal' not in df_signaled.columns: df_signaled['signal'] = 0
        if 'confidence' not in df_signaled.columns: df_signaled['confidence'] = 0

        if len(df_signaled) < 2:
            return df_signaled

        # 对于优化过程，我们需要在每次调用时重置状态
        # 因为优化引擎会为每个参数组合使用同一个策略实例
        if hasattr(self, 'is_optimizing') and self.is_optimizing:
            # 重置状态
            self.state = 0
            self.first_peak_price = None
            self.first_peak_rsi = None
            self.first_trough_price = None
            self.first_trough_rsi = None
            self.divergence_confirmed = False
            self.bars_since_alert = 0

        # The state is managed by the instance, so calling next for the last point works
        signal = self.next(df_signaled.iloc[-1], df_signaled.iloc[-2])
        
        if signal != 0:
            df_signaled.iloc[-1, df_signaled.columns.get_loc('signal')] = signal
            df_signaled.iloc[-1, df_signaled.columns.get_loc('confidence')] = 100
            
        return df_signaled

# --- MACDStrategy ---
class MACDStrategy(Strategy):
    """基于MACD指标的策略"""
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9
        }
        if params: 
            # 先更新默认参数
            default_params.update(params)
            
            # 验证参数，确保fast周期小于slow周期
            if default_params['macd_fast'] >= default_params['macd_slow']:
                logger.warning(f"({self.name}) MACD fast周期 ({default_params['macd_fast']}) 大于等于 slow周期 ({default_params['macd_slow']})，自动调整")
                # 调整参数，确保fast < slow
                default_params['macd_fast'] = min(default_params['macd_fast'], default_params['macd_slow'] - 1)
        
        super().__init__(default_params)
        self.name = "MACD策略"
        self.min_history_periods = max(self.params['macd_slow'], self.params['macd_signal']) + 1

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().calculate_all_indicators(df)
        
        try:
            # 计算MACD指标
            fast_period = self.params['macd_fast']
            slow_period = self.params['macd_slow']
            signal_period = self.params['macd_signal']
            
            # 计算EMA
            ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
            ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()
            
            # 计算MACD线
            macd_line = ema_fast - ema_slow
            
            # 计算信号线
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
            
            # 计算柱状图
            macd_hist = macd_line - signal_line
            
            df['macd_line'] = macd_line
            df['macd_signal'] = signal_line
            df['macd_hist'] = macd_hist
            
        except Exception as e:
            logger.error(f"({self.name}) 计算MACD时发生错误: {e}", exc_info=True)
            df['macd_line'] = 0
            df['macd_signal'] = 0
            df['macd_hist'] = 0
        
        return df

    def generate_signals_from_indicators_on_window(self, df_window_with_indicators: pd.DataFrame) -> pd.DataFrame:
        df_out = df_window_with_indicators.copy()
        if 'signal' not in df_out.columns: df_out['signal'] = 0
        if 'confidence' not in df_out.columns: df_out['confidence'] = 0

        if len(df_out) < 2 or 'macd_hist' not in df_out.columns:
            return df_out

        curr_idx = -1
        prev_idx = -2
        
        curr_hist = df_out.iloc[curr_idx]['macd_hist']
        prev_hist = df_out.iloc[prev_idx]['macd_hist']
        
        signal = 0
        confidence_score = 0
        
        # MACD柱状图金叉
        if curr_hist > 0 and prev_hist <= 0:
            signal = 1
            confidence_score = 50 + min(50, abs(curr_hist) * 100)
        # MACD柱状图死叉
        elif curr_hist < 0 and prev_hist >= 0:
            signal = -1
            confidence_score = 50 + min(50, abs(curr_hist) * 100)
        
        df_out.iloc[curr_idx, df_out.columns.get_loc('signal')] = signal
        df_out.iloc[curr_idx, df_out.columns.get_loc('confidence')] = int(max(0, min(100, confidence_score)))
        return df_out

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df_with_indicators = self.calculate_all_indicators(df.copy())
        df_signaled = df_with_indicators.copy()
        if 'signal' not in df_signaled.columns: df_signaled['signal'] = 0
        if 'confidence' not in df_signaled.columns: df_signaled['confidence'] = 0
        
        if 'macd_hist' not in df_signaled.columns:
            return df_signaled
        
        for i in range(1, len(df_signaled)):
            curr_hist = df_signaled.iloc[i]['macd_hist']
            prev_hist = df_signaled.iloc[i-1]['macd_hist']
            
            signal = 0
            confidence_score = 0
            
            if curr_hist > 0 and prev_hist <= 0:
                signal = 1
                confidence_score = 50 + min(50, abs(curr_hist) * 100)
            elif curr_hist < 0 and prev_hist >= 0:
                signal = -1
                confidence_score = 50 + min(50, abs(curr_hist) * 100)
            
            df_signaled.iloc[i, df_signaled.columns.get_loc('signal')] = signal
            df_signaled.iloc[i, df_signaled.columns.get_loc('confidence')] = int(max(0, min(100, confidence_score)))
        
        return df_signaled

# --- BollingerBandsStrategy ---
class BollingerBandsStrategy(Strategy):
    """基于布林带指标的策略"""
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'bb_period': 20,
            'bb_std': 2.0
        }
        if params: default_params.update(params)
        super().__init__(default_params)
        self.name = "布林带策略"
        self.min_history_periods = self.params['bb_period'] + 1

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().calculate_all_indicators(df)
        
        try:
            # 计算布林带指标
            period = self.params['bb_period']
            std_dev = self.params['bb_std']
            
            # 计算移动平均线
            df['bb_mid'] = df['close'].rolling(window=period).mean()
            
            # 计算标准差
            df['bb_std'] = df['close'].rolling(window=period).std()
            
            # 计算上轨和下轨
            df['bb_upper'] = df['bb_mid'] + (df['bb_std'] * std_dev)
            df['bb_lower'] = df['bb_mid'] - (df['bb_std'] * std_dev)
            
            # 计算价格相对于布林带的位置
            df['bb_position'] = (df['close'] - df['bb_mid']) / (df['bb_std'] * std_dev) if 'bb_std' in df.columns else 0
            
        except Exception as e:
            logger.error(f"({self.name}) 计算布林带时发生错误: {e}", exc_info=True)
            df['bb_mid'] = df['close']
            df['bb_upper'] = df['close']
            df['bb_lower'] = df['close']
            df['bb_position'] = 0
        
        return df

    def generate_signals_from_indicators_on_window(self, df_window_with_indicators: pd.DataFrame) -> pd.DataFrame:
        df_out = df_window_with_indicators.copy()
        if 'signal' not in df_out.columns: df_out['signal'] = 0
        if 'confidence' not in df_out.columns: df_out['confidence'] = 0

        if len(df_out) < 2 or 'bb_position' not in df_out.columns:
            return df_out

        curr_idx = -1
        prev_idx = -2
        
        curr_position = df_out.iloc[curr_idx]['bb_position']
        prev_position = df_out.iloc[prev_idx]['bb_position']
        
        signal = 0
        confidence_score = 0
        
        # 价格从下轨上方穿越下轨 -> 买入信号
        if curr_position < -1 and prev_position >= -1:
            signal = 1
            confidence_score = 50 + min(50, abs(curr_position) * 25)
        # 价格从上轨下方穿越上轨 -> 卖出信号
        elif curr_position > 1 and prev_position <= 1:
            signal = -1
            confidence_score = 50 + min(50, abs(curr_position) * 25)
        
        df_out.iloc[curr_idx, df_out.columns.get_loc('signal')] = signal
        df_out.iloc[curr_idx, df_out.columns.get_loc('confidence')] = int(max(0, min(100, confidence_score)))
        return df_out

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df_with_indicators = self.calculate_all_indicators(df.copy())
        df_signaled = df_with_indicators.copy()
        if 'signal' not in df_signaled.columns: df_signaled['signal'] = 0
        if 'confidence' not in df_signaled.columns: df_signaled['confidence'] = 0
        
        if 'bb_position' not in df_signaled.columns:
            return df_signaled
        
        for i in range(1, len(df_signaled)):
            curr_position = df_signaled.iloc[i]['bb_position']
            prev_position = df_signaled.iloc[i-1]['bb_position']
            
            signal = 0
            confidence_score = 0
            
            if curr_position < -1 and prev_position >= -1:
                signal = 1
                confidence_score = 50 + min(50, abs(curr_position) * 25)
            elif curr_position > 1 and prev_position <= 1:
                signal = -1
                confidence_score = 50 + min(50, abs(curr_position) * 25)
            
            df_signaled.iloc[i, df_signaled.columns.get_loc('signal')] = signal
            df_signaled.iloc[i, df_signaled.columns.get_loc('confidence')] = int(max(0, min(100, confidence_score)))
        
        return df_signaled

# --- MAStrategy (Moving Average Crossover) ---
class MAStrategy(Strategy):
    """基于移动平均线交叉的策略"""
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'ma_fast': 5,
            'ma_slow': 20
        }
        if params: default_params.update(params)
        super().__init__(default_params)
        self.name = "移动平均线策略"
        self.min_history_periods = self.params['ma_slow'] + 1

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().calculate_all_indicators(df)
        
        try:
            # 计算移动平均线
            fast_period = self.params['ma_fast']
            slow_period = self.params['ma_slow']
            
            df['ma_fast'] = df['close'].rolling(window=fast_period).mean()
            df['ma_slow'] = df['close'].rolling(window=slow_period).mean()
            
            # 计算均线差
            df['ma_diff'] = df['ma_fast'] - df['ma_slow']
            
        except Exception as e:
            logger.error(f"({self.name}) 计算移动平均线时发生错误: {e}", exc_info=True)
            df['ma_fast'] = df['close']
            df['ma_slow'] = df['close']
            df['ma_diff'] = 0
        
        return df

    def generate_signals_from_indicators_on_window(self, df_window_with_indicators: pd.DataFrame) -> pd.DataFrame:
        df_out = df_window_with_indicators.copy()
        if 'signal' not in df_out.columns: df_out['signal'] = 0
        if 'confidence' not in df_out.columns: df_out['confidence'] = 0

        if len(df_out) < 2 or 'ma_diff' not in df_out.columns:
            return df_out

        curr_idx = -1
        prev_idx = -2
        
        curr_diff = df_out.iloc[curr_idx]['ma_diff']
        prev_diff = df_out.iloc[prev_idx]['ma_diff']
        
        signal = 0
        confidence_score = 0
        
        # 金叉：短期均线上穿长期均线
        if curr_diff > 0 and prev_diff <= 0:
            signal = 1
            confidence_score = 50 + min(50, abs(curr_diff) * 1000)
        # 死叉：短期均线下穿长期均线
        elif curr_diff < 0 and prev_diff >= 0:
            signal = -1
            confidence_score = 50 + min(50, abs(curr_diff) * 1000)
        
        df_out.iloc[curr_idx, df_out.columns.get_loc('signal')] = signal
        df_out.iloc[curr_idx, df_out.columns.get_loc('confidence')] = int(max(0, min(100, confidence_score)))
        return df_out

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df_with_indicators = self.calculate_all_indicators(df.copy())
        df_signaled = df_with_indicators.copy()
        if 'signal' not in df_signaled.columns: df_signaled['signal'] = 0
        if 'confidence' not in df_signaled.columns: df_signaled['confidence'] = 0
        
        if 'ma_diff' not in df_signaled.columns:
            return df_signaled
        
        for i in range(1, len(df_signaled)):
            curr_diff = df_signaled.iloc[i]['ma_diff']
            prev_diff = df_signaled.iloc[i-1]['ma_diff']
            
            signal = 0
            confidence_score = 0
            
            if curr_diff > 0 and prev_diff <= 0:
                signal = 1
                confidence_score = 50 + min(50, abs(curr_diff) * 1000)
            elif curr_diff < 0 and prev_diff >= 0:
                signal = -1
                confidence_score = 50 + min(50, abs(curr_diff) * 1000)
            
            df_signaled.iloc[i, df_signaled.columns.get_loc('signal')] = signal
            df_signaled.iloc[i, df_signaled.columns.get_loc('confidence')] = int(max(0, min(100, confidence_score)))
        
        return df_signaled

# --- get_available_strategies ---
def get_available_strategies() -> List[Dict[str, Any]]:
    """获取所有可用策略的列表"""
    strategies = [
        {
            'id': 'simple_rsi', 'name': '简单RSI策略', 'class': SimpleRSIStrategy,
            'description': '使用RSI指标判断超买超卖',
            'parameters': [
                {'name': 'rsi_period', 'type': 'int', 'default': 14, 'min': 5, 'max': 30, 'step': 1, 'description': 'RSI周期'},
                {'name': 'rsi_overbought', 'type': 'int', 'default': 70, 'min': 1, 'max': 99, 'step': 1, 'description': 'RSI超买阈值'},
                {'name': 'rsi_oversold', 'type': 'int', 'default': 30, 'min': 1, 'max': 99, 'step': 1, 'description': 'RSI超卖阈值'},
            ]
        },
        {
            'id': 'rsi_divergence', 'name': 'RSI背离策略', 'class': RsiDivergenceStrategy,
            'description': '实现RSI穿越警报与价格背离确认的逻辑。',
            'parameters': [
                {'name': 'rsi_period', 'type': 'int', 'default': 14, 'min': 5, 'max': 30, 'step': 1, 'description': 'RSI计算周期'},
                {'name': 'rsi_overbought', 'type': 'int', 'default': 70, 'min': 60, 'max': 85, 'step': 1, 'description': 'RSI超买阈值'},
                {'name': 'rsi_oversold', 'type': 'int', 'default': 30, 'min': 15, 'max': 40, 'step': 1, 'description': 'RSI超卖阈值'},
                {'name': 'divergence_lookback', 'type': 'int', 'default': 50, 'min': 1, 'max': 100, 'step': 1, 'description': '进入观察状态后，等待背离形成的最大K线数'},
            ]
        },
        {
            'id': 'macd', 'name': 'MACD策略', 'class': MACDStrategy,
            'description': '使用MACD指标判断趋势变化',
            'parameters': [
                {'name': 'macd_fast', 'type': 'int', 'default': 12, 'min': 5, 'max': 20, 'step': 1, 'description': '快速EMA周期'},
                {'name': 'macd_slow', 'type': 'int', 'default': 26, 'min': 20, 'max': 40, 'step': 1, 'description': '慢速EMA周期'},
                {'name': 'macd_signal', 'type': 'int', 'default': 9, 'min': 5, 'max': 15, 'step': 1, 'description': '信号线周期'},
            ]
        },
        {
            'id': 'bollinger_bands', 'name': '布林带策略', 'class': BollingerBandsStrategy,
            'description': '使用布林带指标判断价格波动范围',
            'parameters': [
                {'name': 'bb_period', 'type': 'int', 'default': 20, 'min': 10, 'max': 50, 'step': 1, 'description': '移动平均线周期'},
                {'name': 'bb_std', 'type': 'float', 'default': 2.0, 'min': 1.0, 'max': 3.0, 'step': 0.1, 'description': '标准差倍数'},
            ]
        },
        {
            'id': 'ma_crossover', 'name': '移动平均线策略', 'class': MAStrategy,
            'description': '使用移动平均线交叉判断趋势变化',
            'parameters': [
                {'name': 'ma_fast', 'type': 'int', 'default': 5, 'min': 2, 'max': 20, 'step': 1, 'description': '短期移动平均线周期'},
                {'name': 'ma_slow', 'type': 'int', 'default': 20, 'min': 10, 'max': 50, 'step': 1, 'description': '长期移动平均线周期'},
            ]
        },
        {
            'id': 'model_prediction', 'name': '模型预测策略', 'class': ModelPredictionStrategy,
            'description': '使用机器学习模型进行价格预测',
            'parameters': [
                {'name': 'model_count', 'type': 'int', 'default': 5, 'min': 1, 'max': 28, 'step': 1, 'description': '使用的模型数量'},
                {'name': 'model_types', 'type': 'string', 'default': 'random_forest,xgboost', 'description': '模型类型，多个类型用逗号分隔'},
                {'name': 'preprocessor_params', 'type': 'object', 'default': {}, 'description': '数据预处理器参数'},
                {'name': 'model_params', 'type': 'object', 'default': {}, 'description': '模型参数'},
            ]
        },
    ]
    ids = [s['id'] for s in strategies]
    if len(ids) != len(set(ids)):
        raise ValueError("策略ID不唯一！请检查 get_available_strategies 函数。")
    return strategies

# --- ModelPredictionStrategy --- 新增模型预测策略
class ModelPredictionStrategy(Strategy):
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'model_count': 5,
            'model_types': 'random_forest,xgboost',
            'preprocessor_params': {},
            'model_params': {}
        }
        if params: 
            # 处理字符串形式的模型类型
            if isinstance(params.get('model_types'), str):
                default_params['model_types'] = params['model_types']
            elif isinstance(params.get('model_types'), list):
                default_params['model_types'] = ','.join(params['model_types'])
            default_params.update(params)
        super().__init__(default_params)
        self.name = "模型预测策略"
        self.min_history_periods = 30  # 需要足够的历史数据进行预测
        self.is_optimizing = False  # 优化模式标志
        self._has_configuration = False  # 配置状态标志
        
        # 延迟导入以避免循环依赖
        from model_prediction.predictor import Predictor
        self.predictor = Predictor({
            'model_count': self.params['model_count'],
            'model_types': self.params['model_types'],
            'preprocessor_params': self.params['preprocessor_params'],
            'model_params': self.params['model_params']
        })
        
        # 检查是否已有配置
        self._check_configuration()
    
    def _check_configuration(self):
        """检查模型是否已配置"""
        try:
            # 检查预测器是否存在
            if hasattr(self, 'predictor'):
                # 检查模型是否已训练或加载
                if hasattr(self.predictor, 'models') and self.predictor.models:
                    self._has_configuration = True
                else:
                    # 尝试检查是否有已保存的模型
                    from model_prediction.predictor import Predictor
                    if Predictor.has_saved_models():
                        self._has_configuration = True
        except Exception as e:
            logger.error(f"({self.name}) 配置检查出错: {str(e)}")
            self._has_configuration = False
    
    def has_configuration(self):
        """检查模型是否已配置"""
        self._check_configuration()
        return self._has_configuration

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().calculate_all_indicators(df)
        
        # 只进行数据验证，不添加额外的技术指标
        # 特征计算由 predictor 类自己处理
        
        return df

    def generate_signals_from_indicators_on_window(self, df_window_with_indicators: pd.DataFrame) -> pd.DataFrame:
        df_out = df_window_with_indicators.copy()
        if 'signal' not in df_out.columns: df_out['signal'] = 0
        if 'confidence' not in df_out.columns: df_out['confidence'] = 0.0
        
        # 确保有足够的数据进行预测
        if len(df_out) < self.min_history_periods:
            return df_out
        
        try:
            # 检查是否已配置
            if not self.has_configuration() and not self.is_optimizing:
                # 未配置，生成明确的错误提示
                error_message = "请先在模型预测模块中完成参数配置"
                logger.warning(f"({self.name}) {error_message}")
                # 生成特殊信号，表明未配置
                df_out.loc[df_out.index[-1], 'signal'] = 0
                df_out.loc[df_out.index[-1], 'confidence'] = -1  # 使用-1表示未配置
                # 在df中添加错误信息列
                if 'error_message' not in df_out.columns:
                    df_out['error_message'] = ''
                df_out.loc[df_out.index[-1], 'error_message'] = error_message
                return df_out
            
            # 尝试使用模型进行预测
            try:
                if self.is_optimizing:
                    # 优化模式：简化预测逻辑，跳过耗时操作
                    # 只使用基本的价格趋势来生成信号
                    # 计算短期和长期移动平均线
                    df_out['short_ma'] = df_out['close'].rolling(window=5).mean()
                    df_out['long_ma'] = df_out['close'].rolling(window=20).mean()
                    
                    # 基于移动平均线交叉生成信号
                    if len(df_out) >= 20:
                        if df_out['short_ma'].iloc[-1] > df_out['long_ma'].iloc[-1] and df_out['short_ma'].iloc[-2] <= df_out['long_ma'].iloc[-2]:
                            df_out.loc[df_out.index[-1], 'signal'] = 1
                            df_out.loc[df_out.index[-1], 'confidence'] = 0.7
                        elif df_out['short_ma'].iloc[-1] < df_out['long_ma'].iloc[-1] and df_out['short_ma'].iloc[-2] >= df_out['long_ma'].iloc[-2]:
                            df_out.loc[df_out.index[-1], 'signal'] = -1
                            df_out.loc[df_out.index[-1], 'confidence'] = 0.7
                        else:
                            df_out.loc[df_out.index[-1], 'signal'] = 0
                            df_out.loc[df_out.index[-1], 'confidence'] = 0
                else:
                    # 正常模式：使用完整的预测逻辑
                    prediction_result = self.predictor.predict(df_out)
                    predictions = prediction_result['predictions']
                    
                    if predictions:
                        # 基于预测结果生成信号
                        last_prediction = predictions[-1]
                        last_close = df_out['close'].iloc[-1]
                        
                        # 计算价格变化百分比
                        price_change_pct = (last_prediction - last_close) / last_close
                        
                        # 生成信号
                        if price_change_pct > 0.001:  # 上涨超过0.1%
                            df_out.loc[df_out.index[-1], 'signal'] = 1
                            df_out.loc[df_out.index[-1], 'confidence'] = min(abs(price_change_pct) * 100, 1.0)
                        elif price_change_pct < -0.001:  # 下跌超过0.1%
                            df_out.loc[df_out.index[-1], 'signal'] = -1
                            df_out.loc[df_out.index[-1], 'confidence'] = min(abs(price_change_pct) * 100, 1.0)
                        else:
                            df_out.loc[df_out.index[-1], 'signal'] = 0
                            df_out.loc[df_out.index[-1], 'confidence'] = 0
            except ValueError as e:
                if "Model not trained or loaded" in str(e) or "No models trained or loaded" in str(e):
                    if not self.is_optimizing:
                        # 模型未训练或加载，生成明确的错误提示
                        error_message = "请先在模型预测模块中完成参数配置"
                        logger.warning(f"({self.name}) {error_message}")
                        df_out.loc[df_out.index[-1], 'signal'] = 0
                        df_out.loc[df_out.index[-1], 'confidence'] = -1  # 使用-1表示未配置
                        if 'error_message' not in df_out.columns:
                            df_out['error_message'] = ''
                        df_out.loc[df_out.index[-1], 'error_message'] = error_message
                else:
                    # 其他ValueError，继续抛出
                    raise
        except Exception as e:
            logger.error(f"({self.name}) 预测过程出错: {str(e)}")
            # 出错时使用默认信号
            df_out.loc[df_out.index[-1], 'signal'] = 0
            df_out.loc[df_out.index[-1], 'confidence'] = 0
        
        return df_out

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df_with_indicators = self.calculate_all_indicators(df.copy())
        df_signaled = df_with_indicators.copy()
        if 'signal' not in df_signaled.columns: df_signaled['signal'] = 0
        if 'confidence' not in df_signaled.columns: df_signaled['confidence'] = 0
        
        # 为每一行生成信号
        for i in range(self.min_history_periods, len(df_signaled)):
            window_df = df_signaled.iloc[:i+1].copy()
            window_result = self.generate_signals_from_indicators_on_window(window_df)
            df_signaled.loc[window_result.index[-1], 'signal'] = window_result['signal'].iloc[-1]
            df_signaled.loc[window_result.index[-1], 'confidence'] = window_result['confidence'].iloc[-1]
        
        return df_signaled
