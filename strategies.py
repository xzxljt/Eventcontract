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
        # logger.info(f"({self.name}) 基类 calculate_all_indicators 调用，检查数据类型。")
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

# --- RsiBollingerBandsStrategy ---
class RsiBollingerBandsStrategy(Strategy):
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            # --- Indicator Switches ---
            'use_bb': True,
            'use_rsi': True,
            'use_td_seq': False, # Default to off

            # --- Trigger Logic Control ---
            'use_continuous_trigger': False,  # False = 穿越触发(推荐), True = 持续触发(旧逻辑)

            # --- Bollinger Bands Params ---
            'bb_period': 20,
            'bb_std_dev': 2.0,

            # --- RSI Params ---
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,

            # --- TD Sequential Params ---
            'td_seq_buy_setup': 9,  # Can be 9 or 13
            'td_seq_sell_setup': 9, # Can be 9 or 13
        }
        if params: default_params.update(params)
        super().__init__(default_params)
        
        active_indicators = []
        if self.params.get('use_bb'): active_indicators.append('BB')
        if self.params.get('use_rsi'): active_indicators.append('RSI')
        if self.params.get('use_td_seq'): active_indicators.append('TD')
        self.name = f"Flexible ({'+'.join(active_indicators)})"
        
        self.min_history_periods = 0
        if self.params.get('use_bb'): self.min_history_periods = max(self.min_history_periods, self.params['bb_period'])
        if self.params.get('use_rsi'): self.min_history_periods = max(self.min_history_periods, self.params['rsi_period'])
        if self.params.get('use_td_seq'): self.min_history_periods = max(self.min_history_periods, 13) # TD needs at least a few bars
        if self.min_history_periods == 0: self.min_history_periods = 1


    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().calculate_all_indicators(df)
        try:
            if self.params.get('use_bb'):
                df.ta.bbands(length=self.params['bb_period'], std=self.params['bb_std_dev'], append=True)
            if self.params.get('use_rsi'):
                df.ta.rsi(length=self.params['rsi_period'], append=True)
            if self.params.get('use_td_seq'):
                # pandas-ta calculates both setup and countdown. We check for the final signal.
                # It creates columns like 'TD_SEQ_UPa' and 'TD_SEQ_DNa'
                df.ta.td_seq(append=True)
        except Exception as e:
            logger.error(f"({self.name}) Error calculating indicators: {e}", exc_info=True)
        return df

    def generate_signals_from_indicators_on_window(self, df_window_with_indicators: pd.DataFrame) -> pd.DataFrame:
        df_out = df_window_with_indicators.copy()
        if 'signal' not in df_out.columns: df_out['signal'] = 0
        if 'confidence' not in df_out.columns: df_out['confidence'] = 0

        if len(df_out) < 1:
            return df_out

        # 对于穿越逻辑，需要至少2行数据
        if not self.params.get('use_continuous_trigger', False) and len(df_out) < 2:
            return df_out

        last_row = df_out.iloc[-1]
        active_signals = []

        # --- Calculate signal for each enabled indicator ---

        # Bollinger Bands Signal
        if self.params.get('use_bb'):
            bb_signal = 0
            std_dev_str = str(float(self.params['bb_std_dev']))
            bb_upper_col = f"BBU_{self.params['bb_period']}_{std_dev_str}"
            bb_lower_col = f"BBL_{self.params['bb_period']}_{std_dev_str}"

            if self.params.get('use_continuous_trigger', False):
                # 旧逻辑：持续触发
                if all(c in last_row for c in [bb_upper_col, bb_lower_col, 'close']) and not last_row[[bb_upper_col, bb_lower_col, 'close']].isna().any():
                    if last_row['close'] <= last_row[bb_lower_col]: bb_signal = 1
                    elif last_row['close'] >= last_row[bb_upper_col]: bb_signal = -1
            else:
                # 新逻辑：穿越触发（布林带回归逻辑）
                if len(df_out) >= 2:
                    curr_row = df_out.iloc[-1]
                    prev_row = df_out.iloc[-2]

                    required_cols = [bb_upper_col, bb_lower_col, 'close']
                    if (all(c in curr_row for c in required_cols) and all(c in prev_row for c in required_cols) and
                        not curr_row[required_cols].isna().any() and not prev_row[required_cols].isna().any()):

                        # 买入信号：价格从下轨下方回到上方（均值回归）
                        if (prev_row['close'] <= prev_row[bb_lower_col] and
                            curr_row['close'] > curr_row[bb_lower_col]):
                            bb_signal = 1
                        # 卖出信号：价格从上轨上方回到下方（均值回归）
                        elif (prev_row['close'] >= prev_row[bb_upper_col] and
                              curr_row['close'] < curr_row[bb_upper_col]):
                            bb_signal = -1

            active_signals.append(bb_signal)

        # RSI Signal
        if self.params.get('use_rsi'):
            rsi_signal = 0
            rsi_col = f"RSI_{self.params['rsi_period']}"

            if self.params.get('use_continuous_trigger', False):
                # 旧逻辑：持续触发
                if rsi_col in last_row and pd.notna(last_row[rsi_col]):
                    if last_row[rsi_col] <= self.params['rsi_oversold']: rsi_signal = 1
                    elif last_row[rsi_col] >= self.params['rsi_overbought']: rsi_signal = -1
            else:
                # 新逻辑：穿越触发
                if len(df_out) >= 2:
                    curr_row = df_out.iloc[-1]
                    prev_row = df_out.iloc[-2]

                    if (rsi_col in curr_row and rsi_col in prev_row and
                        pd.notna(curr_row[rsi_col]) and pd.notna(prev_row[rsi_col])):

                        curr_rsi = curr_row[rsi_col]
                        prev_rsi = prev_row[rsi_col]

                        # 买入信号：RSI 从超卖阈值以上穿越到以下（刚进入超卖）
                        if curr_rsi <= self.params['rsi_oversold'] and prev_rsi > self.params['rsi_oversold']:
                            rsi_signal = 1
                        # 卖出信号：RSI 从超买阈值以下穿越到以上（刚进入超买）
                        elif curr_rsi >= self.params['rsi_overbought'] and prev_rsi < self.params['rsi_overbought']:
                            rsi_signal = -1

            active_signals.append(rsi_signal)

        # TD Sequential Signal
        if self.params.get('use_td_seq'):
            td_signal = 0
            # pandas-ta uses 'TD_SEQ_UPa' for buy setup count and 'TD_SEQ_DNa' for sell setup count
            td_buy_col = 'TD_SEQ_UPa'
            td_sell_col = 'TD_SEQ_DNa'
            if td_buy_col in last_row and last_row[td_buy_col] == self.params['td_seq_buy_setup']:
                td_signal = 1
            elif td_sell_col in last_row and last_row[td_sell_col] == self.params['td_seq_sell_setup']:
                td_signal = -1
            active_signals.append(td_signal)

        # --- Combine Signals ---
        final_signal = 0
        if not active_signals: # No indicators enabled
            pass
        elif len(active_signals) == 1:
            # 只有一个指标启用，直接使用其信号
            final_signal = active_signals[0]
        else:
            # 多个指标启用，使用状态确认逻辑
            if self.params.get('use_continuous_trigger', False):
                # 旧逻辑：AND 逻辑（同时穿越）
                if all(s == 1 for s in active_signals):
                    final_signal = 1
                elif all(s == -1 for s in active_signals):
                    final_signal = -1
            else:
                # 新逻辑：状态确认逻辑
                final_signal = self._evaluate_state_confirmation_logic(df_out)
        
        # Simple confidence for now
        confidence = 100 if final_signal != 0 else 0

        df_out.iloc[-1, df_out.columns.get_loc('signal')] = final_signal
        df_out.iloc[-1, df_out.columns.get_loc('confidence')] = confidence
        
        return df_out

    def _evaluate_state_confirmation_logic(self, df_out: pd.DataFrame) -> int:
        """
        状态确认逻辑：一个指标处于极值状态，另一个指标发生穿越即可触发信号

        买入信号逻辑：
        - RSI ≤ 30（处于超卖状态）且 布林带发生向上穿越
        - 或 布林带 ≤ 下轨（处于超卖状态）且 RSI 发生向下穿越

        卖出信号逻辑：
        - RSI ≥ 70（处于超买状态）且 布林带发生向下穿越
        - 或 布林带 ≥ 上轨（处于超买状态）且 RSI 发生向上穿越
        """
        if len(df_out) < 2:
            return 0

        curr_row = df_out.iloc[-1]
        prev_row = df_out.iloc[-2]

        # 检查各指标的状态和穿越情况
        rsi_state = self._get_rsi_state_and_crossover(curr_row, prev_row)
        bb_state = self._get_bb_state_and_crossover(curr_row, prev_row)

        # 状态确认逻辑
        # 买入信号：RSI超卖状态 + BB向上穿越，或 BB超卖状态 + RSI向下穿越
        if ((rsi_state['is_oversold'] and bb_state['crossover'] == 1) or
            (bb_state['is_oversold'] and rsi_state['crossover'] == 1)):
            return 1

        # 卖出信号：RSI超买状态 + BB向下穿越，或 BB超买状态 + RSI向上穿越
        if ((rsi_state['is_overbought'] and bb_state['crossover'] == -1) or
            (bb_state['is_overbought'] and rsi_state['crossover'] == -1)):
            return -1

        return 0

    def _get_rsi_state_and_crossover(self, curr_row, prev_row):
        """获取RSI的状态和穿越情况"""
        result = {
            'is_oversold': False,
            'is_overbought': False,
            'crossover': 0  # 1=向下穿越(进入超卖), -1=向上穿越(进入超买), 0=无穿越
        }

        if not self.params.get('use_rsi'):
            return result

        rsi_col = f"RSI_{self.params['rsi_period']}"
        if (rsi_col not in curr_row or rsi_col not in prev_row or
            pd.isna(curr_row[rsi_col]) or pd.isna(prev_row[rsi_col])):
            return result

        curr_rsi = curr_row[rsi_col]
        prev_rsi = prev_row[rsi_col]

        # 状态检查
        result['is_oversold'] = curr_rsi <= self.params['rsi_oversold']
        result['is_overbought'] = curr_rsi >= self.params['rsi_overbought']

        # 穿越检查
        if curr_rsi <= self.params['rsi_oversold'] and prev_rsi > self.params['rsi_oversold']:
            result['crossover'] = 1  # 向下穿越进入超卖
        elif curr_rsi >= self.params['rsi_overbought'] and prev_rsi < self.params['rsi_overbought']:
            result['crossover'] = -1  # 向上穿越进入超买

        return result

    def _get_bb_state_and_crossover(self, curr_row, prev_row):
        """获取布林带的状态和穿越情况"""
        result = {
            'is_oversold': False,
            'is_overbought': False,
            'crossover': 0  # 1=向上穿越(回归), -1=向下穿越(回归), 0=无穿越
        }

        if not self.params.get('use_bb'):
            return result

        std_dev_str = str(float(self.params['bb_std_dev']))
        bb_upper_col = f"BBU_{self.params['bb_period']}_{std_dev_str}"
        bb_lower_col = f"BBL_{self.params['bb_period']}_{std_dev_str}"

        required_cols = [bb_upper_col, bb_lower_col, 'close']
        if (not all(c in curr_row for c in required_cols) or
            not all(c in prev_row for c in required_cols) or
            curr_row[required_cols].isna().any() or prev_row[required_cols].isna().any()):
            return result

        curr_close = curr_row['close']
        prev_close = prev_row['close']
        curr_bb_upper = curr_row[bb_upper_col]
        curr_bb_lower = curr_row[bb_lower_col]
        prev_bb_upper = prev_row[bb_upper_col]
        prev_bb_lower = prev_row[bb_lower_col]

        # 状态检查
        result['is_oversold'] = curr_close <= curr_bb_lower
        result['is_overbought'] = curr_close >= curr_bb_upper

        # 穿越检查（布林带回归逻辑）
        if prev_close <= prev_bb_lower and curr_close > curr_bb_lower:
            result['crossover'] = 1  # 从下轨下方回到上方
        elif prev_close >= prev_bb_upper and curr_close < curr_bb_upper:
            result['crossover'] = -1  # 从上轨上方回到下方

        return result

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        # This method is for full backtesting. The main logic is in the windowed version.
        # We can implement a vectorized version here for speed if needed.
        df_with_indicators = self.calculate_all_indicators(df.copy())
        df_signaled = df_with_indicators.copy()
        if 'signal' not in df_signaled.columns: df_signaled['signal'] = 0
        if 'confidence' not in df_signaled.columns: df_signaled['confidence'] = 0

        buy_conditions = pd.Series(True, index=df_signaled.index)
        sell_conditions = pd.Series(True, index=df_signaled.index)

        # BB Conditions
        if self.params.get('use_bb'):
            std_dev_str = str(float(self.params['bb_std_dev']))
            bb_upper_col = f"BBU_{self.params['bb_period']}_{std_dev_str}"
            bb_lower_col = f"BBL_{self.params['bb_period']}_{std_dev_str}"

            if all(c in df_signaled.columns for c in [bb_upper_col, bb_lower_col]):
                if self.params.get('use_continuous_trigger', False):
                    # 旧逻辑：持续触发
                    buy_conditions &= (df_signaled['close'] <= df_signaled[bb_lower_col])
                    sell_conditions &= (df_signaled['close'] >= df_signaled[bb_upper_col])
                else:
                    # 新逻辑：穿越触发（布林带回归逻辑）
                    prev_close = df_signaled['close'].shift(1)
                    prev_bb_upper = df_signaled[bb_upper_col].shift(1)
                    prev_bb_lower = df_signaled[bb_lower_col].shift(1)

                    # 买入：价格从下轨下方回到上方
                    bb_buy_condition = ((prev_close <= prev_bb_lower) &
                                       (df_signaled['close'] > df_signaled[bb_lower_col]))
                    # 卖出：价格从上轨上方回到下方
                    bb_sell_condition = ((prev_close >= prev_bb_upper) &
                                        (df_signaled['close'] < df_signaled[bb_upper_col]))

                    buy_conditions &= bb_buy_condition
                    sell_conditions &= bb_sell_condition
            else: # If column missing, condition is false
                buy_conditions &= False
                sell_conditions &= False

        # RSI Conditions
        if self.params.get('use_rsi'):
            rsi_col = f"RSI_{self.params['rsi_period']}"
            if rsi_col in df_signaled.columns:
                if self.params.get('use_continuous_trigger', False):
                    # 旧逻辑：持续触发
                    buy_conditions &= (df_signaled[rsi_col] <= self.params['rsi_oversold'])
                    sell_conditions &= (df_signaled[rsi_col] >= self.params['rsi_overbought'])
                else:
                    # 新逻辑：穿越触发
                    prev_rsi = df_signaled[rsi_col].shift(1)

                    # 买入：RSI 从超卖阈值以上穿越到以下
                    rsi_buy_condition = ((df_signaled[rsi_col] <= self.params['rsi_oversold']) &
                                        (prev_rsi > self.params['rsi_oversold']))
                    # 卖出：RSI 从超买阈值以下穿越到以上
                    rsi_sell_condition = ((df_signaled[rsi_col] >= self.params['rsi_overbought']) &
                                         (prev_rsi < self.params['rsi_overbought']))

                    buy_conditions &= rsi_buy_condition
                    sell_conditions &= rsi_sell_condition
            else:
                buy_conditions &= False
                sell_conditions &= False

        # TD Sequential Conditions
        if self.params.get('use_td_seq'):
            td_buy_col = 'TD_SEQ_UPa'
            td_sell_col = 'TD_SEQ_DNa'
            if td_buy_col in df_signaled.columns:
                buy_conditions &= (df_signaled[td_buy_col] == self.params['td_seq_buy_setup'])
            else:
                buy_conditions &= False
            if td_sell_col in df_signaled.columns:
                sell_conditions &= (df_signaled[td_sell_col] == self.params['td_seq_sell_setup'])
            else:
                sell_conditions &= False
        
        # Apply signals
        if not self.params.get('use_continuous_trigger', False) and len([p for p in [self.params.get('use_bb'), self.params.get('use_rsi')] if p]) > 1:
            # 对于状态确认逻辑，使用逐行处理（因为向量化实现较复杂）
            for i in range(1, len(df_signaled)):
                window_df = df_signaled.iloc[i-1:i+1].copy()
                result_df = self.generate_signals_from_indicators_on_window(window_df)
                if len(result_df) > 0:
                    df_signaled.iloc[i, df_signaled.columns.get_loc('signal')] = result_df.iloc[-1]['signal']
                    df_signaled.iloc[i, df_signaled.columns.get_loc('confidence')] = result_df.iloc[-1]['confidence']
        else:
            # 使用原有的向量化逻辑
            df_signaled.loc[buy_conditions, 'signal'] = 1
            df_signaled.loc[sell_conditions, 'signal'] = -1
            df_signaled.loc[buy_conditions | sell_conditions, 'confidence'] = 100

        return df_signaled


# --- EnhancedRSIStrategy ---
class EnhancedRSIStrategy(Strategy):
    """
    加强版RSI策略：能够识别市场趋势并相应调整交易方向
    - 上涨趋势：只做多，过滤做空信号
    - 下跌趋势：只做空，过滤做多信号
    - 震荡市场：传统RSI逆势交易
    """
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            # RSI参数 (针对1分钟K线优化)
            'rsi_period': 9,        # 缩短至9周期，更敏感
            'rsi_overbought': 75,   # 提高阈值，减少假信号
            'rsi_oversold': 25,     # 降低阈值，减少假信号

            # 趋势识别参数 (针对10分钟事件合约优化)
            'ema_fast': 5,          # 5分钟快线，捕捉短期趋势
            'ema_slow': 15,         # 15分钟慢线，确认中期趋势
            'adx_period': 8,        # 缩短ADX周期，更快响应
            'adx_threshold': 20,    # 降低阈值，适应短期波动

            # 辅助指标参数 (短期优化)
            'macd_fast': 5,         # 快速MACD设置
            'macd_slow': 13,        # 适合1分钟K线
            'macd_signal': 4,       # 更快的信号线
            'volume_period': 10,    # 10分钟成交量均线
            'roc_period': 5,        # 5分钟动量
            'atr_period': 8,        # 8分钟波动率

            # 顺势交易RSI阈值 (针对短期调整)
            'trend_rsi_buy_min': 35,   # 上涨趋势中RSI回调到此值以上才考虑做多
            'trend_rsi_buy_max': 55,   # 上涨趋势中RSI在此值以下才考虑做多
            'trend_rsi_sell_min': 45,  # 下跌趋势中RSI反弹到此值以上才考虑做空
            'trend_rsi_sell_max': 65,  # 下跌趋势中RSI在此值以下才考虑做空

            # 置信度权重 (强化趋势和动量权重)
            'trend_weight': 0.35,      # 提高趋势权重
            'rsi_weight': 0.25,
            'macd_weight': 0.2,
            'volume_weight': 0.1,      # 降低成交量权重(1分钟成交量噪音大)
            'momentum_weight': 0.1,

            # 10分钟事件合约特殊参数
            'min_confidence_threshold': 65,  # 最低置信度阈值
            'trend_confirmation_periods': 3, # 趋势确认需要的周期数
        }
        if params: default_params.update(params)
        super().__init__(default_params)
        self.name = "加强版RSI策略"
        self.min_history_periods = max(
            self.params['rsi_period'],
            self.params['ema_slow'],
            self.params['adx_period'],
            self.params['volume_period']
        ) + 5

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = super().calculate_all_indicators(df)

            # 确保数据有效
            if 'close' not in df.columns or len(df) < self.min_history_periods:
                logger.warning(f"({self.name}) 数据不足或缺少close列")
                return self._add_default_indicators(df)

            # 1. RSI指标
            df.ta.rsi(length=self.params['rsi_period'], append=True)

            # 2. EMA均线
            df.ta.ema(length=self.params['ema_fast'], append=True)
            df.ta.ema(length=self.params['ema_slow'], append=True)

            # 3. ADX趋势强度
            df.ta.adx(length=self.params['adx_period'], append=True)

            # 4. MACD
            df.ta.macd(fast=self.params['macd_fast'],
                      slow=self.params['macd_slow'],
                      signal=self.params['macd_signal'], append=True)

            # 5. 成交量指标
            if 'volume' in df.columns:
                df.ta.sma(close=df['volume'], length=self.params['volume_period'], append=True)
                # 重命名成交量均线列
                volume_sma_col = f"SMA_{self.params['volume_period']}"
                if volume_sma_col in df.columns:
                    df[f"Volume_SMA_{self.params['volume_period']}"] = df[volume_sma_col]

            # 6. 价格动量ROC
            df.ta.roc(length=self.params['roc_period'], append=True)

            # 7. ATR波动率
            df.ta.atr(length=self.params['atr_period'], append=True)

            # 8. 计算自定义指标
            self._calculate_custom_indicators(df)

        except Exception as e:
            logger.error(f"({self.name}) 计算指标时发生错误: {e}", exc_info=True)
            return self._add_default_indicators(df)

        return df

    def _add_default_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加默认指标值以防计算失败"""
        default_indicators = {
            f'RSI_{self.params["rsi_period"]}': 50.0,
            f'EMA_{self.params["ema_fast"]}': df['close'].iloc[-1] if len(df) > 0 else 0,
            f'EMA_{self.params["ema_slow"]}': df['close'].iloc[-1] if len(df) > 0 else 0,
            f'ADX_{self.params["adx_period"]}': 20.0,
            f'MACD_{self.params["macd_fast"]}_{self.params["macd_slow"]}_{self.params["macd_signal"]}': 0.0,
            f'MACDs_{self.params["macd_fast"]}_{self.params["macd_slow"]}_{self.params["macd_signal"]}': 0.0,
            f'ROC_{self.params["roc_period"]}': 0.0,
            f'ATRr_{self.params["atr_period"]}': 1.0,
            'market_regime': 'sideways',
            'trend_strength': 0.0
        }

        for col, default_val in default_indicators.items():
            if col not in df.columns:
                df[col] = default_val

        return df

    def _calculate_custom_indicators(self, df: pd.DataFrame) -> None:
        """计算自定义指标 - 针对1分钟K线优化"""
        try:
            # 市场形态识别
            ema_fast_col = f'EMA_{self.params["ema_fast"]}'
            ema_slow_col = f'EMA_{self.params["ema_slow"]}'
            adx_col = f'ADX_{self.params["adx_period"]}'

            # 初始化
            df['market_regime'] = 'sideways'
            df['trend_strength'] = 0.0
            df['trend_confirmed'] = False

            if all(col in df.columns for col in [ema_fast_col, ema_slow_col, adx_col]):
                # 趋势强度 (标准化ADX，针对短期调整)
                df['trend_strength'] = np.clip(df[adx_col] / 40.0, 0, 1)  # 降低分母适应短期

                # 基础趋势条件
                ema_bullish = df[ema_fast_col] > df[ema_slow_col]
                ema_bearish = df[ema_fast_col] < df[ema_slow_col]
                strong_trend = df[adx_col] > self.params['adx_threshold']
                price_above_fast = df['close'] > df[ema_fast_col]
                price_below_fast = df['close'] < df[ema_fast_col]

                # 计算价格相对于均线的距离（过滤噪音）
                price_ema_distance = abs(df['close'] - df[ema_fast_col]) / df[ema_fast_col]
                significant_distance = price_ema_distance > 0.002  # 0.2%以上的距离

                # 趋势确认：需要连续几个周期满足条件
                confirmation_periods = self.params.get('trend_confirmation_periods', 3)

                # 上涨趋势条件（更严格）
                uptrend_base = ema_bullish & strong_trend & price_above_fast & significant_distance
                # 下跌趋势条件（更严格）
                downtrend_base = ema_bearish & strong_trend & price_below_fast & significant_distance

                # 使用滚动窗口确认趋势
                if len(df) >= confirmation_periods:
                    uptrend_confirmed = uptrend_base.rolling(window=confirmation_periods, min_periods=confirmation_periods).sum() >= confirmation_periods
                    downtrend_confirmed = downtrend_base.rolling(window=confirmation_periods, min_periods=confirmation_periods).sum() >= confirmation_periods

                    df.loc[uptrend_confirmed, 'market_regime'] = 'uptrend'
                    df.loc[downtrend_confirmed, 'market_regime'] = 'downtrend'
                    df.loc[uptrend_confirmed | downtrend_confirmed, 'trend_confirmed'] = True

                # 额外的趋势强度计算（结合价格动量）
                if 'close' in df.columns and len(df) > 1:
                    price_momentum = df['close'].pct_change(periods=5).fillna(0)  # 5分钟动量
                    momentum_strength = np.clip(abs(price_momentum) * 100, 0, 1)  # 标准化动量强度
                    df['trend_strength'] = np.maximum(df['trend_strength'], momentum_strength)

        except Exception as e:
            logger.warning(f"({self.name}) 计算自定义指标失败: {e}")

    def generate_signals_from_indicators_on_window(self, df_window_with_indicators: pd.DataFrame) -> pd.DataFrame:
        df_out = df_window_with_indicators.copy()
        if 'signal' not in df_out.columns: df_out['signal'] = 0
        if 'confidence' not in df_out.columns: df_out['confidence'] = 0

        if len(df_out) < 2:
            return df_out

        # 获取当前和前一行数据
        curr_row = df_out.iloc[-1]
        prev_row = df_out.iloc[-2]

        # 获取指标列名
        rsi_col = f'RSI_{self.params["rsi_period"]}'
        macd_col = f'MACD_{self.params["macd_fast"]}_{self.params["macd_slow"]}_{self.params["macd_signal"]}'
        macd_signal_col = f'MACDs_{self.params["macd_fast"]}_{self.params["macd_slow"]}_{self.params["macd_signal"]}'

        # 检查必要指标是否存在
        if not all(col in curr_row for col in [rsi_col, 'market_regime']):
            return df_out

        # 获取市场形态和RSI值
        market_regime = curr_row['market_regime']
        curr_rsi = curr_row[rsi_col]
        prev_rsi = prev_row[rsi_col] if rsi_col in prev_row else curr_rsi

        if pd.isna(curr_rsi) or pd.isna(prev_rsi):
            return df_out

        # 根据市场形态生成信号
        signal = 0
        confidence_components = {}

        # 检查趋势确认（针对1分钟K线的噪音过滤）
        trend_confirmed = curr_row.get('trend_confirmed', False)

        if market_regime == 'uptrend' and trend_confirmed:
            signal, confidence_components = self._generate_uptrend_signal(curr_row, prev_row, curr_rsi, prev_rsi)
        elif market_regime == 'downtrend' and trend_confirmed:
            signal, confidence_components = self._generate_downtrend_signal(curr_row, prev_row, curr_rsi, prev_rsi)
        else:  # sideways or unconfirmed trend
            signal, confidence_components = self._generate_sideways_signal(curr_row, prev_row, curr_rsi, prev_rsi)

        # 计算综合置信度
        confidence = self._calculate_confidence(confidence_components, curr_row)

        # 应用最低置信度阈值（10分钟事件合约需要高置信度）
        min_confidence = self.params.get('min_confidence_threshold', 65)
        if confidence < min_confidence:
            signal = 0  # 置信度不足，不产生信号
            confidence = 0

        df_out.iloc[-1, df_out.columns.get_loc('signal')] = signal
        df_out.iloc[-1, df_out.columns.get_loc('confidence')] = int(max(0, min(100, confidence)))

        return df_out

    def _generate_uptrend_signal(self, curr_row, prev_row, curr_rsi, prev_rsi):
        """上涨趋势中的信号生成 - 只做多"""
        signal = 0
        confidence_components = {'trend': 1.0, 'rsi': 0.0, 'macd': 0.0, 'volume': 0.0, 'momentum': 0.0}

        # 在上涨趋势中，只在RSI回调时寻找做多机会
        if (self.params['trend_rsi_buy_min'] <= curr_rsi <= self.params['trend_rsi_buy_max'] and
            prev_rsi < curr_rsi):  # RSI开始回升
            signal = 1
            # RSI置信度：越接近买入区间中值越好
            rsi_optimal = (self.params['trend_rsi_buy_min'] + self.params['trend_rsi_buy_max']) / 2
            rsi_distance = abs(curr_rsi - rsi_optimal)
            confidence_components['rsi'] = max(0, 1 - rsi_distance / 20)

        return signal, confidence_components

    def _generate_downtrend_signal(self, curr_row, prev_row, curr_rsi, prev_rsi):
        """下跌趋势中的信号生成 - 只做空"""
        signal = 0
        confidence_components = {'trend': 1.0, 'rsi': 0.0, 'macd': 0.0, 'volume': 0.0, 'momentum': 0.0}

        # 在下跌趋势中，只在RSI反弹时寻找做空机会
        if (self.params['trend_rsi_sell_min'] <= curr_rsi <= self.params['trend_rsi_sell_max'] and
            prev_rsi > curr_rsi):  # RSI开始回落
            signal = -1
            # RSI置信度：越接近卖出区间中值越好
            rsi_optimal = (self.params['trend_rsi_sell_min'] + self.params['trend_rsi_sell_max']) / 2
            rsi_distance = abs(curr_rsi - rsi_optimal)
            confidence_components['rsi'] = max(0, 1 - rsi_distance / 20)

        return signal, confidence_components

    def _generate_sideways_signal(self, curr_row, prev_row, curr_rsi, prev_rsi):
        """震荡市场中的信号生成 - 传统RSI逆势"""
        signal = 0
        confidence_components = {'trend': 0.5, 'rsi': 0.0, 'macd': 0.0, 'volume': 0.0, 'momentum': 0.0}

        # 传统RSI逆势交易
        if curr_rsi < self.params['rsi_oversold'] and prev_rsi >= self.params['rsi_oversold']:
            signal = 1
            rsi_distance = self.params['rsi_oversold'] - curr_rsi
            confidence_components['rsi'] = min(1.0, rsi_distance / 20)
        elif curr_rsi > self.params['rsi_overbought'] and prev_rsi <= self.params['rsi_overbought']:
            signal = -1
            rsi_distance = curr_rsi - self.params['rsi_overbought']
            confidence_components['rsi'] = min(1.0, rsi_distance / 20)

        return signal, confidence_components

    def _calculate_confidence(self, confidence_components, curr_row):
        """计算综合置信度"""
        try:
            # 基础置信度
            base_confidence = 50

            # 各组件置信度加权
            weighted_confidence = (
                confidence_components.get('trend', 0) * self.params['trend_weight'] +
                confidence_components.get('rsi', 0) * self.params['rsi_weight'] +
                confidence_components.get('macd', 0) * self.params['macd_weight'] +
                confidence_components.get('volume', 0) * self.params['volume_weight'] +
                confidence_components.get('momentum', 0) * self.params['momentum_weight']
            ) * 50  # 转换为0-50的加分

            # 添加MACD确认
            macd_col = f'MACD_{self.params["macd_fast"]}_{self.params["macd_slow"]}_{self.params["macd_signal"]}'
            macd_signal_col = f'MACDs_{self.params["macd_fast"]}_{self.params["macd_slow"]}_{self.params["macd_signal"]}'

            if all(col in curr_row for col in [macd_col, macd_signal_col]):
                macd_val = curr_row[macd_col]
                macd_signal = curr_row[macd_signal_col]
                if not (pd.isna(macd_val) or pd.isna(macd_signal)):
                    if macd_val > macd_signal:  # MACD金叉
                        confidence_components['macd'] = 0.8
                    elif macd_val < macd_signal:  # MACD死叉
                        confidence_components['macd'] = 0.8

            # 添加成交量确认
            volume_sma_col = f"Volume_SMA_{self.params['volume_period']}"
            if 'volume' in curr_row and volume_sma_col in curr_row:
                if not (pd.isna(curr_row['volume']) or pd.isna(curr_row[volume_sma_col])):
                    volume_ratio = curr_row['volume'] / curr_row[volume_sma_col]
                    if volume_ratio > 1.2:  # 放量
                        confidence_components['volume'] = min(1.0, volume_ratio / 2)

            # 添加动量确认
            roc_col = f'ROC_{self.params["roc_period"]}'
            if roc_col in curr_row and not pd.isna(curr_row[roc_col]):
                roc_val = abs(curr_row[roc_col])
                confidence_components['momentum'] = min(1.0, roc_val / 5)  # ROC绝对值越大动量越强

            # 重新计算加权置信度
            weighted_confidence = (
                confidence_components.get('trend', 0) * self.params['trend_weight'] +
                confidence_components.get('rsi', 0) * self.params['rsi_weight'] +
                confidence_components.get('macd', 0) * self.params['macd_weight'] +
                confidence_components.get('volume', 0) * self.params['volume_weight'] +
                confidence_components.get('momentum', 0) * self.params['momentum_weight']
            ) * 50

            # 趋势强度调整
            if 'trend_strength' in curr_row and not pd.isna(curr_row['trend_strength']):
                trend_bonus = curr_row['trend_strength'] * 10
                weighted_confidence += trend_bonus

            return base_confidence + weighted_confidence

        except Exception as e:
            logger.warning(f"({self.name}) 计算置信度失败: {e}")
            return 50

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """完整的信号生成方法"""
        df_with_indicators = self.calculate_all_indicators(df.copy())
        df_signaled = df_with_indicators.copy()
        if 'signal' not in df_signaled.columns: df_signaled['signal'] = 0
        if 'confidence' not in df_signaled.columns: df_signaled['confidence'] = 0

        # 逐行生成信号（从有足够历史数据开始）
        for i in range(self.min_history_periods, len(df_signaled)):
            # 创建窗口数据
            window_start = max(0, i - 1)
            window_df = df_signaled.iloc[window_start:i+1].copy()

            # 生成信号
            result_df = self.generate_signals_from_indicators_on_window(window_df)

            if len(result_df) > 0:
                df_signaled.iloc[i, df_signaled.columns.get_loc('signal')] = result_df.iloc[-1]['signal']
                df_signaled.iloc[i, df_signaled.columns.get_loc('confidence')] = result_df.iloc[-1]['confidence']

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

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates the necessary indicators (RSI)."""
        df = super().calculate_all_indicators(df)
        rsi_period = self.params.get('rsi_period', 14)
        rsi_col = f"RSI_{rsi_period}"
        
        if rsi_col not in df.columns:
            df.ta.rsi(length=rsi_period, append=True)
        
        # For simplicity in the logic, create a generic 'rsi' column
        if rsi_col in df.columns:
            df['rsi'] = df[rsi_col]
        else:
            df['rsi'] = 50.0  # Fallback
        return df

    def next(self, data, prev_data):
        """
        This method implements the core state machine logic as requested by the user.
        It's designed to be called iteratively.
        """
        close = data['close']
        rsi = data['rsi']
        prev_rsi = prev_data['rsi']
        
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
            if close < self.first_trough_price and rsi > self.first_trough_rsi:
                self.divergence_confirmed = True

            # 如果背离已确认，且RSI回升到超卖线以上，产生买入信号
            if self.divergence_confirmed and rsi > rsi_oversold and prev_rsi <= rsi_oversold:
                signal = 1
                self.state = 0  # 重置状态
            # 如果RSI回升到50中线，趋势可能结束，重置状态
            elif rsi > 50:
                self.state = 0
        
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
        """
        df_signaled = df_window_with_indicators.copy()
        if 'signal' not in df_signaled.columns: df_signaled['signal'] = 0
        if 'confidence' not in df_signaled.columns: df_signaled['confidence'] = 0

        if len(df_signaled) < 2:
            return df_signaled

        # The state is managed by the instance, so calling next for the last point works
        signal = self.next(df_signaled.iloc[-1], df_signaled.iloc[-2])
        
        if signal != 0:
            df_signaled.iloc[-1, df_signaled.columns.get_loc('signal')] = signal
            df_signaled.iloc[-1, df_signaled.columns.get_loc('confidence')] = 100
            
        return df_signaled


class RsiBollingerStrategy(Strategy):
    """
    RSI布林带策略。
    该策略首先计算RSI指标，然后对RSI指标本身计算布林带。
    信号逻辑:
    - 看涨 (Long): 当RSI值从布林带下轨下方上穿下轨时触发。
    - 看跌 (Short): 当RSI值从布林带上轨上方下穿上轨时触发。
    """
    def __init__(self, params: Dict[str, Any] = None):
        """
        初始化策略并设置默认参数。
        """
        default_params = {
            'rsi_period': 14,      # RSI计算周期
            'bb_period': 20,       # 对RSI计算布林带的周期
            'bb_std_dev': 2.0,     # 布林带的标准差
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)
        
        self.name = "RSI布林带策略"
        # 最小历史数据需求：RSI计算所需周期 + 布林带计算所需周期
        self.min_history_periods = self.params['rsi_period'] + self.params['bb_period']

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        在完整的DataFrame上计算所有需要的技术指标。
        1. 计算RSI。
        2. 基于RSI计算布林带。
        """
        df = super().calculate_all_indicators(df)
        
        try:
            # 1. 计算RSI
            rsi_period = self.params['rsi_period']
            df.ta.rsi(length=rsi_period, append=True)
            rsi_col = f"RSI_{rsi_period}"

            if rsi_col not in df.columns or df[rsi_col].isnull().all():
                logger.warning(f"({self.name}) RSI列 '{rsi_col}' 计算失败或全为空值。")
                # 创建默认的布林带列以避免后续错误
                df[f'RSI_BBU_{self.params["bb_period"]}_{self.params["bb_std_dev"]}'] = np.nan
                df[f'RSI_BBM_{self.params["bb_period"]}_{self.params["bb_std_dev"]}'] = np.nan
                df[f'RSI_BBL_{self.params["bb_period"]}_{self.params["bb_std_dev"]}'] = np.nan
                return df

            # 2. 基于RSI计算布林带
            bb_period = self.params['bb_period']
            bb_std_dev = self.params['bb_std_dev']
            
            rsi_series = df[rsi_col]
            rolling_mean = rsi_series.rolling(window=bb_period).mean()
            rolling_std = rsi_series.rolling(window=bb_period).std()
            
            df[f'RSI_BBU_{bb_period}_{bb_std_dev}'] = rolling_mean + (rolling_std * bb_std_dev)
            df[f'RSI_BBM_{bb_period}_{bb_std_dev}'] = rolling_mean
            df[f'RSI_BBL_{bb_period}_{bb_std_dev}'] = rolling_mean - (rolling_std * bb_std_dev)

        except Exception as e:
            logger.error(f"({self.name}) 在 calculate_all_indicators 中发生错误: {e}", exc_info=True)

        return df

    def generate_signals_from_indicators_on_window(self, df_window_with_indicators: pd.DataFrame) -> pd.DataFrame:
        """
        在给定的、已包含指标的窗口数据上，为窗口的最后一行生成信号。
        """
        df_out = df_window_with_indicators.copy()
        if 'signal' not in df_out.columns: df_out['signal'] = 0
        if 'confidence' not in df_out.columns: df_out['confidence'] = 0

        if len(df_out) < 2:
            return df_out

        rsi_col = f"RSI_{self.params['rsi_period']}"
        bb_upper_col = f"RSI_BBU_{self.params['bb_period']}_{self.params['bb_std_dev']}"
        bb_lower_col = f"RSI_BBL_{self.params['bb_period']}_{self.params['bb_std_dev']}"

        required_cols = [rsi_col, bb_upper_col, bb_lower_col]
        if not all(col in df_out.columns for col in required_cols):
            return df_out

        curr_row = df_out.iloc[-1]
        prev_row = df_out.iloc[-2]

        if pd.isna(curr_row[required_cols]).any() or pd.isna(prev_row[required_cols]).any():
            return df_out

        curr_rsi = curr_row[rsi_col]
        prev_rsi = prev_row[rsi_col]
        curr_bb_upper = curr_row[bb_upper_col]
        curr_bb_lower = curr_row[bb_lower_col]
        prev_bb_upper = prev_row[bb_upper_col]
        prev_bb_lower = prev_row[bb_lower_col]
        
        signal = 0
        confidence = 0

        # 看涨信号：前一根K线的RSI在下轨下方，当前K线的RSI上穿下轨
        if prev_rsi <= prev_bb_lower and curr_rsi > curr_bb_lower:
            signal = 1
            confidence = 100

        # 看跌信号：前一根K线的RSI在上轨上方，当前K线的RSI下穿上轨
        elif prev_rsi >= prev_bb_upper and curr_rsi < curr_bb_upper:
            signal = -1
            confidence = 100
        
        df_out.iloc[-1, df_out.columns.get_loc('signal')] = signal
        df_out.iloc[-1, df_out.columns.get_loc('confidence')] = confidence
        
        return df_out

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        为整个DataFrame生成信号（主要用于回测）。
        """
        df_with_indicators = self.calculate_all_indicators(df.copy())
        df_signaled = df_with_indicators.copy()
        df_signaled['signal'] = 0
        df_signaled['confidence'] = 0

        rsi_col = f"RSI_{self.params['rsi_period']}"
        bb_upper_col = f"RSI_BBU_{self.params['bb_period']}_{self.params['bb_std_dev']}"
        bb_lower_col = f"RSI_BBL_{self.params['bb_period']}_{self.params['bb_std_dev']}"

        required_cols = [rsi_col, bb_upper_col, bb_lower_col]
        if not all(col in df_signaled.columns for col in required_cols):
            logger.warning(f"({self.name}) 缺少必要的指标列，无法生成信号。")
            return df_signaled

        prev_rsi = df_signaled[rsi_col].shift(1)
        prev_bb_upper = df_signaled[bb_upper_col].shift(1)
        prev_bb_lower = df_signaled[bb_lower_col].shift(1)

        long_condition = (prev_rsi <= prev_bb_lower) & (df_signaled[rsi_col] > df_signaled[bb_lower_col])
        
        short_condition = (prev_rsi >= prev_bb_upper) & (df_signaled[rsi_col] < df_signaled[bb_upper_col])

        df_signaled.loc[long_condition, 'signal'] = 1
        df_signaled.loc[short_condition, 'signal'] = -1
        df_signaled.loc[long_condition | short_condition, 'confidence'] = 100

        return df_signaled


# --- RsiMomentumExhaustionStrategy ---
class RsiMomentumExhaustionStrategy(Strategy):
    """
    RSI动能衰竭与成交量确认策略 (智能切换版)。
    该策略能够根据成交量环境，在“进场触发”和“离场触发”两种模式间智能切换。
    1. 极端信号：RSI穿越极端阈值（如90/10），立即执行。
    2. 标准信号：
        - 在高成交量环境，采用“离场触发”：RSI从超买区回落或从超卖区回升，且必须有成交量萎缩确认。
        - 在低成交量环境，可选择采用“进场触发”：RSI首次进入超买/超卖区就触发信号，无需等待回头。
    """
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'rsi_period': 14,
            'overbought_level': 80,
            'oversold_level': 20,
            'extreme_overbought_level': 90,
            'extreme_oversold_level': 10,
            'vma_period': 20,
            'volume_confirmation_multiplier': 0.9,
            'volume_bypass_threshold': 0, # 新增：成交量绕过阈值，0表示禁用
            'low_volume_on_entry': True, # 新增：在低成交量时使用“进场触发”模式
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)
        
        self.name = "RSI动能衰竭策略(智能版)"
        self.min_history_periods = max(self.params['rsi_period'], self.params['vma_period']) + 1

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().calculate_all_indicators(df)
        try:
            rsi_period = self.params['rsi_period']
            df.ta.rsi(length=rsi_period, append=True)
            self.rsi_col = f"RSI_{rsi_period}"

            if 'volume' in df.columns:
                vma_period = self.params['vma_period']
                df.ta.ema(close=df['volume'], length=vma_period, append=True)
                self.vma_col = f"EMA_{vma_period}"
            else:
                logger.warning(f"({self.name}) 'volume' 列不存在，无法计算VMA。")
                self.vma_col = None
        except Exception as e:
            logger.error(f"({self.name}) 在 calculate_all_indicators 中发生错误: {e}", exc_info=True)
            self.rsi_col = None
            self.vma_col = None
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df_with_indicators = self.calculate_all_indicators(df.copy())
        df_signaled = df_with_indicators.copy()
        df_signaled['signal'] = 0
        df_signaled['confidence'] = 0

        if not hasattr(self, 'rsi_col') or not self.rsi_col or self.rsi_col not in df_signaled.columns:
            return df_signaled
        
        p = self.params
        current_rsi = df_signaled[self.rsi_col]
        previous_rsi = df_signaled[self.rsi_col].shift(1)

        # --- 信号条件 ---
        # 极端信号 (总是“进场”触发)
        extreme_sell_signal = (previous_rsi < p['extreme_overbought_level']) & (current_rsi >= p['extreme_overbought_level'])
        extreme_buy_signal = (previous_rsi > p['extreme_oversold_level']) & (current_rsi <= p['extreme_oversold_level'])

        # 标准信号的RSI穿越条件
        standard_sell_exit_trigger = (previous_rsi > p['overbought_level']) & (current_rsi <= p['overbought_level'])
        standard_buy_exit_trigger = (previous_rsi < p['oversold_level']) & (current_rsi >= p['oversold_level'])
        
        standard_sell_entry_trigger = (previous_rsi < p['overbought_level']) & (current_rsi >= p['overbought_level'])
        standard_buy_entry_trigger = (previous_rsi > p['oversold_level']) & (current_rsi <= p['oversold_level'])

        # --- 成交量环境判断 ---
        is_low_volume_env = pd.Series(False, index=df_signaled.index)
        volume_check_passed = pd.Series(False, index=df_signaled.index)

        if hasattr(self, 'vma_col') and self.vma_col and self.vma_col in df_signaled.columns and 'volume' in df_signaled.columns:
            current_vma = df_signaled[self.vma_col]
            current_volume = df_signaled['volume']
            
            # 检查是否为低成交量环境
            if p['volume_bypass_threshold'] > 0:
                is_low_volume_env = current_vma < p['volume_bypass_threshold']
            
            # 常规成交量确认
            volume_check_passed = current_volume < (p['volume_confirmation_multiplier'] * current_vma)
        else:
            logger.warning(f"({self.name}) 无法进行成交量确认，标准信号将被禁用。")

        # --- 根据环境组合最终信号 ---
        # 在低成交量且启用“进场触发”模式时，使用 entry_trigger
        use_entry_trigger = is_low_volume_env & p['low_volume_on_entry']
        
        # 做空信号
        sell_trigger = pd.Series(False, index=df_signaled.index)
        sell_trigger.loc[use_entry_trigger] = standard_sell_entry_trigger[use_entry_trigger]
        sell_trigger.loc[~use_entry_trigger] = standard_sell_exit_trigger[~use_entry_trigger]
        
        standard_sell_signal = sell_trigger & (is_low_volume_env | volume_check_passed)

        # 做多信号
        buy_trigger = pd.Series(False, index=df_signaled.index)
        buy_trigger.loc[use_entry_trigger] = standard_buy_entry_trigger[use_entry_trigger]
        buy_trigger.loc[~use_entry_trigger] = standard_buy_exit_trigger[~use_entry_trigger]

        standard_buy_signal = buy_trigger & (is_low_volume_env | volume_check_passed)

        # --- 合并所有信号 ---
        df_signaled.loc[standard_buy_signal, 'signal'] = 1
        df_signaled.loc[standard_sell_signal, 'signal'] = -1
        # 极端信号具有最高优先级，会覆盖标准信号
        df_signaled.loc[extreme_buy_signal, 'signal'] = 1
        df_signaled.loc[extreme_sell_signal, 'signal'] = -1
        
        df_signaled.loc[df_signaled['signal'] != 0, 'confidence'] = 100
        return df_signaled

    def generate_signals_from_indicators_on_window(self, df_window_with_indicators: pd.DataFrame) -> pd.DataFrame:
        df_out = df_window_with_indicators.copy()
        if 'signal' not in df_out.columns: df_out['signal'] = 0
        if 'confidence' not in df_out.columns: df_out['confidence'] = 0

        if len(df_out) < 2 or not hasattr(self, 'rsi_col') or not self.rsi_col or self.rsi_col not in df_out.columns:
            return df_out

        current = df_out.iloc[-1]
        previous = df_out.iloc[-2]
        
        if pd.isna(current[self.rsi_col]) or pd.isna(previous[self.rsi_col]):
            return df_out
        
        p = self.params
        current_rsi = current[self.rsi_col]
        previous_rsi = previous[self.rsi_col]
        signal = 0

        # --- 极端信号检查 (最高优先级) ---
        if previous_rsi < p['extreme_overbought_level'] and current_rsi >= p['extreme_overbought_level']:
            signal = -1
        elif previous_rsi > p['extreme_oversold_level'] and current_rsi <= p['extreme_oversold_level']:
            signal = 1
        
        # --- 标准信号检查 (仅在无极端信号时) ---
        if signal == 0:
            # 检查成交量环境
            is_low_volume_env = False
            volume_check_passed = False
            if hasattr(self, 'vma_col') and self.vma_col and self.vma_col in current and 'volume' in current and pd.notna(current[self.vma_col]) and pd.notna(current['volume']):
                if p['volume_bypass_threshold'] > 0 and current[self.vma_col] < p['volume_bypass_threshold']:
                    is_low_volume_env = True
                if current['volume'] < p['volume_confirmation_multiplier'] * current[self.vma_col]:
                    volume_check_passed = True

            # 确定是否满足成交量条件
            volume_condition_met = is_low_volume_env or volume_check_passed
            
            if volume_condition_met:
                # 根据环境确定RSI触发模式
                use_entry_trigger = is_low_volume_env and p['low_volume_on_entry']
                
                # 做空信号
                sell_trigger_met = False
                if use_entry_trigger: # 进场触发
                    if previous_rsi < p['overbought_level'] and current_rsi >= p['overbought_level']:
                        sell_trigger_met = True
                else: # 离场触发
                    if previous_rsi > p['overbought_level'] and current_rsi <= p['overbought_level']:
                        sell_trigger_met = True
                
                if sell_trigger_met:
                    signal = -1

                # 做多信号 (仅在无做空信号时)
                if signal == 0:
                    buy_trigger_met = False
                    if use_entry_trigger: # 进场触发
                        if previous_rsi > p['oversold_level'] and current_rsi <= p['oversold_level']:
                            buy_trigger_met = True
                    else: # 离场触发
                        if previous_rsi < p['oversold_level'] and current_rsi >= p['oversold_level']:
                            buy_trigger_met = True
                    
                    if buy_trigger_met:
                        signal = 1

        if signal != 0:
            df_out.iloc[-1, df_out.columns.get_loc('signal')] = signal
            df_out.iloc[-1, df_out.columns.get_loc('confidence')] = 100
            
        return df_out


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
            'id': 'rsi_with_price_change_filter',
            'name': 'RSI价格变化过滤策略',
            'class': RSIWithPriceChangeFilter,
            'description': '在RSI信号基础上，增加价格在N周期内的变化幅度作为额外过滤条件。',
            'parameters': [
                {'name': 'rsi_period', 'type': 'int', 'default': 14, 'min': 5, 'max': 30, 'step': 1, 'description': 'RSI和价格变化计算周期'},
                {'name': 'rsi_overbought', 'type': 'int', 'default': 70, 'min': 1, 'max': 99, 'step': 1, 'description': 'RSI超买阈值'},
                {'name': 'rsi_oversold', 'type': 'int', 'default': 30, 'min': 1, 'max': 99, 'step': 1, 'description': 'RSI超卖阈值'},
                {'name': 'price_change_percentage', 'type': 'float', 'default': 1.0, 'min': -10.0, 'max': 10.0, 'step': 0.1, 'description': '价格变化百分比阈值'},
                {'name': 'price_change_operator', 'type': 'select', 'default': '>', 'options': ['>', '<'], 'description': '价格变化比较操作符'},
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
            'id': 'enhanced_rsi', 'name': '加强版RSI策略(1分钟K线)', 'class': EnhancedRSIStrategy,
            'description': '专为1分钟K线和10分钟事件合约优化。智能识别市场趋势，在上涨趋势中只做多，下跌趋势中只做空，震荡市场中做逆势交易。结合快速MACD、成交量等多个指标提高胜率。',
            'parameters': [
                # --- RSI参数 (1分钟K线优化) ---
                {'name': 'rsi_period', 'type': 'int', 'default': 9, 'min': 5, 'max': 15, 'description': 'RSI周期(推荐9，适合1分钟)'},
                {'name': 'rsi_overbought', 'type': 'int', 'default': 75, 'min': 70, 'max': 85, 'description': 'RSI超买阈值（震荡市场）'},
                {'name': 'rsi_oversold', 'type': 'int', 'default': 25, 'min': 15, 'max': 30, 'description': 'RSI超卖阈值（震荡市场）'},

                # --- 趋势识别参数 (短期优化) ---
                {'name': 'ema_fast', 'type': 'int', 'default': 5, 'min': 3, 'max': 8, 'description': '快速EMA周期(5分钟)'},
                {'name': 'ema_slow', 'type': 'int', 'default': 15, 'min': 10, 'max': 20, 'description': '慢速EMA周期(15分钟)'},
                {'name': 'adx_period', 'type': 'int', 'default': 8, 'min': 6, 'max': 12, 'description': 'ADX周期(快速响应)'},
                {'name': 'adx_threshold', 'type': 'int', 'default': 20, 'min': 15, 'max': 30, 'description': 'ADX趋势阈值'},

                # --- 顺势交易RSI阈值 (10分钟事件合约优化) ---
                {'name': 'trend_rsi_buy_min', 'type': 'int', 'default': 35, 'min': 25, 'max': 45, 'description': '上涨趋势RSI买入下限'},
                {'name': 'trend_rsi_buy_max', 'type': 'int', 'default': 55, 'min': 45, 'max': 65, 'description': '上涨趋势RSI买入上限'},
                {'name': 'trend_rsi_sell_min', 'type': 'int', 'default': 45, 'min': 35, 'max': 55, 'description': '下跌趋势RSI卖出下限'},
                {'name': 'trend_rsi_sell_max', 'type': 'int', 'default': 65, 'min': 55, 'max': 75, 'description': '下跌趋势RSI卖出上限'},

                # --- 快速辅助指标参数 ---
                {'name': 'macd_fast', 'type': 'int', 'default': 5, 'min': 3, 'max': 8, 'description': 'MACD快线周期(快速设置)'},
                {'name': 'macd_slow', 'type': 'int', 'default': 13, 'min': 10, 'max': 18, 'description': 'MACD慢线周期'},
                {'name': 'macd_signal', 'type': 'int', 'default': 4, 'min': 3, 'max': 6, 'description': 'MACD信号线周期'},
                {'name': 'volume_period', 'type': 'int', 'default': 10, 'min': 5, 'max': 15, 'description': '成交量均线周期(10分钟)'},
                {'name': 'roc_period', 'type': 'int', 'default': 5, 'min': 3, 'max': 8, 'description': '价格动量周期(5分钟)'},
                {'name': 'atr_period', 'type': 'int', 'default': 8, 'min': 5, 'max': 12, 'description': 'ATR周期'},

                # --- 10分钟事件合约特殊参数 ---
                {'name': 'min_confidence_threshold', 'type': 'int', 'default': 65, 'min': 50, 'max': 80, 'description': '最低置信度阈值(%)'},
                {'name': 'trend_confirmation_periods', 'type': 'int', 'default': 3, 'min': 2, 'max': 5, 'description': '趋势确认周期数'},
            ]
        },
        {
            'id': 'flexible_signal', 'name': '灵活信号组合策略', 'class': RsiBollingerBandsStrategy,
            'description': '结合RSI,布林带和TD Sequential指标，可灵活配置。',
            'parameters': [
                # --- Indicator Switches ---
                {'name': 'use_bb', 'type': 'boolean', 'default': True, 'description': '启用布林带指标'},
                {'name': 'use_rsi', 'type': 'boolean', 'default': True, 'description': '启用RSI指标'},
                {'name': 'use_td_seq', 'type': 'boolean', 'default': False, 'description': '启用TD Sequential指标'},

                # --- Trigger Logic Control ---
                {'name': 'use_continuous_trigger', 'type': 'boolean', 'default': False, 'description': '使用持续触发逻辑（不推荐，可能导致频繁信号）'},

                # --- BB Params ---
                {'name': 'bb_period', 'type': 'int', 'default': 20, 'min': 10, 'max': 50, 'description': '布林带周期'},
                {'name': 'bb_std_dev', 'type': 'float', 'default': 2.0, 'min': 1.5, 'max': 3.0, 'step': 0.1, 'description': '布林带标准差'},

                # --- RSI Params ---
                {'name': 'rsi_period', 'type': 'int', 'default': 14, 'min': 5, 'max': 30, 'description': 'RSI周期'},
                {'name': 'rsi_overbought', 'type': 'int', 'default': 70, 'min': 55, 'max': 99, 'description': 'RSI超买阈值'},
                {'name': 'rsi_oversold', 'type': 'int', 'default': 30, 'min': 1, 'max': 45, 'description': 'RSI超卖阈值'},

                # --- TD Sequential Params ---
                {'name': 'td_seq_buy_setup', 'type': 'select', 'default': 9, 'options':[9, 13], 'description': 'TD买入计数'},
                {'name': 'td_seq_sell_setup', 'type': 'select', 'default': 9, 'options':[9, 13], 'description': 'TD卖出计数'},
            ]
        },
        {
            'id': 'rsi_momentum_exhaustion',
            'name': 'RSI动能衰竭与成交量确认',
            'class': RsiMomentumExhaustionStrategy,
            'description': '结合极端RSI反转和标准RSI反转+成交量确认的双层信号策略。',
            'parameters': [
                {'name': 'rsi_period', 'type': 'int', 'default': 14, 'min': 5, 'max': 50, 'step': 1, 'description': 'RSI计算周期'},
                {'name': 'overbought_level', 'type': 'int', 'default': 80, 'min': 60, 'max': 95, 'step': 1, 'description': '标准超买阈值'},
                {'name': 'oversold_level', 'type': 'int', 'default': 20, 'min': 5, 'max': 40, 'step': 1, 'description': '标准超卖阈值'},
                {'name': 'extreme_overbought_level', 'type': 'int', 'default': 90, 'min': 80, 'max': 99, 'step': 1, 'description': '极端超买阈值'},
                {'name': 'extreme_oversold_level', 'type': 'int', 'default': 10, 'min': 1, 'max': 20, 'step': 1, 'description': '极端超卖阈值'},
                {'name': 'vma_period', 'type': 'int', 'default': 20, 'min': 5, 'max': 100, 'step': 1, 'description': '成交量移动平均(VMA)周期'},
                {'name': 'volume_confirmation_multiplier', 'type': 'float', 'default': 0.9, 'min': 0.1, 'max': 2.0, 'step': 0.1, 'description': '成交量确认乘数 (<1表示缩量)'},
            ]
        },
        {
            'id': 'rsi_bollinger',
            'name': 'RSI布林带策略',
            'class': RsiBollingerStrategy,
            'description': '在RSI指标上应用布林带，根据RSI穿越其布林带轨道来产生信号。',
            'parameters': [
                {'name': 'rsi_period', 'type': 'int', 'default': 14, 'min': 5, 'max': 30, 'description': 'RSI计算周期'},
                {'name': 'bb_period', 'type': 'int', 'default': 20, 'min': 10, 'max': 50, 'description': 'RSI布林带周期'},
                {'name': 'bb_std_dev', 'type': 'float', 'default': 2.0, 'min': 1.0, 'max': 5.0, 'step': 0.1, 'description': 'RSI布林带标准差'},
            ]
        }
    ]
    ids = [s['id'] for s in strategies]
    if len(ids) != len(set(ids)):
        raise ValueError("策略ID不唯一！请检查 get_available_strategies 函数。")
    return strategies


class RSIWithPriceChangeFilter(Strategy):
    """
    结合RSI与价格变化过滤器的策略。
    仅当RSI条件和价格变化条件同时满足时，才生成买入信号。
    卖出信号（平仓）逻辑保持不变。
    """
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'price_change_percentage': 1.0,
            'price_change_operator': '>'
        }
        if params: default_params.update(params)
        super().__init__(default_params)
        self.name = "RSI价格变化过滤策略"
        self.min_history_periods = self.params['rsi_period'] + 1

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().calculate_all_indicators(df)
        rsi_period = self.params['rsi_period']
        
        # 1. 计算RSI
        df.ta.rsi(length=rsi_period, append=True)
        self.rsi_col = f"RSI_{rsi_period}"

        # 2. 计算价格变化百分比
        price_n_periods_ago = df['close'].shift(rsi_period)
        price_change = ((df['close'] - price_n_periods_ago) / price_n_periods_ago) * 100
        df['price_change_pct'] = price_change.fillna(0)
        
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df_with_indicators = self.calculate_all_indicators(df.copy())
        df_signaled = df_with_indicators.copy()
        df_signaled['signal'] = 0
        df_signaled['confidence'] = 0

        p = self.params
        rsi_col = getattr(self, 'rsi_col', f"RSI_{p['rsi_period']}")

        if rsi_col not in df_signaled.columns:
            logger.warning(f"({self.name}) RSI列 '{rsi_col}' 不存在，无法生成信号。")
            return df_signaled

        # 条件1: 价格变化过滤
        price_cond_met = pd.Series(True, index=df_signaled.index) # 默认满足
        if p['price_change_operator'] == '>':
            price_cond_met = df_signaled['price_change_pct'] > p['price_change_percentage']
        elif p['price_change_operator'] == '<':
            price_cond_met = df_signaled['price_change_pct'] < p['price_change_percentage']

        # 条件2: RSI穿越信号
        prev_rsi = df_signaled[rsi_col].shift(1)
        buy_rsi_crossover = (df_signaled[rsi_col] < p['rsi_oversold']) & (prev_rsi >= p['rsi_oversold'])
        sell_rsi_crossover = (df_signaled[rsi_col] > p['rsi_overbought']) & (prev_rsi <= p['rsi_overbought'])

        # 组合信号
        # 买入信号: RSI穿越超卖区 且 满足价格变化条件
        buy_signal = buy_rsi_crossover & price_cond_met
        
        # 卖出信号: RSI穿越超买区 (平仓逻辑，不应用价格过滤器)
        sell_signal = sell_rsi_crossover

        df_signaled.loc[buy_signal, 'signal'] = 1
        df_signaled.loc[sell_signal, 'signal'] = -1
        df_signaled.loc[buy_signal | sell_signal, 'confidence'] = 100

        return df_signaled

    def generate_signals_from_indicators_on_window(self, df_window_with_indicators: pd.DataFrame) -> pd.DataFrame:
        # 对于需要状态的策略或实时交易，这个方法更重要。
        # 这里我们用一个简单的方法，直接在窗口上调用完整的generate_signals，然后取最后一行。
        # 注意：这在性能上不是最优的，但对于回测框架是兼容的。
        if len(df_window_with_indicators) < self.min_history_periods:
            df_out = df_window_with_indicators.copy()
            if 'signal' not in df_out.columns: df_out['signal'] = 0
            if 'confidence' not in df_out.columns: df_out['confidence'] = 0
            return df_out

        # 在小窗口上运行向量化信号生成
        signaled_window = self.generate_signals(df_window_with_indicators)
        
        # 只保留最后一行信号
        df_out = df_window_with_indicators.copy()
        if 'signal' not in df_out.columns: df_out['signal'] = 0
        if 'confidence' not in df_out.columns: df_out['confidence'] = 0
        
        df_out.iloc[-1, df_out.columns.get_loc('signal')] = signaled_window.iloc[-1]['signal']
        df_out.iloc[-1, df_out.columns.get_loc('confidence')] = signaled_window.iloc[-1]['confidence']
        
        return df_out
