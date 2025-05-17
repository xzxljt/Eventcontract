# --- START OF FILE backtester.py ---

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from strategies import Strategy # 确保 strategies.py 在同一目录或PYTHONPATH中
from investment_strategies import BaseInvestmentStrategy, get_available_investment_strategies # 确保 investment_strategies.py

class Backtester:
    def __init__(self,
                 df: pd.DataFrame,
                 strategy: Strategy,
                 event_period: str,
                 confidence_threshold: float = 0,
                 investment_strategy_id: str = 'fixed',
                 investment_strategy_params: Optional[Dict[str, Any]] = None,
                 initial_balance: float = 1000.0,
                 profit_rate_pct: float = 80.0,
                 loss_rate_pct: float = 100.0,
                 min_investment_amount: float = 5.0,
                 max_investment_amount: float = 250.0,
                 kline_fetch_limit_for_signal: int = 100,
                 use_close_for_end_price: bool = True,
                 slippage_pct: float = 0.000002 # 默认滑点比例 (0.002%)
                ):
        self.df = df.copy() # 原始数据副本
        self.strategy = strategy
        self.event_period = event_period
        self.confidence_threshold = confidence_threshold
        self.initial_balance = initial_balance
        self.profit_rate = profit_rate_pct / 100.0
        self.loss_rate = loss_rate_pct / 100.0
        self.kline_fetch_limit_for_signal = kline_fetch_limit_for_signal
        self.use_close_for_end_price = use_close_for_end_price
        self.slippage_pct = slippage_pct

        available_inv_strategies = get_available_investment_strategies()
        inv_strategy_info = next((s for s in available_inv_strategies if s['id'] == investment_strategy_id), None)
        if not inv_strategy_info:
            raise ValueError(f"未找到ID为 {investment_strategy_id} 的投资策略")

        # 合并通用投资参数和策略特定参数
        merged_inv_params = {
            'minAmount': min_investment_amount, # 这些通用参数也可能被特定策略内部使用
            'maxAmount': max_investment_amount,
            **(investment_strategy_params or {}) # 策略特定参数覆盖通用参数（如果名称相同）
        }
        self.investment_strategy: BaseInvestmentStrategy = inv_strategy_info['class'](
            params=merged_inv_params, # 传递合并后的参数给策略构造函数
            min_amount=min_investment_amount, # 也将通用限制传递给基类（或策略自行处理）
            max_amount=max_investment_amount
        )
        self.results = None
        self.period_minutes = self._convert_period_to_minutes(event_period)
        self.df_with_indicators = None # 将在此处存储带有全局指标的DataFrame

        # 初始化日志
        print(f"初始化回测器: 事件周期={event_period} ({self.period_minutes}分钟), 预测策略={strategy.name}, 投资策略={self.investment_strategy.name}")
        print(f"  模拟实测信号生成窗口: {self.kline_fetch_limit_for_signal} 条K线")
        print(f"  事件结束价格使用: {'收盘价' if self.use_close_for_end_price else '开盘价'}")
        print(f"  滑点模拟百分比: {self.slippage_pct*100:.4f}%")
        print(f"  投资策略合并后参数 (用于实例化): {merged_inv_params}")
        print(f"  投资策略实际使用的 minAmount: {self.investment_strategy.min_amount}, maxAmount: {self.investment_strategy.max_amount}") # 确认策略实例中这些值的来源
        print(f"初始资金: {initial_balance}, 盈利比率: {profit_rate_pct}%, 亏损比率: {loss_rate_pct}%")
        if not self.df.empty:
            print(f"输入数据范围: {self.df.index.min()} 到 {self.df.index.max()}, 共 {len(self.df)} 条记录")
        else:
            print("警告: 回测器接收到的输入数据为空!")


    def _convert_period_to_minutes(self, period: str) -> int:
        if period == '10m': return 10
        elif period == '30m': return 30
        elif period == '1h': return 60
        elif period == '1d': return 1440 # 24 * 60
        else: raise ValueError(f"不支持的事件周期: {period}")

    def run(self) -> Dict[str, Any]:
        if self.df.empty:
            print("错误: DataFrame 为空，无法运行回测。")
            return self._calculate_statistics([], 0.0, 0, 0)

        # 确保索引是 DatetimeIndex 并且是单调递增的
        if not isinstance(self.df.index, pd.DatetimeIndex):
            try:
                self.df.index = pd.to_datetime(self.df.index)
                print("DataFrame 索引已转换为 DatetimeIndex。")
            except Exception as e:
                print(f"错误: DataFrame 索引无法转换为 DatetimeIndex: {e}")
                return self._calculate_statistics([], 0.0, 0, 0) # 如果转换失败，返回空统计

        if not self.df.index.is_monotonic_increasing:
            print("警告: DataFrame 索引未排序，正在排序...")
            self.df.sort_index(inplace=True)

        # --- 性能优化：一次性计算所有指标 ---
        # 某些策略可能需要基于整个历史数据集来计算指标（例如某些类型的趋势线）
        # 而另一些策略可能只关心当前窗口。这里的 calculate_all_indicators 应该能处理这两种情况。
        # 如果策略的指标计算不依赖于“未来”数据（即只用当前及之前的数据），这是安全的。
        print(f"Backtester: 开始全局计算指标 (策略: {self.strategy.name})...")
        try:
            # 传递 df 的副本以避免策略内部修改原始数据
            self.df_with_indicators = self.strategy.calculate_all_indicators(self.df.copy())
            print(f"Backtester: 全局指标计算完成。DataFrame列: {self.df_with_indicators.columns.tolist()}")
        except Exception as e_calc_all:
            print(f"错误: 全局计算指标时失败: {e_calc_all}")
            import traceback
            traceback.print_exc()
            return self._calculate_statistics([], 0.0, 0, 0)


        predictions = [] # 存储所有预测和交易结果的列表
        current_balance = self.initial_balance # 当前账户余额
        peak_balance = self.initial_balance # 用于计算最大回撤的峰值余额
        max_drawdown = 0.0 # 最大回撤比例

        self.investment_strategy.reset_state() # 重置投资策略的内部状态（例如马丁格尔序列）
        last_trade_result: Optional[bool] = None # 上一次交易的结果 (True for win, False for loss)
        consecutive_wins = 0; max_consecutive_wins = 0 # 连胜统计
        consecutive_losses = 0; max_consecutive_losses = 0 # 连败统计

        # 策略生成信号所需的最小历史K线条数，减去1是因为窗口本身包含当前K线
        min_data_for_signal_logic = getattr(self.strategy, 'min_history_periods', 2) # 策略可以定义这个值，默认为2
        # 确定回测开始的索引位置。需要足够的历史数据来填充第一个信号生成窗口。
        # self.kline_fetch_limit_for_signal 是模拟实时获取K线时，每次用于生成信号的窗口大小。
        start_index = self.kline_fetch_limit_for_signal -1

        if start_index >= len(self.df_with_indicators):
            print(f"错误: 数据量 ({len(self.df_with_indicators)}) 不足以形成一个大小为 {self.kline_fetch_limit_for_signal} 的信号生成窗口。")
            return self._calculate_statistics([], 0.0, 0, 0)

        print(f"Backtester: 开始逐K线评估信号 (窗口大小: {self.kline_fetch_limit_for_signal}). 将从索引 {start_index} 开始评估。")
        num_signals_evaluated = 0 # 评估了多少个K线（时间点）
        num_valid_signals_found = 0 # 找到了多少个满足置信度的有效预测

        # 从 start_index 开始遍历整个带有指标的 DataFrame
        for i in range(start_index, len(self.df_with_indicators)):
            num_signals_evaluated += 1
            current_kline_time = self.df_with_indicators.index[i] # 当前K线的时间戳（通常是K线开始时间）

            # 构建当前用于生成信号的K线窗口
            # 窗口的起始位置是 i - self.kline_fetch_limit_for_signal + 1
            # 窗口的结束位置是 i (包含当前K线)
            window_start_iloc = i - self.kline_fetch_limit_for_signal + 1
            df_window = self.df_with_indicators.iloc[window_start_iloc : i + 1]

            # 确保窗口数据量满足策略的最小需求
            if len(df_window) < min_data_for_signal_logic:
                continue # 跳过这个时间点，因为数据不足

            # 使用策略在当前窗口上生成信号
            # generate_signals_from_indicators_on_window 是一个优化方法，
            # 它假设指标已在 df_with_indicators 中，只在小窗口上应用信号逻辑。
            try:
                temp_signal_df_window = self.strategy.generate_signals_from_indicators_on_window(df_window.copy())
            except Exception as e_strat_win:
                print(f"警告: 在 {current_kline_time} 为窗口数据生成信号时策略出错 (优化路径): {e_strat_win}")
                continue # 如果策略出错，跳过这个时间点

            # 检查策略返回的信号DataFrame是否有效
            if temp_signal_df_window.empty or \
               'signal' not in temp_signal_df_window.columns or \
               'confidence' not in temp_signal_df_window.columns:
                continue # 如果没有信号或置信度列，跳过

            # 确保策略返回的DataFrame长度与输入窗口一致
            if len(temp_signal_df_window) != len(df_window):
                # 这是一个潜在问题，因为我们期望信号是针对窗口中的每一条K线（尤其是最后一条）
                print(f"警告: 策略 {self.strategy.name} 的 generate_signals_from_indicators_on_window 返回的DataFrame长度 ({len(temp_signal_df_window)}) 与输入窗口 ({len(df_window)}) 不匹配。")
                continue

            # 获取当前K线（即窗口的最后一条K线）的信号数据
            signal_data_for_current_kline = temp_signal_df_window.iloc[-1]
            signal_val = int(signal_data_for_current_kline.get('signal', 0)) # 信号方向: 1 (做多), -1 (做空), 0 (无信号)
            confidence_val = float(signal_data_for_current_kline.get('confidence', 0.0)) # 信号置信度

            # 如果没有信号或置信度低于阈值，则忽略
            if signal_val == 0 or confidence_val < self.confidence_threshold:
                continue

            num_valid_signals_found += 1 # 有效预测信号计数增加
            current_kline_data_row = self.df_with_indicators.iloc[i] # 获取当前K线的完整数据行

            signal_time_obj = current_kline_time # 信号发生时间
            signal_price_original = current_kline_data_row['close'] # 信号发生时的价格（使用收盘价）

            # 计算考虑滑点后的实际信号价格 (用于盈亏计算)
            effective_signal_price = signal_price_original
            if self.slippage_pct > 0:
                if signal_val == 1: # 做多时，成交价比信号价高
                    effective_signal_price = signal_price_original * (1 + self.slippage_pct)
                elif signal_val == -1: # 做空时，成交价比信号价低
                    effective_signal_price = signal_price_original * (1 - self.slippage_pct)

            # 计算事件合约的预期结束时间
            end_time_dt_expected = signal_time_obj + timedelta(minutes=self.period_minutes)
            # 在DataFrame中查找预期结束时间对应的K线索引位置
            # 'left' 表示如果精确时间点不存在，则取其右侧（之后）的第一个K线
            future_kline_index_pos = self.df_with_indicators.index.searchsorted(end_time_dt_expected, side='left')

            trade_occurred_successfully = False # 标记本次信号是否成功执行了交易
            current_investment_this_trade = 0.0 # 本次交易的投资额
            pnl_amount = 0.0 # 本次交易的盈亏金额
            prediction_correct_for_trade: Optional[bool] = None # 预测方向是否正确

            # 检查是否能找到未来的K线来确定事件结果
            if future_kline_index_pos < len(self.df_with_indicators):
                actual_end_kline_row = self.df_with_indicators.iloc[future_kline_index_pos] # 获取事件结束时的K线数据
                # 根据配置决定使用收盘价还是开盘价作为事件结束价格
                end_price = actual_end_kline_row['close'] if self.use_close_for_end_price else actual_end_kline_row['open']
                actual_end_time_obj = self.df_with_indicators.index[future_kline_index_pos] # 实际的事件结束时间

                # 计算价格变化百分比 (基于考虑滑点后的信号价格)
                price_change_pct = (end_price - effective_signal_price) / effective_signal_price * 100 if effective_signal_price != 0 else 0

                # 判断预测方向是否正确
                if signal_val == 1 and price_change_pct > 0: prediction_correct_for_trade = True
                elif signal_val == -1 and price_change_pct < 0: prediction_correct_for_trade = True
                else: prediction_correct_for_trade = False

                # 使用投资策略计算本次的投资金额
                calculated_investment_by_strategy = self.investment_strategy.calculate_investment(
                    current_balance=current_balance,
                    previous_trade_result=last_trade_result # 将上次交易结果传递给策略
                )

                # 投资金额的调整逻辑
                if current_balance < self.investment_strategy.min_amount: # 如果余额连最小投资额都不到
                    current_investment_this_trade = 0.0 # 则不投资
                elif current_balance < calculated_investment_by_strategy: # 如果余额不足以支持策略计算的投资额
                    # 则投资金额调整为：min(当前余额, 策略允许的最大投资额)，但不能低于策略允许的最小投资额
                    current_investment_this_trade = min(current_balance, self.investment_strategy.max_amount)
                    current_investment_this_trade = max(self.investment_strategy.min_amount, current_investment_this_trade)
                    # 再次确保调整后的投资额不超过当前余额（处理 minAmount > current_balance 的极端情况）
                    current_investment_this_trade = min(current_balance, current_investment_this_trade)
                else: # 余额充足
                    current_investment_this_trade = calculated_investment_by_strategy

                # 应用全局的最小/最大投资限制（这些是硬性限制，策略内部的min/max是建议）
                current_investment_this_trade = max(self.investment_strategy.min_amount, min(self.investment_strategy.max_amount, current_investment_this_trade))
                # 再次确保投资额不超过当前余额
                current_investment_this_trade = min(current_balance, current_investment_this_trade)

                # 如果最终确定的投资额仍然低于策略的最小允许投资额（例如余额过少），则不进行此交易
                if current_investment_this_trade < self.investment_strategy.min_amount:
                    current_investment_this_trade = 0.0

                # 如果确定要投资 (投资额 > 0)
                if current_investment_this_trade > 0:
                    trade_occurred_successfully = True # 标记交易成功发生
                    if prediction_correct_for_trade: # 如果预测正确
                        pnl_amount = current_investment_this_trade * self.profit_rate # 计算盈利
                        consecutive_wins += 1; consecutive_losses = 0 # 更新连胜/连败计数
                        max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                    else: # 如果预测错误
                        pnl_amount = - (current_investment_this_trade * self.loss_rate) # 计算亏损
                        consecutive_losses += 1; consecutive_wins = 0 # 更新连胜/连败计数
                        max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                    current_balance += pnl_amount # 更新账户余额

                if trade_occurred_successfully: # 如果交易发生了，更新上次交易结果
                    last_trade_result = prediction_correct_for_trade

                # 更新峰值余额和最大回撤
                if current_balance > peak_balance: peak_balance = current_balance
                drawdown = (peak_balance - current_balance) / peak_balance if peak_balance > 0 else 0.0
                if drawdown > max_drawdown: max_drawdown = drawdown

                # 记录本次预测/交易的详细信息
                predictions.append({
                    'signal_time': signal_time_obj, # 信号时间
                    'signal_price': float(signal_price_original), # 原始信号价格
                    'effective_signal_price_for_calc': float(effective_signal_price), # 计算滑点后的价格
                    'signal': signal_val, 'confidence': confidence_val, # 信号方向和置信度
                    'end_time_expected': end_time_dt_expected, # 预期结束时间
                    'end_time_actual': actual_end_time_obj, 'end_price': float(end_price), # 实际结束时间和价格
                    'price_change_pct': float(price_change_pct), # 价格变化百分比
                    'result': prediction_correct_for_trade if trade_occurred_successfully else None, # 交易结果
                    'investment_amount': float(round(current_investment_this_trade, 2)), # 投资额
                    'pnl_amount': float(round(pnl_amount, 2)), # 盈亏金额
                    'balance_after_trade': float(round(current_balance, 2)), # 交易后余额
                })
            else: # 如果找不到未来的K线来确定事件结果 (例如回测数据在此结束)
                predictions.append({
                    'signal_time': signal_time_obj,
                    'signal_price': float(signal_price_original),
                    'effective_signal_price_for_calc': float(effective_signal_price),
                    'signal': signal_val, 'confidence': confidence_val,
                    'end_time_expected': end_time_dt_expected,
                    'end_time_actual': None, 'end_price': None, 'price_change_pct': None,
                    'result': None, 'investment_amount': 0.0, 'pnl_amount': 0.0, # 未能验证，无投资和盈亏
                    'balance_after_trade': float(round(current_balance, 2)), # 余额保持不变
                })

            # 如果余额耗尽，提前结束回测
            if current_balance <= 0 and self.initial_balance > 0 : # 检查 initial_balance > 0 防止初始就为0时误判
                print(f"余额耗尽在 {signal_time_obj.strftime('%Y-%m-%d %H:%M:%S')}。停止回测。")
                break # 跳出主循环

        print(f"回测循环优化版结束。总共评估 {num_signals_evaluated} 个K线时间点，找到 {num_valid_signals_found} 个有效预测信号，生成 {len(predictions)} 条交易记录。")
        self.results = self._calculate_statistics(predictions, max_drawdown, max_consecutive_wins, max_consecutive_losses)
        return self.results

    def _calculate_statistics(self, predictions: List[Dict[str, Any]], max_drawdown_val: float, max_wins: int, max_losses: int) -> Dict[str, Any]:
        # 筛选出实际发生交易的预测记录
        actual_trades_predictions = [p for p in predictions if p.get('investment_amount', 0) > 0 and p.get('result') is not None]

        daily_pnl_summary = {} # 用于存储每日盈亏和交易统计

        # --- 处理没有实际交易发生的情况 ---
        if not actual_trades_predictions:
            # 即使没有实际交易，也可能存在预测信号，我们需要根据这些信号的时间范围来初始化 daily_pnl_summary
            if predictions:
                try:
                    # 筛选出包含有效 'signal_time' 的预测记录
                    valid_predictions_for_dates = [p for p in predictions if isinstance(p.get('signal_time'), (datetime, pd.Timestamp))]
                    if valid_predictions_for_dates:
                        # 将所有 signal_time 转换为 datetime 对象以进行比较
                        all_signal_times = [pd.to_datetime(p['signal_time']) for p in valid_predictions_for_dates]
                        start_date_obj = min(all_signal_times).date() # 回测期内的最早信号日期
                        end_date_obj = max(all_signal_times).date()   # 回测期内的最晚信号日期
                        current_event_date = start_date_obj
                        # 为回测范围内的每一天生成空的每日统计
                        while current_event_date <= end_date_obj:
                            daily_pnl_summary[current_event_date.strftime('%Y-%m-%d')] = {
                                'pnl': 0.0,
                                'trades': 0,
                                'balance': round(self.initial_balance, 2) # 如果没有交易，每日余额设为初始余额
                            }
                            current_event_date += timedelta(days=1)
                except Exception as e:
                    print(f"生成空每日盈亏时出错(no actual trades): {e}. Predictions sample: {predictions[:2]}")

            # 返回不包含交易的统计结果
            return {
                'total_predictions': 0, 'total_wins': 0, 'total_losses': 0, 'win_rate': 0.0,
                'long_predictions': 0, 'long_wins': 0, 'long_win_rate': 0.0,
                'short_predictions': 0, 'short_wins': 0, 'short_win_rate': 0.0,
                'initial_balance': float(round(self.initial_balance, 2)),
                'final_balance': float(round(self.initial_balance, 2)), # 如果没有交易，最终余额也是初始余额
                'total_investment_volume': 0.0, 'total_pnl_amount': 0.0, 'roi_percentage': 0.0,
                'average_pnl_per_trade': 0.0, 'profit_factor': 0.0,
                'max_drawdown_percentage': 0.0,
                'max_consecutive_wins': 0, 'max_consecutive_losses': 0,
                'predictions': [], # 交易记录列表为空
                'daily_pnl': daily_pnl_summary # 返回空的或基于初始余额的每日统计
            }

        # --- 如果有实际交易发生，则进行详细统计 ---
        pred_df = pd.DataFrame(actual_trades_predictions) # 将实际交易记录转换为DataFrame以便分析

        total_predictions_count = len(pred_df) # 总交易次数
        total_wins_count = pred_df['result'].sum() # 总胜利次数 (True计为1, False计为0)
        total_losses_count = total_predictions_count - total_wins_count # 总失败次数
        win_rate_val = total_wins_count / total_predictions_count * 100 if total_predictions_count > 0 else 0.0 # 胜率

        # 做多交易统计
        long_df = pred_df[pred_df['signal'] == 1]
        long_predictions_count = len(long_df)
        long_wins_count = long_df['result'].sum() if not long_df.empty else 0
        long_win_rate_val = long_wins_count / long_predictions_count * 100 if long_predictions_count > 0 else 0.0

        # 做空交易统计
        short_df = pred_df[pred_df['signal'] == -1]
        short_predictions_count = len(short_df)
        short_wins_count = short_df['result'].sum() if not short_df.empty else 0
        short_win_rate_val = short_wins_count / short_predictions_count * 100 if short_predictions_count > 0 else 0.0

        # 资金相关统计
        final_balance_val = pred_df['balance_after_trade'].iloc[-1] if not pred_df.empty else self.initial_balance # 最终余额
        total_investment_volume_val = pred_df['investment_amount'].sum() # 总投资额
        total_pnl_amount_val = pred_df['pnl_amount'].sum() # 总盈亏金额

        roi_percentage_val = (total_pnl_amount_val / self.initial_balance) * 100 if self.initial_balance > 0 else 0.0 # 投资回报率
        average_pnl_per_trade_val = total_pnl_amount_val / total_predictions_count if total_predictions_count > 0 else 0.0 # 平均每笔交易盈亏

        # 盈利因子计算
        total_profit_from_wins_val = pred_df[pred_df['pnl_amount'] > 0]['pnl_amount'].sum() # 所有盈利交易的总利润
        total_loss_from_losses_val = abs(pred_df[pred_df['pnl_amount'] < 0]['pnl_amount'].sum()) # 所有亏损交易的总损失（取绝对值）
        profit_factor_val = total_profit_from_wins_val / total_loss_from_losses_val if total_loss_from_losses_val > 0 else float('inf') # 盈利因子

        # --- 计算每日盈亏和余额 ---
        if not pred_df.empty:
            temp_df_for_daily = pred_df.copy() # 复制DataFrame以进行每日聚合
            # 确保 'signal_time' 列是 datetime 类型
            if not pd.api.types.is_datetime64_any_dtype(temp_df_for_daily['signal_time']):
                 temp_df_for_daily['signal_time'] = pd.to_datetime(temp_df_for_daily['signal_time'])

            # 将 signal_time 从 UTC 转换为 'Asia/Shanghai' 时区
            # 确保 signal_time 已经带有 UTC 时区信息
            if temp_df_for_daily['signal_time'].dt.tz is None:
                 # 如果没有时区信息，假设它是 UTC
                 temp_df_for_daily['signal_time'] = temp_df_for_daily['signal_time'].dt.tz_localize('UTC')
            temp_df_for_daily['signal_time'] = temp_df_for_daily['signal_time'].dt.tz_convert('Asia/Shanghai')

            temp_df_for_daily = temp_df_for_daily.set_index('signal_time') # 将信号时间设为索引以进行重采样

            # 定义每日聚合操作
            daily_aggregations = {
                'pnl_amount': 'sum', # 每日盈亏总和
                'investment_amount': 'count', # 每日交易次数 (使用 investment_amount 列计数，假设每笔有效交易都有投资额)
                'balance_after_trade': 'last'  # 每日的最后一个余额作为当日结束余额
            }
            # 按天重采样并聚合，使用本地时间进行分组
            daily_summary_raw = temp_df_for_daily.resample('D').agg(daily_aggregations)
            # 重命名聚合后的列
            daily_summary_raw.rename(columns={'pnl_amount': 'pnl', 'investment_amount': 'trades', 'balance_after_trade': 'balance'}, inplace=True)

            # 确保回测范围内的所有天都存在于每日统计中，即使某些天没有交易
            if predictions: # 使用完整的 'predictions' 列表（包含未交易的信号）来确定日期范围
                valid_predictions_for_dates = [p for p in predictions if isinstance(p.get('signal_time'), (datetime, pd.Timestamp))]
                if valid_predictions_for_dates:
                    # 将所有 signal_time 转换为带有 'Asia/Shanghai' 时区的 datetime 对象以进行比较
                    all_signal_times = []
                    for p in valid_predictions_for_dates:
                        dt_obj = pd.to_datetime(p['signal_time'])
                        if dt_obj.tz is None:
                            # 如果是时区无感的，先本地化为UTC，再转换为目标时区
                            all_signal_times.append(dt_obj.tz_localize('UTC').tz_convert('Asia/Shanghai'))
                        else:
                            # 如果是时区感知的，直接转换为目标时区
                            all_signal_times.append(dt_obj.tz_convert('Asia/Shanghai'))

                    overall_start_date = min(all_signal_times).normalize() # 规范化到日期开始（00:00:00），此时已是本地时间
                    overall_end_date = max(all_signal_times).normalize()
                    # 创建包含回测期间所有日期的 DatetimeIndex，使用本地时区
                    all_days_in_simulation_range = pd.date_range(start=overall_start_date, end=overall_end_date, freq='D', tz='Asia/Shanghai')

                    # 使用完整日期范围重新索引每日摘要，填充缺失日期的NaN
                    daily_summary_reindexed = daily_summary_raw.reindex(all_days_in_simulation_range)

                    # 填充NaN值
                    daily_summary_reindexed['pnl'] = daily_summary_reindexed['pnl'].fillna(0.0) # 没有交易的日期，pnl为0
                    daily_summary_reindexed['trades'] = daily_summary_reindexed['trades'].fillna(0).astype(int) # 没有交易的日期，trades为0

                    # 为没有交易的日期的 'balance' 填充正确的余额
                    # 逻辑：如果某天 balance 是 NaN，它应该等于前一天交易结束后的余额。
                    # 如果回测开始的第一天就没有交易，其余额应为初始余额。
                    last_known_balance = self.initial_balance # 从初始余额开始
                    for date_idx, row in daily_summary_reindexed.iterrows():
                        if pd.notna(row['balance']): # 如果当天有交易，记录其结束余额
                            last_known_balance = row['balance']
                        else: # 如果当天没有交易 (balance 是 NaN)
                            # 检查 pnl 和 trades 是否都为0，以确认是真正的无交易日
                            if row['pnl'] == 0.0 and row['trades'] == 0:
                                daily_summary_reindexed.loc[date_idx, 'balance'] = last_known_balance # 使用上一个已知余额
                            # else: 如果 pnl/trades 不为0但 balance 是 NaN，这是异常情况，目前余额会是 last_known_balance

                    # 再次检查并确保第一个日期的余额是有效的
                    if pd.isna(daily_summary_reindexed['balance'].iloc[0]):
                        daily_summary_reindexed['balance'].iloc[0] = self.initial_balance
                    # 向前填充以处理可能因其他原因（理论上不应发生）导致的中间NaN
                    daily_summary_reindexed['balance'] = daily_summary_reindexed['balance'].ffill()

                    # 将处理后的每日摘要存入 daily_pnl_summary 字典
                    for date_index, row_data in daily_summary_reindexed.iterrows():
                        daily_pnl_summary[date_index.strftime('%Y-%m-%d')] = {
                            'pnl': round(row_data['pnl'], 2),
                            'trades': int(row_data['trades']),
                            'balance': round(row_data['balance'], 2) if pd.notna(row_data['balance']) else round(self.initial_balance, 2) # 添加余额，如果仍为NaN则用初始余额
                        }
                else: # 如果 'predictions' 列表为空或不包含有效日期 (理论上不太可能在此阶段进入)
                    for date_index, row_data in daily_summary_raw.iterrows(): # 直接使用原始的每日摘要
                         daily_pnl_summary[date_index.strftime('%Y-%m-%d')] = {
                            'pnl': round(row_data['pnl'], 2),
                            'trades': int(row_data['trades']),
                            'balance': round(row_data['balance'], 2) if pd.notna(row_data['balance']) else round(self.initial_balance, 2)
                        }
            elif not daily_summary_raw.empty: # 如果 pred_df 为空，但 daily_summary_raw 不为空 (异常情况)
                 for date_index, row_data in daily_summary_raw.iterrows():
                    daily_pnl_summary[date_index.strftime('%Y-%m-%d')] = {
                        'pnl': round(row_data['pnl'], 2),
                        'trades': int(row_data['trades']),
                        'balance': round(row_data['balance'], 2) if pd.notna(row_data['balance']) else round(self.initial_balance, 2)
                    }

        # --- 准备最终返回的交易记录列表，确保数据类型正确 ---
        processed_predictions_list = []
        for pred_item_dict_orig in predictions: # 使用包含所有信号（包括未交易的）的 'predictions' 列表
            processed_pred_item = {}
            for key, value in pred_item_dict_orig.items():
                if isinstance(value, pd.Timestamp): # Pandas Timestamp -> Python datetime
                    processed_pred_item[key] = value.to_pydatetime()
                elif isinstance(value, np.datetime64): # NumPy datetime64 -> Python datetime
                    processed_pred_item[key] = pd.Timestamp(value).to_pydatetime()
                elif isinstance(value, (np.int64, np.int32, np.int16, np.int8)): # NumPy int -> Python int
                    processed_pred_item[key] = int(value)
                elif isinstance(value, (np.float64, np.float32)): # NumPy float -> Python float
                    processed_pred_item[key] = float(value)
                elif isinstance(value, np.bool_): # NumPy bool -> Python bool
                    processed_pred_item[key] = bool(value)
                else: # 其他类型保持不变
                    processed_pred_item[key] = value
            processed_predictions_list.append(processed_pred_item)

        # --- 构建最终的统计结果字典 ---
        return {
            'total_predictions': int(total_predictions_count), # 总交易次数（基于实际发生的交易）
            'total_wins': int(total_wins_count),
            'total_losses': int(total_losses_count),
            'win_rate': float(round(win_rate_val, 2)),
            'long_predictions': int(long_predictions_count),
            'long_wins': int(long_wins_count),
            'long_win_rate': float(round(long_win_rate_val, 2)),
            'short_predictions': int(short_predictions_count),
            'short_wins': int(short_wins_count),
            'short_win_rate': float(round(short_win_rate_val, 2)),
            'initial_balance': float(round(self.initial_balance, 2)),
            'final_balance': float(round(final_balance_val, 2)),
            'total_investment_volume': float(round(total_investment_volume_val, 2)),
            'total_pnl_amount': float(round(total_pnl_amount_val, 2)),
            'roi_percentage': float(round(roi_percentage_val, 2)),
            'average_pnl_per_trade': float(round(average_pnl_per_trade_val,2)),
            'profit_factor': float(round(profit_factor_val,2)) if profit_factor_val != float('inf') else 'inf', # 盈利因子可能为无穷大
            'max_drawdown_percentage': float(round(max_drawdown_val * 100, 2)), # 最大回撤百分比
            'max_consecutive_wins': int(max_wins), # 最大连胜次数
            'max_consecutive_losses': int(max_losses), # 最大连败次数
            'predictions': processed_predictions_list, # 完整的预测/交易记录列表
            'daily_pnl': daily_pnl_summary # 包含每日盈亏、交易次数和每日结束余额的字典
        }
# --- END OF FILE backtester.py ---