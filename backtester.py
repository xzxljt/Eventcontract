import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# --- START OF FILE backtester.py ---

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
from strategies import Strategy, get_available_strategies # 确保 strategies.py 在同一目录或PYTHONPATH中
from investment_strategies import BaseInvestmentStrategy, get_available_investment_strategies # 确保 investment_strategies.py
from binance_client import BinanceClient

logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self,
                 df: pd.DataFrame,
                 df_index_price: Optional[pd.DataFrame],
                 strategy: Strategy,
                 symbol: str,
                 interval: str,
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
                 slippage_pct: float = 0.0, # 默认滑点比例 (0.002%)
                 min_trade_interval_minutes: float = 0
                ):
        self.df = df.copy() # 原始数据副本
        self.df_index_price = df_index_price.copy() if df_index_price is not None else None
        self.strategy = strategy
        self.symbol = symbol
        self.interval = interval
        self.event_period = event_period
        self.confidence_threshold = confidence_threshold
        self.initial_balance = initial_balance
        self.profit_rate = profit_rate_pct / 100.0
        self.loss_rate = loss_rate_pct / 100.0
        self.kline_fetch_limit_for_signal = kline_fetch_limit_for_signal
        self.use_close_for_end_price = use_close_for_end_price
        self.slippage_pct = slippage_pct
        self.min_trade_interval_minutes = min_trade_interval_minutes

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
        self.binance_client = BinanceClient() # 初始化Binance客户端

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
        if period == '3m': return 3
        elif period == '10m': return 10
        elif period == '30m': return 30
        elif period == '1h': return 60
        elif period == '1d': return 1440 # 24 * 60
        else: raise ValueError(f"不支持的事件周期: {period}")

    def run(self) -> Dict[str, Any]:
        # logger.info("\n[DEBUG] --- Backtester Run Start ---")
        if self.df.empty:
            # logger.info("[DEBUG] 错误: DataFrame 为空，无法运行回测。")
            return self._calculate_statistics([], 0.0, 0, 0)

        # 获取指数价格数据
        # logger.info("[DEBUG] Backtester: 开始获取指数价格K线数据...")
        if self.df_index_price is None:
            logger.info("No pre-loaded index price data found, fetching from client...")
            try:
                # 假设self.df的索引是回测的完整时间范围
                start_time_ms = int(self.df.index.min().timestamp() * 1000)
                end_time_ms = int((self.df.index.max() + timedelta(minutes=self.period_minutes)).timestamp() * 1000)
                df_index_price = self.binance_client.get_index_price_klines(
                    symbol=self.symbol,
                    interval=self.interval,
                    start_time=start_time_ms,
                    end_time=end_time_ms
                )
                if df_index_price.empty:
                    raise ValueError("未能获取到回测时间范围内的指数价格数据。")
                # logger.info(f"[DEBUG] Backtester: 成功获取 {len(df_index_price)} 条指数价格K线。")
                # logger.info(f"[DEBUG]   指数价格数据范围: {df_index_price.index.min()} to {df_index_price.index.max()}")
            except Exception as e_idx_price:
                # logger.info(f"错误: 获取指数价格数据失败: {e_idx_price}")
                import traceback
                traceback.print_exc()
                return self._calculate_statistics([], 0.0, 0, 0)
        else:
            # logger.info("Using pre-loaded index price data.")
            df_index_price = self.df_index_price


        if not isinstance(self.df.index, pd.DatetimeIndex):
            try:
                self.df.index = pd.to_datetime(self.df.index)
                print("DataFrame 索引已转换为 DatetimeIndex。")
            except Exception as e:
                print(f"错误: DataFrame 索引无法转换为 DatetimeIndex: {e}")
                return self._calculate_statistics([], 0.0, 0, 0)

        if not self.df.index.is_monotonic_increasing:
            print("警告: DataFrame 索引未排序，正在排序...")
            self.df.sort_index(inplace=True)

        print(f"Backtester: 开始全局计算指标 (策略: {self.strategy.name})...")
        try:
            self.df_with_indicators = self.strategy.calculate_all_indicators(self.df.copy())
            print(f"Backtester: 全局指标计算完成。")
        except Exception as e_calc_all:
            print(f"错误: 全局计算指标时失败: {e_calc_all}")
            import traceback
            traceback.print_exc()
            return self._calculate_statistics([], 0.0, 0, 0)

        # 初始化状态变量
        predictions = []
        pending_trades = []
        current_balance = self.initial_balance
        peak_balance = self.initial_balance
        max_drawdown = 0.0
        self.investment_strategy.reset_state()
        last_trade_result: Optional[bool] = None
        consecutive_wins = 0
        max_consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_losses = 0
        last_actual_trade_time: Optional[datetime] = None

        min_data_for_signal_logic = getattr(self.strategy, 'min_history_periods', 2)
        start_index = self.kline_fetch_limit_for_signal - 1

        if start_index >= len(self.df_with_indicators):
            print(f"错误: 数据量 ({len(self.df_with_indicators)}) 不足以形成一个大小为 {self.kline_fetch_limit_for_signal} 的信号生成窗口。")
            return self._calculate_statistics([], 0.0, 0, 0)

        print(f"Backtester: 开始逐K线评估信号 (窗口大小: {self.kline_fetch_limit_for_signal}). 将从索引 {start_index} 开始评估。")
        num_signals_evaluated = 0
        num_valid_signals_found = 0

        # 主循环
        for i in range(start_index, len(self.df_with_indicators)):
            current_kline_time = self.df_with_indicators.index[i]
            # logger.info(f"\n[DEBUG] Main loop: i={i}, current_kline_time={current_kline_time}")
            state_updated_this_step = False # 标记本轮K线中，投资策略的状态是否已被更新
            
            # 1. 处理已结算的交易
            settled_trades_this_step = []
            remaining_pending_trades = []
            
            # 按预期结束时间排序，确保按时序处理
            pending_trades.sort(key=lambda x: x['end_time_expected'])

            for trade in pending_trades:
                if trade['end_time_expected'] <= current_kline_time:
                    # logger.info(f"  [DEBUG] Settling trade: SignalTime={trade['signal_time']}, ExpectedEnd={trade['end_time_expected']}")
                    # 查找结算价格
                    # --- MODIFICATION: Use Index Price for backtest validation ---
                    future_kline_index_pos = df_index_price.index.searchsorted(trade['end_time_expected'], side='left')
                    
                    if future_kline_index_pos < len(df_index_price):
                        actual_end_kline_row = df_index_price.iloc[future_kline_index_pos]
                        end_price = actual_end_kline_row['open'] # 指数价格只使用收盘价
                        actual_end_time_obj = df_index_price.index[future_kline_index_pos]
                        # logger.info(f"    [DEBUG] Settlement Found: EndPrice={end_price} at IndexTime={actual_end_time_obj}")
                        
                        price_change_pct = (end_price - trade['effective_signal_price_for_calc']) / trade['effective_signal_price_for_calc'] * 100 if trade['effective_signal_price_for_calc'] != 0 else 0
                        
                        prediction_correct = (trade['signal'] == 1 and price_change_pct > 0) or \
                                             (trade['signal'] == -1 and price_change_pct < 0)
                        
                        pnl_amount = 0.0
                        if prediction_correct:
                            pnl_amount = trade['investment_amount'] * self.profit_rate
                            consecutive_wins += 1
                            consecutive_losses = 0
                            max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                        else:
                            pnl_amount = - (trade['investment_amount'] * self.loss_rate)
                            consecutive_losses += 1
                            consecutive_wins = 0
                            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                        
                        current_balance += pnl_amount
                        last_trade_result = prediction_correct
                        state_updated_this_step = True

                        # 立即通知投资策略以更新其内部状态（例如，连胜/连败计数）。
                        # 我们调用此方法主要是为了其更新状态的副作用，而非获取返回值。
                        self.investment_strategy.calculate_investment(
                            current_balance=current_balance,
                            previous_trade_result=last_trade_result
                        )

                        if current_balance > peak_balance: peak_balance = current_balance
                        drawdown = (peak_balance - current_balance) / peak_balance if peak_balance > 0 else 0.0
                        if drawdown > max_drawdown: max_drawdown = drawdown

                        trade.update({
                            'end_time_actual': actual_end_time_obj,
                            'end_price': float(end_price),
                            'price_change_pct': float(price_change_pct),
                            'result': prediction_correct,
                            'pnl_amount': float(round(pnl_amount, 2)),
                            'balance_after_trade': float(round(current_balance, 2)),
                        })
                        predictions.append(trade)
                    else:
                        # 无法结算，保留在待处理列表或单独处理
                        # logger.info(f"    [DEBUG] Settlement FAILED: ExpectedEnd={trade['end_time_expected']} is out of index price data range (max: {df_index_price.index.max()}).")
                        trade.update({'result': None, 'pnl_amount': 0.0, 'balance_after_trade': current_balance})
                        predictions.append(trade) # 添加到最终结果，但标记为未验证
                else:
                    remaining_pending_trades.append(trade)

            pending_trades = remaining_pending_trades

            # 如果余额耗尽，提前结束
            if current_balance <= 0 and self.initial_balance > 0:
                print(f"余额耗尽在 {current_kline_time.strftime('%Y-%m-%d %H:%M:%S')}。停止回测。")
                break

            # 2. 生成新信号
            num_signals_evaluated += 1
            window_start_iloc = i - self.kline_fetch_limit_for_signal + 1
            df_window = self.df_with_indicators.iloc[window_start_iloc : i + 1]

            if len(df_window) < min_data_for_signal_logic: continue

            try:
                temp_signal_df_window = self.strategy.generate_signals_from_indicators_on_window(df_window.copy())
            except Exception as e_strat_win:
                print(f"警告: 在 {current_kline_time} 为窗口数据生成信号时策略出错: {e_strat_win}")
                continue

            if temp_signal_df_window.empty or 'signal' not in temp_signal_df_window.columns or 'confidence' not in temp_signal_df_window.columns:
                continue
            
            if len(temp_signal_df_window) != len(df_window):
                print(f"警告: 策略 {self.strategy.name} 返回的DataFrame长度与输入窗口不匹配。")
                continue

            signal_data = temp_signal_df_window.iloc[-1]
            signal_val = int(signal_data.get('signal', 0))
            confidence_val = float(signal_data.get('confidence', 0.0))

            if signal_val == 0 or confidence_val < self.confidence_threshold:
                continue

            # 3. 创建新交易并添加到待处理列表
            num_valid_signals_found += 1

            # --- MODIFICATION: Adjust entry price logic and remove slippage ---

            # Boundary check: ensure we can access the next kline
            if i + 1 >= len(self.df_with_indicators):
                # logger.warning(f"  [DEBUG] Signal generated on the last kline ({current_kline_time}). Cannot determine entry price from next kline. Skipping trade.")
                continue

            # --- 新增：最小开单间隔检查 (使用下一根K线的时间进行判断) ---
            next_kline_time = self.df_with_indicators.index[i + 1]
            if self.min_trade_interval_minutes > 0 and last_actual_trade_time is not None:
                time_diff = (next_kline_time - last_actual_trade_time).total_seconds() / 60
                if time_diff < self.min_trade_interval_minutes:
                    # print(f"DEBUG: Skipping signal at {current_kline_time} due to min interval. Last trade at {last_actual_trade_time}. Diff: {time_diff:.2f}m")
                    continue # 跳过此信号

            # Use the open price of the NEXT kline as the entry price
            effective_signal_price = self.df_with_indicators.iloc[i + 1]['open']
            # logger.info(f"  [DEBUG] Signal on kline {current_kline_time}. Entry planned for next kline {next_kline_time} at open price: {effective_signal_price}")

            # --- MODIFICATION: Fetch Index Price for signal recording (using the signal kline time) ---
            index_price_at_signal = None
            signal_kline_index_pos = df_index_price.index.searchsorted(current_kline_time, side='left')
            # logger.info(f"    [DEBUG] Searching for index price to record. Signal time: {current_kline_time}, Found position in df_index_price: {signal_kline_index_pos}")

            if signal_kline_index_pos < len(df_index_price):
                index_kline_time = df_index_price.index[signal_kline_index_pos]
                if abs((index_kline_time - current_kline_time).total_seconds()) < 60:
                    index_price_at_signal = df_index_price.iloc[signal_kline_index_pos]['close']
                    # logger.info(f"    [DEBUG] Successfully fetched index price for recording: {index_price_at_signal}")
                else:
                    pass
                    # logger.warning(f"    [DEBUG] Index kline time mismatch for recording. Falling back to market entry price.")
            else:
                pass
                #  logger.warning(f"    [DEBUG] Index search for recording out of bounds. Falling back to market entry price.")
            
            # --- END MODIFICATION ---

            # effective_signal_price is now the clean market entry price (next kline's open) without slippage
            
            # 如果投资策略的状态在本轮K线中已经被结算的交易更新过，
            # 我们传递 None 以避免重复计算最后一次的交易结果。
            # 否则，我们传递上一次（在之前K线中）的交易结果。
            result_for_calc = None if state_updated_this_step else last_trade_result
            investment_amount = self.investment_strategy.calculate_investment(
                current_balance=current_balance,
                previous_trade_result=result_for_calc
            )

            # 投资金额调整逻辑
            if current_balance < self.investment_strategy.min_amount:
                investment_amount = 0.0
            elif current_balance < investment_amount:
                investment_amount = min(current_balance, self.investment_strategy.max_amount)
                investment_amount = max(self.investment_strategy.min_amount, investment_amount)
                investment_amount = min(current_balance, investment_amount)
            
            investment_amount = min(current_balance, investment_amount)
            if investment_amount < self.investment_strategy.min_amount:
                investment_amount = 0.0

            # 仅当成功获取到指数价格且投资金额大于0时，才创建交易
            if investment_amount > 0 and index_price_at_signal is not None:
                new_trade = {
                    'signal_time': next_kline_time, # Use next kline's time
                    'signal_price': float(index_price_at_signal), # 显示价格和计算价格都使用指数价格
                    'effective_signal_price_for_calc': float(index_price_at_signal), # PnL is calculated from index price
                    'signal': signal_val,
                    'confidence': confidence_val,
                    'end_time_expected': next_kline_time + timedelta(minutes=self.period_minutes), # End time is based on actual trade time
                    'investment_amount': float(round(investment_amount, 2)),
                    'actual_trade_time': next_kline_time, # Record actual trade time as next kline's time
                }
                pending_trades.append(new_trade)
                last_actual_trade_time = next_kline_time # Update last trade time
                # logger.info(f"  [DEBUG] New trade generated and pending: ActualSignalTime={new_trade['signal_time']}, SignalPrice(Index)={new_trade['signal_price']}, MarketPrice(for PnL)={new_trade['effective_signal_price_for_calc']}, ExpectedEnd={new_trade['end_time_expected']}")

        # 循环结束后，处理所有剩余的待处理交易（标记为未验证）
        for trade in pending_trades:
            trade.update({'result': None, 'pnl_amount': 0.0, 'balance_after_trade': current_balance})
            predictions.append(trade)

        # logger.info(f"回测循环结束。总共评估 {num_signals_evaluated} 个K线时间点，找到 {num_valid_signals_found} 个有效预测信号，生成 {len(predictions)} 条交易记录。")
        self.results = self._calculate_statistics(predictions, max_drawdown, max_consecutive_wins, max_consecutive_losses)
        # logger.info("[DEBUG] --- Backtester Run End ---")
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
                                'balance': round(self.initial_balance, 2), # 如果没有交易，每日余额设为初始余额
                                'daily_return_pct': 0.0
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
                    if pd.isna(daily_summary_reindexed['balance'].iloc):
                        daily_summary_reindexed['balance'].iloc = self.initial_balance
                    # 向前填充以处理可能因其他原因（理论上不应发生）导致的中间NaN
                    daily_summary_reindexed['balance'] = daily_summary_reindexed['balance'].ffill()

                    # 计算前一天的余额
                    previous_day_balance = daily_summary_reindexed['balance'].shift(1).fillna(self.initial_balance)

                    # 计算每日回报率百分比，避免除以零的错误
                    daily_summary_reindexed['daily_return_pct'] = np.where(
                        previous_day_balance > 0,
                        ((daily_summary_reindexed['balance'] - previous_day_balance) / previous_day_balance) * 100,
                        0.0 # 如果前一天余额为0，则回报率为0
                    )

                    # 将处理后的每日摘要存入 daily_pnl_summary 字典
                    for date_index, row_data in daily_summary_reindexed.iterrows():
                        daily_pnl_summary[date_index.strftime('%Y-%m-%d')] = {
                            'pnl': round(row_data['pnl'], 2),
                            'trades': int(row_data['trades']),
                            'balance': round(row_data['balance'], 2) if pd.notna(row_data['balance']) else round(self.initial_balance, 2), # 添加余额，如果仍为NaN则用初始余额
                            'daily_return_pct': round(row_data['daily_return_pct'], 2) if pd.notna(row_data['daily_return_pct']) else 0.0
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
            # Exclude the internal calculation field before sending to frontend
            pred_item_dict_orig.pop('effective_signal_price_for_calc', None)
            
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

def run_single_backtest(df_kline: pd.DataFrame, 
                        df_index_price: pd.DataFrame, 
                        strategy_id: str, 
                        strategy_params: dict, 
                        backtest_config: dict) -> dict:
    """
    一个独立的函数，用于运行单次回测。
    这是为了兼容 optimizer.py 的 worker_function。
    """
    # 1. 获取策略类
    available_strategies = get_available_strategies()
    strategy_info = next((s for s in available_strategies if s['id'] == strategy_id), None)
    if not strategy_info:
        raise ValueError(f"Strategy with id '{strategy_id}' not found.")
    
    strategy_class = strategy_info['class']
    strategy = strategy_class(params=strategy_params)

    # 2. 准备 Backtester 的配置
    # 将 backtest_config 和 strategy_params 合并到 backtester 的初始化参数中
    # backtest_config 包含了 symbol, interval, event_period 等
    # strategy_params 包含了策略的参数，但 Backtester 不直接使用它们
    
    # 从 backtest_config 中提取 Backtester 需要的参数
    bt_params = {
        'df': df_kline,
        'df_index_price': df_index_price,
        'strategy': strategy,
        'symbol': backtest_config.get('symbol'),
        'interval': backtest_config.get('interval'),
        'event_period': backtest_config.get('event_period'),
        'initial_balance': backtest_config.get('initial_balance', 1000.0),
        'profit_rate_pct': backtest_config.get('profit_rate_pct', 80.0),
        'loss_rate_pct': backtest_config.get('loss_rate_pct', 100.0),
        'investment_strategy_id': backtest_config.get('investment_strategy_id', 'fixed'),
        'investment_strategy_params': backtest_config.get('investment_strategy_params'),
        'min_investment_amount': backtest_config.get('min_investment_amount', 5.0),
    }

    # 3. 实例化并运行 Backtester
    try:
        backtester = Backtester(**bt_params)
        results = backtester.run()
        
        # 确保返回的结果中包含参数信息，以便于分析
        results['params'] = strategy_params
        return results

    except Exception as e:
        logger.error(f"Backtest failed for strategy {strategy_id} with params {strategy_params}. Error: {e}", exc_info=True)
        # 返回一个表示失败的字典结构，与 optimizer.py 中的 worker_function 错误处理保持一致
        return {
            'params': strategy_params,
            'win_rate': -1,
            'roi_percentage': -1,
            'total_predictions': 0,
            'error': str(e)
        }