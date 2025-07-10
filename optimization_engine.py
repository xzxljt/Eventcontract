"""
策略参数优化引擎
提供网格搜索、多线程并行计算、综合评估体系等功能
"""

import os
import time
import uuid
import logging
import itertools
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

# 导入现有模块
from backtester import Backtester, run_single_backtest
from strategies import get_available_strategies
from investment_strategies import get_available_investment_strategies
from binance_client import BinanceClient

# 配置日志
logger = logging.getLogger(__name__)

@dataclass
class OptimizationProgress:
    """优化进度数据类"""
    optimization_id: str
    status: str = "not_started"  # not_started, running, completed, stopped, error
    current: int = 0
    total: int = 0
    percentage: float = 0.0
    elapsed_time: float = 0.0
    estimated_remaining: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None

@dataclass
class OptimizationResult:
    """单次优化结果数据类"""
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    backtest_details: Dict[str, Any]
    rank: int = 0
    composite_score: float = 0.0

class ParameterValidator:
    """参数验证器"""
    
    @staticmethod
    def validate_optimization_config(config: Dict[str, Any]) -> Tuple[bool, str]:
        """验证优化配置参数"""
        try:
            # 检查必需字段
            required_fields = ['symbol', 'interval', 'start_date', 'end_date', 
                             'strategy_id', 'strategy_params_ranges']
            for field in required_fields:
                if field not in config:
                    return False, f"缺少必需字段: {field}"
            
            # 验证日期格式
            try:
                start_date = pd.to_datetime(config['start_date'])
                end_date = pd.to_datetime(config['end_date'])
                if start_date >= end_date:
                    return False, "开始日期必须早于结束日期"
            except Exception as e:
                return False, f"日期格式错误: {e}"
            
            # 验证策略ID
            available_strategies = get_available_strategies()
            strategy_ids = [s['id'] for s in available_strategies]
            if config['strategy_id'] not in strategy_ids:
                return False, f"无效的策略ID: {config['strategy_id']}"
            
            # 验证参数范围
            param_ranges = config['strategy_params_ranges']
            if not isinstance(param_ranges, dict) or not param_ranges:
                return False, "策略参数范围不能为空"
            
            for param_name, param_range in param_ranges.items():
                if not isinstance(param_range, dict):
                    return False, f"参数 {param_name} 的范围定义必须是字典"
                
                required_range_fields = ['min', 'max', 'step']
                for field in required_range_fields:
                    if field not in param_range:
                        return False, f"参数 {param_name} 缺少范围字段: {field}"
                
                if param_range['min'] > param_range['max']:
                    return False, f"参数 {param_name} 的最小值不能大于最大值"
                
                if param_range['step'] <= 0:
                    return False, f"参数 {param_name} 的步长必须大于0"
            
            return True, "验证通过"
            
        except Exception as e:
            return False, f"验证过程中发生错误: {e}"
    
    @staticmethod
    def calculate_total_combinations(param_ranges: Dict[str, Dict[str, Any]]) -> int:
        """计算参数组合总数"""
        total = 1
        for param_name, param_range in param_ranges.items():
            min_val = param_range['min']
            max_val = param_range['max']
            step = param_range['step']
            
            # 计算该参数的可能值数量
            count = int((max_val - min_val) / step) + 1
            total *= count
        
        return total
    
    @staticmethod
    def validate_resource_limits(total_combinations: int, max_combinations: int = 10000) -> Tuple[bool, str]:
        """验证计算资源限制"""
        if total_combinations > max_combinations:
            return False, f"参数组合数量 ({total_combinations}) 超过限制 ({max_combinations})"
        
        if total_combinations <= 0:
            return False, "参数组合数量必须大于0"
        
        return True, "资源限制验证通过"

class EvaluationMetrics:
    """评估指标计算器"""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """初始化评估指标计算器"""
        self.weights = weights or {
            'total_return': 0.25,
            'win_rate': 0.25,
            'sharpe_ratio': 0.20,
            'max_drawdown': 0.15,
            'profit_factor': 0.15
        }
        
        # 确保权重总和为1
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.001:
            logger.warning(f"评估权重总和不为1 ({total_weight})，将进行标准化")
            for key in self.weights:
                self.weights[key] /= total_weight
    
    def calculate_all_metrics(self, backtest_result: Dict[str, Any]) -> Dict[str, float]:
        """计算所有评估指标"""
        try:
            metrics = {}
            
            # 基础指标
            metrics['total_return'] = backtest_result.get('roi_percentage', 0.0)
            metrics['win_rate'] = backtest_result.get('win_rate', 0.0)
            metrics['max_drawdown'] = backtest_result.get('max_drawdown_percentage', 0.0)
            metrics['profit_factor'] = self._safe_float(backtest_result.get('profit_factor', 0.0))
            metrics['total_trades'] = backtest_result.get('total_predictions', 0)
            
            # 计算夏普比率
            metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(backtest_result)
            
            # 计算年化收益率
            metrics['annualized_return'] = self._calculate_annualized_return(backtest_result)
            
            # 计算波动率
            metrics['volatility'] = self._calculate_volatility(backtest_result)
            
            # 计算平均交易持续时间
            metrics['avg_trade_duration'] = self._calculate_avg_trade_duration(backtest_result)
            
            # 计算VaR
            metrics['var_95'] = self._calculate_var(backtest_result)
            
            return metrics
            
        except Exception as e:
            logger.error(f"计算评估指标时发生错误: {e}")
            return self._get_default_metrics()
    
    def calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """计算综合评分 (0-100)"""
        try:
            score = 0.0
            
            # 收益率评分 (0-100)
            return_score = min(100, max(0, metrics.get('total_return', 0) * 2))  # 50%收益率 = 100分
            
            # 胜率评分 (0-100)
            win_rate_score = metrics.get('win_rate', 0)
            
            # 夏普比率评分 (0-100)
            sharpe_ratio = metrics.get('sharpe_ratio', 0)
            sharpe_score = min(100, max(0, sharpe_ratio * 25))  # 4.0夏普比率 = 100分
            
            # 最大回撤评分 (0-100，回撤越小分数越高)
            max_drawdown = metrics.get('max_drawdown', 0)
            drawdown_score = max(0, 100 - max_drawdown * 2)  # 50%回撤 = 0分
            
            # 盈利因子评分 (0-100)
            profit_factor = metrics.get('profit_factor', 0)
            if profit_factor == float('inf'):
                profit_factor_score = 100
            else:
                profit_factor_score = min(100, max(0, profit_factor * 25))  # 4.0盈利因子 = 100分
            
            # 加权计算综合评分
            score = (
                return_score * self.weights['total_return'] +
                win_rate_score * self.weights['win_rate'] +
                sharpe_score * self.weights['sharpe_ratio'] +
                drawdown_score * self.weights['max_drawdown'] +
                profit_factor_score * self.weights['profit_factor']
            )
            
            return round(score, 2)
            
        except Exception as e:
            logger.error(f"计算综合评分时发生错误: {e}")
            return 0.0
    
    def _safe_float(self, value: Union[float, str]) -> float:
        """安全转换为浮点数"""
        if value == 'inf' or value == float('inf'):
            return float('inf')
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def _calculate_sharpe_ratio(self, backtest_result: Dict[str, Any]) -> float:
        """计算夏普比率"""
        try:
            daily_pnl = backtest_result.get('daily_pnl', {})
            if not daily_pnl:
                return 0.0
            
            # 提取每日收益率
            daily_returns = []
            prev_balance = backtest_result.get('initial_balance', 1000.0)
            
            for date_str, day_data in daily_pnl.items():
                current_balance = day_data.get('balance', prev_balance)
                daily_return = (current_balance - prev_balance) / prev_balance if prev_balance > 0 else 0
                daily_returns.append(daily_return)
                prev_balance = current_balance
            
            if len(daily_returns) < 2:
                return 0.0
            
            # 计算夏普比率
            mean_return = np.mean(daily_returns)
            std_return = np.std(daily_returns, ddof=1)
            
            if std_return == 0:
                return 0.0
            
            # 假设无风险利率为0
            sharpe_ratio = mean_return / std_return * np.sqrt(252)  # 年化
            return round(sharpe_ratio, 4)
            
        except Exception as e:
            logger.warning(f"计算夏普比率失败: {e}")
            return 0.0
    
    def _calculate_annualized_return(self, backtest_result: Dict[str, Any]) -> float:
        """计算年化收益率"""
        try:
            total_return = backtest_result.get('roi_percentage', 0.0) / 100.0
            
            # 估算交易天数（简化处理）
            predictions = backtest_result.get('predictions', [])
            if not predictions:
                return 0.0
            
            # 从第一笔到最后一笔交易的天数
            first_trade = min(predictions, key=lambda x: x.get('signal_time', ''))
            last_trade = max(predictions, key=lambda x: x.get('signal_time', ''))
            
            first_date = pd.to_datetime(first_trade.get('signal_time'))
            last_date = pd.to_datetime(last_trade.get('signal_time'))
            
            days = (last_date - first_date).days
            if days <= 0:
                return 0.0
            
            years = days / 365.25
            annualized_return = (1 + total_return) ** (1 / years) - 1
            
            return round(annualized_return * 100, 2)
            
        except Exception as e:
            logger.warning(f"计算年化收益率失败: {e}")
            return 0.0
    
    def _calculate_volatility(self, backtest_result: Dict[str, Any]) -> float:
        """计算波动率"""
        try:
            daily_pnl = backtest_result.get('daily_pnl', {})
            if not daily_pnl:
                return 0.0
            
            # 计算每日收益率
            daily_returns = []
            prev_balance = backtest_result.get('initial_balance', 1000.0)
            
            for date_str, day_data in daily_pnl.items():
                current_balance = day_data.get('balance', prev_balance)
                daily_return = (current_balance - prev_balance) / prev_balance if prev_balance > 0 else 0
                daily_returns.append(daily_return)
                prev_balance = current_balance
            
            if len(daily_returns) < 2:
                return 0.0
            
            # 年化波动率
            volatility = np.std(daily_returns, ddof=1) * np.sqrt(252)
            return round(volatility * 100, 2)
            
        except Exception as e:
            logger.warning(f"计算波动率失败: {e}")
            return 0.0
    
    def _calculate_avg_trade_duration(self, backtest_result: Dict[str, Any]) -> float:
        """计算平均交易持续时间（分钟）"""
        try:
            predictions = backtest_result.get('predictions', [])
            if not predictions:
                return 0.0
            
            durations = []
            for trade in predictions:
                signal_time = pd.to_datetime(trade.get('signal_time'))
                end_time = pd.to_datetime(trade.get('end_time_actual', trade.get('end_time_expected')))
                
                if pd.notna(signal_time) and pd.notna(end_time):
                    duration = (end_time - signal_time).total_seconds() / 60  # 转换为分钟
                    durations.append(duration)
            
            if not durations:
                return 0.0
            
            return round(np.mean(durations), 2)
            
        except Exception as e:
            logger.warning(f"计算平均交易持续时间失败: {e}")
            return 0.0
    
    def _calculate_var(self, backtest_result: Dict[str, Any], confidence: float = 0.95) -> float:
        """计算VaR (Value at Risk)"""
        try:
            predictions = backtest_result.get('predictions', [])
            if not predictions:
                return 0.0
            
            # 提取每笔交易的盈亏百分比
            trade_returns = []
            for trade in predictions:
                pnl = trade.get('pnl_amount', 0)
                investment = trade.get('investment_amount', 0)
                if investment > 0:
                    return_pct = pnl / investment
                    trade_returns.append(return_pct)
            
            if not trade_returns:
                return 0.0
            
            # 计算VaR
            var_value = np.percentile(trade_returns, (1 - confidence) * 100)
            return round(var_value * 100, 2)
            
        except Exception as e:
            logger.warning(f"计算VaR失败: {e}")
            return 0.0
    
    def _get_default_metrics(self) -> Dict[str, float]:
        """获取默认指标值"""
        return {
            'total_return': 0.0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'profit_factor': 0.0,
            'total_trades': 0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'avg_trade_duration': 0.0,
            'var_95': 0.0
        }

class GridSearchOptimizer:
    """网格搜索优化器"""

    def __init__(self, max_workers: Optional[int] = None):
        """初始化网格搜索优化器"""
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.is_running = False
        self.should_stop = False
        self._lock = threading.Lock()

    def generate_parameter_combinations(self, param_ranges: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """生成所有参数组合"""
        try:
            param_names = list(param_ranges.keys())
            param_values = []

            for param_name in param_names:
                param_range = param_ranges[param_name]
                min_val = param_range['min']
                max_val = param_range['max']
                step = param_range['step']

                # 生成参数值序列
                values = []
                current = min_val
                while current <= max_val:
                    values.append(current)
                    current += step

                param_values.append(values)

            # 生成所有组合
            combinations = []
            for combination in itertools.product(*param_values):
                param_dict = dict(zip(param_names, combination))
                combinations.append(param_dict)

            logger.info(f"生成了 {len(combinations)} 个参数组合")
            return combinations

        except Exception as e:
            logger.error(f"生成参数组合时发生错误: {e}")
            return []

    def optimize_parallel(self,
                         combinations: List[Dict[str, Any]],
                         backtest_func: Callable,
                         progress_callback: Optional[Callable] = None) -> List[OptimizationResult]:
        """并行优化执行"""
        try:
            with self._lock:
                self.is_running = True
                self.should_stop = False

            results = []
            completed_count = 0
            total_count = len(combinations)

            logger.info(f"开始并行优化，共 {total_count} 个组合，使用 {self.max_workers} 个线程")

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交所有任务
                future_to_params = {
                    executor.submit(backtest_func, params): params
                    for params in combinations
                }

                # 处理完成的任务
                for future in as_completed(future_to_params):
                    # 检查是否需要停止
                    if self.should_stop:
                        logger.info("优化被用户停止")
                        break

                    params = future_to_params[future]

                    try:
                        backtest_result = future.result()

                        # 创建优化结果
                        if backtest_result and not backtest_result.get('error'):
                            # 计算评估指标
                            evaluator = EvaluationMetrics()
                            metrics = evaluator.calculate_all_metrics(backtest_result)
                            composite_score = evaluator.calculate_composite_score(metrics)

                            result = OptimizationResult(
                                parameters=params,
                                metrics=metrics,
                                backtest_details=backtest_result,
                                composite_score=composite_score
                            )
                            results.append(result)
                        else:
                            logger.warning(f"回测失败，参数: {params}, 错误: {backtest_result.get('error', '未知错误')}")

                    except Exception as e:
                        logger.error(f"处理回测结果时发生错误，参数: {params}, 错误: {e}")

                    completed_count += 1

                    # 调用进度回调
                    if progress_callback:
                        try:
                            progress_callback(completed_count, total_count)
                        except Exception as e:
                            logger.warning(f"进度回调函数执行失败: {e}")

            # 按综合评分排序
            results.sort(key=lambda x: x.composite_score, reverse=True)

            # 设置排名
            for i, result in enumerate(results):
                result.rank = i + 1

            logger.info(f"优化完成，共处理 {completed_count} 个组合，有效结果 {len(results)} 个")

            with self._lock:
                self.is_running = False

            return results

        except Exception as e:
            logger.error(f"并行优化过程中发生错误: {e}")
            with self._lock:
                self.is_running = False
            return []

    def stop_optimization(self):
        """停止优化"""
        with self._lock:
            self.should_stop = True
            logger.info("已请求停止优化")

    def is_optimization_running(self) -> bool:
        """检查优化是否正在运行"""
        with self._lock:
            return self.is_running

class ProgressTracker:
    """进度跟踪器"""

    def __init__(self):
        """初始化进度跟踪器"""
        self._progresses: Dict[str, OptimizationProgress] = {}
        self._lock = threading.Lock()

    def create_progress(self, optimization_id: str, total: int) -> OptimizationProgress:
        """创建新的进度跟踪"""
        with self._lock:
            progress = OptimizationProgress(
                optimization_id=optimization_id,
                status="running",
                total=total,
                start_time=datetime.now()
            )
            self._progresses[optimization_id] = progress
            return progress

    def update_progress(self, optimization_id: str, current: int):
        """更新进度"""
        with self._lock:
            if optimization_id in self._progresses:
                progress = self._progresses[optimization_id]
                progress.current = current
                progress.percentage = (current / progress.total * 100) if progress.total > 0 else 0

                # 计算已用时间
                if progress.start_time:
                    progress.elapsed_time = (datetime.now() - progress.start_time).total_seconds()

                # 估算剩余时间
                if current > 0 and progress.elapsed_time > 0:
                    avg_time_per_item = progress.elapsed_time / current
                    remaining_items = progress.total - current
                    progress.estimated_remaining = avg_time_per_item * remaining_items
                else:
                    progress.estimated_remaining = 0.0

    def complete_progress(self, optimization_id: str, status: str = "completed", error_message: Optional[str] = None):
        """完成进度跟踪"""
        with self._lock:
            if optimization_id in self._progresses:
                progress = self._progresses[optimization_id]
                progress.status = status
                progress.end_time = datetime.now()
                progress.error_message = error_message

                if status == "completed":
                    progress.current = progress.total
                    progress.percentage = 100.0
                    progress.estimated_remaining = 0.0

    def get_progress(self, optimization_id: str) -> Optional[OptimizationProgress]:
        """获取进度信息"""
        with self._lock:
            return self._progresses.get(optimization_id)

    def remove_progress(self, optimization_id: str):
        """移除进度跟踪"""
        with self._lock:
            if optimization_id in self._progresses:
                del self._progresses[optimization_id]

class ResultsManager:
    """结果管理器"""

    def __init__(self):
        """初始化结果管理器"""
        self._results: Dict[str, List[OptimizationResult]] = {}
        self._lock = threading.Lock()

    def store_results(self, optimization_id: str, results: List[OptimizationResult]):
        """存储优化结果"""
        with self._lock:
            self._results[optimization_id] = results

    def get_results(self, optimization_id: str, limit: Optional[int] = None) -> List[OptimizationResult]:
        """获取优化结果"""
        with self._lock:
            results = self._results.get(optimization_id, [])
            if limit:
                return results[:limit]
            return results

    def get_best_result(self, optimization_id: str) -> Optional[OptimizationResult]:
        """获取最佳结果"""
        with self._lock:
            results = self._results.get(optimization_id, [])
            if results:
                return results[0]  # 结果已按评分排序
            return None

    def generate_scatter_plot_data(self, optimization_id: str,
                                 x_axis: str = 'win_rate',
                                 y_axis: str = 'total_return') -> Dict[str, Any]:
        """生成散点图数据"""
        try:
            with self._lock:
                results = self._results.get(optimization_id, [])

            if not results:
                return {'x_axis': x_axis, 'y_axis': y_axis, 'points': []}

            points = []
            for result in results:
                x_value = result.metrics.get(x_axis, 0)
                y_value = result.metrics.get(y_axis, 0)

                point = {
                    'x': x_value,
                    'y': y_value,
                    'parameters': result.parameters,
                    'rank': result.rank,
                    'composite_score': result.composite_score,
                    'metrics': result.metrics
                }
                points.append(point)

            return {
                'x_axis': x_axis,
                'y_axis': y_axis,
                'points': points
            }

        except Exception as e:
            logger.error(f"生成散点图数据时发生错误: {e}")
            return {'x_axis': x_axis, 'y_axis': y_axis, 'points': []}

    def export_results_to_csv(self, optimization_id: str, file_path: str) -> bool:
        """导出结果到CSV文件"""
        try:
            with self._lock:
                results = self._results.get(optimization_id, [])

            if not results:
                logger.warning(f"没有找到优化结果: {optimization_id}")
                return False

            # 准备数据
            data = []
            for result in results:
                row = {
                    'rank': result.rank,
                    'composite_score': result.composite_score,
                    **result.parameters,
                    **result.metrics
                }
                data.append(row)

            # 创建DataFrame并保存
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False, encoding='utf-8-sig')

            logger.info(f"结果已导出到: {file_path}")
            return True

        except Exception as e:
            logger.error(f"导出结果到CSV时发生错误: {e}")
            return False

    def remove_results(self, optimization_id: str):
        """移除结果"""
        with self._lock:
            if optimization_id in self._results:
                del self._results[optimization_id]

class OptimizationEngine:
    """策略参数优化引擎主类"""

    def __init__(self):
        """初始化优化引擎"""
        self.optimizer = GridSearchOptimizer()
        self.progress_tracker = ProgressTracker()
        self.results_manager = ResultsManager()
        self.binance_client = BinanceClient()

        # 活跃的优化任务
        self._active_optimizations: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    def optimize_strategy(self, optimization_config: Dict[str, Any]) -> str:
        """开始策略优化"""
        try:
            # 1. 验证配置
            is_valid, error_msg = ParameterValidator.validate_optimization_config(optimization_config)
            if not is_valid:
                raise ValueError(f"配置验证失败: {error_msg}")

            # 2. 计算参数组合数量
            param_ranges = optimization_config['strategy_params_ranges']
            total_combinations = ParameterValidator.calculate_total_combinations(param_ranges)

            # 3. 验证资源限制
            max_combinations = optimization_config.get('max_combinations', 10000)
            is_valid, error_msg = ParameterValidator.validate_resource_limits(total_combinations, max_combinations)
            if not is_valid:
                raise ValueError(f"资源限制验证失败: {error_msg}")

            # 4. 生成优化ID
            optimization_id = str(uuid.uuid4())

            # 5. 创建进度跟踪
            self.progress_tracker.create_progress(optimization_id, total_combinations)

            # 6. 启动优化线程
            optimization_thread = threading.Thread(
                target=self._run_optimization,
                args=(optimization_id, optimization_config),
                daemon=True
            )

            with self._lock:
                self._active_optimizations[optimization_id] = optimization_thread

            optimization_thread.start()

            logger.info(f"优化任务已启动，ID: {optimization_id}, 参数组合数: {total_combinations}")
            return optimization_id

        except Exception as e:
            logger.error(f"启动优化时发生错误: {e}")
            raise

    def get_optimization_progress(self, optimization_id: str) -> Optional[Dict[str, Any]]:
        """获取优化进度"""
        try:
            progress = self.progress_tracker.get_progress(optimization_id)
            if not progress:
                return None

            return {
                'optimization_id': progress.optimization_id,
                'status': progress.status,
                'current': progress.current,
                'total': progress.total,
                'percentage': progress.percentage,
                'elapsed_time': progress.elapsed_time,
                'estimated_remaining': progress.estimated_remaining,
                'start_time': progress.start_time.isoformat() if progress.start_time else None,
                'end_time': progress.end_time.isoformat() if progress.end_time else None,
                'error_message': progress.error_message
            }

        except Exception as e:
            logger.error(f"获取优化进度时发生错误: {e}")
            return None

    def stop_optimization(self, optimization_id: str) -> bool:
        """停止优化"""
        try:
            # 停止优化器
            self.optimizer.stop_optimization()

            # 更新进度状态
            self.progress_tracker.complete_progress(optimization_id, "stopped")

            # 清理活跃优化记录
            with self._lock:
                if optimization_id in self._active_optimizations:
                    del self._active_optimizations[optimization_id]

            logger.info(f"优化任务已停止，ID: {optimization_id}")
            return True

        except Exception as e:
            logger.error(f"停止优化时发生错误: {e}")
            return False

    def get_optimization_results(self, optimization_id: str, limit: Optional[int] = None) -> Dict[str, Any]:
        """获取优化结果"""
        try:
            # 获取进度信息
            progress = self.progress_tracker.get_progress(optimization_id)
            if not progress:
                return {'error': '未找到优化任务'}

            # 获取结果
            results = self.results_manager.get_results(optimization_id, limit)
            best_result = self.results_manager.get_best_result(optimization_id)

            # 生成散点图数据
            scatter_plot_data = self.results_manager.generate_scatter_plot_data(optimization_id)

            # 构建返回数据
            response = {
                'optimization_id': optimization_id,
                'status': progress.status,
                'progress': {
                    'current': progress.current,
                    'total': progress.total,
                    'percentage': progress.percentage,
                    'elapsed_time': progress.elapsed_time,
                    'estimated_remaining': progress.estimated_remaining
                },
                'summary': {
                    'total_combinations_tested': progress.current,
                    'valid_results': len(results),
                    'optimization_time': progress.elapsed_time,
                    'best_score': best_result.composite_score if best_result else 0.0
                },
                'scatter_plot_data': scatter_plot_data
            }

            # 添加最佳结果
            if best_result:
                response['best_result'] = {
                    'parameters': best_result.parameters,
                    'metrics': best_result.metrics,
                    'composite_score': best_result.composite_score,
                    'rank': best_result.rank
                }

            # 添加所有结果（如果请求）
            if limit is None or limit > 0:
                response['all_results'] = [
                    {
                        'parameters': result.parameters,
                        'metrics': result.metrics,
                        'composite_score': result.composite_score,
                        'rank': result.rank
                    }
                    for result in results
                ]

            return response

        except Exception as e:
            logger.error(f"获取优化结果时发生错误: {e}")
            return {'error': str(e)}

    def _run_optimization(self, optimization_id: str, optimization_config: Dict[str, Any]):
        """运行优化的内部方法"""
        try:
            logger.info(f"开始执行优化任务: {optimization_id}")

            # 1. 准备数据
            df_kline, df_index_price = self._prepare_data(optimization_config)
            if df_kline is None or df_index_price is None:
                raise ValueError("数据准备失败")

            # 2. 生成参数组合
            param_ranges = optimization_config['strategy_params_ranges']
            combinations = self.optimizer.generate_parameter_combinations(param_ranges)

            if not combinations:
                raise ValueError("未能生成有效的参数组合")

            # 3. 创建回测函数
            backtest_func = self._create_backtest_function(
                df_kline, df_index_price, optimization_config
            )

            # 4. 创建进度回调
            def progress_callback(current: int, total: int):
                self.progress_tracker.update_progress(optimization_id, current)

            # 5. 执行并行优化
            results = self.optimizer.optimize_parallel(
                combinations, backtest_func, progress_callback
            )

            # 6. 存储结果
            self.results_manager.store_results(optimization_id, results)

            # 7. 完成进度跟踪
            self.progress_tracker.complete_progress(optimization_id, "completed")

            logger.info(f"优化任务完成: {optimization_id}, 有效结果: {len(results)}")

        except Exception as e:
            logger.error(f"优化任务执行失败: {optimization_id}, 错误: {e}")
            self.progress_tracker.complete_progress(optimization_id, "error", str(e))

        finally:
            # 清理活跃优化记录
            with self._lock:
                if optimization_id in self._active_optimizations:
                    del self._active_optimizations[optimization_id]

    def _prepare_data(self, optimization_config: Dict[str, Any]) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """准备回测数据"""
        try:
            symbol = optimization_config['symbol']
            interval = optimization_config['interval']
            start_date = pd.to_datetime(optimization_config['start_date'])
            end_date = pd.to_datetime(optimization_config['end_date'])

            logger.info(f"准备数据: {symbol}, {interval}, {start_date} - {end_date}")

            # 转换日期为时间戳
            start_timestamp = int(start_date.timestamp() * 1000)
            end_timestamp = int(end_date.timestamp() * 1000)

            # 获取K线数据
            df_kline = self.binance_client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_time=start_timestamp,
                end_time=end_timestamp
            )

            if df_kline is None or df_kline.empty:
                logger.error("获取K线数据失败")
                return None, None

            # 获取指数价格数据（用于回测验证）
            df_index_price = self.binance_client.get_index_price_klines(
                symbol=symbol,
                interval=interval,
                start_time=start_timestamp,
                end_time=end_timestamp
            )

            if df_index_price is None or df_index_price.empty:
                logger.error("获取指数价格数据失败")
                return None, None

            # 应用时间排除逻辑
            df_kline = self._apply_time_exclusions(df_kline, optimization_config)
            df_index_price = self._apply_time_exclusions(df_index_price, optimization_config)

            logger.info(f"数据准备完成，K线数据: {len(df_kline)} 条，指数数据: {len(df_index_price)} 条")
            return df_kline, df_index_price

        except Exception as e:
            logger.error(f"准备数据时发生错误: {e}")
            return None, None

    def _apply_time_exclusions(self, df: pd.DataFrame, optimization_config: Dict[str, Any]) -> pd.DataFrame:
        """应用时间排除逻辑"""
        try:
            if df.empty:
                return df

            # 排除特定时间段
            exclude_time_ranges = optimization_config.get('exclude_time_ranges', [])
            for time_range in exclude_time_ranges:
                start_time = time_range['start']
                end_time = time_range['end']

                # 过滤时间段
                mask = ~((df.index.time >= pd.to_datetime(start_time).time()) &
                        (df.index.time <= pd.to_datetime(end_time).time()))
                df = df[mask]

            # 排除特定星期
            exclude_weekdays = optimization_config.get('exclude_weekdays', [])
            if exclude_weekdays:
                mask = ~df.index.weekday.isin(exclude_weekdays)
                df = df[mask]

            return df

        except Exception as e:
            logger.warning(f"应用时间排除逻辑时发生错误: {e}")
            return df

    def _create_backtest_function(self,
                                df_kline: pd.DataFrame,
                                df_index_price: pd.DataFrame,
                                optimization_config: Dict[str, Any]) -> Callable:
        """创建回测函数"""
        def backtest_func(strategy_params: Dict[str, Any]) -> Dict[str, Any]:
            try:
                # 准备回测配置
                backtest_config = {
                    'symbol': optimization_config['symbol'],
                    'interval': optimization_config['interval'],
                    'event_period': optimization_config.get('event_period', '10m'),
                    'initial_balance': optimization_config.get('initial_balance', 1000.0),
                    'profit_rate_pct': optimization_config.get('profit_rate_pct', 80.0),
                    'loss_rate_pct': optimization_config.get('loss_rate_pct', 100.0),
                    'investment_strategy_id': optimization_config.get('investment_strategy_id', 'fixed'),
                    'investment_strategy_params': optimization_config.get('investment_strategy_params', {'amount': 20.0}),
                    'min_investment_amount': optimization_config.get('min_investment_amount', 5.0),
                    'max_investment_amount': optimization_config.get('max_investment_amount', 250.0),
                }

                # 执行回测
                result = run_single_backtest(
                    df_kline=df_kline.copy(),
                    df_index_price=df_index_price.copy(),
                    strategy_id=optimization_config['strategy_id'],
                    strategy_params=strategy_params,
                    backtest_config=backtest_config
                )

                # 检查最小交易次数
                min_trades = optimization_config.get('min_trades', 10)
                if result.get('total_predictions', 0) < min_trades:
                    result['error'] = f"交易次数不足 ({result.get('total_predictions', 0)} < {min_trades})"

                return result

            except Exception as e:
                logger.error(f"回测执行失败，参数: {strategy_params}, 错误: {e}")
                return {
                    'params': strategy_params,
                    'error': str(e),
                    'win_rate': 0,
                    'roi_percentage': 0,
                    'total_predictions': 0
                }

        return backtest_func

    def get_strategy_parameter_ranges(self, strategy_id: str) -> Dict[str, Any]:
        """获取策略的参数范围信息"""
        try:
            available_strategies = get_available_strategies()
            strategy_info = next((s for s in available_strategies if s['id'] == strategy_id), None)

            if not strategy_info:
                return {'error': f'策略 {strategy_id} 不存在'}

            return {
                'strategy_id': strategy_id,
                'strategy_name': strategy_info['name'],
                'description': strategy_info['description'],
                'parameters': strategy_info['parameters']
            }

        except Exception as e:
            logger.error(f"获取策略参数范围时发生错误: {e}")
            return {'error': str(e)}

    def get_parameter_presets(self, strategy_id: str) -> Dict[str, Any]:
        """获取参数预设"""
        try:
            presets = {
                'simple_rsi': {
                    'conservative': {
                        'description': '保守型参数范围，适合稳定收益',
                        'ranges': {
                            'rsi_period': {'min': 12, 'max': 16, 'step': 2},
                            'rsi_overbought': {'min': 75, 'max': 85, 'step': 5},
                            'rsi_oversold': {'min': 15, 'max': 25, 'step': 5}
                        }
                    },
                    'balanced': {
                        'description': '平衡型参数范围，收益与风险平衡',
                        'ranges': {
                            'rsi_period': {'min': 10, 'max': 20, 'step': 2},
                            'rsi_overbought': {'min': 65, 'max': 80, 'step': 5},
                            'rsi_oversold': {'min': 20, 'max': 35, 'step': 5}
                        }
                    },
                    'aggressive': {
                        'description': '激进型参数范围，追求高收益',
                        'ranges': {
                            'rsi_period': {'min': 5, 'max': 15, 'step': 2},
                            'rsi_overbought': {'min': 60, 'max': 75, 'step': 5},
                            'rsi_oversold': {'min': 25, 'max': 40, 'step': 5}
                        }
                    }
                },
                'enhanced_rsi': {
                    'conservative': {
                        'description': '保守型参数范围，适合稳定收益',
                        'ranges': {
                            'rsi_period': {'min': 8, 'max': 12, 'step': 2},
                            'rsi_overbought': {'min': 75, 'max': 85, 'step': 5},
                            'rsi_oversold': {'min': 15, 'max': 25, 'step': 5},
                            'ema_fast': {'min': 4, 'max': 6, 'step': 1},
                            'ema_slow': {'min': 12, 'max': 18, 'step': 3}
                        }
                    },
                    'balanced': {
                        'description': '平衡型参数范围，收益与风险平衡',
                        'ranges': {
                            'rsi_period': {'min': 6, 'max': 12, 'step': 2},
                            'rsi_overbought': {'min': 70, 'max': 80, 'step': 5},
                            'rsi_oversold': {'min': 20, 'max': 30, 'step': 5},
                            'ema_fast': {'min': 3, 'max': 7, 'step': 1},
                            'ema_slow': {'min': 10, 'max': 20, 'step': 2}
                        }
                    },
                    'aggressive': {
                        'description': '激进型参数范围，追求高收益',
                        'ranges': {
                            'rsi_period': {'min': 5, 'max': 10, 'step': 1},
                            'rsi_overbought': {'min': 65, 'max': 75, 'step': 5},
                            'rsi_oversold': {'min': 25, 'max': 35, 'step': 5},
                            'ema_fast': {'min': 3, 'max': 8, 'step': 1},
                            'ema_slow': {'min': 8, 'max': 15, 'step': 1}
                        }
                    }
                },
                'flexible_signal': {
                    'conservative': {
                        'description': '保守型参数范围，适合稳定收益',
                        'ranges': {
                            'bb_period': {'min': 18, 'max': 25, 'step': 2},
                            'bb_std_dev': {'min': 2.0, 'max': 2.5, 'step': 0.1},
                            'rsi_period': {'min': 12, 'max': 18, 'step': 2},
                            'rsi_overbought': {'min': 75, 'max': 85, 'step': 5},
                            'rsi_oversold': {'min': 15, 'max': 25, 'step': 5}
                        }
                    },
                    'balanced': {
                        'description': '平衡型参数范围，收益与风险平衡',
                        'ranges': {
                            'bb_period': {'min': 15, 'max': 25, 'step': 2},
                            'bb_std_dev': {'min': 1.8, 'max': 2.5, 'step': 0.1},
                            'rsi_period': {'min': 10, 'max': 20, 'step': 2},
                            'rsi_overbought': {'min': 65, 'max': 80, 'step': 5},
                            'rsi_oversold': {'min': 20, 'max': 35, 'step': 5}
                        }
                    },
                    'aggressive': {
                        'description': '激进型参数范围，追求高收益',
                        'ranges': {
                            'bb_period': {'min': 10, 'max': 20, 'step': 2},
                            'bb_std_dev': {'min': 1.5, 'max': 2.2, 'step': 0.1},
                            'rsi_period': {'min': 8, 'max': 16, 'step': 2},
                            'rsi_overbought': {'min': 60, 'max': 75, 'step': 5},
                            'rsi_oversold': {'min': 25, 'max': 40, 'step': 5}
                        }
                    }
                }
            }

            strategy_presets = presets.get(strategy_id, {})
            if not strategy_presets:
                return {'error': f'策略 {strategy_id} 没有预设参数'}

            return {
                'strategy_id': strategy_id,
                'presets': strategy_presets
            }

        except Exception as e:
            logger.error(f"获取参数预设时发生错误: {e}")
            return {'error': str(e)}

    def export_optimization_results(self, optimization_id: str, file_path: str, format: str = 'csv') -> bool:
        """导出优化结果"""
        try:
            if format.lower() == 'csv':
                return self.results_manager.export_results_to_csv(optimization_id, file_path)
            else:
                logger.error(f"不支持的导出格式: {format}")
                return False

        except Exception as e:
            logger.error(f"导出优化结果时发生错误: {e}")
            return False

    def cleanup_optimization(self, optimization_id: str):
        """清理优化相关数据"""
        try:
            self.progress_tracker.remove_progress(optimization_id)
            self.results_manager.remove_results(optimization_id)

            with self._lock:
                if optimization_id in self._active_optimizations:
                    del self._active_optimizations[optimization_id]

            logger.info(f"已清理优化数据: {optimization_id}")

        except Exception as e:
            logger.error(f"清理优化数据时发生错误: {e}")

# 全局优化引擎实例
optimization_engine = OptimizationEngine()

def get_optimization_engine() -> OptimizationEngine:
    """获取全局优化引擎实例"""
    return optimization_engine
