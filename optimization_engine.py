"""
策略参数优化引擎
提供网格搜索、多线程并行计算、综合评估体系等功能
"""

import os
import uuid
import logging
import itertools
import threading
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

# 导入现有模块
from backtester import run_single_backtest
from strategies import get_available_strategies
from binance_client import BinanceClient
from optimization_database import OptimizationDatabase, OptimizationRecord, get_optimization_db

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
    # 新增：详细阶段信息
    current_stage: str = "preparing"  # preparing, data_loading, backtesting, completed
    stage_description: str = "准备中"
    data_loading_completed: bool = False

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

            # 检查是否至少有一个参数启用优化或有固定值
            has_optimization_param = False
            has_any_param = False

            for param_name, param_range in param_ranges.items():
                if not isinstance(param_range, dict):
                    return False, f"参数 {param_name} 的范围定义必须是字典"

                has_any_param = True

                # 检查参数是否启用优化
                if param_range.get('enabled', True):
                    # 启用优化的参数需要验证范围字段
                    has_optimization_param = True
                    required_range_fields = ['min', 'max', 'step']
                    for field in required_range_fields:
                        if field not in param_range:
                            return False, f"参数 {param_name} 缺少范围字段: {field}"

                    if param_range['min'] > param_range['max']:
                        return False, f"参数 {param_name} 的最小值不能大于最大值"

                    if param_range['step'] <= 0:
                        return False, f"参数 {param_name} 的步长必须大于0"
                else:
                    # 固定值参数需要验证fixed_value字段
                    if 'fixed_value' not in param_range:
                        return False, f"固定参数 {param_name} 缺少 fixed_value 字段"

            if not has_any_param:
                return False, "至少需要配置一个参数"
            
            return True, "验证通过"
            
        except Exception as e:
            return False, f"验证过程中发生错误: {e}"
    
    @staticmethod
    def calculate_total_combinations(param_ranges: Dict[str, Dict[str, Any]]) -> int:
        """计算参数组合总数，只计算启用优化的参数"""
        total = 1
        optimization_param_count = 0

        for param_name, param_range in param_ranges.items():
            # 只计算启用优化的参数
            if param_range.get('enabled', True):
                optimization_param_count += 1
                min_val = param_range['min']
                max_val = param_range['max']
                step = param_range['step']

                # 计算该参数的可能值数量
                count = int((max_val - min_val) / step) + 1
                total *= count

        # 如果没有启用优化的参数，返回1（只有一个固定值组合）
        if optimization_param_count == 0:
            return 1

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
            

            
            # 计算波动率
            metrics['volatility'] = self._calculate_volatility(backtest_result)

            # 多单和空单胜率
            metrics['long_win_rate'] = backtest_result.get('long_win_rate', 0.0)
            metrics['short_win_rate'] = backtest_result.get('short_win_rate', 0.0)
            
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
            'volatility': 0.0,
            'long_win_rate': 0.0,
            'short_win_rate': 0.0,
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
        """生成所有参数组合，支持部分参数固定值"""
        try:
            # 分离优化参数和固定参数
            optimization_params = {}
            fixed_params = {}

            for param_name, param_config in param_ranges.items():
                if param_config.get('enabled', True):
                    # 启用优化的参数
                    optimization_params[param_name] = param_config
                else:
                    # 固定值参数
                    fixed_params[param_name] = param_config.get('fixed_value', 0)

            logger.info(f"优化参数: {list(optimization_params.keys())}")
            logger.info(f"固定参数: {list(fixed_params.keys())}")

            # 如果没有需要优化的参数，返回单个组合（全部固定值）
            if not optimization_params:
                if fixed_params:
                    return [fixed_params]
                else:
                    logger.warning("没有任何参数配置")
                    return []

            # 生成优化参数的组合
            param_names = list(optimization_params.keys())
            param_values = []

            for param_name in param_names:
                param_range = optimization_params[param_name]
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
                # 创建参数字典，包含优化参数和固定参数
                param_dict = dict(zip(param_names, combination))
                param_dict.update(fixed_params)  # 添加固定参数
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
                start_time=datetime.now(),
                current_stage="preparing",
                stage_description="准备中",
                data_loading_completed=False
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

    def update_stage(self, optimization_id: str, stage: str, description: str):
        """更新当前阶段信息"""
        with self._lock:
            if optimization_id in self._progresses:
                progress = self._progresses[optimization_id]
                progress.current_stage = stage
                progress.stage_description = description

                # 特殊处理数据加载完成标记
                if stage == "backtesting":
                    progress.data_loading_completed = True

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
                    # 更新阶段信息为已完成
                    progress.current_stage = "completed"
                    progress.stage_description = "优化完成"
                elif status == "error":
                    # 更新阶段信息为错误
                    progress.current_stage = "error"
                    progress.stage_description = "优化失败"
                elif status == "stopped":
                    # 更新阶段信息为已停止
                    progress.current_stage = "stopped"
                    progress.stage_description = "已停止"

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

    def __init__(self, main_loop: Optional[asyncio.AbstractEventLoop] = None):
        """初始化优化引擎"""
        self.main_loop = main_loop
        self.optimizer = GridSearchOptimizer()
        self.progress_tracker = ProgressTracker()
        self.results_manager = ResultsManager()
        self.binance_client = BinanceClient()

        # 单任务管理
        self._current_optimization_id: Optional[str] = None
        self._current_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # 数据库实例（延迟初始化）
        self._db: Optional[OptimizationDatabase] = None

    async def _get_db(self) -> OptimizationDatabase:
        """获取数据库实例"""
        if self._db is None:
            self._db = await get_optimization_db()
        return self._db

    async def _create_optimization_record(self, optimization_id: str, config: Dict[str, Any], total_combinations: int):
        """创建优化记录"""
        try:
            db = await self._get_db()

            # 获取策略名称
            strategy_name = "未知策略"
            try:
                strategies = get_available_strategies()
                strategy = next((s for s in strategies if s['id'] == config.get('strategy_id')), None)
                if strategy:
                    strategy_name = strategy['name']
            except:
                pass

            record = OptimizationRecord(
                id=optimization_id,
                symbol=config.get('symbol', ''),
                interval=config.get('interval', ''),
                strategy_id=config.get('strategy_id', ''),
                strategy_name=strategy_name,
                start_date=config.get('start_date', ''),
                end_date=config.get('end_date', ''),
                status='running',
                progress={
                    'current': 0,
                    'total': total_combinations,
                    'percentage': 0.0,
                    'elapsed_time': 0.0,
                    'estimated_remaining': 0.0
                },
                config=config
            )

            await db.save_record(record)
            logger.info(f"优化记录已创建: {optimization_id}")

        except Exception as e:
            logger.error(f"创建优化记录失败: {e}")

    async def _update_record_status(self, optimization_id: str, status: str,
                                  progress: Optional[Dict[str, Any]] = None,
                                  results: Optional[Dict[str, Any]] = None,
                                  error_message: Optional[str] = None):
        """更新记录状态"""
        try:
            db = await self._get_db()
            await db.update_record_status(optimization_id, status, progress, results, error_message)
            logger.info(f"记录状态已更新: {optimization_id} -> {status}")
        except Exception as e:
            logger.error(f"更新记录状态失败: {e}")

    def _schedule_db_update(self, optimization_id: str, status: str,
                          progress: Optional[Dict[str, Any]] = None,
                          results: Optional[Dict[str, Any]] = None,
                          error_message: Optional[str] = None):
        """线程安全的数据库更新调度"""
        try:
            # 使用线程池执行器来运行异步任务
            import concurrent.futures
            import threading

            def run_async_update():
                try:
                    # 创建新的事件循环
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(
                            self._update_record_status(optimization_id, status, progress, results, error_message)
                        )
                    finally:
                        loop.close()
                except Exception as e:
                    logger.error(f"异步数据库更新失败: {e}")

            # 在新线程中执行
            thread = threading.Thread(target=run_async_update, daemon=True)
            thread.start()

        except Exception as e:
            logger.warning(f"调度数据库更新失败: {e}")

    def _schedule_record_creation(self, optimization_id: str, config: Dict[str, Any], total_combinations: int):
        """线程安全的记录创建调度"""
        try:
            import threading

            def run_async_creation():
                try:
                    # 创建新的事件循环
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(
                            self._create_optimization_record(optimization_id, config, total_combinations)
                        )
                    finally:
                        loop.close()
                except Exception as e:
                    logger.error(f"异步记录创建失败: {e}")

            # 在新线程中执行
            thread = threading.Thread(target=run_async_creation, daemon=True)
            thread.start()

        except Exception as e:
            logger.warning(f"调度记录创建失败: {e}")

    def optimize_strategy(self, optimization_config: Dict[str, Any], progress_update_callback: Optional[Callable] = None) -> str:
        """开始策略优化"""
        try:
            # 1. 检查是否有正在运行的任务
            with self._lock:
                if self._current_optimization_id is not None:
                    raise ValueError("已有优化任务正在运行，请等待完成或停止当前任务")

            # 2. 验证配置
            is_valid, error_msg = ParameterValidator.validate_optimization_config(optimization_config)
            if not is_valid:
                raise ValueError(f"配置验证失败: {error_msg}")

            # 3. 计算参数组合数量
            param_ranges = optimization_config['strategy_params_ranges']
            total_combinations = ParameterValidator.calculate_total_combinations(param_ranges)

            # 4. 验证资源限制
            max_combinations = optimization_config.get('max_combinations', 10000)
            is_valid, error_msg = ParameterValidator.validate_resource_limits(total_combinations, max_combinations)
            if not is_valid:
                raise ValueError(f"资源限制验证失败: {error_msg}")

            # 5. 生成优化ID
            optimization_id = str(uuid.uuid4())

            # 6. 创建进度跟踪
            self.progress_tracker.create_progress(optimization_id, total_combinations)

            # 7. 创建数据库记录
            self._schedule_record_creation(optimization_id, optimization_config, total_combinations)

            # 8. 启动优化线程
            optimization_thread = threading.Thread(
                target=self._run_optimization,
                args=(optimization_id, optimization_config, progress_update_callback),
                daemon=True
            )

            with self._lock:
                self._current_optimization_id = optimization_id
                self._current_thread = optimization_thread

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
            # 检查是否是当前运行的任务
            with self._lock:
                if self._current_optimization_id != optimization_id:
                    logger.warning(f"尝试停止非当前任务: {optimization_id}")
                    return False

                current_thread = self._current_thread

            # 停止优化器
            self.optimizer.stop_optimization()

            # 更新进度状态
            self.progress_tracker.complete_progress(optimization_id, "stopped")

            # 更新数据库记录，包含完整的进度信息
            final_progress = self.progress_tracker.get_progress(optimization_id)
            final_progress_data = None
            if final_progress:
                final_progress_data = {
                    'current': final_progress.current,
                    'total': final_progress.total,
                    'percentage': final_progress.percentage,
                    'elapsed_time': final_progress.elapsed_time,
                    'estimated_remaining': final_progress.estimated_remaining,
                    'current_stage': final_progress.current_stage,
                    'stage_description': final_progress.stage_description,
                    'data_loading_completed': final_progress.data_loading_completed
                }

            self._schedule_db_update(optimization_id, "stopped", final_progress_data)

            # 等待线程结束（设置超时避免无限等待）
            if current_thread and current_thread.is_alive():
                logger.info(f"等待优化线程结束，最多等待10秒...")
                current_thread.join(timeout=10)
                if current_thread.is_alive():
                    logger.warning(f"优化线程未能在10秒内结束，可能存在资源泄漏")

            # 清理当前任务记录
            with self._lock:
                self._current_optimization_id = None
                self._current_thread = None

            logger.info(f"优化任务已停止，ID: {optimization_id}")
            return True

        except Exception as e:
            logger.error(f"停止优化时发生错误: {e}")
            return False

    def stop_all_optimizations(self) -> bool:
        """停止所有正在运行的优化任务"""
        try:
            with self._lock:
                current_optimization_id = self._current_optimization_id
                current_thread = self._current_thread

            if current_optimization_id:
                logger.info(f"正在停止当前优化任务: {current_optimization_id}")
                success = self.stop_optimization(current_optimization_id)
                if not success:
                    logger.warning(f"停止优化任务 {current_optimization_id} 失败")

                    # 强制停止优化器
                    self.optimizer.stop_optimization()

                    # 如果线程仍在运行，尝试强制结束
                    if current_thread and current_thread.is_alive():
                        logger.warning("尝试强制结束优化线程...")
                        current_thread.join(timeout=5)
                        if current_thread.is_alive():
                            logger.error("优化线程未能在5秒内结束，可能存在资源泄漏")

                    # 强制清理状态
                    with self._lock:
                        self._current_optimization_id = None
                        self._current_thread = None

                return True
            else:
                logger.info("没有正在运行的优化任务需要停止")
                return True

        except Exception as e:
            logger.error(f"停止所有优化任务时发生错误: {e}")
            return False

    async def get_current_optimization(self) -> Optional[Dict[str, Any]]:
        """获取当前正在运行的优化任务"""
        try:
            db = await self._get_db()
            record = await db.get_running_record()

            if record:
                return {
                    'id': record.id,
                    'symbol': record.symbol,
                    'interval': record.interval,
                    'strategy_name': record.strategy_name,
                    'status': record.status,
                    'progress': record.progress,
                    'created_at': record.created_at
                }
            return None

        except Exception as e:
            logger.error(f"获取当前优化任务失败: {e}")
            return None

    async def get_optimization_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取优化历史记录"""
        try:
            db = await self._get_db()
            records = await db.get_all_records(limit)

            return [
                {
                    'id': record.id,
                    'symbol': record.symbol,
                    'interval': record.interval,
                    'strategy_name': record.strategy_name,
                    'status': record.status,
                    'progress': record.progress,
                    'created_at': record.created_at,
                    'completed_at': record.completed_at,
                    'error_message': record.error_message,
                    'start_date': record.start_date,
                    'end_date': record.end_date
                }
                for record in records
            ]

        except Exception as e:
            logger.error(f"获取优化历史失败: {e}")
            return []

    async def delete_optimization_record(self, record_id: str) -> bool:
        """删除优化记录"""
        try:
            db = await self._get_db()
            return await db.delete_record(record_id)
        except Exception as e:
            logger.error(f"删除优化记录失败: {e}")
            return False

    async def create_database_backup(self) -> bool:
        """创建数据库备份"""
        try:
            db = await self._get_db()
            return await db.create_backup()
        except Exception as e:
            logger.error(f"创建数据库备份失败: {e}")
            return False

    async def get_database_backups(self) -> List[Dict[str, Any]]:
        """获取数据库备份列表"""
        try:
            db = await self._get_db()
            return await db.get_backup_list()
        except Exception as e:
            logger.error(f"获取备份列表失败: {e}")
            return []

    async def restore_database_from_backup(self, backup_path: str) -> bool:
        """从备份恢复数据库"""
        try:
            db = await self._get_db()
            return await db.restore_from_backup(backup_path)
        except Exception as e:
            logger.error(f"从备份恢复数据库失败: {e}")
            return False

    async def get_optimization_results(self, optimization_id: str, limit: Optional[int] = None) -> Dict[str, Any]:
        """获取优化结果，优先从内存获取，失败则从数据库获取"""
        try:
            # 优先从内存中获取实时进度和结果
            progress = self.progress_tracker.get_progress(optimization_id)
            
            if progress:
                logger.info(f"从内存中获取优化结果: {optimization_id}")
                results_from_manager = self.results_manager.get_results(optimization_id, limit)
                best_result_from_manager = self.results_manager.get_best_result(optimization_id)
                scatter_plot_data = self.results_manager.generate_scatter_plot_data(optimization_id)

                response = {
                    'optimization_id': optimization_id,
                    'status': progress.status,
                    'progress': {
                        'current': progress.current, 'total': progress.total,
                        'percentage': progress.percentage, 'elapsed_time': progress.elapsed_time,
                        'estimated_remaining': progress.estimated_remaining
                    },
                    'summary': {
                        'total_combinations_tested': progress.current,
                        'valid_results': len(results_from_manager),
                        'optimization_time': progress.elapsed_time,
                        'best_score': best_result_from_manager.composite_score if best_result_from_manager else 0.0
                    },
                    'scatter_plot_data': scatter_plot_data
                }
                if best_result_from_manager:
                    response['best_result'] = {
                        'parameters': best_result_from_manager.parameters, 'metrics': best_result_from_manager.metrics,
                        'composite_score': best_result_from_manager.composite_score, 'rank': best_result_from_manager.rank,
                        'backtest_details': best_result_from_manager.backtest_details
                    }
                if limit is None or limit > 0:
                    response['all_results'] = [
                        {'parameters': r.parameters, 'metrics': r.metrics, 'composite_score': r.composite_score, 'rank': r.rank, 'backtest_details': r.backtest_details}
                        for r in results_from_manager
                    ]
                return response

            # 如果内存中没有，则从数据库中获取历史记录
            logger.info(f"内存中未找到，从数据库获取优化结果: {optimization_id}")
            db = await self._get_db()
            record = await db.get_record(optimization_id)

            if not record:
                return {'error': '未找到优化任务'}

            # 从数据库记录中构建响应
            record_results_data = record.results or {}
            summary = record_results_data.get('summary', {})
            all_results_from_db = record_results_data.get('results', [])
            
            if limit is not None:
                all_results_from_db = all_results_from_db[:limit]

            scatter_points = []
            for res in all_results_from_db:
                scatter_points.append({
                    'x': res.get('metrics', {}).get('win_rate', 0),
                    'y': res.get('metrics', {}).get('total_return', 0),
                    'parameters': res.get('parameters'), 'rank': res.get('rank'),
                    'composite_score': res.get('composite_score'), 'metrics': res.get('metrics')
                })

            response = {
                'optimization_id': record.id, 'status': record.status,
                'progress': record.progress, 'summary': summary,
                'scatter_plot_data': {'x_axis': 'win_rate', 'y_axis': 'total_return', 'points': scatter_points},
                'all_results': all_results_from_db,
                'best_result': all_results_from_db[0] if all_results_from_db else None,
                'start_date': record.start_date,
                'end_date': record.end_date,
                'symbol': record.symbol,
                'interval': record.interval,
                'strategy_name': record.strategy_name
            }
            return response

        except Exception as e:
            logger.error(f"获取优化结果时发生错误: {e}", exc_info=True)
            return {'error': str(e)}

    def _run_optimization(self, optimization_id: str, optimization_config: Dict[str, Any], progress_update_callback: Optional[Callable] = None):
        """运行优化的内部方法"""
        try:
            logger.info(f"开始执行优化任务: {optimization_id}")

            # 创建一个事件，用于在优化完成后停止所有挂起的数据库更新
            _optimization_completed_event = threading.Event()

            # 1. 更新阶段：数据获取
            self.progress_tracker.update_stage(optimization_id, "data_loading", "数据获取中")
            self._send_stage_update(optimization_id, progress_update_callback)

            # 准备数据
            df_kline, df_index_price = self._prepare_data(optimization_config)
            if df_kline is None or df_index_price is None:
                raise ValueError("数据准备失败")

            # 2. 更新阶段：数据获取完成
            self.progress_tracker.update_stage(optimization_id, "data_loaded", "数据获取完成")
            self._send_stage_update(optimization_id, progress_update_callback)

            # 生成参数组合
            param_ranges = optimization_config['strategy_params_ranges']
            combinations = self.optimizer.generate_parameter_combinations(param_ranges)

            if not combinations:
                raise ValueError("未能生成有效的参数组合")

            # 3. 更新阶段：开始回测
            self.progress_tracker.update_stage(optimization_id, "backtesting", "回测执行中")
            self._send_stage_update(optimization_id, progress_update_callback)

            # 创建回测函数
            backtest_func = self._create_backtest_function(
                df_kline, df_index_price, optimization_config
            )

            # 4. 创建进度回调
            def progress_callback(current: int, total: int):
                # 如果优化已完成，则不再调度任何“running”状态的更新
                if _optimization_completed_event.is_set():
                    return

                self.progress_tracker.update_progress(optimization_id, current)
                progress = self.progress_tracker.get_progress(optimization_id)

                # 同步更新数据库进度，但只有在优化未完成时才更新
                if progress and not _optimization_completed_event.is_set():
                    progress_data_for_db = {
                        'current': progress.current, 'total': progress.total,
                        'percentage': progress.percentage, 'elapsed_time': progress.elapsed_time,
                        'estimated_remaining': progress.estimated_remaining
                    }
                    self._schedule_db_update(optimization_id, "running", progress_data_for_db)

                # 通过WebSocket发送进度更新
                if progress_update_callback and progress:
                    ws_data = {
                        "type": "progress_update",
                        "data": {
                            'optimization_id': progress.optimization_id, 'status': progress.status,
                            'current_stage': progress.current_stage,
                            'stage_description': progress.stage_description,
                            'data_loading_completed': progress.data_loading_completed,
                            'current': progress.current, 'total': progress.total,
                            'percentage': round(progress.percentage, 2),
                            'elapsed_time': round(progress.elapsed_time, 2),
                            'estimated_remaining': round(progress.estimated_remaining, 2)
                        }
                    }
                    if self.main_loop:
                        # 使用 run_coroutine_threadsafe 安全地在主事件循环上调度回调
                        asyncio.run_coroutine_threadsafe(
                            progress_update_callback(optimization_id, ws_data),
                            self.main_loop
                        )

            # 5. 执行并行优化
            results = self.optimizer.optimize_parallel(
                combinations, backtest_func, progress_callback
            )

            # 发出信号，停止所有进一步的“running”状态更新
            _optimization_completed_event.set()

            # 等待一小段时间，确保所有挂起的数据库更新完成
            import time
            time.sleep(0.1)

            # 6. 存储结果
            self.results_manager.store_results(optimization_id, results)

            # 7. 完成进度跟踪
            self.progress_tracker.complete_progress(optimization_id, "completed")

            # 8. 更新数据库记录为完成状态
            # 准备要存入数据库的完整结果
            
            # 辅助函数，用于递归地将对象转换为与JSON兼容的格式
            def ensure_serializable(obj):
                if isinstance(obj, dict):
                    return {k: ensure_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [ensure_serializable(i) for i in obj]
                elif isinstance(obj, (datetime, pd.Timestamp)):
                    return obj.isoformat()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj

            results_list_for_db = []
            for r in results:
                # 准备要序列化的完整结果字典
                full_result_dict = {
                    'parameters': r.parameters,
                    'metrics': r.metrics,
                    'composite_score': r.composite_score,
                    'rank': r.rank,
                    'backtest_details': r.backtest_details
                }
                # 在存入列表前进行序列化清理
                results_list_for_db.append(ensure_serializable(full_result_dict))

            results_data = {
                'summary': {
                    'total_combinations_tested': len(combinations),
                    'valid_results': len(results),
                    'best_score': results[0].composite_score if results else 0.0
                },
                'results': results_list_for_db
            }

            # 获取最终进度信息，确保数据库记录包含完整的完成状态
            final_progress = self.progress_tracker.get_progress(optimization_id)
            final_progress_data = None
            if final_progress:
                final_progress_data = {
                    'current': final_progress.current,
                    'total': final_progress.total,
                    'percentage': final_progress.percentage,
                    'elapsed_time': final_progress.elapsed_time,
                    'estimated_remaining': final_progress.estimated_remaining,
                    'current_stage': final_progress.current_stage,
                    'stage_description': final_progress.stage_description,
                    'data_loading_completed': final_progress.data_loading_completed
                }

            # 同步更新最终状态以避免竞态条件
            # 直接运行异步更新函数，确保在所有进度更新之后执行
            final_update_loop = None
            try:
                final_update_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(final_update_loop)
                final_update_loop.run_until_complete(
                    self._update_record_status(optimization_id, "completed", final_progress_data, results_data)
                )
            except Exception as db_e:
                logger.error(f"同步更新最终数据库状态失败: {db_e}")
            finally:
                if final_update_loop and not final_update_loop.is_closed():
                    final_update_loop.close()

            # 通过WebSocket发送最终完成状态
            if progress_update_callback:
                final_progress = self.progress_tracker.get_progress(optimization_id)
                best_result = results[0] if results else None
                ws_data = {
                    "type": "completed",
                    "data": {
                        'optimization_id': final_progress.optimization_id,
                        'status': final_progress.status,
                        'current_stage': final_progress.current_stage,
                        'stage_description': final_progress.stage_description,
                        'data_loading_completed': final_progress.data_loading_completed,
                        'current': final_progress.current,
                        'total': final_progress.total,
                        'percentage': round(final_progress.percentage, 2),
                        'elapsed_time': round(final_progress.elapsed_time, 2),
                        'estimated_remaining': round(final_progress.estimated_remaining, 2),
                        'summary': {
                            'total_combinations_tested': final_progress.current, 'valid_results': len(results),
                            'optimization_time': final_progress.elapsed_time,
                            'best_score': best_result.composite_score if best_result else 0.0
                        },
                        'best_result': {
                            'parameters': best_result.parameters, 'metrics': best_result.metrics,
                            'composite_score': best_result.composite_score
                        } if best_result else None
                    }
                }
                if self.main_loop:
                    asyncio.run_coroutine_threadsafe(
                        progress_update_callback(optimization_id, ws_data),
                        self.main_loop
                    )

            logger.info(f"优化任务完成: {optimization_id}, 有效结果: {len(results)}")

        except Exception as e:
            logger.error(f"优化任务执行失败: {optimization_id}, 错误: {e}")
            self.progress_tracker.complete_progress(optimization_id, "error", str(e))
            
            # 通过WebSocket发送错误状态
            if progress_update_callback:
                final_progress = self.progress_tracker.get_progress(optimization_id)
                if final_progress:
                    ws_data = {
                        "type": "error",
                        "data": {
                            'optimization_id': final_progress.optimization_id,
                            'status': final_progress.status,
                            'current_stage': final_progress.current_stage,
                            'stage_description': final_progress.stage_description,
                            'data_loading_completed': final_progress.data_loading_completed,
                            'current': final_progress.current,
                            'total': final_progress.total,
                            'percentage': round(final_progress.percentage, 2),
                            'elapsed_time': round(final_progress.elapsed_time, 2),
                            'estimated_remaining': round(final_progress.estimated_remaining, 2),
                            'error_message': final_progress.error_message
                        }
                    }
                    if self.main_loop:
                        asyncio.run_coroutine_threadsafe(
                            progress_update_callback(optimization_id, ws_data),
                            self.main_loop
                        )

            # 更新数据库记录为错误状态，包含完整的进度信息
            final_progress = self.progress_tracker.get_progress(optimization_id)
            final_progress_data = None
            if final_progress:
                final_progress_data = {
                    'current': final_progress.current,
                    'total': final_progress.total,
                    'percentage': final_progress.percentage,
                    'elapsed_time': final_progress.elapsed_time,
                    'estimated_remaining': final_progress.estimated_remaining,
                    'current_stage': final_progress.current_stage,
                    'stage_description': final_progress.stage_description,
                    'data_loading_completed': final_progress.data_loading_completed
                }

            self._schedule_db_update(optimization_id, "error", final_progress_data, None, str(e))

        finally:
            # 清理当前任务记录
            with self._lock:
                if self._current_optimization_id == optimization_id:
                    self._current_optimization_id = None
                    self._current_thread = None

    def _send_stage_update(self, optimization_id: str, progress_update_callback: Optional[Callable] = None):
        """发送阶段更新到前端"""
        if progress_update_callback:
            progress = self.progress_tracker.get_progress(optimization_id)
            if progress:
                ws_data = {
                    "type": "stage_update",
                    "data": {
                        'optimization_id': progress.optimization_id,
                        'status': progress.status,
                        'current_stage': progress.current_stage,
                        'stage_description': progress.stage_description,
                        'data_loading_completed': progress.data_loading_completed,
                        'current': progress.current,
                        'total': progress.total,
                        'percentage': round(progress.percentage, 2),
                        'elapsed_time': round(progress.elapsed_time, 2),
                        'estimated_remaining': round(progress.estimated_remaining, 2)
                    }
                }
                if self.main_loop:
                    asyncio.run_coroutine_threadsafe(
                        progress_update_callback(optimization_id, ws_data),
                        self.main_loop
                    )

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
        """应用时间过滤逻辑"""
        try:
            if df.empty:
                return df

            # 处理包含时间段（新增逻辑）
            include_time_ranges = optimization_config.get('include_time_ranges', [])
            if include_time_ranges:
                # 如果设置了包含时间段，只保留这些时间段内的数据
                combined_mask = pd.Series([False] * len(df), index=df.index)

                for time_range in include_time_ranges:
                    start_time = time_range['start']
                    end_time = time_range['end']

                    try:
                        # 解析时间字符串
                        start_time_obj = pd.to_datetime(start_time).time()
                        end_time_obj = pd.to_datetime(end_time).time()

                        # 处理跨天情况
                        if start_time_obj <= end_time_obj:
                            # 正常情况：开始时间 <= 结束时间
                            mask = (df.index.time >= start_time_obj) & (df.index.time <= end_time_obj)
                        else:
                            # 跨天情况：开始时间 > 结束时间
                            mask = (df.index.time >= start_time_obj) | (df.index.time <= end_time_obj)

                        combined_mask = combined_mask | mask
                        logger.info(f"添加包含时间段 {start_time}-{end_time}")

                    except Exception as time_error:
                        logger.warning(f"解析包含时间范围 {start_time}-{end_time} 时发生错误: {time_error}")
                        continue

                df = df[combined_mask]
                logger.info(f"应用包含时间段过滤后，剩余数据: {len(df)} 条")

            # 排除特定时间段（保留原有逻辑，用于向后兼容）
            exclude_time_ranges = optimization_config.get('exclude_time_ranges', [])
            for time_range in exclude_time_ranges:
                start_time = time_range['start']
                end_time = time_range['end']

                try:
                    # 解析时间字符串
                    start_time_obj = pd.to_datetime(start_time).time()
                    end_time_obj = pd.to_datetime(end_time).time()

                    # 处理跨天情况
                    if start_time_obj <= end_time_obj:
                        # 正常情况：开始时间 <= 结束时间
                        mask = ~((df.index.time >= start_time_obj) & (df.index.time <= end_time_obj))
                    else:
                        # 跨天情况：开始时间 > 结束时间
                        mask = ~((df.index.time >= start_time_obj) | (df.index.time <= end_time_obj))

                    df = df[mask]
                    logger.info(f"应用排除时间段 {start_time}-{end_time}，剩余数据: {len(df)} 条")

                except Exception as time_error:
                    logger.warning(f"解析排除时间范围 {start_time}-{end_time} 时发生错误: {time_error}")
                    continue

            # 排除特定星期
            exclude_weekdays = optimization_config.get('exclude_weekdays', [])
            if exclude_weekdays:
                mask = ~df.index.weekday.isin(exclude_weekdays)
                df = df[mask]
                logger.info(f"应用星期过滤 {exclude_weekdays}，剩余数据: {len(df)} 条")

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
                },
                'rsi_divergence': {
                    'conservative': {
                        'description': '保守型：寻找更可靠的背离信号',
                        'ranges': {
                            'rsi_period': {'min': 12, 'max': 18, 'step': 2},
                            'pivot_lookback_high': {'min': 5, 'max': 10, 'step': 1},
                            'pivot_lookback_low': {'min': 5, 'max': 10, 'step': 1}
                        }
                    },
                    'balanced': {
                        'description': '平衡型：平衡信号频率和可靠性',
                        'ranges': {
                            'rsi_period': {'min': 10, 'max': 16, 'step': 2},
                            'pivot_lookback_high': {'min': 3, 'max': 7, 'step': 1},
                            'pivot_lookback_low': {'min': 3, 'max': 7, 'step': 1}
                        }
                    },
                    'aggressive': {
                        'description': '激进型：捕捉更多潜在的背离机会',
                        'ranges': {
                            'rsi_period': {'min': 8, 'max': 14, 'step': 2},
                            'pivot_lookback_high': {'min': 2, 'max': 5, 'step': 1},
                            'pivot_lookback_low': {'min': 2, 'max': 5, 'step': 1}
                        }
                    }
                },
                'rsi_bollinger': {
                    'conservative': {
                        'description': '保守型：更宽的通道，更少的信号',
                        'ranges': {
                            'rsi_period': {'min': 12, 'max': 18, 'step': 2},
                            'bb_len': {'min': 18, 'max': 25, 'step': 2},
                            'bb_std': {'min': 2.0, 'max': 2.5, 'step': 0.1}
                        }
                    },
                    'balanced': {
                        'description': '平衡型：标准参数，平衡信号和过滤',
                        'ranges': {
                            'rsi_period': {'min': 10, 'max': 16, 'step': 2},
                            'bb_len': {'min': 15, 'max': 22, 'step': 2},
                            'bb_std': {'min': 1.8, 'max': 2.2, 'step': 0.1}
                        }
                    },
                    'aggressive': {
                        'description': '激进型：更窄的通道，更频繁的信号',
                        'ranges': {
                            'rsi_period': {'min': 8, 'max': 14, 'step': 2},
                            'bb_len': {'min': 10, 'max': 18, 'step': 2},
                            'bb_std': {'min': 1.5, 'max': 2.0, 'step': 0.1}
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

def get_optimization_engine(main_loop: Optional[asyncio.AbstractEventLoop] = None) -> OptimizationEngine:
    """获取全局优化引擎实例"""
    # 在首次从主线程调用时设置事件循环
    if main_loop and not optimization_engine.main_loop:
        optimization_engine.main_loop = main_loop
    return optimization_engine
