from typing import Dict, Any
import pandas as pd

class StrategyAdjuster:
    """自适应策略调整器"""
    
    def __init__(self):
        # 策略模式参数预设
        self.strategy_presets = {
            'conservative': {
                'confidence_threshold': 0.7,
                'volatility_max': 0.03,
                'momentum_min': 0.01,
                'model_weight_ml': 0.7,
                'model_weight_tech': 0.3,
                'alert_thresholds': {
                    'volatility': 0.04,
                    'momentum': 0.015,
                    'model_performance': 0.5
                }
            },
            'balanced': {
                'confidence_threshold': 0.6,
                'volatility_max': 0.05,
                'momentum_min': 0.005,
                'model_weight_ml': 0.5,
                'model_weight_tech': 0.5,
                'alert_thresholds': {
                    'volatility': 0.05,
                    'momentum': 0.02,
                    'model_performance': 0.4
                }
            },
            'aggressive': {
                'confidence_threshold': 0.5,
                'volatility_max': 0.08,
                'momentum_min': -0.01,
                'model_weight_ml': 0.3,
                'model_weight_tech': 0.7,
                'alert_thresholds': {
                    'volatility': 0.07,
                    'momentum': 0.025,
                    'model_performance': 0.35
                }
            }
        }
        
        self.current_strategy = 'balanced'
        self.adaptive_enabled = True
    
    def get_strategy_parameters(self, strategy_name: str = None) -> Dict[str, Any]:
        """获取策略参数"""
        strategy = strategy_name or self.current_strategy
        return self.strategy_presets.get(strategy, self.strategy_presets['balanced'])
    
    def set_strategy(self, strategy_name: str):
        """设置策略模式"""
        if strategy_name in self.strategy_presets:
            self.current_strategy = strategy_name
            return True
        return False
    
    def adjust_strategy_based_on_market(self, market_conditions: Dict[str, Any]) -> str:
        """根据市场条件自动调整策略"""
        if not self.adaptive_enabled:
            return self.current_strategy
        
        volatility = market_conditions.get('volatility', 0)
        momentum = abs(market_conditions.get('momentum', 0))
        trend_strength = market_conditions.get('trend_strength', 0)
        
        # 基于市场条件调整策略
        if volatility > 0.06 or momentum > 0.03:
            # 高波动或高动量市场，使用保守策略
            self.set_strategy('conservative')
        elif volatility < 0.03 and trend_strength > 0.7:
            # 低波动且强趋势市场，使用激进策略
            self.set_strategy('aggressive')
        else:
            # 正常市场条件，使用平衡策略
            self.set_strategy('balanced')
        
        return self.current_strategy
    
    def adjust_prediction_based_on_strategy(self, prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        """根据当前策略调整预测结果"""
        strategy_params = self.get_strategy_parameters()
        
        # 获取原始预测结果
        smart_pred = prediction_result.get('smart_combined_prediction', {})
        final_signal = smart_pred.get('final_signal', 0)
        final_confidence = smart_pred.get('final_confidence', 0)
        
        # 根据策略调整信心阈值
        if final_confidence < strategy_params['confidence_threshold']:
            # 信心不足，调整为中立信号
            adjusted_signal = 0
            adjusted_confidence = final_confidence
        else:
            adjusted_signal = final_signal
            adjusted_confidence = final_confidence
        
        # 调整预测结果
        adjusted_prediction = {
            'final_signal': adjusted_signal,
            'final_confidence': adjusted_confidence,
            'original_signal': final_signal,
            'original_confidence': final_confidence,
            'strategy_applied': self.current_strategy,
            'confidence_threshold': strategy_params['confidence_threshold']
        }
        
        # 更新预测结果
        prediction_result['smart_combined_prediction'] = {**smart_pred, **adjusted_prediction}
        prediction_result['strategy_info'] = {
            'current_strategy': self.current_strategy,
            'strategy_parameters': strategy_params
        }
        
        return prediction_result
    
    def get_strategy_recommendation(self, market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """获取策略推荐"""
        recommended_strategy = self.adjust_strategy_based_on_market(market_conditions)
        params = self.get_strategy_parameters(recommended_strategy)
        
        recommendation = {
            'recommended_strategy': recommended_strategy,
            'current_strategy': self.current_strategy,
            'strategy_parameters': params,
            'market_conditions_analysis': {
                'volatility_level': 'high' if market_conditions.get('volatility', 0) > 0.05 else 'medium' if market_conditions.get('volatility', 0) > 0.03 else 'low',
                'momentum_level': 'high' if abs(market_conditions.get('momentum', 0)) > 0.02 else 'medium' if abs(market_conditions.get('momentum', 0)) > 0.01 else 'low',
                'trend_strength_level': 'strong' if market_conditions.get('trend_strength', 0) > 0.6 else 'medium' if market_conditions.get('trend_strength', 0) > 0.4 else 'weak'
            }
        }
        
        return recommendation
    
    def enable_adaptive_strategy(self, enabled: bool):
        """启用或禁用自适应策略调整"""
        self.adaptive_enabled = enabled
    
    def get_current_strategy(self) -> str:
        """获取当前策略"""
        return self.current_strategy
    
    def get_strategy_performance_metrics(self, strategy_name: str, performance_data: pd.DataFrame) -> Dict[str, Any]:
        """获取策略性能指标"""
        if performance_data.empty:
            return {}
        
        # 根据策略参数过滤交易
        params = self.get_strategy_parameters(strategy_name)
        filtered_data = performance_data[performance_data['confidence'] >= params['confidence_threshold']]
        
        if filtered_data.empty:
            return {
                'strategy': strategy_name,
                'total_trades': 0,
                'win_rate': 0,
                'average_profit': 0,
                'max_drawdown': 0
            }
        
        win_rate = (filtered_data['profit'] > 0).mean()
        average_profit = filtered_data['profit'].mean()
        
        # 计算最大回撤
        cumulative_profit = filtered_data['profit'].cumsum()
        running_max = cumulative_profit.cummax()
        drawdown = (cumulative_profit - running_max) / running_max.replace(0, 1)
        max_drawdown = drawdown.min() if not drawdown.empty else 0
        
        return {
            'strategy': strategy_name,
            'total_trades': len(filtered_data),
            'win_rate': win_rate,
            'average_profit': average_profit,
            'max_drawdown': max_drawdown,
            'confidence_threshold': params['confidence_threshold']
        }
    
    def compare_strategies(self, performance_data: pd.DataFrame) -> Dict[str, Any]:
        """比较不同策略的性能"""
        comparisons = {}
        
        for strategy in self.strategy_presets.keys():
            comparisons[strategy] = self.get_strategy_performance_metrics(strategy, performance_data)
        
        # 找出最佳策略
        best_strategy = None
        best_score = -float('inf')
        
        for strategy, metrics in comparisons.items():
            if metrics.get('total_trades', 0) > 0:
                # 简单评分：胜率 * 平均利润
                score = metrics['win_rate'] * metrics['average_profit']
                if score > best_score:
                    best_score = score
                    best_strategy = strategy
        
        return {
            'comparisons': comparisons,
            'best_strategy': best_strategy,
            'best_strategy_score': best_score
        }