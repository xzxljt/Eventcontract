import time
import pandas as pd
from typing import Dict, Any, List

class MarketMonitor:
    """市场监控系统"""
    
    def __init__(self):
        self.market_history = []
        self.model_performance_history = []
        self.prediction_history = []
        self.alert_thresholds = {
            'volatility': 0.05,
            'momentum': 0.02,
            'regime_change': 2,
            'model_performance': 0.4
        }
        
    def monitor_market_conditions(self, df):
        """监控市场条件"""
        try:
            market_conditions = {
                'timestamp': time.time(),
                'close': df['close'].iloc[-1] if 'close' in df.columns else 0,
                'volume': df['volume'].iloc[-1] if 'volume' in df.columns else 0,
                'volatility': df['atr_percent'].iloc[-1] if 'atr_percent' in df.columns else \
                             df['close'].rolling(14).std().iloc[-1] / df['close'].iloc[-1],
                'momentum': df['ret5'].iloc[-1] if 'ret5' in df.columns else \
                            df['close'].pct_change(5).iloc[-1],
                'market_regime': df['market_regime'].iloc[-1] if 'market_regime' in df.columns else 0,
                'trend_strength': df['trend_strength'].iloc[-1] if 'trend_strength' in df.columns else 0.5
            }
            
            self.market_history.append(market_conditions)
            if len(self.market_history) > 1000:
                self.market_history = self.market_history[-1000:]
            
            return market_conditions
        except Exception as e:
            return {
                'timestamp': time.time(),
                'error': str(e)
            }
    
    def monitor_model_performance(self, model_name, predictions, actuals):
        """监控模型性能"""
        try:
            # 计算准确率
            correct = sum(1 for pred, actual in zip(predictions, actuals) if pred == actual)
            accuracy = correct / len(predictions)
            
            performance = {
                'timestamp': time.time(),
                'model_name': model_name,
                'accuracy': accuracy,
                'correct': correct,
                'total': len(predictions)
            }
            
            self.model_performance_history.append(performance)
            if len(self.model_performance_history) > 1000:
                self.model_performance_history = self.model_performance_history[-1000:]
            
            return performance
        except Exception as e:
            return {
                'timestamp': time.time(),
                'model_name': model_name,
                'error': str(e)
            }
    
    def monitor_prediction(self, prediction_result):
        """监控预测结果"""
        try:
            prediction_info = {
                'timestamp': time.time(),
                'final_signal': prediction_result.get('smart_combined_prediction', {}).get('final_signal', 0),
                'final_confidence': prediction_result.get('smart_combined_prediction', {}).get('final_confidence', 0),
                'vote_summary': prediction_result.get('smart_combined_prediction', {}).get('vote_summary', ''),
                'technical_predictions_count': len(prediction_result.get('technical_predictions', [])),
                'ml_predictions_count': len(prediction_result.get('ml_predictions', [])),
                'use_ensemble': prediction_result.get('use_ensemble', False)
            }
            
            self.prediction_history.append(prediction_info)
            if len(self.prediction_history) > 1000:
                self.prediction_history = self.prediction_history[-1000:]
            
            return prediction_info
        except Exception as e:
            return {
                'timestamp': time.time(),
                'error': str(e)
            }
    
    def generate_alerts(self):
        """生成警报"""
        alerts = []
        
        # 检查市场条件警报
        if self.market_history:
            latest = self.market_history[-1]
            
            if latest.get('volatility', 0) > self.alert_thresholds['volatility']:
                alerts.append({
                    'type': 'HIGH_VOLATILITY',
                    'message': f"高波动率警报: {latest['volatility']:.4f}",
                    'timestamp': latest['timestamp'],
                    'severity': 'high'
                })
            
            if abs(latest.get('momentum', 0)) > self.alert_thresholds['momentum']:
                alerts.append({
                    'type': 'STRONG_MOMENTUM',
                    'message': f"强动量警报: {latest['momentum']:.4f}",
                    'timestamp': latest['timestamp'],
                    'severity': 'medium'
                })
        
        # 检查模型性能警报
        if self.model_performance_history:
            recent_perf = self.model_performance_history[-10:]
            avg_accuracy = sum(p.get('accuracy', 0) for p in recent_perf) / len(recent_perf)
            
            if avg_accuracy < self.alert_thresholds['model_performance']:
                alerts.append({
                    'type': 'LOW_MODEL_PERFORMANCE',
                    'message': f"模型性能低下警报: {avg_accuracy:.4f}",
                    'timestamp': time.time(),
                    'severity': 'high'
                })
        
        return alerts
    
    def get_market_summary(self, window=20):
        """获取市场摘要"""
        if not self.market_history:
            return {}
        
        recent_history = self.market_history[-window:]
        
        summary = {
            'window': window,
            'average_volatility': sum(h.get('volatility', 0) for h in recent_history) / len(recent_history),
            'average_momentum': sum(h.get('momentum', 0) for h in recent_history) / len(recent_history),
            'average_trend_strength': sum(h.get('trend_strength', 0) for h in recent_history) / len(recent_history),
            'regime_distribution': {},
            'latest': recent_history[-1]
        }
        
        # 计算市场状态分布
        for h in recent_history:
            regime = h.get('market_regime', 0)
            summary['regime_distribution'][regime] = summary['regime_distribution'].get(regime, 0) + 1
        
        return summary
    
    def get_model_performance_summary(self, window=50):
        """获取模型性能摘要"""
        if not self.model_performance_history:
            return {}
        
        recent_history = self.model_performance_history[-window:]
        
        summary = {
            'window': window,
            'average_accuracy': sum(h.get('accuracy', 0) for h in recent_history) / len(recent_history),
            'total_predictions': sum(h.get('total', 0) for h in recent_history),
            'total_correct': sum(h.get('correct', 0) for h in recent_history)
        }
        
        return summary
    
    def get_prediction_summary(self, window=50):
        """获取预测摘要"""
        if not self.prediction_history:
            return {}
        
        recent_history = self.prediction_history[-window:]
        
        up_signals = sum(1 for h in recent_history if h.get('final_signal', 0) == 1)
        down_signals = sum(1 for h in recent_history if h.get('final_signal', 0) == 0)
        
        summary = {
            'window': window,
            'up_signals': up_signals,
            'down_signals': down_signals,
            'average_confidence': sum(h.get('final_confidence', 0) for h in recent_history) / len(recent_history),
            'signal_ratio': up_signals / (up_signals + down_signals) if up_signals + down_signals > 0 else 0.5
        }
        
        return summary
    
    def reset_history(self):
        """重置历史记录"""
        self.market_history = []
        self.model_performance_history = []
        self.prediction_history = []
    
    def set_alert_thresholds(self, thresholds):
        """设置警报阈值"""
        self.alert_thresholds.update(thresholds)