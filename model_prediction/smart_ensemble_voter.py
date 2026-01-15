import pandas as pd
from typing import Dict, List, Tuple, Any

class SmartEnsembleVoter:
    """智能集成投票器"""
    
    def __init__(self):
        self.model_weights = {}
        self.model_history = {}
        self.adaptive_weights = True
        
    def update_model_performance(self, model_name, correct):
        """更新模型性能历史"""
        if model_name not in self.model_history:
            self.model_history[model_name] = []
        
        self.model_history[model_name].append(correct)
        
        if len(self.model_history[model_name]) > 100:
            self.model_history[model_name] = self.model_history[model_name][-100:]
    
    def get_model_performance(self, model_name):
        """获取模型历史性能"""
        if model_name not in self.model_history or not self.model_history[model_name]:
            return 0.5
        
        recent = self.model_history[model_name][-20:] if len(self.model_history[model_name]) >= 20 else self.model_history[model_name]
        if not recent:
            return 0.5
        
        return sum(recent) / len(recent)
    
    def calculate_adaptive_weights(self, predictions, market_conditions):
        """计算自适应权重"""
        weights = {}
        
        for name, signal, conf in predictions:
            base_weight = conf
            
            # 根据市场条件调整权重
            if 'TREND' in name and market_conditions.get('trend_strength', 0) > 0.6:
                base_weight *= 1.4
            
            if 'MEAN_REVERSION' in name and market_conditions.get('volatility', 0) > 0.7:
                base_weight *= 1.3
            
            if 'MOMENTUM' in name and market_conditions.get('momentum', 0) > 0.5:
                base_weight *= 1.2
            
            # 根据历史性能调整权重
            historical_perf = self.get_model_performance(name)
            base_weight *= (historical_perf * 0.5 + 0.5)
            
            weights[name] = min(base_weight, 1.0)
        
        return weights
    
    def analyze_market_conditions(self, df):
        """分析市场条件"""
        conditions = {
            'trend_strength': abs(df['adx'].iloc[-1] / 100) if 'adx' in df.columns else 0.5,
            'volatility': df['atr_percent'].iloc[-1] if 'atr_percent' in df.columns else 0.5,
            'momentum': df['ret5'].iloc[-1] * 10 if 'ret5' in df.columns else 0,
            'market_regime': df['market_regime'].iloc[-1] if 'market_regime' in df.columns else 0
        }
        return conditions
    
    def weighted_voting(self, ml_predictions, tech_predictions, market_conditions):
        """加权投票"""
        all_predictions = ml_predictions + tech_predictions
        
        if not all_predictions:
            return None, 0.0, "无预测", {}, {}
        
        if self.adaptive_weights:
            weights = self.calculate_adaptive_weights(all_predictions, market_conditions)
        else:
            weights = {name: min(conf, 1.0) for name, _, conf in all_predictions}
        
        up_weight = 0.0
        down_weight = 0.0
        total_weight = 0.0
        
        ml_up = 0.0
        ml_down = 0.0
        ml_total = 0.0
        
        tech_up = 0.0
        tech_down = 0.0
        tech_total = 0.0
        
        vote_details = []
        for name, pred, conf in all_predictions:
            weight = weights.get(name, conf)
            total_weight += weight
            
            if 'ML_' in name or name in ['RF1', 'RF2', 'RF3', 'LGB1', 'LGB2', 'XGB1', 'XGB2', 'CAT1', 'CAT2', 
                                        'ADA', 'GB', 'ET', 'HGB', 'SVM1', 'SVM2', 'KNN', 'MLP', 'LDA', 'QDA', 
                                        'NB', 'ENSEMBLE']:
                ml_total += weight
                if pred == 1:
                    ml_up += weight
                    up_weight += weight
                else:
                    ml_down += weight
                    down_weight += weight
            else:
                tech_total += weight
                if pred == 1:
                    tech_up += weight
                    up_weight += weight
                else:
                    tech_down += weight
                    down_weight += weight
            
            vote_details.append(f"{name}:{'↑' if pred==1 else '↓'}({conf:.2f})")
        
        ml_ratio = ml_up / ml_total if ml_total > 0 else 0
        tech_ratio = tech_up / tech_total if tech_total > 0 else 0
        
        if total_weight == 0:
            return None, 0.0, "无预测", {'ml': 0.5, 'tech': 0.5}, {'ml': 0, 'tech': 0}
        
        vote_ratio = up_weight / total_weight
        
        if vote_ratio > 0.5:
            final_pred = 1
            final_conf = 2 * (vote_ratio - 0.5)
        else:
            final_pred = 0
            final_conf = 2 * (0.5 - vote_ratio)
        
        # 根据波动率调整置信度
        volatility_factor = 1 - market_conditions.get('volatility', 0.5)
        final_conf *= volatility_factor
        
        vote_summary = f"{len(all_predictions)}模型: 涨{up_weight:.2f}/跌{down_weight:.2f} | ML:{ml_ratio:.2f} | Tech:{tech_ratio:.2f}"
        
        model_ratios = {'ml': ml_ratio, 'tech': tech_ratio}
        model_weights = {'ml': ml_total, 'tech': tech_total}
        
        return final_pred, min(max(final_conf, 0.0), 1.0), vote_summary, model_ratios, model_weights
    
    def calculate_model_consistency(self, predictions):
        """计算模型一致性"""
        if not predictions:
            return 0.0
        
        signals = [pred for _, pred, _ in predictions]
        if len(signals) == 0:
            return 0.0
        
        up_count = sum(1 for s in signals if s == 1)
        down_count = sum(1 for s in signals if s == 0)
        
        consistency = abs(up_count - down_count) / len(signals)
        return consistency
