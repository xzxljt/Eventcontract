import time
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .data_preprocessor import DataPreprocessor
from .model_manager import ModelManager
from .technical_indicator_models import TechnicalIndicatorModels
from .smart_ensemble_voter import SmartEnsembleVoter
from .market_monitor import MarketMonitor
from .strategy_adjuster import StrategyAdjuster

class Predictor:
    # 支持的模型类型
    SUPPORTED_MODEL_TYPES = ['random_forest', 'linear_regression', 'ridge', 'svr', 'xgboost']
    
    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or {}
        self.data_preprocessor = DataPreprocessor(self.params.get('preprocessor_params', {}))
        
        # 根据参数决定使用单个模型还是集成模型
        model_count = self.params.get('model_count', 1)
        model_types = self.params.get('model_types', '')
        
        self.use_ensemble = False
        self.models = []
        self.model_type = None
        self.model_types = []
        self.model_count = 1
        self.model_path = self.params.get('model_path', 'model_prediction/models/model.joblib')
        self.preprocessor_path = None
        self.metrics = {}
        self.last_trained = None
        
        if model_count > 1 or ',' in model_types:
            # 使用集成模型
            self.use_ensemble = True
            self.model_types = model_types.split(',') if model_types else ['random_forest', 'xgboost']
            self._validate_model_types()
            # 确保model_types不为空
            if not self.model_types:
                self.model_types = ['random_forest', 'xgboost']
            # 使用用户设置的模型数量
            self.model_count = model_count
            self.model_path = self.params.get('model_path', 'model_prediction/models/ensemble/')
        else:
            # 使用单个模型
            self.use_ensemble = False
            self.model_type = self.params.get('model_params', {}).get('model_type', 'random_forest')
            # 验证模型类型
            if self.model_type not in self.SUPPORTED_MODEL_TYPES:
                raise ValueError(f"Unsupported model type: {self.model_type}. Supported types are: {self.SUPPORTED_MODEL_TYPES}")

        self.preprocessor_path = self.params.get('preprocessor_path') or self._resolve_preprocessor_path(self.model_path)
        
        # 初始化智能投票系统
        self.smart_voter = SmartEnsembleVoter()
        
        # 初始化市场监控系统
        self.market_monitor = MarketMonitor()
        
        # 初始化策略调整器
        self.strategy_adjuster = StrategyAdjuster()
        
        self.cache_enabled = self.params.get('enable_cache', True)
        self.cache_expiry = self.params.get('cache_expiry', 3600)
        self.cache = {}
        self.cache_size_limit = self.params.get('cache_size_limit', 1000)  # 缓存大小限制
        self.feature_names = []
        
        # 尝试自动加载已保存的模型
        self._auto_load_model()

    def _resolve_preprocessor_path(self, model_path: str) -> str:
        """解析预处理器保存路径"""
        if model_path.endswith('.joblib'):
            base_dir = os.path.dirname(model_path)
        else:
            base_dir = model_path
        if not base_dir:
            base_dir = '.'
        return os.path.join(base_dir, 'preprocessor.joblib')
    
    def _validate_model_types(self):
        """验证模型类型是否支持"""
        for model_type in self.model_types:
            if model_type not in self.SUPPORTED_MODEL_TYPES:
                raise ValueError(f"Unsupported model type: {model_type}. Supported types are: {self.SUPPORTED_MODEL_TYPES}")
    
    def _auto_load_model(self):
        """尝试自动加载已保存的模型"""
        try:
            if self.use_ensemble:
                # 检查集成模型目录是否存在
                if os.path.exists(self.model_path):
                    self.load_models()
            else:
                # 检查单个模型文件是否存在
                if os.path.exists(self.model_path):
                    self.load_model()
        except Exception as e:
            # 加载失败不影响系统启动，只是记录错误
            pass

    def _save_preprocessor(self):
        """保存预处理器"""
        if not self.preprocessor_path:
            self.preprocessor_path = self._resolve_preprocessor_path(self.model_path)
        dir_path = os.path.dirname(self.preprocessor_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        self.data_preprocessor.save_preprocessor(self.preprocessor_path)

    def _load_preprocessor(self):
        """加载预处理器"""
        if not self.preprocessor_path:
            self.preprocessor_path = self._resolve_preprocessor_path(self.model_path)
        if os.path.exists(self.preprocessor_path):
            self.data_preprocessor.load_preprocessor(self.preprocessor_path)
            self.feature_names = self.data_preprocessor.feature_names or []
        else:
            self._sync_feature_names_from_models()

    def _sync_feature_names_from_models(self):
        """从已加载模型同步特征名"""
        if not self.models:
            return
        model_feature_names = self.models[0].feature_names
        if model_feature_names:
            self.feature_names = list(model_feature_names)
            self.data_preprocessor.feature_names = list(model_feature_names)
    
    @classmethod
    def has_saved_models(cls):
        """检查是否有已保存的模型"""
        try:
            # 检查默认的单个模型路径
            single_model_path = 'model_prediction/models/model.joblib'
            if os.path.exists(single_model_path):
                return True
            
            # 检查默认的集成模型目录
            ensemble_model_path = 'model_prediction/models/ensemble/'
            if os.path.exists(ensemble_model_path) and os.listdir(ensemble_model_path):
                return True
            
            return False
        except Exception as e:
            # 检查失败不影响系统运行，返回False
            return False
    
    def train(self, df: pd.DataFrame, target_column: str = 'close') -> Dict[str, Any]:
        # 移除目标列和生成的列，保持特征一致性
        X = df.drop([target_column, 'signal', 'confidence'], axis=1, errors='ignore')
        y = df[target_column].shift(-1).dropna()
        X = X.loc[y.index]
        
        X_preprocessed = self.data_preprocessor.preprocess_data(X, is_training=True, target_series=y)
        self.feature_names = list(X_preprocessed.columns)
        self._save_preprocessor()
        
        if self.use_ensemble:
            # 集成模型训练
            return self._train_ensemble(X_preprocessed, y)
        else:
            # 单个模型训练
            return self._train_single_model(X_preprocessed, y)
    
    def _train_single_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """训练单个模型"""
        model = ModelManager({
            'model_type': self.model_type,
            'model_path': self.model_path,
            **self.params.get('model_params', {})
        })
        model.feature_names = self.feature_names
        mse = model.train(X, y)
        
        # 保存模型
        model.save_model(self.model_path)
        
        # 获取特征重要性
        feature_importance = model.get_feature_importance() if hasattr(model, 'get_feature_importance') else {}
        
        self.models = [model]
        self.metrics = model.metrics
        self.last_trained = time.time()
        # 在单个模型模式下设置model_types
        self.model_types = [self.model_type]
        
        return {
            'mse': mse,
            'metrics': self.metrics,
            'feature_importance': feature_importance
        }
    
    def _train_ensemble(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """训练集成模型"""
        self.models = []
        model_metrics = {}
        
        # 确保有模型类型可用
        if not self.model_types:
            self.model_types = ['random_forest', 'xgboost']
        
        for i in range(self.model_count):
            # 循环使用可用的模型类型
            model_type = self.model_types[i % len(self.model_types)]
            model_params = self._get_model_params(model_type)
            
            model = ModelManager({
                'model_type': model_type,
                'model_path': os.path.join(self.model_path, f'{model_type}_{i}.joblib'),
                **model_params
            })
            
            model.feature_names = self.feature_names
            mse = model.train(X, y)
            
            # 累积模型指标
            if model_type not in model_metrics:
                model_metrics[model_type] = {
                    'mse': mse,
                    'metrics': model.metrics
                }
            else:
                # 如果同一模型类型训练多次，使用最后一次的指标
                model_metrics[model_type] = {
                    'mse': mse,
                    'metrics': model.metrics
                }
            
            self.models.append(model)
        
        # 计算集成模型的整体指标
        self._calculate_ensemble_metrics(X, y)
        
        # 保存集成模型
        self.save_models()
        
        # 获取特征重要性
        feature_importance = self.get_feature_importance()
        
        # 确保更新训练时间
        self.last_trained = time.time()
        print(f"DEBUG: 训练完成，更新last_trained为: {self.last_trained}")
        
        return {
            'ensemble_result': {
                'ensemble_metrics': self.metrics,
                'model_metrics': model_metrics
            },
            'feature_importance': feature_importance
        }
    
    def _get_model_params(self, model_type: str) -> Dict[str, Any]:
        """获取模型参数"""
        model_params = self.params.get('model_params', {})
        if isinstance(model_params, dict) and model_type in model_params:
            return model_params[model_type]
        return {}
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        cache_key = self._generate_cache_key(df)
        
        if self.cache_enabled and cache_key in self.cache:
            cached_result = self.cache[cache_key]
            if time.time() - cached_result['timestamp'] < self.cache_expiry:
                return cached_result['result']
        
        # 移除目标列和生成的列，与训练时保持一致
        X = df.drop(['close', 'signal', 'confidence'], axis=1, errors='ignore')
        X_preprocessed = self.data_preprocessor.preprocess_data(X, is_training=False)
        
        if self.use_ensemble:
            predictions = self._predict_ensemble(X_preprocessed)
            model_info = self.get_ensemble_model_info()
        else:
            predictions = self._predict_single_model(X_preprocessed)
            model_info = self.get_single_model_info()
        
        # 获取技术指标模型预测
        tech_predictions = TechnicalIndicatorModels.get_all_predictions(df)
        
        # 准备机器学习预测格式（转换为分类预测）
        ml_predictions = []
        if len(predictions) > 0:
            latest_prediction = predictions[-1]
            last_close = df['close'].iloc[-1]
            # 转换为分类预测（1为上涨，0为下跌）
            signal = 1 if latest_prediction > last_close * 1.001 else 0
            # 基于预测与当前价格的差异计算置信度
            confidence = min(0.9, abs(latest_prediction - last_close) / last_close * 100)
            if self.use_ensemble:
                model_name = "ML_ENSEMBLE"
            else:
                model_name = f"ML_{self.model_type}"
            ml_predictions.append((model_name, signal, confidence))
        
        # 分析市场条件
        market_conditions = self.smart_voter.analyze_market_conditions(df)
        
        # 使用智能投票系统进行组合预测
        final_pred, final_conf, vote_summary, model_ratios, model_weights = self.smart_voter.weighted_voting(
            ml_predictions, tech_predictions, market_conditions
        )
        
        # 计算模型一致性
        all_predictions = ml_predictions + tech_predictions
        consistency = self.smart_voter.calculate_model_consistency(all_predictions)
        
        # 构建智能组合预测结果
        smart_combined_result = {
            'final_signal': final_pred,
            'final_confidence': final_conf,
            'vote_summary': vote_summary,
            'model_ratios': model_ratios,
            'model_weights': model_weights,
            'consistency': consistency,
            'market_conditions': market_conditions,
            'ml_predictions_count': len(ml_predictions),
            'tech_predictions_count': len(tech_predictions)
        }
        
        result = {
            'predictions': predictions.tolist(),
            'model_info': model_info,
            'use_ensemble': self.use_ensemble,
            'technical_predictions': tech_predictions,
            'ml_predictions': ml_predictions,
            'smart_combined_prediction': smart_combined_result,
            'timestamp': time.time(),
            'market_conditions': market_conditions,
            'alerts': self.market_monitor.generate_alerts(),
            'market_summary': self.market_monitor.get_market_summary(),
            'prediction_summary': self.market_monitor.get_prediction_summary()
        }
        
        # 监控市场条件
        self.market_monitor.monitor_market_conditions(df)
        
        # 监控预测结果
        self.market_monitor.monitor_prediction(result)
        
        # 根据市场条件调整策略
        self.strategy_adjuster.adjust_strategy_based_on_market(market_conditions)
        
        # 根据策略调整预测结果
        result = self.strategy_adjuster.adjust_prediction_based_on_strategy(result)
        
        if self.cache_enabled:
            # 缓存大小管理
            if len(self.cache) >= self.cache_size_limit:
                # 删除最旧的缓存项
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
                del self.cache[oldest_key]
            
            self.cache[cache_key] = {
                'result': result,
                'timestamp': time.time()
            }
        
        return result
    
    def _predict_single_model(self, X: pd.DataFrame) -> np.ndarray:
        """单个模型预测"""
        if not self.models:
            raise ValueError("No model trained or loaded")
        return self.models[0].predict(X)
    
    def _predict_ensemble(self, X: pd.DataFrame) -> np.ndarray:
        """集成模型预测"""
        if not self.models:
            raise ValueError("No models trained or loaded")
        
        predictions = np.array([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=0)  # 使用简单平均集成
    
    def _calculate_ensemble_metrics(self, X: pd.DataFrame, y: pd.Series):
        """计算集成模型的整体指标"""
        y_pred = self._predict_ensemble(X)
        
        self.metrics['mse'] = mean_squared_error(y, y_pred)
        self.metrics['rmse'] = np.sqrt(self.metrics['mse'])
        self.metrics['mae'] = mean_absolute_error(y, y_pred)
        self.metrics['r2'] = r2_score(y, y_pred)
    
    def get_single_model_info(self) -> Dict[str, Any]:
        """获取单个模型信息"""
        if not self.models:
            return {}
        
        model = self.models[0]
        return {
            'model_type': self.model_type,
            'params': self.params.get('model_params', {}),
            'metrics': self.metrics,
            'is_trained': True
        }
    
    def get_ensemble_model_info(self) -> Dict[str, Any]:
        """获取集成模型信息"""
        model_infos = []
        for i, model in enumerate(self.models):
            model_type = self.model_types[i % len(self.model_types)]
            model_infos.append({
                'index': i,
                'model_type': model_type,
                'info': {
                    'model_type': model_type,
                    'metrics': model.metrics
                }
            })
        
        return {
            'ensemble_info': {
                'model_count': self.model_count,
                'model_types': self.model_types,
                'metrics': self.metrics
            },
            'model_infos': model_infos
        }
    
    def load_model(self, model_path: Optional[str] = None):
        """加载模型"""
        if self.use_ensemble:
            self.load_models(model_path)
        else:
            load_path = model_path or self.model_path
            model = ModelManager({
                'model_type': self.model_type,
                'model_path': load_path
            })
            model.load_model(load_path)
            self.models = [model]
            self.metrics = model.metrics
            self._load_preprocessor()
    
    def save_model(self, model_path: Optional[str] = None):
        """保存模型"""
        if self.use_ensemble:
            self.save_models(model_path)
        else:
            save_path = model_path or self.model_path
            if self.models:
                self.models[0].save_model(save_path)
                self._save_preprocessor()
    
    def load_models(self, path: Optional[str] = None):
        """加载集成模型"""
        load_path = path or self.model_path
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Models directory not found: {load_path}")
        
        self.models = []
        for i in range(self.model_count):
            model_type = self.model_types[i % len(self.model_types)]
            model_load_path = os.path.join(load_path, f'{model_type}_{i}.joblib')
            
            if os.path.exists(model_load_path):
                model = ModelManager({'model_type': model_type})
                model.load_model(model_load_path)
                self.models.append(model)
            else:
                break
        self._load_preprocessor()
    
    def save_models(self, path: Optional[str] = None):
        """保存集成模型"""
        save_path = path or self.model_path
        os.makedirs(save_path, exist_ok=True)
        
        for i, model in enumerate(self.models):
            model_type = self.model_types[i % len(self.model_types)]
            model_save_path = os.path.join(save_path, f'{model_type}_{i}.joblib')
            model.save_model(model_save_path)
        self._save_preprocessor()
    
    def _generate_cache_key(self, df: pd.DataFrame) -> str:
        if df.empty:
            return "empty_data"
        
        key_parts = []
        key_parts.append(str(df.shape))
        key_parts.append(str(df.columns.tolist()))
        key_parts.append(str(df.iloc[-1].to_dict()) if not df.empty else "")
        
        # 根据模型类型生成不同的缓存键部分
        if self.use_ensemble:
            key_parts.append(str(self.model_types))
        else:
            key_parts.append(str(self.model_type))
        
        return "_".join(key_parts)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性"""
        if not self.models:
            return {}
        
        feature_importances = {}
        for feature in self.feature_names:
            feature_importances[feature] = 0.0
        
        for model in self.models:
            importance = model.get_feature_importance() if hasattr(model, 'get_feature_importance') else {}
            if importance:
                for feature, value in importance.items():
                    if feature in feature_importances:
                        feature_importances[feature] += abs(value)
        
        # 归一化
        total_importance = sum(feature_importances.values())
        if total_importance > 0:
            for feature in feature_importances:
                feature_importances[feature] /= total_importance
        
        return feature_importances
    
    def update_params(self, params: Dict[str, Any]):
        """更新参数"""
        self.params.update(params)
        self.cache_enabled = self.params.get('enable_cache', True)
        self.cache_expiry = self.params.get('cache_expiry', 3600)
        
        if 'preprocessor_params' in params:
            self.data_preprocessor = DataPreprocessor(params['preprocessor_params'])
        
        if 'model_params' in params or 'model_count' in params or 'model_types' in params:
            # 重新确定使用单个模型还是集成模型
            model_count = self.params.get('model_count', 1)
            model_types = self.params.get('model_types', '')
            
            if model_count > 1 or ',' in model_types:
                # 使用集成模型
                self.use_ensemble = True
                self.model_types = model_types.split(',') if model_types else ['random_forest', 'xgboost']
                self._validate_model_types()
                self.model_count = min(model_count, len(self.model_types))
                self.model_path = self.params.get('model_path', 'model_prediction/models/ensemble/')
            else:
                # 使用单个模型
                self.use_ensemble = False
                self.model_type = self.params.get('model_params', {}).get('model_type', 'random_forest')
                # 验证模型类型
                if self.model_type not in self.SUPPORTED_MODEL_TYPES:
                    raise ValueError(f"Unsupported model type: {self.model_type}. Supported types are: {self.SUPPORTED_MODEL_TYPES}")

        if 'preprocessor_path' in params or 'model_path' in params or 'model_count' in params or 'model_types' in params:
            self.preprocessor_path = self.params.get('preprocessor_path') or self._resolve_preprocessor_path(self.model_path)
    
    def clear_cache(self):
        self.cache.clear()
    
    def get_cache_info(self) -> Dict[str, Any]:
        return {
            'cache_size': len(self.cache),
            'cache_enabled': self.cache_enabled,
            'cache_expiry': self.cache_expiry
        }
    
    def get_technical_predictions(self, df: pd.DataFrame) -> List[Tuple[str, int, float]]:
        """获取技术指标模型预测"""
        return TechnicalIndicatorModels.get_all_predictions(df)
    
    def get_combined_prediction(self, df: pd.DataFrame) -> Dict[str, Any]:
        """获取组合预测结果"""
        # 先获取机器学习预测
        prediction_result = self.predict(df)
        return prediction_result.get('smart_combined_prediction', {})
    
    def update_model_performance(self, model_name: str, correct: bool):
        """更新模型性能"""
        self.smart_voter.update_model_performance(model_name, correct)
    
    def get_model_performance(self, model_name: str) -> float:
        """获取模型历史性能"""
        return self.smart_voter.get_model_performance(model_name)
    
    def analyze_market_conditions(self, df: pd.DataFrame) -> Dict[str, float]:
        """分析市场条件"""
        return self.smart_voter.analyze_market_conditions(df)
    
    def get_model_history(self) -> Dict[str, List[bool]]:
        """获取模型历史性能记录"""
        return self.smart_voter.model_history
    
    def reset_model_history(self):
        """重置模型历史性能记录"""
        self.smart_voter.model_history = {}
    
    def set_adaptive_weights(self, enabled: bool):
        """设置是否启用自适应权重"""
        self.smart_voter.adaptive_weights = enabled
    
    def get_market_summary(self, window=20):
        """获取市场摘要"""
        return self.market_monitor.get_market_summary(window)
    
    def get_model_performance_summary(self, window=50):
        """获取模型性能摘要"""
        return self.market_monitor.get_model_performance_summary(window)
    
    def get_prediction_summary(self, window=50):
        """获取预测摘要"""
        return self.market_monitor.get_prediction_summary(window)
    
    def generate_alerts(self):
        """生成警报"""
        return self.market_monitor.generate_alerts()
    
    def reset_monitor_history(self):
        """重置监控历史"""
        self.market_monitor.reset_history()
    
    def set_alert_thresholds(self, thresholds):
        """设置警报阈值"""
        self.market_monitor.set_alert_thresholds(thresholds)
    
    def monitor_model_performance(self, model_name, predictions, actuals):
        """监控模型性能"""
        return self.market_monitor.monitor_model_performance(model_name, predictions, actuals)
    
    def set_strategy(self, strategy_name: str):
        """设置策略模式"""
        return self.strategy_adjuster.set_strategy(strategy_name)
    
    def get_current_strategy(self) -> str:
        """获取当前策略"""
        return self.strategy_adjuster.get_current_strategy()
    
    def enable_adaptive_strategy(self, enabled: bool):
        """启用或禁用自适应策略调整"""
        self.strategy_adjuster.enable_adaptive_strategy(enabled)
    
    def get_strategy_recommendation(self, market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """获取策略推荐"""
        return self.strategy_adjuster.get_strategy_recommendation(market_conditions)
    
    def compare_strategies(self, performance_data: pd.DataFrame) -> Dict[str, Any]:
        """比较不同策略的性能"""
        return self.strategy_adjuster.compare_strategies(performance_data)
