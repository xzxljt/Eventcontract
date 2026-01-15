import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from typing import Dict, Any, Optional
import joblib

class ModelManager:
    # 支持的模型类型
    SUPPORTED_MODEL_TYPES = ['random_forest', 'linear_regression', 'ridge', 'svr', 'xgboost']
    
    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or {}
        self.model_type = self.params.get('model_type', 'random_forest')
        self.model = None
        self.model_path = self.params.get('model_path', 'model_prediction/models/model.joblib')
        self.metrics = {}
        
        # 验证模型类型
        if self.model_type not in self.SUPPORTED_MODEL_TYPES:
            raise ValueError(f"Unsupported model type: {self.model_type}. Supported types are: {self.SUPPORTED_MODEL_TYPES}")
    
    def _initialize_model(self):
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=self.params.get('n_estimators', 50),  # 减少树的数量以提高速度
                max_depth=self.params.get('max_depth', 10),  # 设置最大深度以避免过拟合
                min_samples_split=self.params.get('min_samples_split', 5),  # 增加最小样本分割以提高速度
                n_jobs=self.params.get('n_jobs', -1),  # 使用所有CPU核心
                random_state=self.params.get('random_state', 42)
            )
        elif self.model_type == 'linear_regression':
            self.model = LinearRegression()
        elif self.model_type == 'ridge':
            self.model = Ridge(alpha=self.params.get('alpha', 1.0))
        elif self.model_type == 'svr':
            self.model = SVR(
                kernel=self.params.get('kernel', 'rbf'),
                C=self.params.get('C', 1.0),
                gamma=self.params.get('gamma', 'scale')
            )
        elif self.model_type == 'xgboost':
            self.model = XGBRegressor(
                n_estimators=self.params.get('n_estimators', 50),  # 减少树的数量以提高速度
                max_depth=self.params.get('max_depth', 5),  # 设置适当的最大深度
                learning_rate=self.params.get('learning_rate', 0.1),
                n_jobs=self.params.get('n_jobs', -1),  # 使用所有CPU核心
                random_state=self.params.get('random_state', 42),
                tree_method='hist',  # 使用直方图方法提高速度
                enable_categorical=False  # 禁用分类特征以提高速度
            )
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> float:
        if self.model is None:
            self._initialize_model()
        
        self.model.fit(X, y)
        
        y_pred = self.model.predict(X)
        self._calculate_metrics(y, y_pred)
        
        return self.metrics['mse']
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        return self.model.predict(X)
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray):
        self.metrics['mse'] = mean_squared_error(y_true, y_pred)
        self.metrics['rmse'] = np.sqrt(self.metrics['mse'])
        self.metrics['mae'] = mean_absolute_error(y_true, y_pred)
        self.metrics['r2'] = r2_score(y_true, y_pred)
    
    def save_model(self, path: Optional[str] = None):
        save_path = path or self.model_path
        # 确保目录存在，即使save_path是一个文件名
        dir_path = os.path.dirname(save_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'model_type': self.model_type,
            'params': self.params,
            'metrics': self.metrics
        }, save_path)
    
    def load_model(self, path: Optional[str] = None):
        load_path = path or self.model_path
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        data = joblib.load(load_path)
        self.model = data['model']
        self.model_type = data['model_type']
        self.params = data['params']
        self.metrics = data.get('metrics', {})
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'model_type': self.model_type,
            'params': self.params,
            'metrics': self.metrics,
            'is_trained': self.model is not None
        }
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_names, self.model.feature_importances_))
        elif hasattr(self.model, 'coef_'):
            return dict(zip(self.feature_names, self.model.coef_))
        return None
    
    @property
    def feature_names(self):
        return self.params.get('feature_names', [])
    
    @feature_names.setter
    def feature_names(self, names):
        self.params['feature_names'] = names