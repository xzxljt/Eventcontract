import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from typing import Dict, Any, Optional, List

class AdvancedFeatureEngineer:
    """高级特征工程"""
    
    @staticmethod
    def calculate_hurst_exponent(series, lags=range(2, 100)):
        """计算Hurst指数"""
        try:
            tau = []
            for lag in lags:
                differences = series.diff(lag).dropna()
                if len(differences) > 10:
                    tau.append(np.std(differences))
                else:
                    tau.append(np.nan)
            
            if len([t for t in tau if not np.isnan(t)]) > 10:
                poly = np.polyfit(np.log(lags[:len(tau)]), np.log(tau), 1)
                return poly[0]
            else:
                return 0.5
        except:
            return 0.5
    
    @staticmethod
    def calculate_fractal_dimension(series, window=50):
        """计算分形维度"""
        try:
            fd_values = []
            for i in range(len(series) - window):
                window_data = series.iloc[i:i+window].values
                L = np.sum(np.abs(np.diff(window_data)))
                price_range = window_data[-1] - window_data[0]
                
                if price_range == 0:
                    fd_values.append(1.5)
                else:
                    try:
                        d = np.log(window) / (np.log(window) + np.log(L/(abs(price_range) + 1e-9)))
                        if np.isfinite(d):
                            fd_values.append(d)
                        else:
                            fd_values.append(1.5)
                    except:
                        fd_values.append(1.5)
            
            fd_series = pd.Series(fd_values, index=series.index[window:])
            return fd_series
        except:
            return pd.Series([1.5]*len(series), index=series.index)
    
    @staticmethod
    def calculate_market_regime(series):
        """计算市场状态"""
        try:
            trend = np.where(series > series.rolling(50).mean(), 1, -1)
            volatility = series.rolling(20).std()
            high_vol = volatility > volatility.quantile(0.7)
            low_vol = volatility < volatility.quantile(0.3)
            
            regime = trend + np.where(high_vol, -0.5, np.where(low_vol, 0.5, 0))
            return np.where(regime > 1, 2, np.where(regime < -1, -2, np.where(regime > 0, 1, -1)))
        except:
            return np.zeros(len(series))
    
    @staticmethod
    def calculate_trend_strength(series):
        """计算趋势强度"""
        try:
            adx = AdvancedFeatureEngineer.calculate_adx(series)
            return adx / 100
        except:
            return np.zeros(len(series))
    
    @staticmethod
    def calculate_adx(series, period=14):
        """计算ADX指标"""
        try:
            high = series
            low = series
            close = series
            
            high_diff = high.diff()
            low_diff = -low.diff()
            
            plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
            
            tr = np.zeros(len(series))
            for i in range(1, len(series)):
                tr[i] = max(
                    high.iloc[i] - low.iloc[i],
                    abs(high.iloc[i] - close.iloc[i-1]),
                    abs(low.iloc[i] - close.iloc[i-1])
                )
            
            atr = pd.Series(tr).rolling(period).mean()
            plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
            minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
            adx = dx.rolling(period).mean()
            
            return adx.fillna(20)
        except:
            return pd.Series([20]*len(series))
    
    @staticmethod
    def add_advanced_features(df):
        """添加高级特征"""
        df_copy = df.copy()
        
        if 'close' in df_copy.columns:
            close = df_copy['close']
            
            # 添加Hurst指数
            hurst = AdvancedFeatureEngineer.calculate_hurst_exponent(close)
            df_copy['hurst_exponent'] = hurst
            
            # 添加分形维度
            fd_series = AdvancedFeatureEngineer.calculate_fractal_dimension(close)
            df_copy['fractal_dimension'] = fd_series.reindex(df_copy.index).ffill().fillna(1.5)
            
            # 添加市场状态
            regime = AdvancedFeatureEngineer.calculate_market_regime(close)
            df_copy['market_regime'] = regime
            
            # 添加趋势强度
            trend_strength = AdvancedFeatureEngineer.calculate_trend_strength(close)
            df_copy['trend_strength'] = trend_strength
            
            # 添加波动率状态
            atr_percent = df_copy['atr_percent'] if 'atr_percent' in df_copy.columns else \
                          df_copy['close'].rolling(14).std() / df_copy['close']
            df_copy['volatility_regime'] = np.where(
                atr_percent > atr_percent.rolling(50).mean() * 1.5, 2,
                np.where(atr_percent < atr_percent.rolling(50).mean() * 0.7, -2,
                         np.where(atr_percent > atr_percent.rolling(50).mean(), 1, -1))
            )
            
            # 添加成交量状态
            if 'volume' in df_copy.columns:
                vol_over_ma = df_copy['volume'] / df_copy['volume'].rolling(21).mean()
                df_copy['volume_regime'] = np.where(
                    vol_over_ma > 1.5, 2,
                    np.where(vol_over_ma < 0.7, -2,
                             np.where(vol_over_ma > 1.0, 1, -1))
                )
            
            # 添加价格动量
            df_copy['price_accel'] = close.diff().diff()
            df_copy['price_velocity'] = close.diff()
            
            # 添加相关性特征
            if 'volume' in df_copy.columns:
                df_copy['price_vol_corr'] = close.rolling(20).corr(df_copy['volume'])
                df_copy['price_vol_corr_ma'] = df_copy['price_vol_corr'].rolling(5).mean()
        
        # 处理NaN值
        df_copy = df_copy.ffill().bfill().fillna(0)
        
        return df_copy

class DataPreprocessor:
    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or {}
        self.scaler_type = self.params.get('scaler_type', 'standard')
        self.scaler = None
        self.feature_selector = None
        self.selected_features = None
        self.feature_names = None
    
    def _initialize_scaler(self):
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unsupported scaler type: {self.scaler_type}")
    
    def preprocess_data(self, df: pd.DataFrame, is_training: bool = True, target_series: pd.Series = None) -> pd.DataFrame:
        df_copy = df.copy()
        
        if 'open_time' in df_copy.columns:
            df_copy = df_copy.drop('open_time', axis=1)
        if 'close_time' in df_copy.columns:
            df_copy = df_copy.drop('close_time', axis=1)
        
        # 添加高级特征
        if self.params.get('use_advanced_features', True):
            df_copy = AdvancedFeatureEngineer.add_advanced_features(df_copy)
        
        # 确保预测时使用与训练时相同的特征集
        if not is_training and hasattr(self, 'feature_names') and self.feature_names:
            # 删除训练时没有的特征
            extra_features = [col for col in df_copy.columns if col not in self.feature_names]
            if extra_features:
                df_copy = df_copy.drop(extra_features, axis=1)
            
            # 添加训练时有的但预测时没有的特征，设置为默认值
            missing_features = [col for col in self.feature_names if col not in df_copy.columns]
            if missing_features:
                for feature in missing_features:
                    df_copy[feature] = 0.0
            
            # 确保列顺序与训练时一致
            df_copy = df_copy[self.feature_names]
        else:
            # 训练时更新特征名称
            self.feature_names = list(df_copy.columns)
        
        if is_training:
            self._initialize_scaler()
            scaled_data = self.scaler.fit_transform(df_copy)
            
            if self.params.get('use_feature_selection', False) and target_series is not None:
                k = self.params.get('n_features', min(10, len(df_copy.columns)))
                self.feature_selector = SelectKBest(f_regression, k=k)
                scaled_data = self.feature_selector.fit_transform(scaled_data, target_series)
                self.selected_features = [self.feature_names[i] for i in self.feature_selector.get_support(indices=True)]
        else:
            if self.scaler is None:
                self._initialize_scaler()
                self.scaler.fit(df_copy)
            scaled_data = self.scaler.transform(df_copy)
            
            if self.feature_selector is not None:
                scaled_data = self.feature_selector.transform(scaled_data)
        
        if self.selected_features:
            feature_names = self.selected_features
        else:
            feature_names = self.feature_names
        
        return pd.DataFrame(scaled_data, columns=feature_names, index=df_copy.index)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if self.feature_selector is not None:
            return dict(zip(self.feature_names, self.feature_selector.scores_))
        return None
    
    def save_preprocessor(self, path: str):
        import joblib
        joblib.dump({
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'selected_features': self.selected_features,
            'feature_names': self.feature_names,
            'params': self.params
        }, path)
    
    def load_preprocessor(self, path: str):
        import joblib
        data = joblib.load(path)
        self.scaler = data['scaler']
        self.feature_selector = data['feature_selector']
        self.selected_features = data['selected_features']
        self.feature_names = data['feature_names']
        self.params = data['params']
        self.scaler_type = self.params.get('scaler_type', 'standard')