import pandas as pd
from typing import List, Tuple

class TechnicalIndicatorModels:
    """技术指标模型集合"""
    
    @staticmethod
    def trend_following_model(df):
        try:
            trend_score = 0
            for period in [5, 20, 50]:
                ma_key = f'ma{period}'
                if ma_key in df.columns:
                    ma = df[ma_key].iloc[-1]
                    ma_prev = df[ma_key].iloc[-2] if len(df) > 1 else ma
                    trend_score += 1 if ma > ma_prev else -1
            
            momentum = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10]
            trend_score += 2 if momentum > 0.01 else -2 if momentum < -0.01 else 0
            
            if 'adx_trend' in df.columns:
                trend_score += df['adx_trend'].iloc[-1] * 1.5
            
            signal = 1 if trend_score > 0 else 0
            confidence = min(0.9, abs(trend_score) * 0.2)
            
            return signal, confidence, "TREND_FOLLOW"
        except:
            return 0, 0.5, "TREND_FOLLOW"
    
    @staticmethod
    def mean_reversion_model(df):
        try:
            mean_reversion_score = 0
            
            if 'bb_position' in df.columns:
                bb_pos = df['bb_position'].iloc[-1]
                if bb_pos > 0.8:
                    mean_reversion_score -= 2
                elif bb_pos < 0.2:
                    mean_reversion_score += 2
            
            if 'rsi' in df.columns:
                rsi = df['rsi'].iloc[-1]
                if rsi > 75:
                    mean_reversion_score -= 1.5
                elif rsi < 25:
                    mean_reversion_score += 1.5
            
            signal = 1 if mean_reversion_score > 0 else 0
            confidence = min(0.85, abs(mean_reversion_score) * 0.25)
            
            return signal, confidence, "MEAN_REVERSION"
        except:
            return 0, 0.5, "MEAN_REVERSION"
    
    @staticmethod
    def momentum_model(df):
        try:
            momentum_score = 0
            
            ret5 = df['ret5'].iloc[-1] if 'ret5' in df.columns else 0
            ret10 = df['ret10'].iloc[-1] if 'ret10' in df.columns else 0
            
            momentum_score += ret5 * 100
            momentum_score += ret10 * 50
            
            if len(df) > 10 and 'ret5' in df.columns:
                mom_accel = ret5 - df['ret5'].iloc[-5]
                momentum_score += mom_accel * 200
            
            if 'macd_hist' in df.columns:
                macd_hist = df['macd_hist'].iloc[-1]
                momentum_score += macd_hist * 10
            
            signal = 1 if momentum_score > 0 else 0
            confidence = min(0.8, abs(momentum_score) * 0.3)
            
            return signal, confidence, "MOMENTUM"
        except:
            return 0, 0.5, "MOMENTUM"
    
    @staticmethod
    def volume_model(df):
        try:
            volume_score = 0
            
            price_change = df['ret1'].iloc[-1] if 'ret1' in df.columns else 0
            volume_change = df['vol_chg'].iloc[-1] if 'vol_chg' in df.columns else 0
            
            if price_change > 0 and volume_change > 0.5:
                volume_score += 2
            elif price_change < 0 and volume_change > 0.5:
                volume_score -= 2
            
            if 'obv_ma_diff' in df.columns:
                obv_trend = df['obv_ma_diff'].iloc[-1]
                volume_score += obv_trend * 0.1
            
            if 'vol_over_ma' in df.columns:
                vol_ratio = df['vol_over_ma'].iloc[-1]
                if vol_ratio > 2.0:
                    volume_score += 1 if price_change > 0 else -1
            
            signal = 1 if volume_score > 0 else 0
            confidence = min(0.75, abs(volume_score) * 0.4)
            
            return signal, confidence, "VOLUME"
        except:
            return 0, 0.5, "VOLUME"
    
    @staticmethod
    def volatility_model(df):
        try:
            volatility_score = 0
            
            if 'atr_percent' in df.columns:
                atr_pct = df['atr_percent'].iloc[-1]
                atr_ma = df['atr_percent'].rolling(20).mean().iloc[-1]
                
                if atr_pct > atr_ma * 1.5:
                    volatility_score -= 1.5
                elif atr_pct < atr_ma * 0.7:
                    volatility_score += 1
            
            price_std = df['close'].rolling(20).std().iloc[-1] / df['close'].rolling(20).mean().iloc[-1]
            volatility_score -= price_std * 10
            
            signal = 1 if volatility_score > 0 else 0
            confidence = min(0.7, abs(volatility_score) * 0.5)
            
            return signal, confidence, "VOLATILITY"
        except:
            return 0, 0.5, "VOLATILITY"
    
    @staticmethod
    def breakout_model(df):
        try:
            breakout_score = 0
            
            resistance = df['high'].rolling(20).max().iloc[-1]
            support = df['low'].rolling(20).min().iloc[-1]
            current_price = df['close'].iloc[-1]
            
            if current_price > resistance * 1.002:
                breakout_score += 2
            elif current_price < support * 0.998:
                breakout_score -= 2
            
            if 'vol_over_ma' in df.columns:
                vol_ratio = df['vol_over_ma'].iloc[-1]
                if vol_ratio > 1.5:
                    breakout_score *= 1.5
            
            signal = 1 if breakout_score > 0 else 0
            confidence = min(0.85, abs(breakout_score) * 0.4)
            
            return signal, confidence, "BREAKOUT"
        except:
            return 0, 0.5, "BREAKOUT"
    
    @staticmethod
    def pattern_model(df):
        try:
            pattern_score = 0
            
            if 'body_ratio' in df.columns:
                body_ratio = df['body_ratio'].iloc[-1]
                if body_ratio < 0.1:
                    pattern_score -= 0.5
            
            if len(df) > 10:
                up_count = sum(1 for i in range(1, 6) if df['close'].iloc[-i] > df['close'].iloc[-i-1])
                if up_count >= 4:
                    pattern_score -= 0.5
                elif up_count <= 1:
                    pattern_score += 0.5
            
            signal = 1 if pattern_score > 0 else 0
            confidence = min(0.7, abs(pattern_score) * 0.6)
            
            return signal, confidence, "PATTERN"
        except:
            return 0, 0.5, "PATTERN"
    
    @staticmethod
    def market_sentiment_model(df):
        try:
            sentiment_score = 0
            
            if 'market_regime' in df.columns:
                regime = df['market_regime'].iloc[-1]
                sentiment_score += regime * 0.5
            
            if 'trend_strength' in df.columns:
                trend_strength = df['trend_strength'].iloc[-1]
                sentiment_score += trend_strength
            
            signal = 1 if sentiment_score > 0 else 0
            confidence = min(0.65, abs(sentiment_score) * 0.8)
            
            return signal, confidence, "SENTIMENT"
        except:
            return 0, 0.5, "SENTIMENT"
    
    @staticmethod
    def get_all_predictions(df):
        models = [
            TechnicalIndicatorModels.trend_following_model,
            TechnicalIndicatorModels.mean_reversion_model,
            TechnicalIndicatorModels.momentum_model,
            TechnicalIndicatorModels.volume_model,
            TechnicalIndicatorModels.volatility_model,
            TechnicalIndicatorModels.breakout_model,
            TechnicalIndicatorModels.pattern_model,
            TechnicalIndicatorModels.market_sentiment_model
        ]
        
        predictions = []
        for model_func in models:
            try:
                signal, confidence, name = model_func(df)
                predictions.append((name, signal, confidence))
            except Exception:
                continue
        
        return predictions