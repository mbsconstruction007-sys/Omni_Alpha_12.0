'''Step 6: Machine Learning Predictions - Real Implementation'''

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import talib
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
import os

class MLPredictionEngine:
    def __init__(self, api):
        self.api = api
        self.models = {}
        self.scaler = StandardScaler()
        self.prediction_cache = {}
        
    def prepare_features(self, symbol, days=30):
        '''Extract technical indicators as features'''
        try:
            # Get historical data
            bars = self.api.get_bars(
                symbol, 
                '1Day',
                start=(datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
                end=datetime.now().strftime('%Y-%m-%d')
            ).df
            
            if len(bars) < 20:
                return None
                
            # Calculate technical indicators
            features = pd.DataFrame()
            
            # Price-based features
            features['returns'] = bars['close'].pct_change()
            features['volatility'] = features['returns'].rolling(5).std()
            
            # Moving averages
            features['sma_5'] = bars['close'].rolling(5).mean()
            features['sma_20'] = bars['close'].rolling(20).mean()
            features['sma_ratio'] = features['sma_5'] / features['sma_20']
            
            # RSI
            features['rsi'] = talib.RSI(bars['close'].values, timeperiod=14)
            
            # MACD
            macd, signal, hist = talib.MACD(bars['close'].values)
            features['macd'] = macd
            features['macd_signal'] = signal
            features['macd_hist'] = hist
            
            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(bars['close'].values)
            features['bb_upper'] = upper
            features['bb_lower'] = lower
            features['bb_width'] = upper - lower
            features['bb_position'] = (bars['close'] - lower) / (upper - lower)
            
            # Volume features
            features['volume_ratio'] = bars['volume'] / bars['volume'].rolling(20).mean()
            
            # Clean data
            features = features.dropna()
            
            return features
        except Exception as e:
            print(f"Error preparing features for {symbol}: {e}")
            return None
    
    def train_model(self, symbol):
        '''Train a model for specific symbol'''
        features = self.prepare_features(symbol, days=90)
        
        if features is None or len(features) < 50:
            return False
        
        try:
            # Create target (1 if price goes up next day, 0 if down)
            target = (features['returns'].shift(-1) > 0).astype(int)[:-1]
            features = features[:-1]  # Remove last row (no target)
            
            # Split features
            X = features.drop(['returns'], axis=1)
            y = target
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
            
            # Fit scaler and model
            X_scaled = self.scaler.fit_transform(X)
            model.fit(X_scaled, y)
            
            # Store model
            self.models[symbol] = model
            
            # Calculate accuracy
            accuracy = model.score(X_scaled, y)
            
            return {
                'symbol': symbol,
                'accuracy': accuracy,
                'features': list(X.columns),
                'samples': len(X)
            }
        except Exception as e:
            print(f"Error training model for {symbol}: {e}")
            return False
    
    def predict(self, symbol):
        '''Make prediction for symbol'''
        try:
            # Check cache
            cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d')}"
            if cache_key in self.prediction_cache:
                return self.prediction_cache[cache_key]
            
            # Train model if not exists
            if symbol not in self.models:
                training_result = self.train_model(symbol)
                if not training_result:
                    return None
            
            # Prepare current features
            features = self.prepare_features(symbol, days=30)
            if features is None or len(features) == 0:
                return None
            
            # Get last row for prediction
            X = features.drop(['returns'], axis=1).iloc[-1:].values
            X_scaled = self.scaler.transform(X)
            
            # Predict
            model = self.models[symbol]
            prediction = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0]
            
            # Get feature importance
            feature_importance = dict(zip(
                features.drop(['returns'], axis=1).columns,
                model.feature_importances_
            ))
            
            # Create recommendation
            if probability[1] > 0.65:
                action = 'STRONG BUY'
            elif probability[1] > 0.55:
                action = 'BUY'
            elif probability[1] < 0.35:
                action = 'STRONG SELL'
            elif probability[1] < 0.45:
                action = 'SELL'
            else:
                action = 'HOLD'
            
            result = {
                'symbol': symbol,
                'prediction': 'UP' if prediction == 1 else 'DOWN',
                'confidence': max(probability) * 100,
                'action': action,
                'probability_up': probability[1] * 100,
                'probability_down': probability[0] * 100,
                'top_features': sorted(feature_importance.items(), 
                                      key=lambda x: x[1], reverse=True)[:3],
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache result
            self.prediction_cache[cache_key] = result
            
            return result
        except Exception as e:
            print(f"Error predicting for {symbol}: {e}")
            return None
    
    def save_models(self, path='data/models/'):
        '''Save trained models'''
        try:
            os.makedirs(path, exist_ok=True)
            for symbol, model in self.models.items():
                joblib.dump(model, f'{path}{symbol}_model.pkl')
            joblib.dump(self.scaler, f'{path}scaler.pkl')
        except Exception as e:
            print(f"Error saving models: {e}")
    
    def load_models(self, symbols, path='data/models/'):
        '''Load saved models'''
        for symbol in symbols:
            try:
                self.models[symbol] = joblib.load(f'{path}{symbol}_model.pkl')
            except:
                pass
        try:
            self.scaler = joblib.load(f'{path}scaler.pkl')
        except:
            pass
