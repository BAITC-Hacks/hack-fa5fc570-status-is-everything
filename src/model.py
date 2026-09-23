"""
Machine Learning & Physics-Informed Forecasting Core.
Trains LightGBM models for Turbine 1 and Turbine 2 on historical SCADA data (2023-2026),
evaluates metrics (MAE, RMSE, WAPE, R2), and saves model artifacts.
"""

import os
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_loader import get_hourly_data, compute_power_curve
from src.physics_features import extract_physics_features

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_COLS = [
    'wind_speed',
    'wind_speed_iec',
    'wind_power_potential',
    'wind_power_potential_iec',
    'wind_speed_sq',
    'air_density',
    'temperature',
    'hour',
    'sin_hour',
    'cos_hour',
    'regime_cut_in',
    'regime_ramp',
    'regime_rated',
    'regime_cut_out_risk'
]

class WindTurbineForecaster:
    def __init__(self, turbine_num: int):
        self.turbine_num = turbine_num
        self.model = None
        self.power_curve = None
        self.metrics = {}
        self.model_path = os.path.join(MODELS_DIR, f"turbine_{turbine_num}_model.joblib")

    def train(self, test_size_days: int = 60) -> dict:
        """
        Trains model using historical hourly data with chronological train/validation split.
        Last `test_size_days` (e.g. Dec 2025 - Jan 2026) used for out-of-time validation.
        """
        raw_hourly = get_hourly_data(self.turbine_num)
        featured_df = extract_physics_features(raw_hourly, 'wind_speed', 'temperature')
        self.power_curve = compute_power_curve(raw_hourly)
        
        # Chronological split
        split_date = featured_df['timestamp'].max() - pd.Timedelta(days=test_size_days)
        train_df = featured_df[featured_df['timestamp'] < split_date].dropna()
        val_df = featured_df[featured_df['timestamp'] >= split_date].dropna()
        
        X_train, y_train = train_df[FEATURE_COLS], train_df['power']
        X_val, y_val = val_df[FEATURE_COLS], val_df['power']
        
        # High-performance LightGBM regressor
        self.model = LGBMRegressor(
            n_estimators=350,
            learning_rate=0.04,
            max_depth=7,
            num_leaves=63,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            verbosity=-1
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate on out-of-time validation set
        val_preds = self.predict_features(X_val)
        
        mae = mean_absolute_error(y_val, val_preds)
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        r2 = r2_score(y_val, val_preds)
        wape = (np.sum(np.abs(y_val - val_preds)) / max(np.sum(y_val), 1e-5)) * 100.0
        
        self.metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'wape_pct': float(wape),
            'train_samples': len(train_df),
            'val_samples': len(val_df)
        }
        
        # Save model and artifacts
        joblib.dump({
            'model': self.model,
            'power_curve': self.power_curve,
            'metrics': self.metrics,
            'features': FEATURE_COLS
        }, self.model_path)
        
        return self.metrics

    def load_or_train(self):
        """Loads model from disk if exists, otherwise trains."""
        if os.path.exists(self.model_path):
            artifact = joblib.load(self.model_path)
            self.model = artifact['model']
            self.power_curve = artifact['power_curve']
            self.metrics = artifact.get('metrics', {})
            return self.metrics
        return self.train()

    def predict_features(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts normalized power output with physical boundary guards."""
        raw_pred = self.model.predict(X[FEATURE_COLS])
        
        # Physical guardrails
        # 1. Clip to [0, 1]
        clipped = np.clip(raw_pred, 0.0, 1.0)
        
        # 2. Hard physical limit: cut-in speed < 2.5 m/s -> 0 power
        wind_speeds = X['wind_speed'].values
        clipped[wind_speeds < 2.5] = 0.0
        
        # 3. Storm cut-out speed > 24.0 m/s -> 0 power
        clipped[wind_speeds > 24.0] = 0.0
        return clipped

    def forecast_weather(self, weather_df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes hourly weather forecast (from Open-Meteo) and outputs power forecast.
        Expects columns: 'timestamp', 'wind_speed_100m', 'temperature_2m', 'surface_pressure'.
        """
        temp_weather = weather_df.copy()
        temp_weather = temp_weather.rename(columns={
            'wind_speed_100m': 'wind_speed',
            'temperature_2m': 'temperature'
        })
        feat_df = extract_physics_features(
            temp_weather, 
            wind_col='wind_speed', 
            temp_col='temperature', 
            pressure_col='surface_pressure'
        )
        predictions = self.predict_features(feat_df)
        
        res = weather_df[['timestamp']].copy()
        res['wind_speed_100m'] = weather_df['wind_speed_100m']
        res['temperature_2m'] = weather_df['temperature_2m']
        res[f'power_t{self.turbine_num}'] = predictions
        
        # Add 90% confidence intervals based on validation error
        std_err = max(self.metrics.get('mae', 0.06) * 1.5, 0.05)
        res[f'power_t{self.turbine_num}_lower'] = np.clip(predictions - std_err, 0.0, 1.0)
        res[f'power_t{self.turbine_num}_upper'] = np.clip(predictions + std_err, 0.0, 1.0)
        return res

def train_all_models():
    """Trains models for both Turbine 1 and Turbine 2."""
    print("=== Training Wind Turbine Forecasting Models ===")
    forecaster1 = WindTurbineForecaster(1)
    m1 = forecaster1.train()
    print(f"Turbine 1: R2={m1['r2']:.4f}, MAE={m1['mae']:.4f}, WAPE={m1['wape_pct']:.2f}%")
    
    forecaster2 = WindTurbineForecaster(2)
    m2 = forecaster2.train()
    print(f"Turbine 2: R2={m2['r2']:.4f}, MAE={m2['mae']:.4f}, WAPE={m2['wape_pct']:.2f}%")

if __name__ == '__main__':
    train_all_models()
