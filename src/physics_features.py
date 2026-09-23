"""
Physics-Informed Feature Engineering for Wind Turbines.
Implements IEC 61400-12 air density correction, kinetic wind power potential,
and meteorological feature transformations.
"""

import numpy as np
import pandas as pd

R_SPECIFIC_AIR = 287.058  # J / (kg * K)
STANDARD_AIR_DENSITY = 1.225  # kg / m^3 at sea level, 15°C

def calculate_air_density(temp_c: np.ndarray, pressure_hpa: np.ndarray = None) -> np.ndarray:
    """
    Calculates air density rho (kg/m^3) using ideal gas law:
    rho = P / (R * T)
    """
    temp_k = temp_c + 273.15
    if pressure_hpa is None:
        # Default typical elevation pressure for Shelek plateau (~850-950 hPa)
        pressure_pa = 940.0 * 100.0
    else:
        pressure_pa = pressure_hpa * 100.0
    
    rho = pressure_pa / (R_SPECIFIC_AIR * temp_k)
    return np.clip(rho, 1.0, 1.45)

def normalize_wind_speed_iec(wind_speed: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """
    Applies standard IEC 61400-12 air density correction:
    v_norm = v * (rho / rho_0)^(1/3)
    """
    return wind_speed * (rho / STANDARD_AIR_DENSITY) ** (1.0 / 3.0)

def extract_physics_features(df: pd.DataFrame, wind_col: str = 'wind_speed', temp_col: str = 'temperature', pressure_col: str = None) -> pd.DataFrame:
    """
    Enriches dataset with physics-informed aerodynamic and meteorological features.
    """
    df = df.copy()
    
    # Wind speed base
    v = df[wind_col].values.astype(float)
    t = df[temp_col].values.astype(float)
    p = df[pressure_col].values.astype(float) if pressure_col and pressure_col in df.columns else None
    
    # 1. Air density
    rho = calculate_air_density(t, p)
    df['air_density'] = rho
    
    # 2. IEC Density-corrected wind speed
    v_iec = normalize_wind_speed_iec(v, rho)
    df['wind_speed_iec'] = v_iec
    
    # 3. Kinetic wind power potential (v^3)
    df['wind_power_potential'] = (v ** 3) / 1000.0
    df['wind_power_potential_iec'] = (v_iec ** 3) / 1000.0
    
    # 4. Wind speed polynomial terms for aerodynamic drag and lift
    df['wind_speed_sq'] = v ** 2
    
    # 5. Temporal / diurnal cycle features (mountain-valley wind cycles in Shelek)
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'])
        hours = ts.dt.hour.values
        df['hour'] = hours
        df['sin_hour'] = np.sin(2 * np.pi * hours / 24.0)
        df['cos_hour'] = np.cos(2 * np.pi * hours / 24.0)
        df['day_of_week'] = ts.dt.dayofweek.values
    else:
        df['hour'] = 12
        df['sin_hour'] = 0.0
        df['cos_hour'] = -1.0
    
    # 6. Turbine operational regimes (physics indicators)
    df['regime_cut_in'] = (v < 3.0).astype(int)              # Below cut-in: 0 power
    df['regime_ramp'] = ((v >= 3.0) & (v < 11.5)).astype(int) # Cubic power curve ramp
    df['regime_rated'] = ((v >= 11.5) & (v <= 22.0)).astype(int) # Rated output plateau (~1.0)
    df['regime_cut_out_risk'] = (v > 22.0).astype(int)       # Storm high-wind safety cut-out
    
    # 7. Icing risk indicator
    humidity = df['relative_humidity_2m'].values if 'relative_humidity_2m' in df.columns else np.full(len(df), 70.0)
    df['icing_risk'] = ((t < -1.0) & (humidity > 80.0)).astype(int)
    
    return df

if __name__ == '__main__':
    # Test feature extraction
    sample_df = pd.DataFrame({
        'timestamp': pd.date_range('2026-02-01 00:00', periods=5, freq='h'),
        'wind_speed': [2.5, 6.0, 12.0, 18.0, 23.5],
        'temperature': [-5.0, -3.0, 0.0, 2.0, 5.0],
        'surface_pressure': [945, 946, 947, 948, 949]
    })
    feat_df = extract_physics_features(sample_df)
    print("Engineered Physics Features:")
    print(feat_df[['timestamp', 'wind_speed', 'air_density', 'wind_speed_iec', 'regime_ramp', 'regime_rated', 'regime_cut_out_risk']])
