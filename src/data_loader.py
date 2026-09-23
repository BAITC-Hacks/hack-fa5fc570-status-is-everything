"""
Data loader module for Samruk WindPilot AI.
Loads 10-minute SCADA measurements, aggregates to 1-hour intervals,
and calculates empirical wind power curves.
"""

import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

def load_turbine_raw(turbine_num: int = 1) -> pd.DataFrame:
    """Loads raw 10-minute data for turbine 1 or 2, checking teammates folders first."""
    candidates = [
        # Teammates folder structure
        os.path.join(os.path.dirname(DATA_DIR), 'ВЭС', 'data', 'raw', f"Dataset HackAlemAI для участников 11.03.2023-28.02.2026 - turbine {turbine_num}.csv"),
        os.path.join(DATA_DIR, f"turbine_{turbine_num}.csv"),
        os.path.join(os.path.expanduser('~'), 'Downloads', f"Dataset HackAlemAI для участников 11.03.2023-28.02.2026 - turbine {turbine_num}.csv")
    ]
    filepath = None
    for c in candidates:
        if os.path.exists(c):
            filepath = c
            break
            
    if not filepath:
        raise FileNotFoundError(f"Turbine {turbine_num} raw CSV file not found in candidates: {candidates}")
    
    print(f"[data_loader] Loading raw data from: {filepath}")
    df = pd.read_csv(filepath)
    df = df.rename(columns={
        'Статистическое время': 'timestamp',
        'Средняя скорость ветра(m/s)': 'wind_speed',
        'Нормализованная активная мощность': 'power',
        'Средняя температура окружающей среды(°C)': 'temperature'
    })
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

def aggregate_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates 10-minute records to 1-hour intervals."""
    df = df.copy()
    df['datetime_hour'] = df['timestamp'].dt.floor('h')
    
    hourly = df.groupby('datetime_hour').agg(
        wind_speed=('wind_speed', 'mean'),
        power=('power', 'mean'),
        temperature=('temperature', 'mean'),
        sample_count=('wind_speed', 'count')
    ).reset_index()
    
    hourly = hourly.rename(columns={'datetime_hour': 'timestamp'})
    # Filter out hours with insufficient samples (< 3 out of 6)
    hourly = hourly[hourly['sample_count'] >= 3].reset_index(drop=True)
    hourly['power'] = hourly['power'].clip(lower=0.0, upper=1.0)
    return hourly

def get_hourly_data(turbine_num: int = 1, use_cache: bool = True) -> pd.DataFrame:
    """Gets hourly aggregated data, using cache if available."""
    cache_path = os.path.join(DATA_DIR, f"hourly_turbine_{turbine_num}.csv")
    if use_cache and os.path.exists(cache_path):
        cached_df = pd.read_csv(cache_path)
        cached_df['timestamp'] = pd.to_datetime(cached_df['timestamp'])
        return cached_df
    
    raw_df = load_turbine_raw(turbine_num)
    hourly_df = aggregate_to_hourly(raw_df)
    hourly_df.to_csv(cache_path, index=False)
    return hourly_df

def compute_power_curve(df: pd.DataFrame, bin_width: float = 0.5) -> pd.DataFrame:
    """
    Computes IEC-standard empirical power curve:
    average power for each wind speed bin.
    """
    bins = np.arange(0, 25 + bin_width, bin_width)
    labels = bins[:-1] + bin_width / 2.0
    df = df.copy()
    df['wind_bin'] = pd.cut(df['wind_speed'], bins=bins, labels=labels)
    
    curve = df.groupby('wind_bin', observed=False).agg(
        power_mean=('power', 'mean'),
        power_std=('power', 'std'),
        count=('power', 'count')
    ).reset_index()
    
    curve['wind_bin'] = curve['wind_bin'].astype(float)
    # Fill gaps by interpolation
    curve['power_mean'] = curve['power_mean'].interpolate(method='linear').bfill().ffill()
    curve['power_mean'] = curve['power_mean'].clip(0.0, 1.0)
    return curve

if __name__ == '__main__':
    print("Pre-processing hourly data for both turbines...")
    h1 = get_hourly_data(1, use_cache=False)
    h2 = get_hourly_data(2, use_cache=False)
    print(f"Turbine 1 hourly rows: {len(h1)}, range: {h1['timestamp'].min()} to {h1['timestamp'].max()}")
    print(f"Turbine 2 hourly rows: {len(h2)}, range: {h2['timestamp'].min()} to {h2['timestamp'].max()}")
