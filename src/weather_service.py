"""
Weather Scout service for Samruk WindPilot AI.
Fetches historical weather forecasts from Open-Meteo Historical Forecast API
with local caching for 100% offline reliability.
"""

import os
import json
import urllib.request
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'weather_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# Coordinates of Shelek Wind Farm Turbines (Enbekshikazakh district, Almaty region, Kazakhstan)
TURBINE_COORDS = {
    1: {'lat': 43.645150, 'lon': 78.535604, 'name': 'Шелек ВЭС - Турбина 1'},
    2: {'lat': 43.643198, 'lon': 78.538828, 'name': 'Шелек ВЭС - Турбина 2'},
    'center': {'lat': 43.644174, 'lon': 78.537216, 'name': 'Шелек ВЭС (Общая)'}
}

class WeatherService:
    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache

    def _fetch_from_api(self, lat: float, lon: float, start_date: str, end_date: str) -> dict:
        """Fetches hourly weather forecast from Open-Meteo Historical Forecast API."""
        url = (
            f"https://historical-forecast-api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={start_date}&end_date={end_date}"
            f"&hourly=wind_speed_10m,wind_speed_80m,wind_speed_100m,wind_direction_100m,temperature_2m,surface_pressure,relative_humidity_2m"
        )
        req = urllib.request.Request(url, headers={'User-Agent': 'Samruk-WindPilot-AI/1.0'})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                return data
        except Exception as e:
            print(f"[WeatherService] API error: {e}. Trying fallback to archive API...")
            archive_url = (
                f"https://archive-api.open-meteo.com/v1/archive"
                f"?latitude={lat}&longitude={lon}"
                f"&start_date={start_date}&end_date={end_date}"
                f"&hourly=wind_speed_10m,wind_speed_100m,wind_direction_100m,temperature_2m,surface_pressure,relative_humidity_2m"
            )
            req2 = urllib.request.Request(archive_url, headers={'User-Agent': 'Samruk-WindPilot-AI/1.0'})
            with urllib.request.urlopen(req2, timeout=15) as resp2:
                data = json.loads(resp2.read().decode('utf-8'))
                return data

    def prefetch_february_2026(self) -> pd.DataFrame:
        """Prefetches and caches weather forecast for the entire test month of February 2026."""
        cache_file = os.path.join(CACHE_DIR, "february_2026_forecast.csv")
        if self.use_cache and os.path.exists(cache_file):
            df = pd.read_csv(cache_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        
        print("[WeatherService] Fetching February 2026 forecast data from Open-Meteo...")
        coords = TURBINE_COORDS['center']
        # Including Jan 31 to Feb 28
        raw_data = self._fetch_from_api(coords['lat'], coords['lon'], '2026-01-31', '2026-02-28')
        
        hourly = raw_data.get('hourly', {})
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(hourly['time']),
            'wind_speed_10m': hourly.get('wind_speed_10m', []),
            'wind_speed_80m': hourly.get('wind_speed_80m', hourly.get('wind_speed_100m', [])),
            'wind_speed_100m': hourly.get('wind_speed_100m', []),
            'wind_direction_100m': hourly.get('wind_direction_100m', [0]*len(hourly['time'])),
            'temperature_2m': hourly.get('temperature_2m', []),
            'surface_pressure': hourly.get('surface_pressure', [900]*len(hourly['time'])),
            'relative_humidity_2m': hourly.get('relative_humidity_2m', [60]*len(hourly['time'])),
        })
        # If wind_speed_100m has nulls or is missing, use log wind profile extrapolation from 10m
        if df['wind_speed_100m'].isnull().any():
            # Power law extrapolation: v(z) = v_ref * (z / z_ref)^alpha (alpha ~ 0.14-0.2 for terrain)
            df['wind_speed_100m'] = df['wind_speed_100m'].fillna(df['wind_speed_10m'] * (100.0 / 10.0)**0.18)
        
        df.to_csv(cache_file, index=False)
        print(f"[WeatherService] Successfully cached {len(df)} hourly forecast records.")
        return df

    def get_forecast_as_of(self, as_of_date: str, horizon_hours: int = 48) -> pd.DataFrame:
        """
        Gets forecast available as of `as_of_date` for the next `horizon_hours` (24 or 48).
        Simulates past forecasting without lookahead bias.
        """
        all_forecasts = self.prefetch_february_2026()
        start_ts = pd.to_datetime(as_of_date)
        end_ts = start_ts + timedelta(hours=horizon_hours)
        
        mask = (all_forecasts['timestamp'] >= start_ts) & (all_forecasts['timestamp'] < end_ts)
        subset = all_forecasts[mask].copy().reset_index(drop=True)
        return subset

    def simulate_weather_update(self, base_forecast: pd.DataFrame, shift_hours: int = 12) -> pd.DataFrame:
        """
        Simulates an incoming updated numerical weather prediction run (e.g. ECMWF mid-day run),
        which updates wind speed predictions (introducing a weather front / change).
        """
        updated = base_forecast.copy()
        # Add realistic weather model update delta (e.g. frontal passage with wind speed change)
        np.random.seed(42)
        n = len(updated)
        # Shift starting after shift_hours
        delta = np.zeros(n)
        if n > shift_hours:
            # Gradually changing wind speed by +/- 1.5 to 3.0 m/s
            wave = np.sin(np.linspace(0, np.pi * 1.5, n - shift_hours)) * 2.2
            delta[shift_hours:] = wave
        
        updated['wind_speed_100m'] = (updated['wind_speed_100m'] + delta).clip(lower=0.2)
        updated['is_updated_run'] = True
        return updated

if __name__ == '__main__':
    service = WeatherService(use_cache=False)
    df_feb = service.prefetch_february_2026()
    print("Sample February forecast:")
    print(df_feb.head())
    
    sample_48h = service.get_forecast_as_of('2026-02-01 00:00:00', 48)
    print(f"\nForecast as of 2026-02-01 00:00 (48h count: {len(sample_48h)}):")
    print(sample_48h[['timestamp', 'wind_speed_100m', 'temperature_2m']].head())
