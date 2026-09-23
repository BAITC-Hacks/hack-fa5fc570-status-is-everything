"""
Walk-Forward Backtesting Simulator for Samruk WindPilot AI.
Simulates daily sequential forecasting across all days of February 2026
(from Jan 31 to Feb 28, 2026) on 24h and 48h horizons, as required by the hackathon task.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.agent import WindFarmAgent, TURBINE_RATED_CAPACITY_MW

DATA_DIR = os.path.join(REPO_ROOT, 'data')

def run_february_walk_forward_simulation(horizon_hours: int = 48) -> pd.DataFrame:
    """
    Executes walk-forward forecast simulation for each day:
    31 Jan 2026, 01 Feb 2026, ..., 28 Feb 2026.
    """
    agent = WindFarmAgent(use_cache=True)
    
    # Dates list: 31 Jan to 28 Feb
    start_date = datetime(2026, 1, 31)
    end_date = datetime(2026, 2, 28)
    
    current_dt = start_date
    daily_summaries = []
    all_hourly_predictions = []
    
    print(f"=== Starting Walk-Forward Simulation (Horizon: {horizon_hours}h) ===")
    while current_dt <= end_date:
        date_str = current_dt.strftime('%Y-%m-%d 00:00:00')
        print(f"-> Simulating day: {current_dt.strftime('%Y-%m-%d')}...")
        
        try:
            cycle_result = agent.execute_agent_cycle(date_str, horizon_hours=horizon_hours, simulate_update=False)
            f_df = cycle_result['forecast'].copy()
            f_df['forecast_as_of'] = date_str
            all_hourly_predictions.append(f_df)
            
            # Next 24 hours stats for this day
            next_24h = f_df.head(24)
            daily_summaries.append({
                'date': current_dt.strftime('%Y-%m-%d'),
                'avg_wind_speed_100m': next_24h['wind_speed_100m'].mean(),
                'max_wind_speed_100m': next_24h['wind_speed_100m'].max(),
                'avg_temp_c': next_24h['temperature_2m'].mean(),
                'daily_energy_mwh': next_24h['mw_cluster'].sum(),
                'avg_kium_pct': next_24h['power_cluster'].mean() * 100.0,
                'status': cycle_result['audit']['status'],
                'risk_count': len(cycle_result['audit']['risks'])
            })
        except Exception as e:
            print(f"Warning for {date_str}: {e}")
            
        current_dt += timedelta(days=1)
        
    df_daily = pd.DataFrame(daily_summaries)
    df_all_hourly = pd.concat(all_hourly_predictions, ignore_index=True)
    
    # Save results to data folder
    daily_csv = os.path.join(DATA_DIR, f"february_daily_summary_{horizon_hours}h.csv")
    hourly_csv = os.path.join(DATA_DIR, f"february_hourly_walkforward_{horizon_hours}h.csv")
    
    df_daily.to_csv(daily_csv, index=False)
    df_all_hourly.to_csv(hourly_csv, index=False)
    
    print(f"\n=== Simulation Complete ===")
    print(f"Total simulated days: {len(df_daily)}")
    print(f"Total forecast generation for Feb 2026 (first 24h slice): {df_daily['daily_energy_mwh'].sum():.2f} MWh")
    print(f"Average monthly Capacity Factor (КИУМ): {df_daily['avg_kium_pct'].mean():.1f}%")
    print(f"Saved daily summary to: {daily_csv}")
    return df_daily

if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    run_february_walk_forward_simulation(48)
