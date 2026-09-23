"""
Agentic AI Core for Wind Power Forecasting (Samruk WindPilot Agent).
Orchestrates the full closed-loop pipeline:
1. Tool: Fetch Weather (Open-Meteo Historical Forecast API)
2. Tool: Feature Engineering (Physics & IEC 61400-12)
3. Tool: Run Forecast (Turbines 1 & 2 + Farm aggregate)
4. Tool: Audit Risks (Cut-out, Icing, Ramps, KEGOC constraints)
5. Tool: Event-Driven Re-calculation upon Weather Updates
6. Tool: Generate Dispatcher Briefing (RU / KZ)
"""

import os
import sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Add repo root to sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.weather_service import WeatherService
from src.model import WindTurbineForecaster

TURBINE_RATED_CAPACITY_MW = 2.5  # Typical rating for Shelek Wind Farm turbines (2 x 2.5 = 5.0 MW cluster)

class WindFarmAgent:
    def __init__(self, use_cache: bool = True):
        self.weather_service = WeatherService(use_cache=use_cache)
        self.forecaster1 = WindTurbineForecaster(1)
        self.forecaster2 = WindTurbineForecaster(2)
        self.forecaster1.load_or_train()
        self.forecaster2.load_or_train()
        self.trace_logs = []

    def _log(self, step_name: str, action: str, details: str, status: str = "INFO"):
        """Appends step execution trace for full transparency."""
        entry = {
            'step': step_name,
            'action': action,
            'details': details,
            'status': status,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        }
        self.trace_logs.append(entry)

    def tool_fetch_weather(self, as_of_date: str, horizon_hours: int = 48) -> pd.DataFrame:
        """Tool 1: Retrieves weather forecast available strictly as of as_of_date without lookahead."""
        self._log(
            "1. Погодный скаут (Weather Scout)",
            "Запрос архивного прогноза погоды",
            f"Координаты: 43.645°N, 78.535°E (Шелек). Горизонт: {horizon_hours}ч от {as_of_date}. Источник: Open-Meteo Historical Forecast API."
        )
        df_weather = self.weather_service.get_forecast_as_of(as_of_date, horizon_hours)
        if len(df_weather) == 0:
            self._log("1. Погодный скаут", "Ошибка данных", f"Нет прогноза на дату {as_of_date}", "ERROR")
            raise ValueError(f"No forecast data available for {as_of_date}")
        
        v_mean = df_weather['wind_speed_100m'].mean()
        v_max = df_weather['wind_speed_100m'].max()
        t_mean = df_weather['temperature_2m'].mean()
        self._log(
            "1. Погодный скаут",
            "Анализ метеоусловий",
            f"Получено {len(df_weather)} почасовых интервалов. Средняя скорость ветра на 100м: {v_mean:.1f} м/с (пик: {v_max:.1f} м/с), температура: {t_mean:.1f}°C."
        )
        return df_weather

    def tool_run_forecast(self, weather_df: pd.DataFrame) -> pd.DataFrame:
        """Tool 2 & 3: Physics feature engineering and model prediction for both turbines."""
        self._log(
            "2. Физико-математический расчёт",
            "Подготовка IEC признаков и запуск ML",
            "Учёт плотности воздуха rho(T, P), кинетического потенциала v^3, аэродинамической кривой IEC 61400-12."
        )
        f1 = self.forecaster1.forecast_weather(weather_df)
        f2 = self.forecaster2.forecast_weather(weather_df)
        
        combined = f1.copy()
        combined['power_t2'] = f2['power_t2']
        combined['power_t2_lower'] = f2['power_t2_lower']
        combined['power_t2_upper'] = f2['power_t2_upper']
        
        # Cluster aggregate normalized power [0..1]
        combined['power_cluster'] = (combined['power_t1'] + combined['power_t2']) / 2.0
        combined['power_cluster_lower'] = (combined['power_t1_lower'] + combined['power_t2_lower']) / 2.0
        combined['power_cluster_upper'] = (combined['power_t1_upper'] + combined['power_t2_upper']) / 2.0
        
        # Generation in physical megawatts (MW)
        combined['mw_t1'] = combined['power_t1'] * TURBINE_RATED_CAPACITY_MW
        combined['mw_t2'] = combined['power_t2'] * TURBINE_RATED_CAPACITY_MW
        combined['mw_cluster'] = combined['power_cluster'] * (TURBINE_RATED_CAPACITY_MW * 2)
        
        total_mwh = combined['mw_cluster'].sum()
        kium_pct = combined['power_cluster'].mean() * 100.0
        self._log(
            "3. Формирование почасового прогноза",
            "Расчёт генерации ВЭС",
            f"Прогноз сформирован. Суммарная генерация: {total_mwh:.2f} МВт·ч за {len(combined)}ч. Прогнозируемый КИУМ: {kium_pct:.1f}%."
        )
        return combined

    def tool_audit_risks(self, forecast_df: pd.DataFrame) -> dict:
        """Tool 4: Evaluates physical safety thresholds and dispatch constraints."""
        risks = []
        status = "NORMAL"
        
        # 1. Storm cut-out check (> 22 m/s)
        max_v = forecast_df['wind_speed_100m'].max()
        cutout_hours = forecast_df[forecast_df['wind_speed_100m'] > 22.0]
        if len(cutout_hours) > 0:
            status = "CRITICAL_STORM"
            risks.append({
                'type': 'STORM_CUT_OUT',
                'severity': 'HIGH',
                'desc': f"Внимание: {len(cutout_hours)} ч. с порывами ветра выше 22 м/с (пик {max_v:.1f} м/с). Риск аварийного отключения защиты ротора!"
            })
            
        # 2. Blade icing risk (T < -2°C with wind)
        cold_hours = forecast_df[forecast_df['temperature_2m'] < -2.0]
        if len(cold_hours) > 12:
            risks.append({
                'type': 'ICING_RISK',
                'severity': 'MEDIUM',
                'desc': f"Низкие температуры ({forecast_df['temperature_2m'].min():.1f}°C). Вероятность обледенения лопастей и снижения КПД аэродинамики на 5-10%."
            })
            
        # 3. Calm hours (< 3 m/s)
        calm_hours = forecast_df[forecast_df['wind_speed_100m'] < 3.0]
        if len(calm_hours) > 0:
            risks.append({
                'type': 'CALM_PERIOD',
                'severity': 'LOW',
                'desc': f"{len(calm_hours)} ч. штилевой погоды (< 3 м/с). Выработка близка к нулю, необходимы компенсационные мощности энергосистемы."
            })
            
        # 4. Rapid ramp-rate check (> 30% capacity jump in 1h)
        power_diff = forecast_df['power_cluster'].diff().abs()
        max_ramp = power_diff.max()
        if max_ramp > 0.35:
            risks.append({
                'type': 'HIGH_RAMP_RATE',
                'severity': 'MEDIUM',
                'desc': f"Высокий градиент изменения мощности: скачок до {max_ramp*100:.1f}% за 1 час. Требуется заблаговременное уведомление KEGOC."
            })
            
        audit_res = {
            'status': status,
            'max_wind': float(max_v),
            'min_temp': float(forecast_df['temperature_2m'].min()),
            'total_energy_mwh': float(forecast_df['mw_cluster'].sum()),
            'avg_kium_pct': float(forecast_df['power_cluster'].mean() * 100.0),
            'risks': risks
        }
        self._log(
            "4. Аудит физических рисков и ограничений",
            "Техническая экспертиза графика",
            f"Выявлено {len(risks)} факторов внимания. Статус безопасности: {status}."
        )
        return audit_res

    def tool_simulate_recalculation(self, base_forecast: pd.DataFrame, shift_hours: int = 12) -> tuple:
        """
        Tool 5: Event-Driven Re-calculator.
        Simulates arrival of an updated numerical weather run (e.g. evening run) and triggers re-forecast.
        """
        self._log(
            "5. Событийный пересчёт (Re-calculation)",
            "Детекция обновления прогноза погоды",
            f"Поступили обновлённые данные метеомодели (прогон с {shift_hours}-го часа). Инициирован дифференциальный аудит."
        )
        updated_weather = self.weather_service.simulate_weather_update(base_forecast, shift_hours=shift_hours)
        updated_forecast = self.tool_run_forecast(updated_weather)
        
        delta_mwh = updated_forecast['mw_cluster'].sum() - base_forecast['mw_cluster'].sum()
        delta_pct = (delta_mwh / max(base_forecast['mw_cluster'].sum(), 0.1)) * 100.0
        
        self._log(
            "5. Событийный пересчёт",
            "Корректировка суточного графика",
            f"График пересчитан. Корректировка выработки: {delta_mwh:+.2f} МВт·ч ({delta_pct:+.1f}%). Готов обновлённый пакет заявок для KEGOC.",
            "SUCCESS"
        )
        return updated_forecast, delta_mwh

    def tool_generate_dispatch_report(self, forecast_df: pd.DataFrame, audit: dict, delta_mwh: float = None) -> dict:
        """Tool 6: Generates executive dispatch briefing in Russian and Kazakh."""
        dt_start = forecast_df['timestamp'].min().strftime('%d.%m.%Y %H:%M')
        dt_end = forecast_df['timestamp'].max().strftime('%d.%m.%Y %H:%M')
        
        # Top peak generation hour
        peak_row = forecast_df.loc[forecast_df['mw_cluster'].idxmax()]
        peak_time = peak_row['timestamp'].strftime('%d.%m в %H:00')
        peak_val = peak_row['mw_cluster']
        
        ru_report = f"""### 📋 Диспетчерская сводка ВЭС Шелек
**Период планирования:** с {dt_start} по {dt_end} (горизонт {len(forecast_df)} ч)
- **Прогнозируемая выработка:** {audit['total_energy_mwh']:.2f} МВт·ч (средний КИУМ: {audit['avg_kium_pct']:.1f}%)
- **Пик генерации:** {peak_val:.2f} МВт ожидается {peak_time} (ветер {peak_row['wind_speed_100m']:.1f} м/с)
- **Метеопараметры:** макс. скорость ветра {audit['max_wind']:.1f} м/с, мин. температура {audit['min_temp']:.1f}°C
"""
        if delta_mwh is not None:
            ru_report += f"- **Повторный расчёт (Update Run):** корректировка генерации составила {delta_mwh:+.2f} МВт·ч.\n"

        if audit['risks']:
            ru_report += "\n**⚠️ Факторы внимания и рекомендации:**\n"
            for r in audit['risks']:
                ru_report += f"• **[{r['severity']}]** {r['desc']}\n"
        else:
            ru_report += "\n✅ Ветровой режим стабильный, риски штормового сброса и обледенения отсутствуют. Рекомендуется полная подача в рынок на сутки вперёд (БРЭ).\n"

        kz_report = f"""### 📋 Шілік ЖЭС диспетчерлік мәліметі
**Жоспарлау кезеңі:** {dt_start} - {dt_end} ({len(forecast_df)} сағат)
- **Болжамды өндіріс:** {audit['total_energy_mwh']:.2f} МВт·сағ (орташа КИУМ: {audit['avg_kium_pct']:.1f}%)
- **Генерация шыңы:** {peak_val:.2f} МВт ({peak_time}, жел {peak_row['wind_speed_100m']:.1f} м/с)
- **Қауіпсіздік күйі:** {'Тұрақты' if not audit['risks'] else 'Назар аудару қажет'}
"""
        return {'ru': ru_report, 'kz': kz_report}

    def execute_agent_cycle(self, as_of_date: str, horizon_hours: int = 48, simulate_update: bool = False) -> dict:
        """
        Executes complete end-to-end Agentic AI cycle:
        fetch -> process -> forecast -> audit -> (re-calc if updated) -> dispatch report.
        """
        self.trace_logs = []
        self._log("0. Инициализация Агента", "Старт рабочего цикла", f"Запуск агента диспетчеризации на {as_of_date}, горизонт {horizon_hours}ч.")
        
        # 1. Fetch
        weather_df = self.tool_fetch_weather(as_of_date, horizon_hours)
        
        # 2 & 3. Forecast
        forecast_df = self.tool_run_forecast(weather_df)
        
        # 4. Audit
        audit = self.tool_audit_risks(forecast_df)
        
        # 5. Optional Re-calculation simulation
        updated_forecast = None
        delta_mwh = None
        if simulate_update:
            updated_forecast, delta_mwh = self.tool_simulate_recalculation(forecast_df)
        
        # 6. Report
        active_forecast = updated_forecast if updated_forecast is not None else forecast_df
        reports = self.tool_generate_dispatch_report(active_forecast, audit, delta_mwh)
        
        return {
            'as_of_date': as_of_date,
            'horizon_hours': horizon_hours,
            'forecast': forecast_df,
            'updated_forecast': updated_forecast,
            'audit': audit,
            'delta_mwh': delta_mwh,
            'reports': reports,
            'trace_logs': self.trace_logs
        }

if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    agent = WindFarmAgent(use_cache=True)
    res = agent.execute_agent_cycle('2026-02-01 00:00:00', 48, simulate_update=True)
    print("\n=== Agent Trace Logs ===")
    for l in res['trace_logs']:
        print(f"[{l['timestamp']}] {l['step']}: {l['action']} -> {l['details']}")
    print("\n=== Dispatch Report RU ===")
    print(res['reports']['ru'])
