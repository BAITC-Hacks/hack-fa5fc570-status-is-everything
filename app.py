"""Samruk WindPilot: audited models and documented weather releases."""
import json
from datetime import date
import pandas as pd
import plotly.express as px
import streamlit as st
import requests
from src.agent import WindFarmAgent
from src.previous_runs import exploratory_power, load_archive
from src.reporting import forecast_audit, physical_power
from src.settings import ROOT, load_config

st.set_page_config(page_title="Samruk WindPilot AI", page_icon="💨", layout="wide")
st.title("💨 Samruk WindPilot AI")
st.caption("Почасовая нормализованная мощность каждой турбины • горизонт 24–48 часов")
config = load_config()
st.sidebar.header("Параметры расчёта")
selected = st.sidebar.date_input("Дата расчёта", value=date(2026, 1, 31), min_value=date(2026, 1, 31))
hour = st.sidebar.selectbox("Час расчёта", list(range(24)))
horizon = st.sidebar.radio("Горизонт, часов", [24, 48], index=1)
st.sidebar.caption(f"Местное время: {config['timezone'] or 'не установлено'}")
auto = st.sidebar.checkbox("Проверять новые выпуски каждые 30 секунд", value=False)
run = st.sidebar.button("Проверить погоду и рассчитать", type="primary")
st.sidebar.caption("Автопроверка работает, пока открыта эта страница. Для фонового процесса используйте python -m src.agent --watch.")
origin = f"{selected.isoformat()} {hour:02d}:00:00"
exploratory_tab, forecast_tab, weather_metrics_tab, metrics_tab, data_tab, logs_tab = st.tabs(["Оценка за февраль", "Проверенный выпуск", "Проверка на прогнозной погоде", "Проверка модели (фактическая погода)", "Качество данных", "Журнал действий"])

with forecast_tab:
    st.info("Единицы — средняя нормализованная мощность за час. Перевод в МВт, суммирование турбин и экономический эффект требуют подтверждённой шкалы нормализации.")
    @st.fragment(run_every=30 if auto else None)
    def forecast_panel():
        key = (origin, horizon)
        if run or auto:
            try:
                result = WindFarmAgent(config=config).execute_agent_cycle(origin, horizon)
                st.session_state["forecast_result"] = (key, result)
                st.session_state.pop("forecast_error", None)
            except (ValueError, FileNotFoundError, OSError, KeyError) as exc:
                st.session_state["forecast_error"] = str(exc)
                st.session_state.pop("forecast_result", None)
        if st.session_state.get("forecast_error"):
            st.warning(st.session_state["forecast_error"])
        stored = st.session_state.get("forecast_result")
        if stored and stored[0] == key:
            result = stored[1]
            df = result["forecast"].copy()
            df["valid_time"] = pd.to_datetime(df.valid_time, utc=True).dt.tz_convert(config["timezone"])
            st.caption(f"Версия {result['revision'][:12]} • расчёт на {result['forecast_origin']} • {result['horizon_hours']} часов")
            st.plotly_chart(px.line(df, x="valid_time", y="predicted_power", color="turbine_id", labels={"valid_time": "Местное время", "predicted_power": "Нормализованная мощность", "turbine_id": "Турбина"}), use_container_width=True)
            st.dataframe(pd.DataFrame(result["audit"]).T, width="stretch")
            st.download_button("Скачать почасовой прогноз CSV", df.to_csv(index=False), "forecast.csv", "text/csv")
        else:
            st.info("Выберите дату и запустите расчёт. Без документированного погодного выпуска прогноз не формируется.")
        if not (ROOT / config["weather_file"]).exists():
            st.warning("Источник погоды ещё не подключён. Старый февральский кэш не подтверждает время доступности и исключён из расчётов.")
            st.caption("Контракт подключения описан в файле проекта config/WEATHER_INPUT.md.")
    forecast_panel()

with exploratory_tab:
    st.subheader("Предварительная оценка с 31 января")
    st.warning("Срезы Open-Meteo сделаны с номинальным горизонтом 24 или 48 часов. API не подтверждает время конкретного выпуска и его доступность на выбранный момент. Это иллюстрация расчёта «погода → мощность», не проверенный исторический прогноз для диспетчеризации.")
    st.caption("Выбранные дата, час и горизонт находятся слева. Срезы взяты из gfs_seamless, ветер на 100 м и температура на 2 м. Высота измерения ветра в SCADA пока не подтверждена. Обе турбины попадают в одну ячейку погодной модели.")
    st.caption(f"Модель обучается только на завершённых часах до момента расчёта. Предположение о задержке SCADA: {config['scada_delay_hours']} ч после окончания часа. Оно требует подтверждения оператора.")
    try:
        slices = load_archive(config)
        estimate = exploratory_power(slices, pd.Timestamp(origin, tz=config["timezone"]), horizon, config=config)
        estimate["local_time"] = pd.to_datetime(estimate.valid_time, utc=True).dt.tz_convert(config["timezone"])
        st.plotly_chart(px.line(estimate, x="local_time", y="predicted_power", color="turbine_id", labels={"local_time": "Местное время", "predicted_power": "Оценка нормализованной мощности", "turbine_id": "Турбина"}), use_container_width=True)
        st.dataframe(estimate[["turbine_id", "local_time", "nominal_lead_time_hours", "predicted_power"]], hide_index=True, width="stretch")
        st.download_button("Скачать исследовательскую оценку CSV", estimate.to_csv(index=False), "exploratory_estimate.csv", "text/csv")
        with st.expander("Проверки результата и обучающая история"):
            st.json(forecast_audit(estimate, estimate))
        try:
            _, plant = physical_power(estimate, config)
            st.dataframe(plant, hide_index=True)
        except ValueError:
            st.caption("МВт и общая выработка не показаны: коэффициент нормализации пока не подтверждён.")
    except (ValueError, FileNotFoundError, KeyError, requests.RequestException) as exc:
        st.info(f"Нет полной оценки на выбранный период: {exc}")

with weather_metrics_tab:
    st.subheader("Walk-forward: январь, архивная прогнозная погода")
    st.warning("Исследовательская оценка по фиксированным срезам 24/48 ч, не строгий replay доступных выпусков. Высота SCADA-ветра и историческое время публикации не подтверждены. Метрики февраля невозможны: фактической мощности февраля в исходниках нет.")
    report_path = ROOT / "reports/january_fixed_lead.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        st.caption(f"Успешных расчётов: {report['origins_succeeded']}/{report['origins_total']}. Для каждого дня — отдельная модель без последующих наблюдений.")
        metric_rows = [{"Турбина": m["turbine_id"], "Номинальный горизонт": m["nominal_lead_hours"], "Часов": m["rows"], "MAE": m["model"]["mae"], "RMSE": m["model"]["rmse"], "MAE baseline": m["baseline"]["mae"], "Макс. ошибка": m["max_absolute_error"], "MAE ветра, м/с": m["wind_mae_ms"]} for m in report.get("metrics", {}).values()]
        st.dataframe(pd.DataFrame(metric_rows), hide_index=True, width="stretch")
        if any(m["model"]["mae"] >= m["baseline"]["mae"] for m in report.get("metrics", {}).values()):
            st.error("На части проверок модель не превосходит baseline. Прототип пока нельзя считать готовым к диспетчерскому планированию.")
        st.caption("Номинальный lead time относится к погодному срезу. Столбец hours_ahead в CSV показывает расстояние от момента расчёта до целевого часа.")
    else:
        st.info("Выполните python -m src.evaluation --january для расчёта метрик на прогнозной погоде.")

with metrics_tab:
    path = ROOT / "reports/metrics.json"
    st.warning("Проверка на фактическом ветре и температуре оценивает зависимость «погода → мощность». Это не точность прогноза на 24–48 часов.")
    if path.exists():
        metrics = json.loads(path.read_text(encoding="utf-8"))
        rows = []
        for tid, record in metrics.items():
            for method in ["model", "baseline"]:
                rows.append({"Турбина": tid, "Метод": method, "MAE": record[method]["mae"], "RMSE": record[method]["rmse"], "Часов проверки": record["validation_rows"], "Начало": record["validation_start"], "Конец (не включая)": record["validation_end_exclusive"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        validation = pd.read_csv(ROOT / "reports/validation_predictions.csv")
        turbine = st.selectbox("Турбина для сравнения", list(metrics))
        subset = validation[validation.turbine_id == turbine]
        st.plotly_chart(px.line(subset, x="timestamp", y=["power_normalized", "model_prediction", "baseline_prediction"], labels={"timestamp": "Местное время", "value": "Нормализованная мощность"}), use_container_width=True)
        st.caption("После проверки финальные модели переобучены на доступной истории до 1 февраля 2026. Февраль в обучение не входит.")
    else:
        st.info("Для подготовки и обучения выполните python -m src.train.")

with data_tab:
    path = ROOT / "reports/data_quality.json"
    if path.exists():
        quality = json.loads(path.read_text(encoding="utf-8"))
        st.dataframe(pd.DataFrame([{"Турбина": tid, "Строк": q["rows"], "Начало": q["date_min"], "Конец": q["date_max"], "Полных часов": q["complete_hours"], "Доля полных часов": q["complete_hour_fraction"], "Дубликатов": q["duplicate_timestamp_rows"], "Конфликтов": q["conflicting_timestamps"]} for tid, q in quality.items()]), hide_index=True)
        for tid, q in quality.items():
            with st.expander(tid):
                st.json(q)
    st.caption("Предположение: timestamp — начало 10-минутного интервала. Нули сохранены; шкала не обрезается до [0, 1]. По умолчанию для обучения нужны все 6 валидных измерений часа.")

with logs_tab:
    path = ROOT / "reports/agent_events.jsonl"
    if path.exists():
        st.dataframe(pd.DataFrame([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()[-100:]]), width="stretch")
    else:
        st.info("Журнал появится после первого расчёта.")
    st.caption("Это журнал выполненных операций и ошибок. Пересчёт запускается при изменении допустимого выпуска погоды, модели или настроек.")
