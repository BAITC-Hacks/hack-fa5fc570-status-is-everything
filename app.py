"""Samruk WindPilot: audited models and documented weather releases."""
import json
from datetime import date
import pandas as pd
import plotly.express as px
import streamlit as st
from src.agent import WindFarmAgent
from src.settings import ROOT, load_config

st.set_page_config(page_title="Samruk WindPilot AI", page_icon="💨", layout="wide")
st.title("💨 Samruk WindPilot AI")
st.caption("Почасовая нормализованная мощность каждой турбины • горизонт 24–48 часов")
config = load_config()
st.sidebar.header("Параметры расчёта")
selected = st.sidebar.date_input("Дата расчёта", value=date(2026, 2, 1), min_value=date(2026, 2, 1))
hour = st.sidebar.selectbox("Час расчёта", list(range(24)))
horizon = st.sidebar.radio("Горизонт, часов", [24, 48], index=1)
st.sidebar.caption(f"Местное время: {config['timezone'] or 'не установлено'}")
auto = st.sidebar.checkbox("Проверять новые выпуски каждые 30 секунд", value=False)
run = st.sidebar.button("Проверить погоду и рассчитать", type="primary")
st.sidebar.caption("Автопроверка работает, пока открыта эта страница. Для фонового процесса используйте python -m src.agent --watch.")
origin = f"{selected.isoformat()} {hour:02d}:00:00"
forecast_tab, metrics_tab, data_tab, logs_tab = st.tabs(["Прогноз", "Проверка модели", "Качество данных", "Журнал действий"])

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
