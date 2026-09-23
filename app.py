"""
Streamlit Web Dashboard for Samruk WindPilot AI.
Interactive dispatch console, Agentic AI execution trace,
walk-forward simulator for February 2026, and economic calculations for Samruk-Kazyna.
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set repo path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.agent import WindFarmAgent, TURBINE_RATED_CAPACITY_MW
from src.data_loader import get_hourly_data, compute_power_curve

st.set_page_config(
    page_title="Samruk WindPilot AI | Прогнозирование ВЭС",
    page_icon="💨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #0f2b48; margin-bottom: 0px; }
    .sub-title { font-size: 1.05rem; color: #4a5568; margin-bottom: 20px; }
    .metric-box { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px; text-align: center; }
    .status-normal { background: #e6fffa; border: 1px solid #38b2ac; border-radius: 8px; padding: 10px; color: #234e52; font-weight: 600; }
    .status-alert { background: #fff5f5; border: 1px solid #e53e3e; border-radius: 8px; padding: 10px; color: #742a2a; font-weight: 600; }
    .status-recalc { background: #ebf8ff; border: 1px solid #3182ce; border-radius: 8px; padding: 10px; color: #2b6cb0; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_agent():
    return WindFarmAgent(use_cache=True)

agent = load_agent()

# Header
col_logo, col_head = st.columns([1, 6])
with col_head:
    st.markdown('<div class="main-title">💨 Samruk WindPilot AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Автономная Agentic AI система почасового прогнозирования выработки ВЭС (Шелекский ветровой коридор) на горизонте 24–48 ч | <b>АО «Самрук-Қазына»</b></div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Пульт управления агентом")

selected_date = st.sidebar.date_input(
    "📅 Дата среза прогноза (Машина времени):",
    value=datetime(2026, 2, 1),
    min_value=datetime(2026, 1, 31),
    max_value=datetime(2026, 2, 28)
)
selected_hour = st.sidebar.selectbox("⏰ Время среза:", ["00:00", "06:00", "12:00", "18:00"], index=0)
as_of_str = f"{selected_date.strftime('%Y-%m-%d')} {selected_hour}:00"

horizon = st.sidebar.radio("⏱️ Горизонт прогнозирования:", [24, 48], index=1)

st.sidebar.markdown("---")
st.sidebar.subheader("🕹️ Действия AI-Агента")
run_agent_btn = st.sidebar.button("🚀 Запустить цикл прогноза", type="primary", use_container_width=True)
simulate_recalc_btn = st.sidebar.button("🔄 Симулировать обновление погоды", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Параметры рынка (БРЭ РК)")
tariff_kzt = st.sidebar.number_input("Тариф продажи э/э (₸/кВт·ч):", value=22.68, step=0.5)
penalty_kzt = st.sidebar.number_input("Штраф за небаланс БРЭ (₸/кВт·ч):", value=7.50, step=0.5)

st.sidebar.markdown("""
<small>
<b>Параметры станции:</b><br>
• Шелекская ВЭС, Алматинская обл.<br>
• Турбина 1: 43.645150°N, 78.535604°E<br>
• Турбина 2: 43.643198°N, 78.538828°E<br>
• Мощность кластера: 2 × 2.5 = 5.0 МВт
</small>
""", unsafe_allow_html=True)

# Run Agent
if 'agent_result' not in st.session_state or run_agent_btn:
    with st.spinner("Агент выполняет сбор погодных данных и расчёт физико-математической модели..."):
        st.session_state.agent_result = agent.execute_agent_cycle(as_of_str, horizon, simulate_update=False)

if simulate_recalc_btn:
    with st.spinner("Агент получил новый прогон метеомодели и производит повторный расчёт..."):
        st.session_state.agent_result = agent.execute_agent_cycle(as_of_str, horizon, simulate_update=True)

res = st.session_state.agent_result
forecast_df = res['forecast']
audit = res['audit']
updated_df = res.get('updated_forecast')
delta_mwh = res.get('delta_mwh')

# Main Tabs
tab_disp, tab_trace, tab_month, tab_physics = st.tabs([
    "📊 Диспетчерский пульт (Dispatch Console)",
    "🧠 Лог Agentic AI (Chain-of-Thought)",
    "📈 Обзор февраля 2026 (Walk-Forward)",
    "⚡ Физика и Модель (Power Curve)"
])

# ----------------- TAB 1: DISPATCH CONSOLE -----------------
with tab_disp:
    # Summary Metrics Row
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Прогноз выработки", f"{audit['total_energy_mwh']:.2f} МВт·ч", 
                  f"{delta_mwh:+.2f} МВт·ч (Update)" if delta_mwh else None)
    with m2:
        peak_mw = forecast_df['mw_cluster'].max()
        st.metric("Пиковая мощность", f"{peak_mw:.2f} МВт", f"из {TURBINE_RATED_CAPACITY_MW * 2:.1f} МВт")
    with m3:
        st.metric("Прогнозируемый КИУМ", f"{audit['avg_kium_pct']:.1f}%")
    with m4:
        st.metric("Макс. ветер на 100м", f"{audit['max_wind']:.1f} м/с")
    with m5:
        revenue_kzt = audit['total_energy_mwh'] * 1000.0 * tariff_kzt
        st.metric("Ожидаемая выручка", f"{revenue_kzt/1e6:.2f} млн ₸")

    # Safety status banner
    if audit['status'] == 'CRITICAL_STORM':
        st.markdown(f'<div class="status-alert">⚠️ ВНИМАНИЕ: Зафиксирован штормовой ветер >22 м/с. Прогнозируется автоматический сброс защиты ротора (Cut-out)!</div>', unsafe_allow_html=True)
    elif delta_mwh is not None:
        st.markdown(f'<div class="status-recalc">🔄 ВЫПОЛНЕН ПОВТОРНЫЙ РАСЧЁТ: Поступил обновлённый прогноз погоды. Корректировка суточного плана: {delta_mwh:+.2f} МВт·ч. Заявка обновлена.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-normal">✅ РЕЖИМ СТАБИЛЬНЫЙ: Параметры ветропарка в пределах нормы. График готов для отправки в Системный оператор KEGOC.</div>', unsafe_allow_html=True)

    st.write("")

    # Interactive Generation Chart
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(f"Почасовой график выработки ВЭС (горизонт {len(forecast_df)} ч)", "Метеоусловия: Скорость ветра на 100м и Температура"),
        row_heights=[0.65, 0.35]
    )

    # Power Generation Traces
    fig.add_trace(
        go.Scatter(
            x=forecast_df['timestamp'], y=forecast_df['mw_cluster'],
            name="Суммарная ВЭС (5.0 МВт)",
            line=dict(color="#0066cc", width=3.5)
        ), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df['timestamp'], y=forecast_df['mw_t1'],
            name="Турбина 1 (2.5 МВт)",
            line=dict(color="#10b981", width=1.5, dash='dash')
        ), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df['timestamp'], y=forecast_df['mw_t2'],
            name="Турбина 2 (2.5 МВт)",
            line=dict(color="#f59e0b", width=1.5, dash='dot')
        ), row=1, col=1
    )

    # Show updated forecast if re-calculated
    if updated_df is not None:
        fig.add_trace(
            go.Scatter(
                x=updated_df['timestamp'], y=updated_df['mw_cluster'],
                name="Обновлённый прогноз (Re-calc)",
                line=dict(color="#dc2626", width=2.5, dash='dashdot')
            ), row=1, col=1
        )

    # Weather Traces
    fig.add_trace(
        go.Scatter(
            x=forecast_df['timestamp'], y=forecast_df['wind_speed_100m'],
            name="Скорость ветра 100м (м/с)",
            line=dict(color="#6366f1", width=2.5)
        ), row=2, col=1
    )
    # Rated speed line (11.5 m/s) and Cut-out (22 m/s)
    fig.add_hline(y=11.5, line=dict(color="#10b981", dash="dot"), annotation_text="Номинал (11.5 м/с)", row=2, col=1)
    fig.add_hline(y=22.0, line=dict(color="#ef4444", dash="dot"), annotation_text="Шторм Cut-out (22 м/с)", row=2, col=1)

    fig.update_layout(
        height=580,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=30)
    )
    fig.update_yaxes(title_text="Мощность (МВт)", row=1, col=1)
    fig.update_yaxes(title_text="Ветер (м/с)", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # Dispatcher Reports (RU / KZ)
    c_ru, c_kz = st.columns(2)
    with c_ru:
        st.markdown(res['reports']['ru'])
    with c_kz:
        st.markdown(res['reports']['kz'])

# ----------------- TAB 2: AGENT EXECUTION TRACE -----------------
with tab_trace:
    st.subheader("🧠 Пошаговый лог рассуждений и действий AI-Агента (Chain-of-Thought)")
    st.markdown("Здесь отображается каждый внутренний шаг автономного цикла агента, вызовы инструментов (*Tool Calling*) и принятые решения:")
    
    for entry in res['trace_logs']:
        status_icon = "🟢" if entry['status'] == "SUCCESS" else ("🔴" if entry['status'] == "ERROR" else "🔵")
        with st.expander(f"{status_icon} [{entry['timestamp']}] {entry['step']} — {entry['action']}", expanded=True):
            st.write(f"**Детали:** {entry['details']}")

# ----------------- TAB 3: MONTHLY WALK-FORWARD -----------------
with tab_month:
    st.subheader("📈 Ретроспективное моделирование (Walk-Forward) за февраль 2026 года")
    st.markdown("Последовательный расчёт прогнозов день за днём с 31 января по 28 февраля 2026 года без заглядывания в будущее:")
    
    daily_csv_path = os.path.join(REPO_ROOT, 'data', 'february_daily_summary_48h.csv')
    if os.path.exists(daily_csv_path):
        df_month = pd.read_csv(daily_csv_path)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Суммарная выработка за февраль", f"{df_month['daily_energy_mwh'].sum():.1f} МВт·ч")
        c2.metric("Средний КИУМ за месяц", f"{df_month['avg_kium_pct'].mean():.1f}%")
        c3.metric("Экономия на небалансах БРЭ", f"~{df_month['daily_energy_mwh'].sum() * 1000 * 0.08 * penalty_kzt / 1e6:.2f} млн ₸")
        
        fig_month = go.Figure()
        fig_month.add_trace(go.Bar(
            x=df_month['date'], y=df_month['daily_energy_mwh'],
            name="Суточная генерация (МВт·ч)",
            marker_color="#0066cc"
        ))
        fig_month.update_layout(
            title="Прогнозируемая суточная выработка Шелекской ВЭС (Февраль 2026)",
            xaxis_title="Дата", yaxis_title="Выработка (МВт·ч)",
            height=400, margin=dict(l=40, r=40, t=40, b=30)
        )
        st.plotly_chart(fig_month, use_container_width=True)
        st.dataframe(df_month, use_container_width=True)
    else:
        st.info("Файл месячной симуляции не найден. Запустите `python src/backtest.py`.")

# ----------------- TAB 4: PHYSICS & POWER CURVE -----------------
with tab_physics:
    st.subheader("⚡ Физико-математический профиль турбин Шелекской ВЭС")
    st.markdown("""
    Эмпирическая аэродинамическая характеристика зависимости активной мощности от скорости ветра,
    рассчитанная на 142 000+ реальных измерениях SCADA (2023–2026 гг.):
    """)
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        curve1 = agent.forecaster1.power_curve
        if curve1 is not None:
            fig_curve = go.Figure()
            fig_curve.add_trace(go.Scatter(
                x=curve1['wind_bin'], y=curve1['power_mean'],
                mode='lines+markers', name='Турбина 1',
                line=dict(color="#10b981", width=3)
            ))
            fig_curve.update_layout(
                title="Кривая мощности Турбины 1 (IEC Power Curve)",
                xaxis_title="Скорость ветра (м/с)", yaxis_title="Нормализованная мощность [0..1]",
                height=350
            )
            st.plotly_chart(fig_curve, use_container_width=True)
            
    with col_p2:
        curve2 = agent.forecaster2.power_curve
        if curve2 is not None:
            fig_curve2 = go.Figure()
            fig_curve2.add_trace(go.Scatter(
                x=curve2['wind_bin'], y=curve2['power_mean'],
                mode='lines+markers', name='Турбина 2',
                line=dict(color="#f59e0b", width=3)
            ))
            fig_curve2.update_layout(
                title="Кривая мощности Турбины 2 (IEC Power Curve)",
                xaxis_title="Скорость ветра (м/с)", yaxis_title="Нормализованная мощность [0..1]",
                height=350
            )
            st.plotly_chart(fig_curve2, use_container_width=True)
            
    st.markdown("""
    **Ключевые физические закономерности:**
    1. **Скорость включения (Cut-in):** 3.0 м/с — при слабом ветре ротор находится в режиме ожидания.
    2. **Рабочий диапазон (Ramp):** 3.0–11.5 м/с — кубический рост кинетической энергии ветра ($P \propto v^3$).
    3. **Номинальная полка (Rated Plateau):** 11.5–22.0 м/с — турбина выдаёт номинальную мощность (100%), угол атаки лопастей регулируется системой Pitch Control.
    4. **Штормовая отсечка (Cut-out):** >22.0 м/с — автоматический останов для предотвращения механического разрушения.
    """)
