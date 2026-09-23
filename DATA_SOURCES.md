# 📡 Data Sources

WindMind AI использует два основных типа данных: исторические данные ветрогенераторов и погодные данные.

## Turbine Data

Исторические данные используются для обучения и проверки моделей прогнозирования.

Основные параметры:

- Wind speed
- Active power
- Temperature
- Timestamp
- Turbine ID

Данные обрабатываются и агрегируются до почасового уровня перед использованием в модели.

## Weather Data

Для формирования прогноза на 24–48 часов система получает погодные данные по координатам ветрогенераторов.

Используемые параметры:

- Wind speed
- Temperature
- Atmospheric pressure
- Humidity

Weather data → Feature Engineering → ML Model → Generation Forecast

## Historical Forecast Mode

Для тестирования модели на феврале 2026 года важно использовать погодную информацию, которая была доступна на момент формирования прогноза.

Это позволяет воспроизводить реальный сценарий:

31 Jan → Forecast for 1–2 Feb  
1 Feb → New Forecast for 2–3 Feb  
2 Feb → New Forecast for 3–4 Feb  
...

Такой подход позволяет тестировать систему последовательно, не используя будущую информацию.

## Data Pipeline

Turbine History  
↓  
Data Cleaning  
↓  
Hourly Aggregation  
↓  
Weather Data  
↓  
Feature Engineering  
↓  
Forecast Model  
↓  
24–48H Generation Forecast
