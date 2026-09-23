"""Strict forecast contract. Never infer unknown publication/availability times."""
import numpy as np
import pandas as pd
from src.settings import load_config
from src.model_store import model_for_origin, model_output

TIME_FIELDS = ["forecast_origin", "issued_at", "available_at", "valid_time"]
REQUIRED = ["turbine_id"] + TIME_FIELDS + ["wind_speed", "temperature"]


def aware_timestamp(value):
    ts = pd.Timestamp(value)
    if pd.isna(ts) or ts.tzinfo is None:
        raise ValueError("Weather timestamps require an explicit UTC offset; unknown time is not allowed")
    return ts.tz_convert("UTC")


def validate_weather(frame, config=None):
    config = config or load_config()
    if not config.get("timezone"):
        raise ValueError("SCADA timezone is unresolved")
    if frame is None or frame.empty:
        raise ValueError("Real weather forecast input is missing")
    if set(REQUIRED) - set(frame.columns):
        raise ValueError(f"Missing weather fields: {sorted(set(REQUIRED) - set(frame.columns))}")
    d = frame.copy()
    for col in TIME_FIELDS:
        d[col] = pd.to_datetime(d[col].map(aware_timestamp), utc=True)
    if d.turbine_id.isna().any() or not set(d.turbine_id).issubset(config["turbines"]):
        raise ValueError("Unknown turbine_id")
    if (d.issued_at > d.forecast_origin).any() or (d.available_at > d.forecast_origin).any():
        raise ValueError("Forecast was issued or available after forecast_origin")
    if (d.available_at < d.issued_at).any():
        raise ValueError("available_at precedes issued_at")
    if (d.valid_time <= d.forecast_origin).any():
        raise ValueError("valid_time must be strictly after forecast_origin")
    if (d.valid_time != d.valid_time.dt.floor("h")).any():
        raise ValueError("valid_time must be an hourly boundary")
    if d.duplicated(["turbine_id", "forecast_origin", "valid_time"]).any():
        raise ValueError("Duplicate forecast hour")
    for col in ["wind_speed", "temperature"]:
        d[col] = pd.to_numeric(d[col], errors="raise")
        if not np.isfinite(d[col]).all():
            raise ValueError(f"Missing or non-finite {col}")
    if (d.wind_speed < 0).any() or (d.temperature <= -273.15).any():
        raise ValueError("Physically invalid weather input")
    return d


def predict_power(weather_dataframe, config=None):
    config = config or load_config()
    d = validate_weather(weather_dataframe, config)
    results = []
    for (tid, origin), group in d.groupby(["turbine_id", "forecast_origin"]):
        artifact, _ = model_for_origin(tid, origin, config)
        result = group[["turbine_id", "forecast_origin", "issued_at", "available_at", "valid_time"]].copy()
        result["lead_time_hours"] = (group.valid_time - group.issued_at).dt.total_seconds() / 3600
        result = result.join(model_output(artifact, group))
        result["model_training_max"] = artifact["training_max"]
        result["model_training_available_at"] = artifact["training_available_at"]
        result["model_id"] = artifact["identity"]
        results.append(result)
    return pd.concat(results, ignore_index=True)
