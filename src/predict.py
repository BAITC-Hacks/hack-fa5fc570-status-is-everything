"""Strict forecast contract. Never infer unknown publication/availability times."""
import joblib
import numpy as np
import pandas as pd
from src.settings import ROOT, load_config

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
    for tid, group in d.groupby("turbine_id"):
        path = ROOT / "models/validated" / f"{tid}.joblib"
        if not path.exists():
            raise FileNotFoundError("Validated model missing; run python -m src.train")
        artifact = joblib.load(path)
        if artifact.get("schema_version") != 2 or artifact["turbine_id"] != tid:
            raise ValueError("Incompatible model artifact")
        for key in ["timezone", "timestamp_meaning", "measurement_minutes", "min_hourly_coverage", "history_cutoff"]:
            if artifact["config"][key] != config[key]:
                raise ValueError(f"Configuration changed ({key}); retrain models")
        cutoff = pd.Timestamp(artifact["history_cutoff"]).tz_localize(config["timezone"]).tz_convert("UTC")
        if (group.forecast_origin < cutoff).any():
            raise ValueError("Model includes observations later than this forecast origin")
        result = group[["turbine_id", "forecast_origin", "valid_time"]].copy()
        result["predicted_power"] = artifact["model"].predict(group[artifact["features"]])
        if not np.isfinite(result.predicted_power).all():
            raise ValueError("Non-finite model output")
        results.append(result)
    return pd.concat(results, ignore_index=True)
