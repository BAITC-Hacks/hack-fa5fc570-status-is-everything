"""Output diagnostics and explicitly configured physical-unit conversion."""
import numpy as np
import pandas as pd


def physical_power(forecast, config):
    """Return turbine MW and plant MW only with documented multiplicative scales."""
    scales = {}
    for tid, settings in config["turbines"].items():
        scale = settings.get("normalization_to_mw")
        if scale is None or not settings.get("normalization_evidence"):
            raise ValueError(f"{tid}: confirmed normalization_to_mw and normalization_evidence are required")
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"{tid}: invalid normalization_to_mw")
        scales[tid] = scale
    keys = ["forecast_origin", "valid_time"]
    if forecast.duplicated(keys + ["turbine_id"]).any():
        raise ValueError("Duplicate turbine hour")
    for _, group in forecast.groupby(keys):
        if set(group.turbine_id) != set(scales):
            raise ValueError("Plant total requires all configured turbines for every hour")
    output = forecast.copy()
    output["predicted_power_mw"] = output.predicted_power * output.turbine_id.map(scales)
    if not np.isfinite(output.predicted_power_mw).all():
        raise ValueError("Non-finite MW output")
    plant = output.groupby(keys, as_index=False).predicted_power_mw.sum()
    # Each row represents one hourly mean, not an instantaneous power reading.
    plant["predicted_energy_mwh"] = plant.predicted_power_mw
    return output, plant


def forecast_audit(forecast, weather):
    audit = {}
    for tid, group in forecast.groupby("turbine_id"):
        source = weather[weather.turbine_id == tid]
        ordered = group.sort_values("valid_time")
        warnings = []
        bounded = int(group.prediction_bounded.sum())
        outside = int(group.weather_out_of_training_range.sum())
        if bounded:
            warnings.append(f"{bounded} model outputs bounded to training power range; raw output retained")
        if outside:
            warnings.append(f"{outside} weather hours outside training range; extrapolation risk")
        warnings.append("No turbine availability/curtailment telemetry: outages cannot be predicted reliably")
        audit[tid] = {
            "hours": len(group), "min_power": float(group.predicted_power.min()),
            "max_power": float(group.predicted_power.max()), "mean_power": float(group.predicted_power.mean()),
            "max_wind_ms": float(source.wind_speed.max()), "min_temperature_c": float(source.temperature.min()),
            "max_hourly_power_change": float(ordered.predicted_power.diff().abs().max()) if len(group) > 1 else 0.,
            "bounded_hours": bounded, "weather_out_of_training_range_hours": outside,
            "model_training_max": str(group.model_training_max.iloc[0]), "warnings": warnings,
        }
    return audit
