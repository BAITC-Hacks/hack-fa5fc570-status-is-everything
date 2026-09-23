"""Audit Open-Meteo fixed-lead slices without inventing model-run timestamps.

Previous Runs is suitable for coverage and fixed-lead research. It does not
identify the underlying complete model run or its historical availability.
Consequently this module must never publish data/weather/forecasts.csv.
"""
import argparse
import hashlib
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np
import requests

from src.settings import ROOT, save_json, load_config

ENDPOINT = "https://previous-runs-api.open-meteo.com/v1/forecast"
MODEL = "gfs_seamless"
VARIABLES = ("wind_speed_10m", "wind_speed_100m", "temperature_2m")
OFFSETS = (1, 2)
HOURLY = tuple(f"{variable}_previous_day{offset}" for variable in VARIABLES for offset in OFFSETS)
EXPECTED_UNITS = {
    **{key: "m/s" for key in HOURLY if key.startswith("wind_speed_")},
    **{key: "°C" for key in HOURLY if key.startswith("temperature_")},
}


def fetch_previous_runs(latitude, longitude, start_date, end_date, session=None):
    """Get one fixed-lead time series; returns API data and its exact request URL."""
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ValueError("Invalid coordinates")
    start, end = date.fromisoformat(str(start_date)), date.fromisoformat(str(end_date))
    if end < start:
        raise ValueError("end_date precedes start_date")
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(HOURLY),
        "models": MODEL,
        "wind_speed_unit": "ms",
        "timezone": "GMT",
    }
    response = (session or requests).get(ENDPOINT, params=params, timeout=45)
    response.raise_for_status()
    payload = response.json()
    if payload.get("error"):
        raise ValueError(f"Open-Meteo: {payload.get('reason', 'unknown error')}")
    return payload, response.url


def inspect_response(payload, turbine_id, requested_latitude, requested_longitude):
    """Return fixed-lead rows and a coverage report. Neither is an issued run."""
    hourly = payload.get("hourly") or {}
    units = payload.get("hourly_units") or {}
    if payload.get("utc_offset_seconds") != 0 or payload.get("timezone") not in ("GMT", "UTC"):
        raise ValueError("Expected UTC/GMT API timestamps")
    for key, unit in EXPECTED_UNITS.items():
        if units.get(key) != unit:
            raise ValueError(f"Unexpected or missing unit for {key}: {units.get(key)}")
    times = hourly.get("time")
    if not isinstance(times, list) or not times:
        raise ValueError("No hourly timestamps")
    stamps = pd.to_datetime(times, utc=True, errors="raise")
    if stamps.has_duplicates or not stamps.is_monotonic_increasing or any(stamps.to_series().diff().iloc[1:] != pd.Timedelta(hours=1)):
        raise ValueError("Missing, duplicated or unordered hourly timestamps")
    columns = {}
    counts = {}
    for key in HOURLY:
        values = hourly.get(key)
        if not isinstance(values, list) or len(values) != len(stamps):
            raise ValueError(f"Missing or truncated series: {key}")
        series = pd.to_numeric(pd.Series(values), errors="raise")
        if np.isinf(series).any():
            raise ValueError(f"Non-finite series: {key}")
        if key.startswith("wind_speed") and (series < 0).any():
            raise ValueError(f"Negative wind speed: {key}")
        if key.startswith("temperature") and (series <= -273.15).any():
            raise ValueError(f"Invalid temperature: {key}")
        columns[key] = series
        counts[key] = {"non_null": int(series.notna().sum()), "null": int(series.isna().sum())}
    frame = pd.DataFrame(columns)
    frame.insert(0, "valid_time", stamps)
    frame.insert(0, "turbine_id", turbine_id)
    report = {
        "turbine_id": turbine_id,
        "requested_point": {"latitude": requested_latitude, "longitude": requested_longitude},
        "returned_grid_point": {"latitude": payload.get("latitude"), "longitude": payload.get("longitude")},
        "model": MODEL,
        "hours": len(stamps),
        "first_valid_time": stamps[0].isoformat(),
        "last_valid_time": stamps[-1].isoformat(),
        "series": counts,
        "limitations": "Fixed lead-time slices only: no exact issued_at, available_at or single-run identity in this API response.",
    }
    return frame, report


def candidate_for_origin(frame, origin, minimum_lead_hours=24):
    """Pick only nominally old-enough fixed-lead values for diagnostics.

    Selection does not establish availability or produce a valid operational
    forecast; do not feed its output to WeatherService or predict_power.
    """
    origin = pd.Timestamp(origin)
    if origin.tzinfo is None:
        raise ValueError("origin needs an explicit UTC offset")
    origin = origin.tz_convert("UTC")
    chosen = []
    for row in frame.itertuples(index=False):
        if row.valid_time <= origin:
            continue
        hours_until_valid = (row.valid_time - origin).total_seconds() / 3600
        offset = next((n for n in OFFSETS if n * 24 >= max(minimum_lead_hours, hours_until_valid)), None)
        if offset is None:
            continue
        chosen.append({
            "turbine_id": row.turbine_id,
            "valid_time": row.valid_time,
            "nominal_lead_time_hours": offset * 24,
            "wind_speed_10m": getattr(row, f"wind_speed_10m_previous_day{offset}"),
            "wind_speed_100m": getattr(row, f"wind_speed_100m_previous_day{offset}"),
            "temperature_2m": getattr(row, f"temperature_2m_previous_day{offset}"),
        })
    return pd.DataFrame(chosen)


def exploratory_power(frame, origin, horizon_hours, model_dir=None, config=None):
    """Weather-to-power illustration; never an as-of compliant forecast."""
    from src.model_store import model_for_origin, model_output

    config = config or load_config()
    if horizon_hours not in (24, 48):
        raise ValueError("horizon_hours must be 24 or 48")
    origin = pd.Timestamp(origin)
    if origin.tzinfo is None:
        raise ValueError("origin needs an explicit UTC offset")
    origin = origin.tz_convert("UTC")
    data = frame.copy()
    data["valid_time"] = pd.to_datetime(data.valid_time, utc=True)
    data = data[(data.valid_time > origin) & (data.valid_time <= origin + pd.Timedelta(hours=horizon_hours))]
    selected = candidate_for_origin(data, origin)
    if selected.empty:
        raise ValueError("No fixed-lead weather slices for this origin")
    if selected.duplicated(["turbine_id", "valid_time"]).any():
        raise ValueError("Duplicate fixed-lead hour")
    if set(selected.turbine_id) != set(config["turbines"]):
        raise ValueError("Fixed-lead archive must contain every configured turbine")
    results = []
    for tid, group in selected.groupby("turbine_id"):
        expected = pd.date_range(origin + pd.Timedelta(hours=1), periods=horizon_hours, freq="h")
        if len(group) != horizon_hours or set(group.valid_time) != set(expected):
            raise ValueError(f"{tid}: incomplete {horizon_hours}h fixed-lead window")
        if not np.isfinite(group[["wind_speed_100m", "temperature_2m"]].to_numpy(dtype=float)).all():
            raise ValueError(f"{tid}: null or non-finite fixed-lead weather")
        artifact, _ = model_for_origin(tid, origin, config, model_dir)
        features = group.rename(columns={"wind_speed_100m": "wind_speed", "temperature_2m": "temperature"})
        output = group[["turbine_id", "valid_time", "nominal_lead_time_hours"]].copy()
        output = output.join(model_output(artifact, features))
        output["baseline_prediction"] = artifact["baseline"].predict(features)
        output["wind_speed"] = features.wind_speed
        output["temperature"] = features.temperature
        output["forecast_origin"] = origin
        output["hours_ahead"] = (output.valid_time - origin).dt.total_seconds() / 3600
        output["model_training_max"] = artifact["training_max"]
        output["model_training_available_at"] = artifact["training_available_at"]
        output["model_id"] = artifact["identity"]
        output["weather_model"] = MODEL
        output["forecast_contract_compatible"] = False
        results.append(output)
    return pd.concat(results, ignore_index=True)


def audit_points(points, start_date, end_date, output_dir=None, session=None):
    """Fetch supplied points and write a diagnostic archive, never production weather."""
    if not points:
        raise ValueError("Provide at least one confirmed or explicitly exploratory coordinate")
    frames, reports = [], {}
    for turbine_id, (latitude, longitude) in points.items():
        payload, url = fetch_previous_runs(latitude, longitude, start_date, end_date, session)
        frame, report = inspect_response(payload, turbine_id, latitude, longitude)
        expected = pd.date_range(start_date, pd.Timestamp(end_date) + pd.Timedelta(hours=23), freq="h", tz="UTC")
        if list(frame.valid_time) != list(expected):
            raise ValueError(f"{turbine_id}: response does not cover requested date range")
        report["source_url"] = url
        frames.append(frame)
        reports[turbine_id] = report
    output = Path(output_dir) if output_dir else ROOT / "data/weather/diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"previous_runs_{start_date}_{end_date}.csv"
    raw = pd.concat(frames, ignore_index=True).to_csv(index=False).encode("utf-8")
    archive.write_bytes(raw)
    summary = {
        "dataset": "Open-Meteo Previous Runs fixed lead-time slices",
        "model": MODEL,
        "start_date": start_date,
        "end_date": end_date,
        "diagnostic_archive": str(archive),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "forecast_contract_compatible": False,
        "reason": "Previous Runs omits exact model-run initialization and historical API availability; issued_at and available_at cannot be derived from these slices.",
        "points": reports,
    }
    save_json(output / f"previous_runs_{start_date}_{end_date}.json", summary)
    return summary


def configured_points(config):
    points = {}
    for tid, turbine in config["turbines"].items():
        lat, lon = turbine.get("latitude"), turbine.get("longitude")
        if lat is None or lon is None:
            raise ValueError(f"{tid}: configure confirmed latitude and longitude")
        points[tid] = (lat, lon)
    return points


def load_archive(config=None, refresh=False):
    """Automatically download missing diagnostic data; verify persisted provenance."""
    import json
    config = config or load_config()
    settings = config["exploratory_weather"]
    path = ROOT / settings["archive"]
    manifest = path.with_suffix(".json")
    points = configured_points(config)
    expected_name = f"previous_runs_{settings['start_date']}_{settings['end_date']}.csv"
    if path.name != expected_name:
        raise ValueError("exploratory_weather.archive must match its configured dates")
    if refresh or not path.exists() or not manifest.exists():
        audit_points(points, settings["start_date"], settings["end_date"], output_dir=path.parent)
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    if metadata["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("Diagnostic archive checksum mismatch; explicitly refresh the archive")
    if metadata["model"] != MODEL:
        raise ValueError("Diagnostic archive model mismatch")
    if (metadata["start_date"], metadata["end_date"]) != (settings["start_date"], settings["end_date"]):
        raise ValueError("Diagnostic archive period mismatch")
    for tid, (lat, lon) in points.items():
        if metadata["points"][tid]["requested_point"] != {"latitude": lat, "longitude": lon}:
            raise ValueError("Diagnostic archive coordinates changed; refresh required")
    frame = pd.read_csv(path, parse_dates=["valid_time"])
    if set(frame.turbine_id) != set(points):
        raise ValueError("Diagnostic archive turbine mismatch")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--turbine-1", nargs=2, metavar=("LAT", "LON"), type=float)
    parser.add_argument("--turbine-2", nargs=2, metavar=("LAT", "LON"), type=float)
    args = parser.parse_args()
    config = load_config()
    points = configured_points(config)
    if args.turbine_1:
        points["turbine_1"] = tuple(args.turbine_1)
    if args.turbine_2:
        points["turbine_2"] = tuple(args.turbine_2)
    result = audit_points(points, args.start_date or config["exploratory_weather"]["start_date"], args.end_date or config["exploratory_weather"]["end_date"])
    for tid, point in result["points"].items():
        print(tid, point["hours"], point["returned_grid_point"], point["series"])
    print("Contract compatible:", result["forecast_contract_compatible"])
    print("Diagnostic archive:", result["diagnostic_archive"])


if __name__ == "__main__":
    main()
