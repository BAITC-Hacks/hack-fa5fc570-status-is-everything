"""Point-in-time model selection shared by strict and exploratory predictions."""
import hashlib
import json
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data import aggregate_to_hourly, read_raw
from src.settings import ROOT, load_config
from src.train import FEATURES, WindBinBaseline, new_model

VERSION = 1
_history_cache = {}


def training_rows(hourly, origin, config):
    origin = pd.Timestamp(origin)
    if origin.tzinfo is None:
        raise ValueError("Model origin requires an explicit UTC offset")
    delay = float(config.get("scada_delay_hours", 1))
    if not np.isfinite(delay) or delay < 0:
        raise ValueError("scada_delay_hours must be finite and non-negative")
    stamps = hourly.timestamp.dt.tz_localize(config["timezone"], ambiguous="NaT", nonexistent="NaT")
    available = stamps + pd.Timedelta(hours=1 + delay)
    mask = hourly.train_eligible & (available <= origin) & (hourly.timestamp < pd.Timestamp(config["history_cutoff"]))
    return hourly.loc[mask].dropna(subset=FEATURES + ["power_normalized"]).copy()


def model_for_origin(turbine_id, origin, config=None, model_dir=None):
    """Cache by source bytes, preprocessing config and safe hourly cutoff.

    SCADA delivery delay is an explicit assumption, not proven publication time.
    No February targets are used, even for March or later origins.
    """
    config = config or load_config()
    origin = pd.Timestamp(origin)
    if origin.tzinfo is None:
        raise ValueError("Model origin requires an explicit UTC offset")
    origin = origin.tz_convert(config["timezone"])
    # Validate delay before constructing the cache key.
    delay = float(config.get("scada_delay_hours", 1))
    if not np.isfinite(delay) or delay < 0:
        raise ValueError("scada_delay_hours must be finite and non-negative")
    cutoff = min((origin - pd.Timedelta(hours=delay)).floor("h").tz_localize(None), pd.Timestamp(config["history_cutoff"]))
    source = ROOT / config["turbines"][turbine_id]["file"]
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    signature = {"version": VERSION, "source": source_hash, "turbine": turbine_id,
                 "config": config, "cutoff": str(cutoff)}
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    path = (Path(model_dir) if model_dir else ROOT / "models/point_in_time") / f"{turbine_id}_{key}.joblib"
    if path.exists():
        artifact = joblib.load(path)
        if artifact.get("identity") != key:
            raise ValueError("Point-in-time model identity mismatch")
        return artifact, path
    history_key = (str(source), source_hash, config["timezone"], config["timestamp_meaning"], config["measurement_minutes"], config["min_hourly_coverage"])
    if history_key not in _history_cache:
        raw, _ = read_raw(source, turbine_id)
        _history_cache[history_key] = aggregate_to_hourly(raw, config)
    train = training_rows(_history_cache[history_key], origin, config)
    if len(train) < 24:
        raise ValueError(f"{turbine_id}: fewer than 24 complete training hours available at origin")
    model = new_model().fit(train[FEATURES], train.power_normalized)
    artifact = {
        "schema_version": 3, "identity": key, "turbine_id": turbine_id,
        "model": model, "baseline": WindBinBaseline().fit(train), "features": FEATURES,
        "config": config, "source_sha256": source_hash, "history_cutoff": str(cutoff),
        "training_rows": len(train), "training_max": str(train.timestamp.max()),
        "training_available_at": (train.timestamp.max().tz_localize(config["timezone"]) + pd.Timedelta(hours=1 + delay)).isoformat(),
        "power_bounds": [float(train.power_normalized.min()), float(train.power_normalized.max())],
        "feature_ranges": {f: [float(train[f].min()), float(train[f].max())] for f in FEATURES},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    joblib.dump(artifact, temporary)
    temporary.replace(path)
    return artifact, path


def model_output(artifact, features):
    raw = artifact["model"].predict(features[artifact["features"]])
    if not np.isfinite(raw).all():
        raise ValueError("Non-finite model output")
    policy = artifact["config"].get("prediction_bounds", "none")
    if policy not in ("none", "training_range"):
        raise ValueError("Unknown prediction_bounds policy")
    # Empirical guard only: does not assert that normalized power must be <= 1.
    predicted = np.clip(raw, *artifact["power_bounds"]) if policy == "training_range" else raw
    out_of_domain = np.zeros(len(features), dtype=bool)
    for field, (low, high) in artifact["feature_ranges"].items():
        out_of_domain |= (features[field].to_numpy() < low) | (features[field].to_numpy() > high)
    return pd.DataFrame({"raw_predicted_power": raw, "predicted_power": predicted,
                         "prediction_bounded": predicted != raw, "weather_out_of_training_range": out_of_domain}, index=features.index)
