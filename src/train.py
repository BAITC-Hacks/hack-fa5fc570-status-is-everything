"""Temporal weather-to-power validation, baseline, then pre-February refit."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.data import prepare
from src.settings import ROOT, load_config, save_json

FEATURES = ["wind_speed", "temperature"]
LIMITATION = "Validation uses observed wind and temperature; this is NOT 24-48h forecast accuracy. February is excluded from training."


class WindBinBaseline:
    def fit(self, frame):
        self.mean = float(frame.power_normalized.mean())
        self.bins = frame.groupby(np.floor(frame.wind_speed / 0.5)).power_normalized.mean().to_dict()
        return self

    def predict(self, frame):
        return np.array([self.bins.get(v, self.mean) for v in np.floor(frame.wind_speed / 0.5)])


def score(actual, predicted):
    return {"mae": float(mean_absolute_error(actual, predicted)), "rmse": float(np.sqrt(mean_squared_error(actual, predicted)))}


def new_model():
    return HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=31, learning_rate=0.08, random_state=42, early_stopping=False)


def validation_month(frame, config):
    preferred = pd.Timestamp(config["validation_start"])
    cutoff = pd.Timestamp(config["history_cutoff"])
    for month in pd.date_range(frame.timestamp.min().to_period("M").start_time, preferred, freq="MS")[::-1]:
        end = month + pd.offsets.MonthBegin(1)
        if end <= cutoff and frame.timestamp.min() <= month and frame.timestamp.max() >= end - pd.Timedelta(hours=1):
            return month, end
    raise ValueError("No fully spanned calendar month available for validation")


def train_all():
    config = load_config()
    hourly, quality = prepare(config)
    cutoff = pd.Timestamp(config["history_cutoff"])
    results, predictions = {}, []
    model_dir = ROOT / "models/validated"
    model_dir.mkdir(parents=True, exist_ok=True)
    for tid in config["turbines"]:
        history = hourly[(hourly.turbine_id == tid) & (hourly.timestamp < cutoff)]
        start, end = validation_month(history, config)
        usable = history[history.train_eligible].dropna(subset=FEATURES + ["power_normalized"])
        train, val = usable[usable.timestamp < start], usable[(usable.timestamp >= start) & (usable.timestamp < end)]
        if train.empty or val.empty:
            raise ValueError(f"{tid}: insufficient training or validation rows")
        model = new_model().fit(train[FEATURES], train.power_normalized)
        baseline = WindBinBaseline().fit(train)
        pred, base = model.predict(val[FEATURES]), baseline.predict(val)
        results[tid] = {"model": score(val.power_normalized, pred), "baseline": score(val.power_normalized, base), "train_rows": len(train), "validation_rows": len(val), "validation_start": str(start), "validation_end_exclusive": str(end), "validation_calendar_hours": int((end-start).total_seconds()/3600), "final_train_rows": len(usable), "final_train_max": str(usable.timestamp.max()), "limitation": LIMITATION}
        comparison = val[["turbine_id", "timestamp", "power_normalized"]].copy()
        comparison["model_prediction"], comparison["baseline_prediction"] = pred, base
        predictions.append(comparison)
        # Refit after evaluation; no February rows are ever included.
        final = new_model().fit(usable[FEATURES], usable.power_normalized)
        artifact = {"schema_version": 2, "turbine_id": tid, "model": final, "features": FEATURES, "metrics": results[tid], "config": config, "source_sha256": quality[tid]["sha256"], "history_cutoff": str(cutoff)}
        joblib.dump(artifact, model_dir / f"{tid}.joblib")
    save_json(ROOT / "reports/metrics.json", results)
    pd.concat(predictions).to_csv(ROOT / "reports/validation_predictions.csv", index=False)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return results


if __name__ == "__main__":
    train_all()
