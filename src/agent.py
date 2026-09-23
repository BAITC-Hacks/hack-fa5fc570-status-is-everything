"""Deterministic forecast coordinator with input versioning and durable audit events."""
import argparse
import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from src.settings import ROOT, load_config, save_json
from src.weather_service import WeatherService
from src.predict import predict_power
from src.model_store import model_for_origin
from src.reporting import forecast_audit, physical_power

class WindFarmAgent:
    def __init__(self, use_cache=True, config=None):
        self.config = config or load_config()
        self.weather_service = WeatherService(use_cache, self.config)

    def _event(self, status, **details):
        path = ROOT / "reports/agent_events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        event = {"at": datetime.now(timezone.utc).isoformat(), "status": status, **details}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event

    def execute_agent_cycle(self, as_of_date, horizon_hours=48, simulate_update=False):
        try:
            if simulate_update:
                raise ValueError("Synthetic updates disabled")
            weather = self.weather_service.get_forecast_as_of(as_of_date, horizon_hours)
            digest = hashlib.sha256(weather.sort_values(["turbine_id", "valid_time"]).to_csv(index=False).encode())
            digest.update(json.dumps(self.config, sort_keys=True).encode())
            for tid in self.config["turbines"]:
                _, model_path = model_for_origin(tid, weather.forecast_origin.iloc[0], self.config)
                digest.update(model_path.read_bytes())
            # Invalidate cached output when postprocessing or audit code changes too.
            for module in ["predict.py", "reporting.py", "model_store.py", "agent.py"]:
                digest.update((Path(__file__).parent / module).read_bytes())
            revision = digest.hexdigest()
            out = ROOT / "data/forecasts"
            out.mkdir(parents=True, exist_ok=True)
            origin = str(weather.forecast_origin.iloc[0])
            key = hashlib.sha256(f"{origin}:{horizon_hours}".encode()).hexdigest()[:20]
            state_path = out / f"{key}.json"
            previous = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None
            if previous and previous["revision"] == revision:
                forecast = pd.read_csv(out / previous["file"])
                return {**previous, "forecast": forecast, "changed": False}
            forecast = predict_power(weather, self.config)
            # Audit is recalculated from the new output on every revision.
            audit = forecast_audit(forecast, weather)
            try:
                forecast, plant = physical_power(forecast, self.config)
                physical_status = "Confirmed multiplicative scales applied; plant hourly MW/MWh saved separately"
            except ValueError as exc:
                plant = None
                physical_status = str(exc)
            filename = f"{key}_{revision[:16]}_{uuid.uuid4().hex[:8]}.csv"
            forecast.to_csv(out / filename, index=False)
            plant_file = filename.replace(".csv", "_plant.csv") if plant is not None else None
            if plant is not None:
                plant.to_csv(out / plant_file, index=False)
            state = {"revision": revision, "file": filename, "plant_file": plant_file, "physical_status": physical_status, "forecast_origin": origin, "horizon_hours": horizon_hours, "audit": audit, "units": "normalized hourly mean power, each turbine separately", "previous_revision": previous["revision"] if previous else None}
            temporary = state_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
            save_json(temporary, state)
            temporary.replace(state_path)
            self._event("recalculated" if previous else "created", **state)
            return {**state, "forecast": forecast, "changed": True}
        except Exception as exc:
            self._event("failed", forecast_origin=str(as_of_date), horizon_hours=horizon_hours, error=str(exc))
            raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", help="Historical origin with offset; default: current full hour")
    parser.add_argument("--horizon", type=int, choices=[24, 48], default=48)
    parser.add_argument("--watch", action="store_true", help="Poll the supplied release file and recalculate on changes")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    if args.interval < 5:
        parser.error("--interval must be at least 5 seconds")
    while True:
        origin = args.origin or pd.Timestamp.now(tz="UTC").floor("h")
        try:
            res = WindFarmAgent().execute_agent_cycle(origin, args.horizon)
            if res["changed"]:
                print(f"Saved {res['file']}", flush=True)
        except (ValueError, FileNotFoundError) as exc:
            print(str(exc), flush=True)
            if not args.watch:
                raise SystemExit(1)
        if not args.watch:
            break
        time.sleep(args.interval)

if __name__ == "__main__":
    main()
