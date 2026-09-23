"""Reproducible fixed-lead research replay, never a strict as-of backtest."""
import argparse
import hashlib

import pandas as pd

from src.data import prepare
from src.previous_runs import exploratory_power, load_archive
from src.reporting import forecast_audit
from src.settings import ROOT, load_config, save_json
from src.train import score

LIMITATION = ("Exploratory fixed-lead weather evaluation, not verified as-of accuracy: "
              "exact issued_at/available_at and SCADA wind-height compatibility are unknown. "
              "SCADA availability follows the configured delay assumption. Outage labels are absent.")


def replay(start, end, horizon=48, name="february_exploratory", refresh=False, evaluate=False):
    config = load_config()
    slices = load_archive(config, refresh=refresh)
    origins = pd.date_range(start, end, freq="D", tz=config["timezone"])
    if origins.empty:
        raise ValueError("Empty origin range")
    results, errors, audits = [], [], {}
    for origin in origins:
        try:
            output = exploratory_power(slices, origin, horizon, config=config)
            results.append(output)
            audits[str(origin)] = forecast_audit(output, output)
        except (ValueError, FileNotFoundError) as exc:
            errors.append({"origin": str(origin), "error": str(exc)})
        print(f"{name}: {origin.date()} ({len(results)} succeeded, {len(errors)} failed)", flush=True)
    joined = pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=["turbine_id", "forecast_origin", "valid_time", "predicted_power"])
    target = ROOT / f"reports/{name}_predictions.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(target, index=False)
    report = {"origins_total": len(origins), "origins_succeeded": len(results), "origins_failed": len(errors),
              "errors": errors, "start": start, "end": end, "horizon_hours": horizon,
              "forecast_contract_compatible": False, "limitation": LIMITATION,
              "weather_sha256": hashlib.sha256((ROOT / config["exploratory_weather"]["archive"]).read_bytes()).hexdigest(),
              "predictions_sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "audit": audits}
    if evaluate and results:
        hourly, _ = prepare(config)
        actual = hourly[hourly.train_eligible][["turbine_id", "timestamp", "power_normalized", "wind_speed", "temperature"]].copy()
        actual["valid_time"] = actual.timestamp.dt.tz_localize(config["timezone"], ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
        actual = actual.rename(columns={"wind_speed": "observed_wind_speed", "temperature": "observed_temperature"})
        comparison = joined.merge(actual.drop(columns="timestamp"), on=["turbine_id", "valid_time"], how="left", validate="many_to_one")
        report["missing_actual_rows"] = int(comparison.power_normalized.isna().sum())
        report["metrics"] = {}
        matched = comparison.dropna(subset=["power_normalized"])
        for (tid, lead), group in matched.groupby(["turbine_id", "nominal_lead_time_hours"]):
            error = (group.predicted_power - group.power_normalized).abs()
            # Flag candidates for engineering review, never label these as confirmed outages.
            possible_stop = (group.observed_wind_speed >= 5) & (group.power_normalized <= .02)
            report["metrics"][f"{tid}_lead{lead}h"] = {
                "turbine_id": tid, "nominal_lead_hours": int(lead), "rows": len(group),
                "model": score(group.power_normalized, group.predicted_power),
                "raw_model": score(group.power_normalized, group.raw_predicted_power),
                "baseline": score(group.power_normalized, group.baseline_prediction),
                "max_absolute_error": float(error.max()), "p95_absolute_error": float(error.quantile(.95)),
                "bounded_outputs": int(group.prediction_bounded.sum()),
                "zero_or_low_power_with_wind_candidates": int(possible_stop.sum()),
                "wind_mae_ms": float((group.wind_speed - group.observed_wind_speed).abs().mean()),
                "beats_baseline_mae": bool(score(group.power_normalized, group.predicted_power)["mae"] < score(group.power_normalized, group.baseline_prediction)["mae"]),
            }
        report["quality_verdict"] = "Research only; model must improve over baseline and weather/SCADA compatibility must be verified before operational use"
        comparison["absolute_error"] = (comparison.predicted_power - comparison.power_normalized).abs()
        comparison.to_csv(ROOT / f"reports/{name}_comparison.csv", index=False)
        comparison.nlargest(20, "absolute_error").to_csv(ROOT / f"reports/{name}_largest_errors.csv", index=False)
    save_json(ROOT / f"reports/{name}.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--january", action="store_true", help="Daily model refits and comparison with January SCADA")
    parser.add_argument("--refresh-weather", action="store_true")
    args = parser.parse_args()
    if args.january:
        # Jan 29 + 48 hours fits inside actual SCADA; Jan 30 would require Feb 1.
        result = replay("2026-01-01", "2026-01-29", name="january_fixed_lead", refresh=args.refresh_weather, evaluate=True)
    else:
        result = replay("2026-01-31", "2026-02-28", refresh=args.refresh_weather)
    print({k: result[k] for k in ["origins_total", "origins_succeeded", "origins_failed", "forecast_contract_compatible"]})
    raise SystemExit(1 if result["origins_failed"] else 0)


if __name__ == "__main__":
    main()
