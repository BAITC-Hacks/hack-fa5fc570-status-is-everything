"""February replay of documented available releases; unavailable days are reported."""
import pandas as pd
from src.agent import WindFarmAgent
from src.settings import ROOT, load_config, save_json

def run_february_walk_forward_simulation(horizon_hours=48):
    config = load_config()
    if not config.get("timezone"):
        raise ValueError("Timezone unresolved")
    agent = WindFarmAgent(config=config)
    rows, errors = [], []
    for day in pd.date_range("2026-01-31", "2026-02-28", freq="D", tz=config["timezone"]):
        try:
            rows.append(agent.execute_agent_cycle(day, horizon_hours)["forecast"])
        except (ValueError, FileNotFoundError) as exc:
            errors.append({"origin": str(day), "error": str(exc)})
    report = {"start_origin": "2026-01-31", "end_origin": "2026-02-28", "origins_total": 29, "origins_succeeded": len(rows), "origins_failed": len(errors), "errors": errors, "limitation": "Strict replay predictions only. Includes Jan 31 origin. February observed power is absent, so no February accuracy metric is reported."}
    save_json(ROOT / "reports/february_replay.json", report)
    # Always overwrite the summary output so a failed rerun cannot expose stale results.
    output = pd.concat(rows) if rows else pd.DataFrame(columns=["turbine_id", "forecast_origin", "valid_time", "predicted_power"])
    output.to_csv(ROOT / "reports/february_predictions.csv", index=False)
    print(report)
    return report

if __name__ == "__main__":
    result = run_february_walk_forward_simulation()
    raise SystemExit(1 if result["origins_failed"] else 0)
