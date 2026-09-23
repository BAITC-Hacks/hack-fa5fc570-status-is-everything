"""Non-destructive CSV audit and explicit hourly aggregation."""
import csv
import hashlib
import io
import numpy as np
import pandas as pd
from src.settings import ROOT, load_config, save_json

ALIASES = {
    "Статистическое время": "timestamp",
    "Средняя скорость ветра(m/s)": "wind_speed",
    "Нормализованная активная мощность": "power_normalized",
    "Нормализованная активная мощность на стороне линии": "power_normalized",
    "Средняя температура окружающей среды(°C)": "temperature",
}
VALUES = ["wind_speed", "power_normalized", "temperature"]


def read_raw(path, turbine_id):
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1251", "utf-16"):
        try:
            decoded = raw.decode(encoding)
            if "timestamp" in decoded[:2000] or "Статистическое время" in decoded[:2000]:
                break
        except UnicodeError:
            continue
    else:
        raise ValueError(f"Cannot detect supported CSV encoding: {path}")
    separator = csv.Sniffer().sniff(decoded[:16000], delimiters=",;\t").delimiter
    df = pd.read_csv(io.StringIO(decoded), sep=separator)
    df.columns = df.columns.str.strip()
    df = df.rename(columns=ALIASES)
    missing = set(["timestamp"] + VALUES) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    meta = {"file": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "encoding": encoding, "separator": separator, "sha256": hashlib.sha256(raw).hexdigest(),
            "rows": len(df), "source_missing": {k: int(v) for k, v in df.isna().sum().items()}}
    df = df[["timestamp"] + VALUES].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", format="mixed")
    for col in VALUES:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ".", regex=False), errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    df["turbine_id"] = turbine_id
    return df, meta


def aggregate_to_hourly(df, config):
    minutes = config["measurement_minutes"]
    if 60 % minutes:
        raise ValueError("measurement_minutes must divide 60")
    d = df.dropna(subset=["timestamp"]).copy()
    d = d.drop_duplicates(["timestamp"] + VALUES)
    conflicts = d.loc[d.duplicated("timestamp", keep=False), "timestamp"].unique()
    d["conflict"] = d.timestamp.isin(conflicts)
    d.loc[d.conflict, VALUES] = np.nan
    d = d.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    if d.empty:
        raise ValueError("No valid timestamps")
    if config["timestamp_meaning"] == "interval_end":
        d.index = d.index - pd.Timedelta(minutes=minutes)
    elif config["timestamp_meaning"] != "interval_start":
        raise ValueError("Choose interval_start or interval_end")
    d["on_grid"] = (d.index.minute % minutes == 0) & (d.index.second == 0) & (d.index.microsecond == 0)
    d["ambiguous_local_time"] = False
    if config.get("timezone"):
        localized = d.index.tz_localize(config["timezone"], ambiguous="NaT", nonexistent="NaT")
        d["ambiguous_local_time"] = localized.isna()
    d["usable"] = d[VALUES].notna().all(axis=1) & d.on_grid & ~d.conflict & ~d.ambiguous_local_time
    usable = d[VALUES].where(d.usable, np.nan)
    hourly = usable.resample("h", closed="left", label="left").mean()
    hourly["sample_count"] = d.resample("h").size()
    hourly["valid_count"] = d.usable.resample("h").sum().astype(int)
    hourly["conflict_count"] = d.conflict.resample("h").sum().astype(int)
    hourly["off_grid_count"] = (~d.on_grid).resample("h").sum().astype(int)
    hourly["ambiguous_local_time_count"] = d.ambiguous_local_time.resample("h").sum().astype(int)
    hourly["coverage"] = hourly.valid_count / (60 // minutes)
    hourly["is_complete"] = (hourly.coverage == 1) & (hourly.conflict_count == 0) & (hourly.off_grid_count == 0)
    hourly["train_eligible"] = (hourly.coverage >= config["min_hourly_coverage"]) & (hourly.conflict_count == 0) & (hourly.off_grid_count == 0) & (hourly.ambiguous_local_time_count == 0)
    hourly["turbine_id"] = df.turbine_id.iloc[0]
    return hourly.reset_index()


def prepare(config=None):
    config = config or load_config()
    frames, reports = [], {}
    for tid, settings in config["turbines"].items():
        d, report = read_raw(ROOT / settings["file"], tid)
        times = d.timestamp.dropna().sort_values()
        unique = d.dropna(subset=["timestamp"]).drop_duplicates(["timestamp"] + VALUES)
        report.update({
            "date_min": str(times.min()), "date_max": str(times.max()),
            "missing_after_parsing": {k: int(v) for k, v in d.isna().sum().items()},
            "duplicate_timestamp_rows": int(times.duplicated().sum()),
            "exact_duplicate_rows": int(d.duplicated(["timestamp"] + VALUES).sum()),
            "conflicting_timestamps": int(unique.loc[unique.duplicated("timestamp", keep=False), "timestamp"].nunique()),
            "intervals": {str(k): int(v) for k, v in times.diff().value_counts().items()},
            "ranges": {c: {"min": None if d[c].notna().sum() == 0 else float(d[c].min()), "max": None if d[c].notna().sum() == 0 else float(d[c].max())} for c in VALUES},
            "suspect_counts": {"negative_wind": int((d.wind_speed < 0).sum()), "wind_over_60": int((d.wind_speed > 60).sum()), "temperature_outside_minus80_60": int(((d.temperature < -80) | (d.temperature > 60)).sum()), "power_negative": int((d.power_normalized < 0).sum()), "power_above_one_not_necessarily_invalid": int((d.power_normalized > 1).sum()), "zero_power_preserved": int((d.power_normalized == 0).sum())},
            "timestamp_meaning": config["timestamp_meaning"], "timezone": config["timezone"],
        })
        h = aggregate_to_hourly(d, config)
        report.update({"hourly_rows": len(h), "complete_hours": int(h.is_complete.sum()), "complete_hour_fraction": float(h.is_complete.mean()), "training_eligible_hours": int(h.train_eligible.sum()), "ambiguous_local_time_measurements": int(h.ambiguous_local_time_count.sum())})
        reports[tid] = report
        frames.append(h)
    hourly = pd.concat(frames, ignore_index=True)
    output = ROOT / "data/processed/hourly.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(output, index=False)
    save_json(ROOT / "reports/data_quality.json", reports)
    return hourly, reports


if __name__ == "__main__":
    hourly, reports = prepare()
    for tid, report in reports.items():
        print(tid, report["rows"], report["date_min"], report["date_max"], "complete:", report["complete_hour_fraction"])
