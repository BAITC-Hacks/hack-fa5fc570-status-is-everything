"""Load documented numerical forecast releases; no reanalysis or synthetic fallback."""
import hashlib
import io
import json
import pandas as pd
from src.predict import aware_timestamp, validate_weather
from src.settings import ROOT, load_config

class WeatherService:
    def __init__(self, use_cache=True, config=None):
        self.config = config or load_config()

    def load_releases(self):
        path = ROOT / self.config["weather_file"]
        manifest_path = ROOT / self.config["weather_manifest"]
        if not path.exists() or not manifest_path.exists():
            raise FileNotFoundError("Проверенные выпуски погоды отсутствуют. Нужны data/weather/forecasts.csv и source.json; старый кэш не используется.")
        meta = json.loads(manifest_path.read_text(encoding="utf-8"))
        required = ["provider", "source_url", "availability_evidence", "wind_height_m", "scada_wind_compatibility"]
        if any(not meta.get(k) for k in required):
            raise ValueError("Weather provenance, wind height or SCADA compatibility is undocumented")
        if meta.get("kind") != "numerical_forecast" or meta.get("wind_speed_unit") != "m/s" or meta.get("temperature_unit") != "degC":
            raise ValueError("Only numerical forecasts in m/s and degC are accepted")
        raw = path.read_bytes()
        if meta.get("sha256") != hashlib.sha256(raw).hexdigest():
            raise ValueError("Weather CSV checksum does not match its provenance manifest")
        df = pd.read_csv(io.BytesIO(raw))
        required_cols = {"turbine_id", "issued_at", "available_at", "valid_time", "wind_speed", "temperature"}
        if required_cols - set(df.columns):
            raise ValueError(f"Missing release fields: {sorted(required_cols - set(df.columns))}")
        for col in ["issued_at", "available_at", "valid_time"]:
            df[col] = pd.to_datetime(df[col].map(aware_timestamp), utc=True)
        if df.empty or df.turbine_id.isna().any() or not set(df.turbine_id).issubset(self.config["turbines"]):
            raise ValueError("Empty weather input or unknown turbine")
        if (df.available_at < df.issued_at).any():
            raise ValueError("Invalid release availability")
        if df.duplicated(["turbine_id", "issued_at", "valid_time"]).any():
            raise ValueError("Conflicting or duplicate release records")
        if (df.groupby(["turbine_id", "issued_at"]).available_at.nunique() != 1).any():
            raise ValueError("One release must have one documented availability timestamp")
        return df

    def get_forecast_as_of(self, as_of_date, horizon_hours=48):
        if horizon_hours not in (24, 48):
            raise ValueError("horizon_hours must be 24 or 48")
        origin = pd.Timestamp(as_of_date)
        if origin.tzinfo is None:
            if not self.config.get("timezone"):
                raise ValueError("SCADA timezone is unresolved")
            origin = origin.tz_localize(self.config["timezone"], ambiguous="raise", nonexistent="raise")
        origin = origin.tz_convert("UTC")
        if origin != origin.floor("h"):
            raise ValueError("Forecast origin must be an hourly boundary")
        releases = self.load_releases()
        expected = pd.date_range(origin + pd.Timedelta(hours=1), periods=horizon_hours, freq="h")
        selected = []
        for tid in self.config["turbines"]:
            eligible = releases[(releases.turbine_id == tid) & (releases.issued_at <= origin) & (releases.available_at <= origin)]
            if eligible.empty:
                raise ValueError(f"{tid}: no release available at {origin}")
            latest = eligible.issued_at.max()
            run = eligible[(eligible.issued_at == latest) & eligible.valid_time.isin(expected)].copy()
            if set(run.valid_time) != set(expected) or len(run) != horizon_hours:
                raise ValueError(f"{tid}: latest available release does not cover all {horizon_hours} hours")
            run["forecast_origin"] = origin
            selected.append(run)
        return validate_weather(pd.concat(selected, ignore_index=True), self.config)

    def simulate_weather_update(self, *args, **kwargs):
        raise ValueError("Synthetic updates disabled; publish a real documented weather release.")
