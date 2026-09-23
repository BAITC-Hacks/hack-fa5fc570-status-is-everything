"""Synthetic fixtures are only for regression tests, never production forecasts."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.data import aggregate_to_hourly, read_raw
from src.predict import validate_weather, predict_power
from src.settings import load_config, ROOT
from src.train import WindBinBaseline
from src.weather_service import WeatherService
from src.agent import WindFarmAgent


def weather_fixture(hours=48, origin="2026-02-01T00:00:00+05:00"):
    start = pd.Timestamp(origin)
    return pd.DataFrame([{"turbine_id": tid, "forecast_origin": start, "issued_at": start - pd.Timedelta(hours=6), "available_at": start - pd.Timedelta(hours=3), "valid_time": stamp, "wind_speed": 5., "temperature": -1.} for tid in ("turbine_1", "turbine_2") for stamp in pd.date_range(start + pd.Timedelta(hours=1), periods=hours, freq="h")])


class DataTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.raw = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=6, freq="10min"), "wind_speed": [5.] * 6, "temperature": [0.] * 6, "power_normalized": [0., 0., 1.8, 1.8, 1.8, 1.8], "turbine_id": ["turbine_1"] * 6})

    def test_exact_duplicate_does_not_change_mean_or_count(self):
        h = aggregate_to_hourly(pd.concat([self.raw, self.raw.iloc[[0]]]), self.config)
        self.assertEqual(h.sample_count.iloc[0], 6)
        self.assertAlmostEqual(h.power_normalized.iloc[0], 1.2)
        self.assertTrue(h.train_eligible.iloc[0])

    def test_conflict_is_marked_not_arbitrarily_selected(self):
        conflict = self.raw.iloc[[0]].copy()
        conflict["power_normalized"] = 10
        h = aggregate_to_hourly(pd.concat([self.raw, conflict]), self.config)
        self.assertEqual(h.conflict_count.iloc[0], 1)
        self.assertFalse(h.train_eligible.iloc[0])

    def test_incomplete_hour_and_missing_target(self):
        h = aggregate_to_hourly(self.raw.iloc[:3], self.config)
        self.assertEqual(h.coverage.iloc[0], .5)
        self.assertFalse(h.train_eligible.iloc[0])
        config = {**self.config, "min_hourly_coverage": .5}
        self.assertTrue(aggregate_to_hourly(self.raw.iloc[:3], config).train_eligible.iloc[0])
        d = self.raw.copy()
        d["power_normalized"] = np.nan
        self.assertTrue(pd.isna(aggregate_to_hourly(d, self.config).power_normalized.iloc[0]))

    def test_missing_hour_is_not_filled(self):
        later = self.raw.copy()
        later.timestamp += pd.Timedelta(hours=2)
        h = aggregate_to_hourly(pd.concat([self.raw, later]), self.config)
        self.assertEqual(len(h), 3)
        self.assertTrue(pd.isna(h.power_normalized.iloc[1]))

    def test_interval_end_and_off_grid(self):
        d = self.raw.copy()
        d.timestamp += pd.Timedelta(minutes=10)
        h = aggregate_to_hourly(d, {**self.config, "timestamp_meaning": "interval_end"})
        self.assertEqual(len(h), 1)
        self.assertTrue(h.is_complete.iloc[0])
        d.timestamp += pd.Timedelta(minutes=1)
        self.assertFalse(aggregate_to_hourly(d, self.config).train_eligible.any())

    def test_kazakhstan_offset_transition_is_not_guessed(self):
        d = self.raw.copy()
        d.timestamp = pd.date_range("2024-02-29 23:00", periods=6, freq="10min")
        h = aggregate_to_hourly(d, self.config)
        self.assertEqual(h.ambiguous_local_time_count.iloc[0], 6)
        self.assertFalse(h.train_eligible.iloc[0])

    def test_encoding_and_separator(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "input.csv"
            # Test fixture only.
            p.write_bytes("Статистическое время;Средняя скорость ветра(m/s);Нормализованная активная мощность;Средняя температура окружающей среды(°C)\n2026-01-01;5;1.2;0\n".encode("cp1251"))
            d, meta = read_raw(p, "turbine_1")
            self.assertEqual(meta["separator"], ";")
            self.assertEqual(meta["encoding"], "cp1251")
            self.assertEqual(d.power_normalized.iloc[0], 1.2)

    def test_baseline_unseen_bin_uses_training_mean(self):
        baseline = WindBinBaseline().fit(self.raw)
        query = self.raw.iloc[[0]].copy()
        query.wind_speed = 99
        self.assertAlmostEqual(baseline.predict(query)[0], self.raw.power_normalized.mean())


class WeatherTests(unittest.TestCase):
    def test_time_contract(self):
        good = weather_fixture()
        self.assertEqual(len(validate_weather(good)), 96)
        for field in ["issued_at", "available_at"]:
            bad = good.copy()
            bad[field] = bad.forecast_origin + pd.Timedelta(hours=1)
            with self.assertRaises(ValueError):
                validate_weather(bad)
        bad = good.copy()
        bad["valid_time"] = bad.forecast_origin
        with self.assertRaises(ValueError):
            validate_weather(bad)

    def test_unknown_time_duplicate_nan_and_unknown_timezone(self):
        good = weather_fixture()
        for bad in [pd.concat([good, good.iloc[[0]]]), good.assign(available_at=None), good.assign(wind_speed=np.nan), good.assign(issued_at="2026-01-31 00:00")]:
            with self.assertRaises(ValueError):
                validate_weather(bad)
        with self.assertRaises(ValueError):
            validate_weather(good, {**load_config(), "timezone": None})

    def test_latest_available_release_and_complete_horizon(self):
        older = weather_fixture()
        later = older.copy()
        later["issued_at"] = later.forecast_origin + pd.Timedelta(hours=1)
        later["available_at"] = later.forecast_origin + pd.Timedelta(hours=2)
        later.wind_speed = 20
        service = WeatherService()
        with patch.object(service, "load_releases", return_value=pd.concat([older, later])):
            selected = service.get_forecast_as_of("2026-02-01T00:00:00+05:00")
            self.assertTrue((selected.wind_speed == 5).all())
        with patch.object(service, "load_releases", return_value=older.iloc[:-1]):
            with self.assertRaisesRegex(ValueError, "48 hours"):
                service.get_forecast_as_of("2026-02-01T00:00:00+05:00")

    def test_last_february_day_needs_march(self):
        fixture = weather_fixture(origin="2026-02-28T00:00:00+05:00")
        service = WeatherService()
        with patch.object(service, "load_releases", return_value=fixture):
            selected = service.get_forecast_as_of("2026-02-28T00:00:00+05:00")
            self.assertEqual(len(selected), 96)
            self.assertEqual(selected.valid_time.max().tz_convert("Asia/Almaty").month, 3)

    def test_missing_real_source_is_an_error(self):
        with tempfile.TemporaryDirectory() as folder:
            config = {**load_config(), "weather_file": str(Path(folder) / "missing.csv"), "weather_manifest": str(Path(folder) / "missing.json")}
            with self.assertRaises(FileNotFoundError):
                WeatherService(config=config).load_releases()

    def test_manifest_units_checksum_and_duplicates(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "weather.csv"
            manifest = Path(folder) / "source.json"
            fixture = weather_fixture()
            fixture.to_csv(path, index=False)
            meta = {"provider": "TEST ONLY", "source_url": "test://fixture", "availability_evidence": "test fixture", "wind_height_m": 100, "scada_wind_compatibility": "test fixture", "kind": "numerical_forecast", "wind_speed_unit": "m/s", "temperature_unit": "degC", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            config = {**load_config(), "weather_file": str(path), "weather_manifest": str(manifest)}
            manifest.write_text(json.dumps(meta), encoding="utf-8")
            self.assertEqual(len(WeatherService(config=config).load_releases()), 96)
            meta["wind_speed_unit"] = "km/h"
            manifest.write_text(json.dumps(meta), encoding="utf-8")
            with self.assertRaises(ValueError):
                WeatherService(config=config).load_releases()
            meta["wind_speed_unit"] = "m/s"
            meta["sha256"] = "wrong"
            manifest.write_text(json.dumps(meta), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                WeatherService(config=config).load_releases()
            pd.concat([fixture, fixture.iloc[[0]]]).to_csv(path, index=False)
            meta["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest.write_text(json.dumps(meta), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                WeatherService(config=config).load_releases()


class ModelIntegrationTests(unittest.TestCase):
    def test_trained_models_and_no_future_training(self):
        self.assertTrue((ROOT / "models/validated/turbine_1.joblib").exists(), "Run python -m src.train first")
        result = predict_power(weather_fixture())
        self.assertEqual(len(result), 96)
        self.assertTrue(np.isfinite(result.predicted_power).all())
        # Earlier origins now select a point-in-time refit, not the final January model.
        earlier = predict_power(weather_fixture(origin="2026-01-31T00:00:00+05:00"))
        available = pd.to_datetime(earlier.model_training_available_at, utc=True)
        self.assertTrue((available <= earlier.forecast_origin).all())
        self.assertTrue((pd.to_datetime(earlier.model_training_max) < pd.Timestamp("2026-01-31")).all())

    def test_agent_persists_versions_and_reaudits_updates(self):
        # Mock only weather transport; run the real saved models and agent.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            agent = WindFarmAgent()
            original_root = ROOT
            # Point model hashing at test-local copies; actual prediction uses project models.
            import shutil
            shutil.copytree(original_root / "models/validated", root / "models/validated")
            weather = validate_weather(weather_fixture())
            with patch("src.agent.ROOT", root), patch.object(agent.weather_service, "get_forecast_as_of", return_value=weather):
                first = agent.execute_agent_cycle("2026-02-01T00:00:00+05:00")
                second = agent.execute_agent_cycle("2026-02-01T00:00:00+05:00")
                self.assertTrue(first["changed"])
                self.assertFalse(second["changed"])
                self.assertEqual(len(list((root / "data/forecasts").glob("*.csv"))), 1)
                weather["wind_speed"] = 9.
                updated = agent.execute_agent_cycle("2026-02-01T00:00:00+05:00")
                self.assertNotEqual(first["revision"], updated["revision"])
                self.assertEqual(updated["audit"]["turbine_1"]["max_wind_ms"], 9.)
                self.assertEqual(updated["previous_revision"], first["revision"])
                self.assertEqual(len(list((root / "data/forecasts").glob("*.csv"))), 2)
                with self.assertRaises(ValueError):
                    agent.execute_agent_cycle("2026-02-01T00:00:00+05:00", simulate_update=True)
                self.assertIn('"status": "failed"', (root / "reports/agent_events.jsonl").read_text())


if __name__ == "__main__":
    unittest.main()
