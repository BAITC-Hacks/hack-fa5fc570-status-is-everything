import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from src.model_store import training_rows, model_for_origin, model_output
from src.previous_runs import exploratory_power, inspect_response, load_archive, audit_points
from src.reporting import physical_power
from src.settings import load_config
from test_previous_runs import response_fixture


class PointInTimeTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.hourly = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-30", periods=72, freq="h"),
            "train_eligible": True, "wind_speed": 5., "temperature": 2., "power_normalized": 1.8,
        })

    def test_completed_hours_and_delivery_delay(self):
        train = training_rows(self.hourly, "2026-01-31T00:00:00+05:00", self.config)
        self.assertEqual(train.timestamp.max(), pd.Timestamp("2026-01-30T22:00"))
        bad_future = self.hourly.copy()
        bad_future.loc[bad_future.timestamp > train.timestamp.max(), "power_normalized"] = 999999
        pd.testing.assert_frame_equal(train, training_rows(bad_future, "2026-01-31T00:00:00+05:00", self.config))

    def test_no_february_targets_even_for_later_origin(self):
        train = training_rows(self.hourly, "2026-03-01T00:00:00+05:00", self.config)
        self.assertEqual(train.timestamp.max(), pd.Timestamp("2026-01-31T23:00"))

    def test_ambiguous_and_incomplete_hours_excluded(self):
        d = self.hourly.iloc[:3].copy()
        d.timestamp = pd.to_datetime(["2024-02-29T22:00", "2024-02-29T23:00", "2024-03-01T00:00"])
        d.loc[d.index[0], "train_eligible"] = False
        train = training_rows(d, "2026-01-31T00:00:00+05:00", self.config)
        self.assertEqual(len(train), 1)

    def test_delay_and_timezone_are_required(self):
        for delay in [-1, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                training_rows(self.hourly, "2026-01-31T00:00:00+05:00", {**self.config, "scada_delay_hours": delay})
        with self.assertRaises(ValueError):
            training_rows(self.hourly, "2026-01-31", self.config)

    def test_bounds_use_training_range_not_assumed_one(self):
        model = Mock()
        model.predict.return_value = np.array([-1., 1.5, 2.1])
        artifact = {"model": model, "features": ["wind_speed", "temperature"], "config": self.config,
                    "power_bounds": [0., 1.8], "feature_ranges": {"wind_speed": [0., 20.], "temperature": [-10., 30.]}}
        features = pd.DataFrame({"wind_speed": [5., 21., 10.], "temperature": [0., 0., 0.]})
        out = model_output(artifact, features)
        self.assertEqual(list(out.predicted_power), [0., 1.5, 1.8])
        self.assertEqual(list(out.raw_predicted_power), [-1., 1.5, 2.1])
        self.assertEqual(list(out.weather_out_of_training_range), [False, True, False])

    def test_real_january_31_model_and_cache_identity(self):
        a, path = model_for_origin("turbine_1", "2026-01-31T00:00:00+05:00", self.config)
        b, second = model_for_origin("turbine_1", "2026-01-31T00:00:00+05:00", self.config)
        self.assertEqual(path, second)
        self.assertEqual(a["identity"], b["identity"])
        self.assertLessEqual(pd.Timestamp(a["training_available_at"]), pd.Timestamp("2026-01-31T00:00:00+05:00"))
        self.assertLess(pd.Timestamp(a["training_max"]), pd.Timestamp("2026-01-31"))


class ArchiveAndUnitsTests(unittest.TestCase):
    def test_incomplete_turbine_set_is_rejected(self):
        frame, _ = inspect_response(response_fixture(), "turbine_1", 43., 78.)
        with self.assertRaisesRegex(ValueError, "every configured turbine"):
            exploratory_power(frame, "2026-02-01T00:00:00Z", 24)

    def test_truncated_requested_period_rejected(self):
        session = Mock()
        session.get.return_value.json.return_value = response_fixture(hours=24)
        session.get.return_value.url = "test://fixture"
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "requested date range"):
                audit_points({"turbine_1": (43., 78.)}, "2026-01-31", "2026-02-01", folder, session)
            self.assertFalse(list(Path(folder).iterdir()))

    def test_infinite_and_negative_weather_rejected(self):
        for value in [np.inf, -1.]:
            payload = response_fixture()
            payload["hourly"]["wind_speed_100m_previous_day1"][0] = value
            with self.assertRaises(ValueError):
                inspect_response(payload, "turbine_1", 43., 78.)

    def test_automatic_download_and_checksum_validation(self):
        config = copy.deepcopy(load_config())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "previous_runs_2026-01-31_2026-02-02.csv"
            config["exploratory_weather"] = {"archive": str(path), "start_date": "2026-01-31", "end_date": "2026-02-02"}
            response = Mock()
            response.json.return_value = response_fixture()
            response.url = "test://fixture"
            with patch("src.previous_runs.requests.get", return_value=response) as get:
                frame = load_archive(config)
                self.assertEqual(len(frame), 144)
                self.assertEqual(get.call_count, 2)
                load_archive(config)
                self.assertEqual(get.call_count, 2)
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_archive(config)

    def test_physical_units_require_documentation_and_full_plant(self):
        config = copy.deepcopy(load_config())
        frame = pd.DataFrame({"turbine_id": ["turbine_1", "turbine_2"], "forecast_origin": ["2026-01-31"] * 2,
                              "valid_time": ["2026-02-01"] * 2, "predicted_power": [.5, .25]})
        with self.assertRaisesRegex(ValueError, "confirmed"):
            physical_power(frame, config)
        for tid, scale in [("turbine_1", 2.), ("turbine_2", 4.)]:
            config["turbines"][tid].update(normalization_to_mw=scale, normalization_evidence="SYNTHETIC TEST ONLY")
        output, plant = physical_power(frame, config)
        self.assertEqual(plant.predicted_power_mw.iloc[0], 2.)
        self.assertEqual(plant.predicted_energy_mwh.iloc[0], 2.)
        with self.assertRaises(ValueError):
            physical_power(frame.iloc[:1], config)


if __name__ == "__main__":
    unittest.main()
