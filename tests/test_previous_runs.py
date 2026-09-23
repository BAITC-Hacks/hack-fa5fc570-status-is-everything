"""Previous Runs API coverage and fixed-lead selection, without network calls."""
import unittest
from unittest.mock import Mock

import pandas as pd

from src.previous_runs import (
    ENDPOINT, HOURLY, MODEL, candidate_for_origin, exploratory_power, fetch_previous_runs,
    inspect_response,
)


def response_fixture(start="2026-01-31T00:00", hours=72):
    times = pd.date_range(start, periods=hours, freq="h", tz="UTC")
    return {
        "timezone": "GMT",
        "utc_offset_seconds": 0,
        "latitude": 43.64,
        "longitude": 78.52,
        "hourly_units": {key: "°C" if key.startswith("temperature_") else "m/s" for key in HOURLY},
        "hourly": {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times],
                   **{key: [3.0] * hours for key in HOURLY}},
    }


class PreviousRunsTests(unittest.TestCase):
    def test_request_fixes_model_units_utc_and_suffix_fields(self):
        session = Mock()
        response = session.get.return_value
        response.json.return_value = response_fixture()
        response.url = "https://example.invalid/test"
        payload, url = fetch_previous_runs(43.64515, 78.535604, "2026-01-31", "2026-02-02", session)
        self.assertEqual(payload["hourly"]["time"][0], "2026-01-31T00:00")
        self.assertEqual(url, response.url)
        args, kwargs = session.get.call_args
        self.assertEqual(args[0], ENDPOINT)
        self.assertEqual(kwargs["params"]["models"], MODEL)
        self.assertEqual(kwargs["params"]["wind_speed_unit"], "ms")
        self.assertEqual(kwargs["params"]["timezone"], "GMT")
        self.assertEqual(set(kwargs["params"]["hourly"].split(",")), set(HOURLY))
        self.assertTrue(all("previous_day" in key for key in kwargs["params"]["hourly"].split(",")))

    def test_reject_missing_100m_and_wrong_units(self):
        payload = response_fixture()
        payload["hourly"]["wind_speed_100m_previous_day2"] = [None] * 72
        frame, report = inspect_response(payload, "turbine_1", 43.64515, 78.535604)
        self.assertEqual(report["series"]["wind_speed_100m_previous_day2"]["null"], 72)
        self.assertEqual(len(frame), 72)
        payload["hourly_units"]["wind_speed_100m_previous_day2"] = "km/h"
        with self.assertRaisesRegex(ValueError, "unit"):
            inspect_response(payload, "turbine_1", 43.64515, 78.535604)

    def test_no_shorter_nominal_lead_for_earlier_origin(self):
        frame, _ = inspect_response(response_fixture(), "turbine_1", 43.64515, 78.535604)
        origin = "2026-01-30T00:00:00Z"
        selected = candidate_for_origin(frame, origin)
        self.assertTrue((selected.nominal_lead_time_hours >= 24).all())
        # Jan 31 01 UTC is 25h ahead of Jan 30 00; day1 is too recent.
        target = selected[selected.valid_time == pd.Timestamp("2026-01-31T01:00:00Z")]
        self.assertEqual(target.nominal_lead_time_hours.iloc[0], 48)
        # Neither day1 nor day2 may be used for >48h ahead.
        self.assertFalse((selected.valid_time > pd.Timestamp(origin) + pd.Timedelta(hours=48)).any())

    def test_no_issued_or_available_time_is_invented(self):
        frame, _ = inspect_response(response_fixture(), "turbine_1", 43.64515, 78.535604)
        self.assertNotIn("issued_at", frame.columns)
        self.assertNotIn("available_at", frame.columns)
        selected = candidate_for_origin(frame, "2026-01-30T00:00:00Z")
        self.assertNotIn("issued_at", selected.columns)
        self.assertNotIn("available_at", selected.columns)

    def test_february_10_exploratory_estimate_uses_saved_models(self):
        left, _ = inspect_response(response_fixture("2026-02-10T00:00", 96), "turbine_1", 43.64515, 78.535604)
        right, _ = inspect_response(response_fixture("2026-02-10T00:00", 96), "turbine_2", 43.643198, 78.538828)
        both = pd.concat([left, right], ignore_index=True)
        for horizon in (24, 48):
            estimate = exploratory_power(both, "2026-02-10T10:00:00+05:00", horizon)
            self.assertEqual(len(estimate), 2 * horizon)
            self.assertTrue((estimate.nominal_lead_time_hours >= 24).all())
            self.assertTrue(estimate.predicted_power.notna().all())
            self.assertNotIn("issued_at", estimate.columns)
            self.assertNotIn("available_at", estimate.columns)


if __name__ == "__main__":
    unittest.main()
