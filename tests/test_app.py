from datetime import date
import unittest

from streamlit.testing.v1 import AppTest
from src.settings import ROOT


class DashboardTests(unittest.TestCase):
    def test_january31_and_february10_have_estimates(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        self.assertFalse(list(app.exception))
        self.assertEqual(app.sidebar.date_input[0].value, date(2026, 1, 31))
        self.assertEqual(len(app.get("plotly_chart")), 2)
        app.sidebar.date_input[0].set_value(date(2026, 2, 10))
        app.sidebar.selectbox[0].set_value(10)
        app.run()
        self.assertFalse(list(app.exception))
        self.assertEqual(len(app.get("plotly_chart")), 2)
        for horizon in [24, 48]:
            app.sidebar.radio[0].set_value(horizon)
            app.run()
            self.assertFalse(list(app.exception))
            # First table is the exploratory hour-by-hour estimate.
            estimate = next(table.value for table in app.dataframe if "nominal_lead_time_hours" in table.value)
            self.assertEqual(len(estimate), 2 * horizon)


if __name__ == "__main__":
    unittest.main()
