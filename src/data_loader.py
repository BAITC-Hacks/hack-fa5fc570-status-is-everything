"""Compatibility helpers using the audited preparation path."""
import numpy as np
from src.data import read_raw, aggregate_to_hourly as aggregate, prepare
from src.settings import ROOT, load_config

def load_turbine_raw(turbine_num=1):
    tid = f"turbine_{turbine_num}"
    return read_raw(ROOT / load_config()["turbines"][tid]["file"], tid)[0]

def aggregate_to_hourly(df):
    return aggregate(df, load_config())

def get_hourly_data(turbine_num=1, use_cache=True):
    # Always rebuild/audit; old unversioned caches are not valid inputs.
    df, _ = prepare()
    return df[df.turbine_id == f"turbine_{turbine_num}"].copy()

def compute_power_curve(df, bin_width=0.5):
    data = df.copy()
    data["wind_bin"] = np.floor(data.wind_speed / bin_width) * bin_width + bin_width / 2
    return data.groupby("wind_bin").power_normalized.agg(["mean", "std", "count"]).reset_index().rename(columns={"mean": "power_mean", "std": "power_std"})
