"""Fetch NOAA GFS forecast releases for the project's strict weather contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

try:
    from herbie import Herbie
except ImportError as exc:
    raise SystemExit(
        "Missing herbie-data. Install with: pip install \"herbie-data[extras]\""
    ) from exc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gfs_archive")


@dataclass(frozen=True)
class Turbine:
    turbine_id: str
    lat: float
    lon: float


TURBINES = (
    Turbine("turbine_1", 43.645150, 78.535604),
    Turbine("turbine_2", 43.643198, 78.538828),
)
CYCLES = ("00", "06", "12", "18")
HUB_HEIGHT_M = 100.0
REF_HEIGHT_M = 10.0
WIND_SHEAR_ALPHA = 0.14


def daterange(start: str, end: str):
    current = datetime.strptime(start, "%Y-%m-%d")
    finish = datetime.strptime(end, "%Y-%m-%d")
    while current <= finish:
        yield current
        current += timedelta(days=1)


def source_url(herbie: Any) -> str | None:
    for attr in ("grib", "grib_source"):
        value = getattr(herbie, attr, None)
        if isinstance(value, str) and value.startswith("http"):
            return value
    sources = getattr(herbie, "SOURCES", None)
    if isinstance(sources, dict):
        for value in sources.values():
            if isinstance(value, str) and value.startswith("http"):
                return value
    return None


def available_at(herbie: Any) -> str:
    url = source_url(herbie)
    if not url:
        raise RuntimeError("Herbie did not expose a source URL for Last-Modified")
    response = requests.head(url, timeout=30, allow_redirects=True)
    response.raise_for_status()
    last_modified = response.headers.get("Last-Modified")
    if not last_modified:
        raise RuntimeError(f"Source has no Last-Modified header: {url}")
    parsed = datetime.strptime(last_modified, "%a, %d %b %Y %H:%M:%S %Z")
    return parsed.replace(tzinfo=timezone.utc).isoformat()


def find_data_var(dataset: Any, names: tuple[str, ...]):
    for name, data_array in dataset.data_vars.items():
        haystack = " ".join(
            [
                str(name),
                str(data_array.attrs.get("GRIB_shortName", "")),
                str(data_array.attrs.get("long_name", "")),
            ]
        ).lower()
        if any(candidate.lower() in haystack for candidate in names):
            return data_array
    return None


def extract_point(data_array: Any, lat: float, lon: float) -> float:
    if data_array is None:
        raise ValueError("Requested GRIB variable was not found")
    lat_name = "latitude" if "latitude" in data_array.coords else "lat"
    lon_name = "longitude" if "longitude" in data_array.coords else "lon"
    point = data_array.sel(
        {lat_name: lat, lon_name: lon % 360},
        method="nearest",
    )
    return float(point.values)


def fetch_run(issued_at: datetime, fxx: int, rows: list[dict], sources: list[dict]) -> None:
    run_id = f"gfs.{issued_at:%Y%m%d}.{issued_at:%H}z.f{fxx:03d}"
    try:
        herbie = Herbie(
            issued_at.replace(tzinfo=None),
            model="gfs",
            product="pgrb2.0p25",
            fxx=fxx,
            verbose=False,
        )
        published_at = available_at(herbie)
        wind_search = ":[UV]GRD:100 m above ground:"
        wind_source = "native_100m"
        try:
            wind_dataset = herbie.xarray(wind_search, remove_grib=False)
            u_data = find_data_var(wind_dataset, ("u100", "100u"))
            v_data = find_data_var(wind_dataset, ("v100", "100v"))
            if u_data is None or v_data is None:
                raise ValueError("native 100 m wind fields not found")
        except Exception:
            wind_source = "extrapolated_from_10m_power_law"
            wind_dataset = herbie.xarray(":[UV]GRD:10 m above ground:", remove_grib=False)
            u_data = find_data_var(wind_dataset, ("u10", "10u"))
            v_data = find_data_var(wind_dataset, ("v10", "10v"))
            if u_data is None or v_data is None:
                raise ValueError("neither 100 m nor 10 m wind fields found")

        temperature_dataset = herbie.xarray(":TMP:2 m above ground:", remove_grib=False)
        temperature_data = find_data_var(temperature_dataset, ("t2m", "2t", "tmp"))
        if temperature_data is None:
            raise ValueError("2 m temperature field not found")

        for turbine in TURBINES:
            u_value = extract_point(u_data, turbine.lat, turbine.lon)
            v_value = extract_point(v_data, turbine.lat, turbine.lon)
            wind_speed = (u_value**2 + v_value**2) ** 0.5
            if wind_source != "native_100m":
                wind_speed *= (HUB_HEIGHT_M / REF_HEIGHT_M) ** WIND_SHEAR_ALPHA
            temperature = extract_point(temperature_data, turbine.lat, turbine.lon) - 273.15
            rows.append(
                {
                    "run_id": run_id,
                    "issued_at": issued_at.isoformat(),
                    "available_at": published_at,
                    "valid_time": (issued_at + timedelta(hours=fxx)).isoformat(),
                    "turbine_id": turbine.turbine_id,
                    "wind_speed": round(wind_speed, 3),
                    "temperature": round(temperature, 2),
                    "hub_wind_source": wind_source,
                }
            )
        sources.append(
            {
                "run_id": run_id,
                "issued_at": issued_at.isoformat(),
                "available_at": published_at,
                "source_url": source_url(herbie),
                "fxx": fxx,
                "hub_wind_source": wind_source,
            }
        )
    except Exception as exc:
        log.warning("%s failed: %s", run_id, exc)


def fetch_run_result(job: tuple[datetime, int]) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    sources: list[dict] = []
    fetch_run(job[0], job[1], rows, sources)
    return rows, sources


def normalize_release_availability(rows: list[dict], sources: list[dict]) -> None:
    """Use the conservative availability of the complete forecast release."""
    by_issue: dict[str, list[pd.Timestamp]] = {}
    for row in rows:
        by_issue.setdefault(row["issued_at"], []).append(pd.Timestamp(row["available_at"]))
    release_times = {
        issued_at: max(times).isoformat()
        for issued_at, times in by_issue.items()
    }
    for row in rows:
        row["available_at"] = release_times[row["issued_at"]]
    for source in sources:
        source["available_at"] = release_times[source["issued_at"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2026-01-30")
    parser.add_argument("--end", default="2026-02-28")
    parser.add_argument("--out-dir", default="data/weather")
    parser.add_argument("--max-fxx", type=int, default=48)
    parser.add_argument("--cycles", default=",".join(CYCLES))
    args = parser.parse_args()
    if args.max_fxx < 1:
        raise SystemExit("--max-fxx must be positive")

    rows: list[dict] = []
    sources: list[dict] = []
    cycles = tuple(args.cycles.split(","))
    requested = 0
    jobs = []
    for day in daterange(args.start, args.end):
        for cycle in cycles:
            issued_at = day.replace(hour=int(cycle), tzinfo=timezone.utc)
            for fxx in range(1, args.max_fxx + 1):
                requested += 1
                jobs.append((issued_at, fxx))
    with ProcessPoolExecutor(max_workers=4) as executor:
        for job_rows, job_sources in executor.map(fetch_run_result, jobs):
            rows.extend(job_rows)
            sources.extend(job_sources)

    expected_rows = requested * len(TURBINES)
    if len(rows) != expected_rows:
        raise SystemExit(
            f"Incomplete archive: collected {len(rows)} of {expected_rows} expected rows. "
            "Fix source errors before using the archive."
        )
    normalize_release_availability(rows, sources)

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "forecasts.csv"
    manifest_path = output_dir / "source.json"
    frame = pd.DataFrame(rows)
    frame.to_csv(csv_path, index=False)
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    manifest = {
        "kind": "numerical_forecast",
        "provider": "NOAA GFS 0.25 degree",
        "source_url": "https://registry.opendata.aws/noaa-gfs-bdp-pds/",
        "availability_evidence": "HTTP Last-Modified header of the exact NOAA source object",
        "wind_speed_unit": "m/s",
        "temperature_unit": "degC",
        "wind_height_m": 100,
        "scada_wind_compatibility": "Native 100 m GFS wind, or documented 10 m power-law extrapolation with hub_wind_source",
        "sha256": digest,
        "date_range": [args.start, args.end],
        "cycles": list(cycles),
        "max_fxx": args.max_fxx,
        "turbines": [turbine.__dict__ for turbine in TURBINES],
        "runs_requested": requested,
        "runs": sources,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Wrote %d rows to %s", len(frame), csv_path)
    log.info("Wrote manifest to %s", manifest_path)


if __name__ == "__main__":
    main()
