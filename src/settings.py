"""Shared paths and explicit, versioned project configuration."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    with open(path or ROOT / "config/settings.json", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
