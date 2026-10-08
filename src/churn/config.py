"""Load configs/config.yaml. Paths in the config are relative to the project root."""

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@lru_cache
def load_config(path: str | Path = ROOT / "configs" / "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(relative: str) -> Path:
    return ROOT / relative
