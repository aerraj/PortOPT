from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and resolve paths relative to the project root."""
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    root = path.parent.parent
    config["_root"] = root
    for key in ("universe", "cache_dir"):
        config["data"][key] = root / config["data"][key]
    return config

