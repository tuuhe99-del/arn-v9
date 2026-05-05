"""ARN user configuration helpers."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(os.environ.get("ARN_CONFIG_FILE", "~/.arn_config.json")).expanduser()


def load_config() -> dict[str, Any]:
    try:
        if CONFIG_PATH.exists():
            with CONFIG_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


def save_config(config: dict[str, Any]) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CONFIG_PATH.with_suffix(CONFIG_PATH.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(CONFIG_PATH)


def get_default_tier(fallback: str = "nano") -> str:
    return os.environ.get("ARN_EMBEDDING_TIER") or load_config().get("embedding_tier") or fallback


def set_default_tier(tier: str) -> None:
    cfg = load_config()
    cfg["embedding_tier"] = tier
    save_config(cfg)
