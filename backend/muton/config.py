from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def repo_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def env_path(name: str, default: str) -> Path:
    value = os.environ.get(name, "").strip()
    if value:
        return Path(value).expanduser()
    return repo_path(default)


def env_str(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default
