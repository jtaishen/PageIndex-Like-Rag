from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DB_PATH = DATA_DIR / "kb.sqlite"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"
SUPPORTED_EXTENSIONS = {
    ".md",
    ".markdown",
    ".txt",
    ".pdf",
    ".docx",
    ".html",
    ".htm",
}

_ENV_LOADED = False


def load_env_file(path: Optional[Path] = None) -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    env_path = path or (PROJECT_ROOT / ".env")
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_db_path(value: Optional[str] = None) -> Path:
    load_env_file()
    raw = value or os.environ.get("KB_AGENT_DB")
    if raw:
        return Path(raw).expanduser().resolve()
    return DEFAULT_DB_PATH


def deepseek_api_key() -> Optional[str]:
    load_env_file()
    return os.environ.get("DEEPSEEK_API_KEY")


def deepseek_base_url() -> str:
    load_env_file()
    return os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL).rstrip("/")


def deepseek_model() -> str:
    load_env_file()
    return os.environ.get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)


def deepseek_temperature() -> float:
    load_env_file()
    raw = os.environ.get("DEEPSEEK_TEMPERATURE", "0.2")
    try:
        return float(raw)
    except ValueError:
        return 0.2


def deepseek_max_tokens() -> int:
    load_env_file()
    raw = os.environ.get("DEEPSEEK_MAX_TOKENS", "1200")
    try:
        return int(raw)
    except ValueError:
        return 1200


def ensure_data_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "parsed").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "state").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "logs").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "eval").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "eval_sets").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "state" / "search_profiles").mkdir(parents=True, exist_ok=True)
