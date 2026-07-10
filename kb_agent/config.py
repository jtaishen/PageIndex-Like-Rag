from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DB_PATH = DATA_DIR / "kb.sqlite"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"
DEFAULT_DEEPSEEK_PROFILE = "default"
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
    return _deepseek_env("API_KEY")


def deepseek_base_url() -> str:
    load_env_file()
    return _deepseek_env("BASE_URL", DEFAULT_DEEPSEEK_BASE_URL).rstrip("/")


def deepseek_model() -> str:
    load_env_file()
    return _deepseek_env("MODEL", DEFAULT_DEEPSEEK_MODEL)


def deepseek_profile() -> str:
    load_env_file()
    return (os.environ.get("DEEPSEEK_PROFILE") or DEFAULT_DEEPSEEK_PROFILE).strip().lower() or DEFAULT_DEEPSEEK_PROFILE


def deepseek_temperature() -> float:
    load_env_file()
    raw = _deepseek_env("TEMPERATURE", "0.2")
    try:
        return float(raw)
    except ValueError:
        return 0.2


def deepseek_max_tokens() -> int:
    load_env_file()
    raw = _deepseek_env("MAX_TOKENS", "1200")
    try:
        return int(raw)
    except ValueError:
        return 1200


def _env_int(name: str, default: int) -> int:
    load_env_file()
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _deepseek_env(suffix: str, default: Optional[str] = None) -> Optional[str]:
    profile = deepseek_profile()
    if profile not in {"", DEFAULT_DEEPSEEK_PROFILE, "default", "v4", "deepseek"}:
        profiled_name = f"DEEPSEEK_{profile.upper()}_{suffix}"
        value = os.environ.get(profiled_name)
        if value is not None and value.strip():
            return value
    return os.environ.get(f"DEEPSEEK_{suffix}", default)


def deepseek_timeout_seconds() -> int:
    return _env_int("DEEPSEEK_TIMEOUT_SECONDS", 45)


def deepseek_probe_timeout_seconds() -> int:
    return _env_int("DEEPSEEK_PROBE_TIMEOUT_SECONDS", 15)


def deepseek_json_retry_count() -> int:
    return _env_int("DEEPSEEK_JSON_RETRY_COUNT", 1)


def mcp_llm_step_timeout_seconds() -> int:
    """Bound one staged MCP LLM request below the client's request timeout."""
    return _env_int("KB_MCP_LLM_STEP_TIMEOUT_SECONDS", 35)


def baseline_llm_timeout_seconds() -> int:
    return _env_int("KB_BASELINE_LLM_TIMEOUT_SECONDS", 420)


def baseline_llm_stage_timeout_seconds() -> int:
    return _env_int("KB_BASELINE_LLM_STAGE_TIMEOUT_SECONDS", 120)


def llm_fact_batch_size() -> int:
    return _env_int("KB_LLM_FACT_BATCH_SIZE", 6)


def llm_fact_max_nodes() -> int:
    return _env_int("KB_LLM_FACT_MAX_NODES", 18)


def llm_compare_evidence_per_doc() -> int:
    return _env_int("KB_LLM_COMPARE_EVIDENCE_PER_DOC", 3)


def ensure_data_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "parsed").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "state").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "logs").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "eval").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "eval_sets").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "eval_suites").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "state" / "search_profiles").mkdir(parents=True, exist_ok=True)
