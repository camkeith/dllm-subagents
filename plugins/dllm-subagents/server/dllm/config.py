from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Mapping, Optional

HARD_MIN_STEPS = 1
HARD_MAX_STEPS = 150
HARD_MIN_SECONDS = 10
HARD_MAX_SECONDS = 1800
MAX_WAIT_SECONDS = 55
DEFAULT_WAIT_SECONDS = 50.0

DEFAULTS_PATH = Path(__file__).resolve().parent.parent / "providers.default.json"


class ConfigError(Exception):
    pass


@dataclass
class Profile:
    name: str
    base_url: str
    api_key_env: str
    model: str
    max_tokens_field: str = "max_completion_tokens"
    max_output_tokens: int = 8192
    extra_body: dict = field(default_factory=dict)
    temperature: Optional[float] = None


@dataclass
class Limits:
    max_steps: int = 40
    max_seconds: int = 300
    max_tokens_total: int = 2_000_000


@dataclass
class Settings:
    default_provider: str
    profiles: Dict[str, Profile]
    limits: Limits
    wait_seconds: float = DEFAULT_WAIT_SECONDS
    pass_env: List[str] = field(default_factory=list)  # secret-looking variables the subagent may see


def config_dir() -> Path:
    override = os.environ.get("DLLM_SUBAGENTS_CONFIG_DIR")
    if override:
        return Path(override)
    return Path.home() / ".config" / "dllm-subagents"


def _read_json(path: Path, required: bool) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        if required:
            raise ConfigError("Missing config file: %s" % path)
        return {}
    except OSError as exc:
        raise ConfigError("Cannot read %s: %s" % (path, exc))
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise ConfigError("Invalid JSON in %s: %s" % (path, exc))
    if not isinstance(data, dict):
        raise ConfigError("%s must contain a JSON object" % path)
    return data


def _merge_provider(raw: Dict[str, dict], name: str, patch: dict, source: Path) -> None:
    if not isinstance(patch, dict):
        raise ConfigError("Provider %r in %s must be an object" % (name, source))
    raw.setdefault(name, {}).update(patch)


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _section(data: dict, key: str, source: Path) -> dict:
    if key not in data:
        return {}
    if not isinstance(data[key], dict):
        raise ConfigError("%s in %s must be an object" % (key, source))
    return data[key]


def _build_profile(name: str, data: dict) -> Profile:
    known = {f.name for f in fields(Profile)} - {"name"}
    unknown = set(data) - known
    if unknown:
        raise ConfigError("Provider %r has unknown fields: %s" % (name, ", ".join(sorted(unknown))))
    missing = [k for k in ("base_url", "api_key_env", "model") if k not in data]
    if missing:
        raise ConfigError("Provider %r is missing: %s" % (name, ", ".join(missing)))
    for key in ("base_url", "api_key_env", "model", "max_tokens_field"):
        if key in data and not isinstance(data[key], str):
            raise ConfigError("Provider %r: %s must be a string" % (name, key))
    if "max_output_tokens" in data and not (_is_int(data["max_output_tokens"]) and data["max_output_tokens"] > 0):
        raise ConfigError("Provider %r: max_output_tokens must be a positive integer" % name)
    if "extra_body" in data and not isinstance(data["extra_body"], dict):
        raise ConfigError("Provider %r: extra_body must be an object" % name)
    if data.get("temperature") is not None and not _is_number(data["temperature"]):
        raise ConfigError("Provider %r: temperature must be a number or null" % name)
    return Profile(name=name, **data)


def load_settings(defaults_path: Optional[Path] = None, user_path: Optional[Path] = None) -> Settings:
    defaults_path = defaults_path or DEFAULTS_PATH
    user_path = user_path or (config_dir() / "config.json")
    defaults = _read_json(defaults_path, required=True)
    user = _read_json(user_path, required=False)

    raw: Dict[str, dict] = {}
    limit_values: dict = {}
    for source, data in ((defaults_path, defaults), (user_path, user)):
        for name, patch in _section(data, "providers", source).items():
            _merge_provider(raw, name, patch, source)
        limit_values.update(_section(data, "limits", source))
    profiles = {name: _build_profile(name, data) for name, data in raw.items()}

    limit_names = {f.name for f in fields(Limits)}
    for key, value in limit_values.items():
        if key in limit_names and not (_is_int(value) and value >= 1):
            raise ConfigError("Invalid limits: %s must be an integer of at least 1, got %r" % (key, value))
    try:
        limits = Limits(**limit_values)
    except TypeError as exc:
        raise ConfigError("Invalid limits: %s" % exc)

    default_provider = user.get("default_provider") or defaults.get("default_provider")
    if not isinstance(default_provider, str) or default_provider not in profiles:
        raise ConfigError("default_provider %r is not a known provider" % (default_provider,))

    wait = user.get("wait_seconds", defaults.get("wait_seconds", DEFAULT_WAIT_SECONDS))
    if not _is_number(wait) or not wait > 0:
        raise ConfigError("wait_seconds must be a positive number of seconds, got %r" % (wait,))
    wait = min(float(wait), MAX_WAIT_SECONDS)

    pass_env = user.get("pass_env", defaults.get("pass_env", []))
    if not isinstance(pass_env, list) or not all(isinstance(n, str) for n in pass_env):
        raise ConfigError("pass_env must be a list of environment variable names, got %r" % (pass_env,))
    return Settings(default_provider, profiles, limits, wait, list(pass_env))


def clamp_limits(base: Limits, max_steps: Optional[int], max_seconds: Optional[int]) -> Limits:
    steps = base.max_steps if max_steps is None else max_steps
    seconds = base.max_seconds if max_seconds is None else max_seconds
    return Limits(
        max_steps=max(HARD_MIN_STEPS, min(int(steps), HARD_MAX_STEPS)),
        max_seconds=max(HARD_MIN_SECONDS, min(int(seconds), HARD_MAX_SECONDS)),
        max_tokens_total=base.max_tokens_total,
    )


def _parse_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8-sig")  # tolerate a BOM from Windows editors
    except UnicodeDecodeError:
        raise ConfigError("cannot read %s: not valid UTF-8" % path)
    except OSError:
        return values
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def resolve_api_key(
    profile: Profile,
    environ: Optional[Mapping[str, str]] = None,
    keys_file: Optional[Path] = None,
) -> str:
    environ = os.environ if environ is None else environ
    keys_file = keys_file or (config_dir() / "keys.env")
    name = profile.api_key_env
    value = (environ.get(name) or "").strip()
    if not value:
        value = _parse_env_file(keys_file).get(name, "").strip()
    if not value:
        raise ConfigError(
            "No API key for provider %r. Set %s in the environment or add the line "
            "%s=<key> to %s" % (profile.name, name, name, keys_file)
        )
    return value
