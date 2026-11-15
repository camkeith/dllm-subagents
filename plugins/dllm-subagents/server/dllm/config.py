from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, Mapping, Optional

HARD_MAX_STEPS = 150
HARD_MAX_SECONDS = 1800
MAX_WAIT_SECONDS = 55

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
    wait_seconds: float = 50.0


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


def _build_profile(name: str, data: dict) -> Profile:
    known = {f.name for f in fields(Profile)} - {"name"}
    unknown = set(data) - known
    if unknown:
        raise ConfigError("Provider %r has unknown fields: %s" % (name, ", ".join(sorted(unknown))))
    missing = [k for k in ("base_url", "api_key_env", "model") if k not in data]
    if missing:
        raise ConfigError("Provider %r is missing: %s" % (name, ", ".join(missing)))
    return Profile(name=name, **data)


def load_settings(defaults_path: Optional[Path] = None, user_path: Optional[Path] = None) -> Settings:
    defaults_path = defaults_path or DEFAULTS_PATH
    user_path = user_path or (config_dir() / "config.json")
    defaults = _read_json(defaults_path, required=True)
    user = _read_json(user_path, required=False)

    raw: Dict[str, dict] = {}
    for source, data in ((defaults_path, defaults), (user_path, user)):
        for name, patch in (data.get("providers") or {}).items():
            _merge_provider(raw, name, patch, source)
    profiles = {name: _build_profile(name, data) for name, data in raw.items()}

    limit_values: dict = {}
    limit_values.update(defaults.get("limits") or {})
    limit_values.update(user.get("limits") or {})
    try:
        limits = Limits(**limit_values)
    except TypeError as exc:
        raise ConfigError("Invalid limits: %s" % exc)

    default_provider = user.get("default_provider") or defaults.get("default_provider")
    if default_provider not in profiles:
        raise ConfigError("default_provider %r is not a known provider" % (default_provider,))

    wait = user.get("wait_seconds", defaults.get("wait_seconds", 50))
    wait = min(float(wait), MAX_WAIT_SECONDS)
    return Settings(default_provider, profiles, limits, wait)


def clamp_limits(base: Limits, max_steps: Optional[int], max_seconds: Optional[int]) -> Limits:
    steps = base.max_steps if max_steps is None else max_steps
    seconds = base.max_seconds if max_seconds is None else max_seconds
    return Limits(
        max_steps=max(1, min(int(steps), HARD_MAX_STEPS)),
        max_seconds=max(10, min(int(seconds), HARD_MAX_SECONDS)),
        max_tokens_total=base.max_tokens_total,
    )


def _parse_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
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
    value = environ.get(name)
    if not value:
        value = _parse_env_file(keys_file).get(name)
    if not value:
        raise ConfigError(
            "No API key for provider %r. Set %s in the environment or add the line "
            "%s=<key> to %s" % (profile.name, name, name, keys_file)
        )
    return value
