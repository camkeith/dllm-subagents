from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dllm import config
from dllm.config import ConfigError, Limits, clamp_limits, load_settings, resolve_api_key


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def user_file(self, data):
        path = self.dir / "config.json"
        path.write_text(json.dumps(data))
        return path

    def test_shipped_profiles_exist(self):
        s = load_settings(user_path=self.dir / "missing.json")
        self.assertTrue({"mercury", "cerebras", "groq", "openrouter"} <= set(s.profiles))
        self.assertEqual(s.default_provider, "mercury")
        self.assertEqual(s.limits, Limits())
        self.assertEqual(s.wait_seconds, 50)

    def test_user_config_merges_fields(self):
        base = load_settings(user_path=self.dir / "missing.json")
        user = self.user_file({"providers": {
            "groq": {"model": "x"},
            "local": {"base_url": "http://localhost:1/v1", "api_key_env": "LOCAL_KEY", "model": "m"},
        }})
        s = load_settings(user_path=user)
        self.assertEqual(s.profiles["groq"].model, "x")
        self.assertEqual(s.profiles["groq"].base_url, base.profiles["groq"].base_url)
        self.assertEqual(s.profiles["local"].api_key_env, "LOCAL_KEY")
        self.assertEqual(s.profiles["local"].max_tokens_field, "max_completion_tokens")

    def test_limits_and_wait_override(self):
        user = self.user_file({"limits": {"max_steps": 10}, "wait_seconds": 90, "default_provider": "groq"})
        s = load_settings(user_path=user)
        self.assertEqual(s.limits.max_steps, 10)
        self.assertEqual(s.limits.max_seconds, 300)
        self.assertEqual(s.wait_seconds, 55)
        self.assertEqual(s.default_provider, "groq")

    def test_bad_types_raise_config_error(self):
        cases = [
            {"providers": []},
            {"providers": None},
            {"limits": [1]},
            {"limits": {"max_steps": "10"}},
            {"limits": {"max_steps": True}},
            {"limits": {"max_seconds": 0}},
            {"limits": {"max_tokens_total": 1.5}},
            {"wait_seconds": "abc"},
            {"wait_seconds": True},
            {"wait_seconds": None},
            {"wait_seconds": 0},
            {"wait_seconds": -5},
            {"default_provider": ["groq"]},
            {"providers": {"groq": {"model": 5}}},
            {"providers": {"groq": {"base_url": None}}},
            {"providers": {"groq": {"api_key_env": ["X"]}}},
            {"providers": {"groq": {"max_tokens_field": 1}}},
            {"providers": {"groq": {"max_output_tokens": "8192"}}},
            {"providers": {"groq": {"max_output_tokens": 0}}},
            {"providers": {"groq": {"max_output_tokens": True}}},
            {"providers": {"groq": {"extra_body": []}}},
            {"providers": {"groq": {"temperature": "hot"}}},
            {"providers": {"groq": {"temperature": True}}},
        ]
        for data in cases:
            with self.subTest(data=data):
                with self.assertRaises(ConfigError):
                    load_settings(user_path=self.user_file(data))

    def test_pass_env(self):
        self.assertEqual(load_settings(user_path=self.dir / "missing.json").pass_env, [])
        s = load_settings(user_path=self.user_file({"pass_env": ["NPM_TOKEN", "GH_TOKEN"]}))
        self.assertEqual(s.pass_env, ["NPM_TOKEN", "GH_TOKEN"])
        for bad in ("NPM_TOKEN", [1], {"a": 1}, None):
            with self.subTest(pass_env=bad):
                with self.assertRaises(ConfigError):
                    load_settings(user_path=self.user_file({"pass_env": bad}))

    def test_valid_optional_types_accepted(self):
        for temp in (None, 0, 0.5):
            s = load_settings(user_path=self.user_file({"providers": {"groq": {"temperature": temp}}}))
            self.assertEqual(s.profiles["groq"].temperature, temp)
        s = load_settings(user_path=self.user_file({"wait_seconds": 12.5, "limits": {"max_steps": 1}}))
        self.assertEqual((s.wait_seconds, s.limits.max_steps), (12.5, 1))

    def test_clamp_limits_caps(self):
        self.assertEqual(clamp_limits(Limits(), 500, 99999), Limits(150, 1800, 2_000_000))
        self.assertEqual(clamp_limits(Limits(), 0, 1), Limits(1, 10, 2_000_000))
        self.assertEqual(clamp_limits(Limits(), None, None), Limits())

    def test_codex_forwards_every_shipped_key_variable(self):
        codex = json.loads((config.DEFAULTS_PATH.parent.parent / "codex.mcp.json").read_text())
        env_vars = codex["mcpServers"]["fast-agent"]["env_vars"]
        shipped = json.loads(config.DEFAULTS_PATH.read_text())["providers"]
        self.assertEqual(sorted(env_vars), sorted({p["api_key_env"] for p in shipped.values()}))
        self.assertEqual(len(env_vars), len(set(env_vars)))

    def test_config_dir_env(self):
        import os
        old = os.environ.get("DLLM_SUBAGENTS_CONFIG_DIR")
        os.environ["DLLM_SUBAGENTS_CONFIG_DIR"] = str(self.dir)
        try:
            self.assertEqual(config.config_dir(), self.dir)
        finally:
            if old is None:
                del os.environ["DLLM_SUBAGENTS_CONFIG_DIR"]
            else:
                os.environ["DLLM_SUBAGENTS_CONFIG_DIR"] = old


class KeyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.keys = Path(self.tmp.name) / "keys.env"
        self.profile = config.Profile("groq", "http://x/v1", "GROQ_API_KEY", "m")

    def test_key_from_env_first(self):
        self.keys.write_text("GROQ_API_KEY=file\n")
        self.assertEqual(resolve_api_key(self.profile, {"GROQ_API_KEY": "env"}, self.keys), "env")

    def test_env_key_is_stripped(self):
        self.assertEqual(resolve_api_key(self.profile, {"GROQ_API_KEY": "env\n"}, self.keys), "env")

    def test_blank_env_key_falls_back_to_file(self):
        self.keys.write_text("GROQ_API_KEY=file\n")
        self.assertEqual(resolve_api_key(self.profile, {"GROQ_API_KEY": " \n"}, self.keys), "file")

    def test_key_from_keys_file(self):
        self.keys.write_text('# comment\n\nOTHER=zzz\nGROQ_API_KEY = "file"\n')
        self.assertEqual(resolve_api_key(self.profile, {}, self.keys), "file")

    def test_keys_file_with_bom(self):
        self.keys.write_bytes(b"\xef\xbb\xbfGROQ_API_KEY=file\n")
        self.assertEqual(resolve_api_key(self.profile, {}, self.keys), "file")

    def test_non_utf8_keys_file_is_config_error(self):
        self.keys.write_bytes(b"GROQ_API_KEY=caf\xe9\n")
        with self.assertRaises(ConfigError) as cm:
            resolve_api_key(self.profile, {}, self.keys)
        self.assertEqual(str(cm.exception), "cannot read %s: not valid UTF-8" % self.keys)

    def test_missing_key_message(self):
        self.keys.write_text("OTHER_KEY=secretvalue\n")
        with self.assertRaises(ConfigError) as cm:
            resolve_api_key(self.profile, {}, self.keys)
        msg = str(cm.exception)
        self.assertIn("GROQ_API_KEY", msg)
        self.assertIn(str(self.keys), msg)
        self.assertNotIn("secretvalue", msg)


if __name__ == "__main__":
    unittest.main()
