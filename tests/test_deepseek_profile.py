from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from kb_agent import config


class DeepSeekProfileTest(unittest.TestCase):
    def setUp(self) -> None:
        self._loaded = config._ENV_LOADED
        config._ENV_LOADED = True

    def tearDown(self) -> None:
        config._ENV_LOADED = self._loaded

    def test_default_profile_keeps_existing_deepseek_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "DEEPSEEK_API_KEY": "default-key",
                "DEEPSEEK_BASE_URL": "http://default.example/v1",
                "DEEPSEEK_MODEL": "deepseek_v4",
            },
            clear=True,
        ):
            self.assertEqual(config.deepseek_profile(), "default")
            self.assertEqual(config.deepseek_api_key(), "default-key")
            self.assertEqual(config.deepseek_base_url(), "http://default.example/v1")
            self.assertEqual(config.deepseek_model(), "deepseek_v4")

    def test_profile_reads_profiled_deepseek_env_first(self) -> None:
        with patch.dict(
            os.environ,
            {
                "DEEPSEEK_PROFILE": "pro",
                "DEEPSEEK_API_KEY": "default-key",
                "DEEPSEEK_BASE_URL": "http://default.example/v1",
                "DEEPSEEK_MODEL": "deepseek_v4",
                "DEEPSEEK_PRO_API_KEY": "pro-key",
                "DEEPSEEK_PRO_BASE_URL": "http://pro.example/v1",
                "DEEPSEEK_PRO_MODEL": "deepseek-v4-pro",
            },
            clear=True,
        ):
            self.assertEqual(config.deepseek_profile(), "pro")
            self.assertEqual(config.deepseek_api_key(), "pro-key")
            self.assertEqual(config.deepseek_base_url(), "http://pro.example/v1")
            self.assertEqual(config.deepseek_model(), "deepseek-v4-pro")


if __name__ == "__main__":
    unittest.main()
