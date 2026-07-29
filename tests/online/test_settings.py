"""Tests for explicit and security-sensitive online configuration."""

from __future__ import annotations

import unittest

from bettercallagent.settings import ExecutionMode, OnlineSettings, SettingsError
from tests.online.helpers import FIXTURE_PATH


class OnlineSettingsTests(unittest.TestCase):
    def test_environment_loader_requires_explicit_mode_and_asset(self) -> None:
        with self.assertRaisesRegex(SettingsError, "BCA_ONLINE_MODE"):
            OnlineSettings.from_environment({})

    def test_wildcard_and_path_cors_origins_are_rejected(self) -> None:
        for origin in ("*", "https://review.example/path"):
            with self.subTest(origin=origin), self.assertRaises(SettingsError):
                OnlineSettings(
                    mode=ExecutionMode.FIXTURE,
                    asset_path=FIXTURE_PATH,
                    allowed_models=("fixture-reviewer",),
                    default_model="fixture-reviewer",
                    cors_origins=(origin,),
                )

    def test_live_mode_requires_tls_and_api_key(self) -> None:
        with self.assertRaisesRegex(SettingsError, "HTTPS"):
            OnlineSettings(
                mode=ExecutionMode.LIVE,
                asset_path=FIXTURE_PATH,
                allowed_models=("review-model",),
                default_model="review-model",
                cors_origins=("https://review.example",),
                llm_base_url="http://provider.example/v1",
                llm_api_key="runtime-secret",
            )
