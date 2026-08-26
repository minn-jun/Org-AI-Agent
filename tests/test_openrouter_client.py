from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

from org_agent_mvp.config import AppConfig
from org_agent_mvp.openrouter_client import OpenRouterClient


class FakeResponse:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


class OpenRouterClientTests(unittest.TestCase):
    def test_raises_runtime_error_for_api_error_payload(self) -> None:
        config = replace(AppConfig.load(), api_key="test-key")
        client = OpenRouterClient(config)

        with patch(
            "urllib.request.urlopen",
            return_value=FakeResponse(b'{"error":{"message":"model unavailable"}}'),
        ):
            with self.assertRaisesRegex(RuntimeError, "OpenRouter API error"):
                client.chat([{"role": "user", "content": "hello"}], [])

    def test_raises_runtime_error_when_choices_missing(self) -> None:
        config = replace(AppConfig.load(), api_key="test-key")
        client = OpenRouterClient(config)

        with patch("urllib.request.urlopen", return_value=FakeResponse(b'{"id":"x"}')):
            with self.assertRaisesRegex(RuntimeError, "missing choices"):
                client.chat([{"role": "user", "content": "hello"}], [])

    def test_returns_message_from_valid_response(self) -> None:
        config = replace(AppConfig.load(), api_key="test-key")
        client = OpenRouterClient(config)
        body = (
            b'{"choices":[{"message":{"role":"assistant","content":"ok"}}]}'
        )

        with patch("urllib.request.urlopen", return_value=FakeResponse(body)):
            self.assertEqual(
                client.chat([{"role": "user", "content": "hello"}], []),
                {"role": "assistant", "content": "ok"},
            )


if __name__ == "__main__":
    unittest.main()
