from __future__ import annotations

import http.client
import unittest
import urllib.error
from dataclasses import replace
from unittest.mock import patch

from org_agent_mvp.config import AppConfig
from org_agent_mvp.openrouter_client import OpenRouterClient

OK_BODY = b'{"choices":[{"message":{"role":"assistant","content":"ok"}}]}'


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

    def test_returns_message_and_usage_from_valid_response(self) -> None:
        config = replace(AppConfig.load(), api_key="test-key")
        client = OpenRouterClient(config)
        body = (
            b'{"choices":[{"message":{"role":"assistant","content":"ok"}}],'
            b'"usage":{"prompt_tokens":120,"completion_tokens":30,"total_tokens":150}}'
        )

        with patch("urllib.request.urlopen", return_value=FakeResponse(body)):
            self.assertEqual(
                client.chat([{"role": "user", "content": "hello"}], []),
                {
                    "message": {"role": "assistant", "content": "ok"},
                    "usage": {
                        "prompt_tokens": 120,
                        "completion_tokens": 30,
                        "total_tokens": 150,
                        "estimated": False,
                    },
                },
            )

    def test_usage_defaults_to_zero_when_missing(self) -> None:
        config = replace(AppConfig.load(), api_key="test-key")
        client = OpenRouterClient(config)
        body = b'{"choices":[{"message":{"role":"assistant","content":"ok"}}]}'

        with patch("urllib.request.urlopen", return_value=FakeResponse(body)):
            response = client.chat([{"role": "user", "content": "hello"}], [])

        self.assertEqual(response["message"]["content"], "ok")
        self.assertEqual(response["usage"]["total_tokens"], 0)
        self.assertFalse(response["usage"]["estimated"])


class NetworkFailureTests(unittest.TestCase):
    """연결이 끊겨도 호출부가 다룰 수 있는 형태로 올라와야 한다 (2026-09-16).

    예전에는 `RemoteDisconnected`가 그대로 터져 나가 에이전트 실행이 그 자리에서 죽었다.
    """

    def setUp(self) -> None:
        self.config = replace(AppConfig.load(), api_key="test-key")
        self.client = OpenRouterClient(self.config)
        sleep = patch("time.sleep")            # 재시도 대기는 건너뛴다
        sleep.start()
        self.addCleanup(sleep.stop)

    def test_remote_disconnect_becomes_runtime_error(self) -> None:
        error = http.client.RemoteDisconnected("Remote end closed connection")
        with patch.dict("os.environ", {"OPENROUTER_RETRIES": "0"}), \
                patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "RemoteDisconnected"):
                self.client.chat([{"role": "user", "content": "hello"}], [])

    def test_retries_until_the_connection_works(self) -> None:
        responses = [http.client.RemoteDisconnected("closed"), FakeResponse(OK_BODY)]
        with patch.dict("os.environ", {"OPENROUTER_RETRIES": "2"}), \
                patch("urllib.request.urlopen", side_effect=responses) as urlopen:
            result = self.client.chat([{"role": "user", "content": "hello"}], [])
        self.assertEqual(result["message"]["content"], "ok")
        self.assertEqual(urlopen.call_count, 2)

    def test_gives_up_after_the_configured_retries(self) -> None:
        error = urllib.error.URLError("EOF occurred in violation of protocol")
        with patch.dict("os.environ", {"OPENROUTER_RETRIES": "2"}), \
                patch("urllib.request.urlopen", side_effect=error) as urlopen:
            with self.assertRaisesRegex(RuntimeError, "OpenRouter request failed"):
                self.client.chat([{"role": "user", "content": "hello"}], [])
        self.assertEqual(urlopen.call_count, 3)   # 첫 시도 + 재시도 2회

    def test_client_error_is_not_retried(self) -> None:
        """4xx는 다시 보내도 같은 답이 온다. 재시도하면 돈만 더 쓴다."""
        error = urllib.error.HTTPError("url", 400, "Bad Request", {}, None)
        with patch.object(error, "read", return_value=b"bad request"), \
                patch.dict("os.environ", {"OPENROUTER_RETRIES": "2"}), \
                patch("urllib.request.urlopen", side_effect=error) as urlopen:
            with self.assertRaisesRegex(RuntimeError, "OpenRouter HTTP 400"):
                self.client.chat([{"role": "user", "content": "hello"}], [])
        self.assertEqual(urlopen.call_count, 1)

    def test_server_error_is_retried(self) -> None:
        error = urllib.error.HTTPError("url", 503, "Service Unavailable", {}, None)
        with patch.object(error, "read", return_value=b"unavailable"), \
                patch.dict("os.environ", {"OPENROUTER_RETRIES": "1"}), \
                patch("urllib.request.urlopen", side_effect=error) as urlopen:
            with self.assertRaisesRegex(RuntimeError, "OpenRouter HTTP 503"):
                self.client.chat([{"role": "user", "content": "hello"}], [])
        self.assertEqual(urlopen.call_count, 2)


if __name__ == "__main__":
    unittest.main()
