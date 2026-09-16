from __future__ import annotations

import http.client
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from .config import AppConfig

#: 연결이 끊길 때 올라오는 예외들. 2026-09-16에 두 가지를 실제로 만났다.
#:   urllib.error.URLError: SSL EOF          호출 중간에 TLS 연결이 끊긴다
#:   http.client.RemoteDisconnected          서버가 응답 없이 연결을 닫는다
#: 뒤의 것은 URLError가 아니라서 예전에는 감싸지지 않고 그대로 터져 나갔다.
#: (URLError는 OSError의 하위라 OSError 하나로도 잡히지만, 무엇을 의도했는지 남긴다.)
NETWORK_ERRORS = (urllib.error.URLError, http.client.HTTPException, OSError)

#: 다시 걸어 볼 만한 HTTP 상태. 4xx는 다시 보내도 같은 답이 오므로 제외한다.
RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})

DEFAULT_RETRIES = 2


def _retries() -> int:
    """재시도 횟수. OPENROUTER_RETRIES=0이면 한 번만 보낸다."""
    try:
        return max(0, int(os.environ.get("OPENROUTER_RETRIES", DEFAULT_RETRIES)))
    except ValueError:
        return DEFAULT_RETRIES


class OpenRouterClient:
    def __init__(self, config: AppConfig):
        if not config.api_key:
            raise ValueError("OPENROUTER_API_KEY is empty. Fill .env or use --mock.")
        self.config = config

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        model: str | None = None,
        temperature: float = 0.2,
        response_format: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        payload = {
            "model": model or self.config.agent_model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = int(max_tokens)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if response_format:
            payload["response_format"] = response_format
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.config.base_url}/chat/completions",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self.config.site_url,
                "X-Title": self.config.app_name,
            },
        )
        retries = _retries()
        for attempt in range(retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    data = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                # HTTPError는 URLError의 하위라 반드시 먼저 잡는다.
                if exc.code in RETRY_STATUS and attempt < retries:
                    time.sleep(2 ** attempt)
                    continue
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
            except NETWORK_ERRORS as exc:
                if attempt < retries:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(
                    f"OpenRouter request failed: {type(exc).__name__}: {exc}"
                ) from exc
        if "error" in data:
            detail = json.dumps(data["error"], ensure_ascii=False)
            raise RuntimeError(f"OpenRouter API error: {detail}")
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            preview = json.dumps(data, ensure_ascii=False)[:1000]
            raise RuntimeError(f"OpenRouter response missing choices: {preview}")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            preview = json.dumps(choices[0], ensure_ascii=False)[:1000]
            raise RuntimeError(f"OpenRouter response missing message: {preview}")
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        return {
            "message": message,
            "usage": {
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "total_tokens": int(usage.get("total_tokens") or 0),
                "estimated": False,
            },
        }

