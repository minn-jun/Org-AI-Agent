from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .config import AppConfig


class OpenRouterClient:
    def __init__(self, config: AppConfig):
        if not config.api_key:
            raise ValueError("OPENROUTER_API_KEY is empty. Fill .env or use --mock.")
        self.config = config

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.2,
        }
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
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenRouter request failed: {exc}") from exc
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
        return message

