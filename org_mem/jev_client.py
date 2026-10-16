"""Jev (TypeSafe System One) 호출. 판정만 돌려주는 모델이다.

    JEV_MODEL=typesafe/jev-1.14      쓸 모델 (기본값)
    JEV_TIMEOUT=30                   한 호출 제한 시간(초)

## chat 클라이언트를 못 쓰는 이유

Jev는 **chat 모델이 아니다.** OpenRouter 모델 메타데이터가 그렇게 말한다.

    modality:          "text+image->decisions"
    output_modalities: ["decisions"]
    supported_parameters: []          temperature·tools·response_format 전부 없음

`/chat/completions`로 보내면 400이 온다. 실제 응답 문구다.

    typesafe/jev-1.14 is a decisions model and cannot be used with the
    chat/completions endpoint. Use the /api/alpha/decisions endpoint instead.

그래서 경로와 몸통이 다르다. `messages` 대신 `state`(판단할 상태)와
`questions`(물어볼 것)를 보낸다.

## 질문 종류

API가 받는 것은 네 가지다 (`type` 판별자에 넣는 값).

    noul    예/아니오. 0~1 확률을 돌려준다. confidence는 없다.
    choice  미리 준 선택지 중 하나. 고른 값 + 선택지별 확률 + confidence.
    score   서술한 단계별 기준에 따른 점수.
    box     구조화된 값 추출.

`noul`과 `choice`는 `instructions`가 필수고, `choice`는 `criteria`
(선택지 이름 -> 설명)도 필수다.

## 실측 (2026-10-16, 7분기 계층 판정)

    지연      0.48 ~ 0.65초
    입력      472 ~ 580 토큰
    비용      호출당 $0.0000198 ~ $0.0000244
    단가      입력 $0.042/M · 출력 무료

입력이 작은 이유는 **문서를 보내지 않기 때문**이다. 계층 판정은 검색
**전**이라 질문과 최근 대화만 들어간다.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from .config import AppConfig
from .openrouter_client import NETWORK_ERRORS, RETRY_STATUS, _retries


DEFAULT_MODEL = "typesafe/jev-1.14"
DEFAULT_TIMEOUT = 30

#: OpenRouter가 decisions 모델을 받는 경로. `/api/v1` 아래가 아니다.
DECISIONS_PATH = "/api/alpha/decisions"

#: API가 아는 질문 종류. 여기 없는 값을 보내면 400이 온다.
QUESTION_TYPES = frozenset({"noul", "choice", "score", "box"})


def model_name() -> str:
    return os.environ.get("JEV_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def timeout_seconds() -> int:
    try:
        return max(1, int(os.environ.get("JEV_TIMEOUT", DEFAULT_TIMEOUT)))
    except ValueError:
        return DEFAULT_TIMEOUT


def decisions_url(base_url: str) -> str:
    """`https://openrouter.ai/api/v1` -> `https://openrouter.ai/api/alpha/decisions`.

    설정의 base_url은 chat용(`/api/v1`)이라 그대로 쓸 수 없다.
    `/api/`까지만 떼어 내고 decisions 경로를 붙인다.
    """
    marker = "/api/"
    index = base_url.find(marker)
    root = base_url[:index] if index != -1 else base_url.rstrip("/")
    return f"{root}{DECISIONS_PATH}"


class JevClient:
    """`decide()` 하나만 있는 얇은 클라이언트.

    한 호출에 여러 질문을 함께 보낸다. 같은 `state`를 공유하므로 질문을
    나눠 보내는 것보다 입력 토큰이 적다. 질문끼리는 독립으로 판정되며,
    앞 질문의 답이 뒤 질문의 입력이 되지는 않는다.
    """

    def __init__(self, config: AppConfig, model: str | None = None):
        if not config.api_key:
            raise ValueError("OPENROUTER_API_KEY가 비어 있다. .env를 채우거나 --mock을 쓴다.")
        self.config = config
        self.model = model or model_name()
        self.url = decisions_url(config.base_url)

    def decide(
        self,
        state: Any,
        questions: dict[str, dict[str, Any]],
        *,
        model: str | None = None,
    ) -> dict[str, Any]:
        """`{"answers": {...}, "usage": {...}, "model": ..., "latency_s": ...}`를 돌려준다.

        `usage`는 이 저장소의 공통 모양(prompt/completion/total)으로 바꿔 담는다.
        Jev는 input_tokens/output_tokens로 보고하므로 그대로 쓰면
        `TokenUsage.add()`가 0으로 센다. 원본 값도 함께 남긴다.
        """
        if not questions:
            raise ValueError("questions가 비어 있다. 최소 한 개가 필요하다.")
        for name, question in questions.items():
            kind = str(question.get("type", ""))
            if kind not in QUESTION_TYPES:
                raise ValueError(
                    f"질문 '{name}'의 type이 {kind!r}이다. "
                    f"가능한 값: {sorted(QUESTION_TYPES)}"
                )

        payload = {
            "model": model or self.model,
            "state": state,
            "questions": questions,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.url,
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
        limit = timeout_seconds()
        started = time.monotonic()
        for attempt in range(retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=limit) as response:
                    data = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                # HTTPError는 URLError의 하위라 반드시 먼저 잡는다.
                if exc.code in RETRY_STATUS and attempt < retries:
                    time.sleep(2 ** attempt)
                    continue
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Jev HTTP {exc.code}: {detail}") from exc
            except NETWORK_ERRORS as exc:
                if attempt < retries:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(
                    f"Jev 호출 실패: {type(exc).__name__}: {exc}"
                ) from exc
        latency = time.monotonic() - started

        if "error" in data:
            detail = json.dumps(data["error"], ensure_ascii=False)
            raise RuntimeError(f"Jev API 오류: {detail}")
        answers = data.get("answers")
        if not isinstance(answers, dict) or not answers:
            preview = json.dumps(data, ensure_ascii=False)[:800]
            raise RuntimeError(f"Jev 응답에 answers가 없다: {preview}")
        missing = [name for name in questions if name not in answers]
        if missing:
            # 일부만 돌아오면 조용히 기본값으로 메우지 않는다. 호출부가 fallback을 고른다.
            raise RuntimeError(f"Jev 응답에 빠진 질문: {missing}")

        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        return {
            "model": str(data.get("model") or payload["model"]),
            "answers": answers,
            "latency_s": round(latency, 3),
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "estimated": False,
                # Jev는 호출당 비용을 직접 알려준다. 출력은 무료다.
                "cost_usd": float(usage.get("cost") or 0.0),
            },
        }


def choice_of(answer: dict[str, Any]) -> tuple[str, dict[str, float], float]:
    """`choice` 답에서 (고른 값, 선택지별 확률, confidence)를 꺼낸다.

    Jev가 argmax를 이미 `choice`로 돌려주므로 다시 고르지 않는다.
    확률은 분석·기록용으로 함께 남긴다.
    """
    picked = str(answer.get("choice") or "")
    raw = answer.get("probabilities")
    probabilities = (
        {str(k): float(v) for k, v in raw.items()} if isinstance(raw, dict) else {}
    )
    confidence = float(answer.get("confidence") or 0.0)
    return picked, probabilities, confidence


def noul_of(answer: dict[str, Any]) -> float:
    """`noul` 답의 확률. Jev는 noul에 confidence를 주지 않으므로 이 값을 쓴다."""
    return float(answer.get("noul") or 0.0)
