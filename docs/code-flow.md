# 코드 기준 동작 흐름

갱신일: 2026-09-01

이 문서는 `org_agent_mvp` 코드가 어떤 책임 분리로 동작하는지 설명한다.
함수 내부의 모든 분기보다, 나중에 수정할 때 어느 파일을 보면 되는지에 초점을 둔다.

수치의 값과 근거는 [`parameters-and-rationale.md`](parameters-and-rationale.md)에 따로 있다.

---

## 호출 흐름 요약

```text
org_agent_mvp/__main__.py
│
├─ main()
│  ├─ CLI 인자 파싱 (--mock, --context-mode, --session-id, --save-log)
│  ├─ AppConfig.load()
│  ├─ build_runtime()
│  └─ ask_once() 또는 interactive()
│
└─ AgentRuntime.run()
   ├─ SessionStore.load_or_create()   전체 기록 로드
   ├─ SessionStore.recent()           컨텍스트용으로 최근 N턴만 잘라냄
   ├─ QueryAnalyzer.analyze()
   ├─ MemoryPrefetcher.prefetch()
   ├─ ContextBuilder.build()
   ├─ [반복] client.chat()
   │     ├─ _execute_tool_call()        retrieve_memory
   │     └─ _execute_expand_evidence()  expand_evidence
   ├─ _save_session_turn()
   └─ _write_turn_log()
```

---

## 주요 파일 역할

| 파일 | 핵심 역할 |
|---|---|
| `__main__.py` | CLI 진입점, 대화형 모드, verbose 출력 |
| `config.py` | `.env`와 기본 설정 로드 |
| `agent_runtime.py` | 한 턴의 전체 실행 흐름, 도구 실행, 토큰 집계 |
| `query_analyzer.py` | 질문 의도 분석, 계층 가중치, 필터 추출, LLM 방어 |
| `prefetch.py` | 근거 선별 (수집, 정규화, 계층 prior, 상대 컷) |
| `context_builder.py` | `[RUNTIME_CONTEXT]` 조립, 컨텍스트 모드 3종 |
| `memory_store.py` | seed 로드, 필터링, 점수 계산, 근거 카드 생성 |
| `session_store.py` | 세션 파일 생성, 로드, 턴 누적 |
| `normalization.py` | 과제명 표기 정규화 |
| `openrouter_client.py` | OpenRouter API 호출, `usage` 수집, 오류 처리 |
| `mock_llm.py` | API 없이 흐름을 재현하는 mock |
| `schemas.py` | `retrieve_memory`, `expand_evidence` 도구 스키마 |
| `prompts.py` | 시스템 프롬프트 |

---

## 1. CLI 진입

`__main__.py`의 `main()`이 시작점이다.

| 인자 | 동작 |
|---|---|
| `--question` | 한 번만 질문. 없으면 대화형 세션 |
| `--mock` | `MockLLMClient` + `RuleBasedQueryAnalyzer` |
| (없음) | `OpenRouterClient` + `LLMQueryAnalyzer` |
| `--context-mode` | `full` / `summary` / `hybrid` (기본은 `.env`) |
| `--verbose` | 실행 이벤트 출력 |
| `--save-log` | 턴 로그 저장 |

`build_runtime()`이 config, 클라이언트, memory store를 묶어 `AgentRuntime`을 만든다.
이때 **컨텍스트 모드에 따라 도구 목록이 달라진다.**

```python
self.tools = [RETRIEVE_MEMORY_TOOL]
if config.context_mode != "full":
    self.tools.append(EXPAND_EVIDENCE_TOOL)
```

`full`은 원문이 이미 전부 들어가므로 확장할 것이 없다.

---

## 2. 세션 로드

```python
session = self.session_store.load_or_create(session_id)
recent_turns = self.session_store.recent(session)   # 최근 N턴만
```

**저장과 주입이 분리되어 있다.**

| | 동작 |
|---|---|
| 세션 파일 | **전체 턴 누적.** 오래된 턴을 지우지 않는다 |
| `recent()` | 컨텍스트 경로에 넘길 최근 구간만 잘라 반환 |

이전에는 `append_turn()`이 최근 N턴만 남기고 잘라내서, 그 이전 대화가
복구 불가능하게 사라졌다. 지금은 저장본이 온전하다.

---

## 3. 질문 분석

`QueryAnalyzer.analyze()`가 `QueryPlan`을 만든다.

```python
QueryPlan(
    intent, can_answer_directly, memory_needed, answer_source,
    use_session_context, memory_weights, query_rewrites, filters, reason,
)
```

### 두 구현

| 구현 | 사용 시점 | 특징 |
|---|---|---|
| `RuleBasedQueryAnalyzer` | `--mock`, 그리고 항상 fallback으로 | 정규식과 키워드. 결정적, 무료 |
| `LLMQueryAnalyzer` | 실모델 | 규칙 결과를 방어 기준으로 삼아 LLM 출력을 정규화 |

`LLMQueryAnalyzer`는 항상 규칙 기반을 먼저 돌린 뒤 LLM을 호출하고,
`_normalize_plan()`에서 다음을 방어한다.

| 방어 | 조건 | 처리 |
|---|---|---|
| 검색 생략 오판 | 규칙은 검색 필요, LLM만 불필요 | `memory_prefetch`로 복원 |
| 과제명 오염 | 규칙이 과제명을 추출함 | 규칙 값 우선 |
| 스키마 밖 필터 | 허용 키 5개 외 | 제거 |
| 가중치 합 0 | `memory_needed=True`인데 전부 0 | fallback 가중치 |
| 예외 / API 오류 | 파싱 실패, 429 등 | 규칙 기반 전체 사용 + `fallback_used=True` |

### query_rewrites에 계층 신호를 넣지 않는다

이전에는 rewrite에 "공식 최종 승인 문서" 같은 단어를 덧붙였다.
그러면 제목에 그 단어가 있는 문서가 **주제 관련성과 무관하게** 상위로 올라온다.
계층 선호는 `memory_weights`(prior) 한 곳에서만 적용한다.

---

## 4. 근거 선별 (`prefetch.py`)

```python
priors = plan.memory_weights

# 1. 계층별로 자르지 않고 넓게 수집
for tier in ("stm", "mtm", "ltm"):
    retrieve(tier, query, filters, top_k=POOL_PER_TIER)

# 2. evidence_id 기준 중복 제거

# 3. 전역 정규화 (계층마다 점수 스케일이 다르다)
norm = raw / max_raw

# 4. 계층 prior를 곱한다 (더하지 않는다)
final = norm * (1 + alpha * tier_prior)

# 5. 상대 임계값으로 자른다
cut = CUT_RATIO * top_score
kept = [c for c in ranked if c.final >= cut]

# 6. 계층별 최소 보장 후 상한 적용
cards = _apply_tier_floor(kept, priors)
```

### 설계 판단

| 항목 | 선택 | 이유 |
|---|---|---|
| 정규화 | 최고점 나누기 | 계층 간 점수 스케일 차이를 없앤다 |
| 결합 | 승산 | 관련성 0인 문서는 계층과 무관하게 0으로 남는다 |
| 임계값 | 상대값 | 절대값은 코퍼스가 바뀌면 의미가 달라진다 |
| 계층 보장 | 컷 통과분 중에서만 | 옛 쿼터처럼 무관한 문서를 끌어오지 않는다 |

`_apply_tier_floor()`에서 **예약분을 앞에 두고 자른 뒤에 정렬**한다.
자르기 전에 전역 정렬하면 예약이 무효가 된다.

---

## 5. 컨텍스트 조립 (`context_builder.py`)

```json
{
  "query_plan":            { ... },
  "context_mode":          "full",
  "recent_session_turns":  [ ... ],
  "prefetched_evidence":   [ ... ]
}
```

`includes_body(index)`가 근거별로 원문 포함 여부를 정한다.

| 모드 | 판정 |
|---|---|
| `full` | 항상 포함 |
| `hybrid` | 상위 `HYBRID_FULL_CARDS`건만 |
| `summary` | 포함하지 않고 `body_available: true` 표시 |

모드에 따라 guidance 문구도 달라진다. `summary`/`hybrid`에서는
"요약으로 판단 가능하면 그대로 답하고, 필요한 근거만 `expand_evidence`로 요청하라"고 지시한다.

---

## 6. LLM 호출 루프

```python
for _ in range(max_tool_calls + 1):     # 최대 4회
    chat_response = client.chat(messages, tools, ...)
    assistant_message = chat_response["message"]
    trace.tokens.add("agent", chat_response["usage"])

    if not tool_calls:
        return 최종 답변

    for tool_call in tool_calls:
        _execute_tool_call(...)          # retrieve_memory 또는 expand_evidence
```

`chat()`은 `{"message": ..., "usage": ...}`를 돌려준다.
`usage`에는 `prompt_tokens`, `completion_tokens`, `total_tokens`, `estimated`가 있다.
mock은 문자 수 기반 근사값이라 `estimated: true`로 표시한다.

### 두 도구의 차이

| | `retrieve_memory` | `expand_evidence` |
|---|---|---|
| 하는 일 | 새로 검색 | 턴 안의 카드 조회 |
| 비용 | 있음 | 없음 (dict 조회) |
| 노출 조건 | 항상 | `context_mode != "full"` |

`evidence_index`는 프리페치 카드를 `{evidence_id: card}`로 턴 동안 들고 있는 것이다.
확장은 이 색인만 본다. 알 수 없는 id는 `unknown_evidence_ids`로 보고한다.

---

## 7. 토큰 집계

```python
@dataclass
class TokenUsage:
    analyzer: dict      # prompt / completion / total
    agent: dict
    analyzer_calls: int
    agent_calls: int
    estimated: bool
```

analyzer와 agent를 나눠 센다. `llm_calls`는 둘의 합이다.
규칙 기반 analyzer는 LLM을 호출하지 않으므로 `analyzer_calls = 0`이 된다.

---

## 8. 저장

| 대상 | 내용 |
|---|---|
| `logs/sessions/*.json` | 전체 턴 누적. `turn_count` 포함 |
| `logs/turns/*.json` | `--save-log` 시. 이벤트, 추론 단계, 도구 실행, 전체 대화 |

---

## 성능 주의점

`MemoryStore.retrieve()`는 쿼리 확장과 토큰화를 **한 번만** 수행한 뒤
문서 루프에 넘긴다(`_prepare_query`).

이전에는 `_score()` 안에서 문서마다 다시 계산해, 후보 수만큼 낭비했다.
문서 115건에서 `_expand_query` 호출이 43회였고 지금은 3회(계층 수)다.
실코퍼스로 가면 이 차이가 그대로 비례한다.
