# 코드 기준 동작 흐름

갱신일: 2026-09-28

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
   ├─ ContextBuilder.build_context()   text + 실제 주입 목록
   ├─ [반복] client.chat()             상한 도달 시 tools=None
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
| `memory_store.py` | seed 로드, 필터링, 점수 계산, 근거 카드 생성, 원문 조회 |
| `ltm_corpus.py` | 실코퍼스 청크 역색인, 문서 단위 집계, 버전 접기, 원문 조회 |
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
| 과제명 오염 | LLM 값이 어휘 목록에 없음 | 규칙 값으로 물러남 |
| 스키마 밖 필터 | 허용 키 4개 외 | 제거 |
| 가중치 합 0 | `memory_needed=True`인데 전부 0 | fallback 가중치 |
| 예외 / API 오류 | 파싱 실패, 429 등 | 규칙 기반 전체 사용 + `fallback_used=True` |

**어휘 목록은 두 구현이 같이 쓴다** (2026-09-28). `LLMQueryAnalyzer`는 내부
fallback에도 같은 `vocabulary`를 넘긴다. 예전에는 빈 분석기를 만들어서,
`PROJECT_RE`의 오탐("이 과제", "총 사업")이 실제 실행 경로에서만 필터로 들어갔다.

`date_range`는 허용 키에서 뺐다. 검색기가 해석하지 않는데 로그에는 날짜 조건이
걸린 것처럼 남았다. 필요해지면 검색기에 구현한 뒤 다시 넣는다.

과제 필터 상속도 좁혔다. 지시어·비교·최근성 신호가 있는 질문에서만 이전 과제를
물려받는다. 예전에는 이전 턴이 있고 검색이 필요하면 전부 물려받아서, 주제가 바뀐
새 질문도 앞 과제로 치우쳤다.

### 문서 유형어가 없으면 검색을 건너뛴다 (2026-10-07 수정)

`memory_needed`는 `MEMORY_MARKERS` · 최근 신호 · 계층 신호 중 하나라도 걸려야 참이 된다.
하나도 없으면 "검색 쪽으로 기운다"는 안전망이 있는데, 그 앞에
`DIRECT_MARKERS`(`개념 · 뜻 · 일반적으로 …`)가 걸리면 **일반 개념 질문으로 분류되어
검색을 아예 하지 않는다.**

실제 과제 문서를 붙이고 라우팅을 재면서 이것이 드러났다 —
*"킥오프에서 말한 LTM 승격 **개념**이 계획서에도 적혀 있어?"*가 후보 0건이었다.
`MEMORY_MARKERS`에 `"문서"`는 있는데 **`"계획서"`가 없어** 신호가 0이었고,
그 틈에 `"개념"`이 걸렸다.

`MEMORY_MARKERS`에 실제 문서 유형어를 넣어 고쳤다 —
`계획서 · 협약 · 매뉴얼 · 보고 · 피드백 · 지표 · 특허 · 공고 · 세미나`.
회귀 테스트는 양방향으로 둔다(문서 질문은 검색하고, `"트랜스포머 어텐션 개념"`
같은 일반 개념 질문은 여전히 검색하지 않는다).

> 이 결함은 공개 평가셋으로는 드러나지 않는다. 거기엔 "계획서"도 "협약"도 없다.

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
#    delivered(앞선 턴에 본문을 준 문서)를 주면 상한을 **새 문서 기준**으로 센다
cards = _apply_tier_floor(kept, priors, delivered)
```

### 상한을 새 문서 기준으로 세는 이유 (2026-10-06)

세션 전달 원장(아래 5절)을 켜면 이미 본문을 준 문서가 **참조 한 줄**로 줄어든다.
그런데 상한(`total_top_k`)을 그대로 두면 그 참조가 본문 카드와 **같은 무게로** 자리를
차지한다. 턴이 쌓일수록 새 근거가 0장이 되고, 2턴째가 1턴째보다 정보가 적어진다.

`_cap()`은 `delivered`가 있을 때 **새 문서가 `total_top_k`장 모일 때까지** 순위를
내려가며, 참조는 그 사이에 따라붙는다. 참조가 무한히 붙지 않게 전체는 상한의 2배에서
끊는다. `delivered`가 비면 예전과 완전히 같다(`cards[:total_top_k]`).

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

### 무엇이 실제로 들어갔는지 돌려준다 (2026-09-28)

`build_context()`는 문자열이 아니라 `BuiltContext`를 돌려준다.

```python
BuiltContext(text, injected, dropped_evidence_ids, evidence_chars)
```

근거 블록이 `max_evidence_chars`를 넘으면 뒷순위 카드가 빠지는데, 예전에는
`build()`가 문자열만 돌려줘서 런타임이 그 사실을 알 수 없었다. 그래서 근거 개수와
`final_sources`에 **검색 후보 전체**를 적었다. Allganize 20문항 중 15건에서
후보 8건 중 6~7건만 들어갔고, 기록은 8건이었다. 그 상태로는
"근거를 줬는데 모델이 못 썼다"와 "근거가 안 들어갔다"를 구분할 수 없다.

`build()`는 문자열만 필요한 곳을 위해 남겨 뒀다.

### 참조 카드 — 앞선 턴에 준 문서 (2026-10-06)

`build_context(..., delivered=)`에 세션 전달 원장을 주면, 이미 본문을 전달한 문서는
**본문 없이** 들어간다.

```json
{ "evidence_id": "...", "title": "...", "source_id": "...",
  "body_available": true, "delivered_in_turn": 1,
  "note": "이전 턴에 본문을 전달한 문서" }
```

목록에서 아예 빼지는 않는다. 빼면 모델이 "그 문서는 없다"로 읽고 같은 검색을 다시
한다(2026-09-29에 도구 결과 5장이 전부 중복이라 빈 목록이 나간 q_180과 같은 모양).

기록도 나눈다. `BuiltContext.reference`는 `injected`와 **별도**이고
`injected_sources` · `final_sources`에 넣지 않는다. 같은 칸에 적으면
"이번 턴에 본문을 전달했다"로 읽힌다.

새로 볼 문서가 `SESSION_LEDGER_MIN_BODIES`보다 적으면 **상위 참조를 본문으로
되돌린다.** 이전 턴의 본문은 메시지에 남지 않으므로(턴마다 messages를 새로 만든다),
전부 참조로 바꾸면 모델이 받는 것은 제목과 `answer_summary` 500자뿐이다.

---

## 6. LLM 호출 루프

```python
for _ in range(max_tool_calls + 1):     # 최대 4회
    # 남은 횟수가 없으면 도구를 아예 넘기지 않는다
    tools_left = max_tool_calls - len(trace.tool_calls)
    chat_response = client.chat(messages, tools if tools_left > 0 else None, ...)
    trace.tokens.add("agent", chat_response["usage"])

    if not tool_calls:
        trace.cited_sources = _cited_sources(content, trace)
        return 최종 답변

    for tool_call in tool_calls:
        if len(trace.tool_calls) >= max_tool_calls:
            messages.append(_tool_limit_message(tool_call))   # 실행하지 않는다
            continue
        _execute_tool_call(...)          # retrieve_memory 또는 expand_evidence
```

### 도구 호출 상한은 실제 제한이다 (2026-09-28)

예전에는 상한에 도달해도 안내 메시지만 덧붙이고 다음 호출에 도구를 계속 넘겼고,
한 응답의 여러 도구 요청을 전부 실행했다. 상한 3회에 8회가 실행되는 것을
재현했다. 지금은 두 곳에서 막는다.

1. 남은 횟수가 0이면 `client.chat`에 `tools`를 넘기지 않는다.
2. 한 응답 안에서도 남은 횟수만 실행하고, 나머지는 실행하지 않은 것으로 응답한다.

`tool_call`마다 응답 메시지는 반드시 붙인다. 빠지면 다음 호출에서 대화 형식이 깨진다.

`chat()`은 `{"message": ..., "usage": ...}`를 돌려준다.
`usage`에는 `prompt_tokens`, `completion_tokens`, `total_tokens`, `estimated`가 있다.
mock은 문자 수 기반 근사값이라 `estimated: true`로 표시한다.

### 두 도구의 차이

| | `retrieve_memory` | `expand_evidence` |
|---|---|---|
| 하는 일 | 새로 검색 | 카드가 가리키는 **원문 조회** |
| 비용 | 있음 | 없음 (역색인 조회) |
| 노출 조건 | 항상 | `context_mode != "full"` |
| 결과 크기 | 도구 예산으로 자름 (아래) | `max_evidence_chars`까지 |

### 도구 결과는 1차 컨텍스트와 다른 예산을 쓴다 (2026-09-29)

`retrieve_memory`는 1차 컨텍스트와 같은 검색 경로를 타므로 카드가 같은 깊이로 만들어진다.
그런데 비용 성격이 다르다 — 1차 컨텍스트는 카드 8장이 턴에 한 번 실리고, 도구 결과는
10~20장이 **호출마다 누적 메시지로 다시 실린다.** 실측에서 1차 컨텍스트를 3배로 키웠을 때
도구 결과도 2.4배가 되어 입력 토큰의 46%를 차지했다.

`_trim_tool_result()`가 **검색 직후**에 카드를 도구용 예산으로 줄인다. 세 가지를 한다.

```python
requested_top_k = int(args.get("top_k") or config.default_top_k)
top_k = min(requested_top_k, config.tool_result_top_k_cap or requested_top_k)
# 중복을 뺄 거라면 빼고 나서도 top_k장이 남도록 넉넉히 받아 온다
fetch_k = max(top_k * 3, top_k + 5) if config.tool_result_dedupe else top_k
result = memory_store.retrieve(..., top_k=fetch_k)
result = self._trim_tool_result(result, trace, limit=top_k)
```

| 무엇 | 설정 |
|---|---|
| 이미 전달한 문서 제거 | `tool_result_dedupe` |
| 장수 제한 | `tool_result_top_k_cap` |
| 본문 길이 · 대목 수 | `tool_result_excerpt_chars` · `tool_result_chunks_per_doc` |

자르는 위치가 카드를 **만드는** 곳이 아니라 **쓰는** 곳이라는 점이 핵심이다. 검색 결과 객체는
그대로 두므로 prefetch가 같은 카드를 깊게 쓸 수 있고, 잘린 쪽이 로그에 남으므로 기록은
여전히 "실제로 전달한 것"이다.

### 중복 제거는 왜 넉넉히 받아 와야 하나

실측에서 도구 카드 96장 중 66장(69%)이 이미 전달한 문서였고, 한 문항은 5장 전부가 중복이었다.
중복만 지우고 끝내면 그 문항의 도구 결과가 **빈 목록**이 되고, 모델은 "찾아도 아무것도 없다"로
읽어 한 번 더 검색한다. 재검색은 컨텍스트 전체를 다시 전송하므로 아낀 것보다 더 쓸 수 있다.
그래서 `fetch_k`를 키워 받아 온 뒤 `limit`으로 자른다.

중복 판정 기준은 **`injected_sources` + 앞선 도구 호출의 `tool_sources`**다. 즉 "이 턴에 이미
모델에게 보낸 문서"다. 이 함수는 `trace.tool_sources`가 갱신되기 **전**에 불린다.

`evidence_index`는 프리페치 카드를 `{evidence_id: card}`로 턴 동안 들고 있는 것이다.
알 수 없는 id는 `unknown_evidence_ids`로 보고한다.

확장은 카드의 발췌가 아니라 `MemoryStore.chunk_context()`로 **청크 전체와 앞뒤
이웃 청크**를 읽는다(2026-09-28). 예전에는 카드에 이미 들어 있던 600자 발췌를 그대로
돌려줬다 — 원문을 요청해도 새로 얻는 글자가 없었다. 카드 발췌는 "이 근거가 무엇인지
알아보는 용도", 원문 조회는 "실제로 읽는 용도"로 나눠 뒀다. 이웃까지 붙이는 이유는
표나 문단이 청크 경계에서 잘려 값과 머리글이 갈라지기 때문이다.

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

## 7-1. 근거 기록 분리 (2026-09-28, 10-06 보강)

`AgentTrace`가 근거를 단계별로 나눠 적는다. 예전에는 `final_sources` 한 칸에
섞여 있어서 "찾았다 / 전달했다 / 썼다"를 구분할 수 없었다.

| 칸 | 뜻 |
|---|---|
| `retrieved_sources` | 검색이 찾아낸 후보 전체 |
| `injected_sources` | 1차 컨텍스트에 **실제로 들어간** 근거 |
| `tool_sources` | 실행 중 `retrieve_memory`로 추가된 근거 |
| `session_delivered_sources` | 앞선 턴에 본문을 준 문서 (전달 원장. 꺼져 있으면 빈 목록) |
| `reference_sources` | 그중 이번 턴에 **참조 카드로만** 들어간 문서 |
| `final_sources` | 모델이 받은 전체 (injected + tool). **reference는 넣지 않는다** |
| `cited_sources` | 답변 본문에 제목이나 출처가 나타난 근거 |
| `dropped_evidence_ids` | 글자 수 제한에 걸려 빠진 카드 |
| `skipped_tool_calls` | 상한 때문에 실행하지 않은 도구 요청 수 |

`cited_sources`는 **추정이다.** 모델이 출처를 구조화해서 돌려주지 않으므로,
제목(8자 이상)이나 `source_id`가 답변 글자에 있는지로 판단한다. 정확한 인용 기록은
모델이 출처를 따로 돌려주게 만들어야 하고, 그건 프롬프트·스키마 변경이 필요하다.

`context_built` 이벤트도 `candidate_count`와 `injected_count`를 나눠 적는다.
예전의 `evidence_count`는 candidate 값이었는데 이름만 보고 주입 건수로 읽혔다.

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
