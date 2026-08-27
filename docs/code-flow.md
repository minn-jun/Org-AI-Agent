# 코드 기준 동작 흐름

갱신일: 2026-08-27

이 문서는 현재 `org_agent_mvp` 코드가 어떤 책임 분리로 동작하는지 설명한다. 함수 내부의 모든 분기보다, 나중에 수정할 때 어느 파일을 보면 되는지에 초점을 둔다.

## 호출 흐름 요약

```text
org_agent_mvp/__main__.py
│
├─ main()
│  ├─ CLI 인자 파싱
│  ├─ AppConfig.load()
│  ├─ build_runtime()
│  └─ ask_once() 또는 interactive()
│
└─ AgentRuntime.run()
   ├─ SessionStore.load_or_create()
   ├─ LLMQueryAnalyzer.analyze() 또는 RuleBasedQueryAnalyzer.analyze()
   ├─ MemoryPrefetcher.prefetch()
   ├─ ContextBuilder.build()
   ├─ LLM client.chat()
   ├─ _execute_tool_call() 반복
   ├─ _save_session_turn()
   └─ _write_turn_log()
```

## 주요 파일 역할

| 파일 | 핵심 역할 |
|---|---|
| `org_agent_mvp/__main__.py` | CLI 진입점, 대화형 모드, verbose 출력 |
| `org_agent_mvp/config.py` | `.env`와 기본 설정 로드 |
| `org_agent_mvp/agent_runtime.py` | 한 턴의 전체 실행 흐름 제어 |
| `org_agent_mvp/query_analyzer.py` | LLM 기반 질문 의도 분석, tier 비율, 검색 필터 결정, fallback |
| `org_agent_mvp/prefetch.py` | LLM 호출 전 메모리 후보 사전 검색 |
| `org_agent_mvp/context_builder.py` | 세션 요약과 근거를 `[RUNTIME_CONTEXT]`로 압축 |
| `org_agent_mvp/memory_store.py` | seed memory 로드, 필터링, 점수 계산, evidence card 생성 |
| `org_agent_mvp/normalization.py` | 과제명 표기 정규화 |
| `org_agent_mvp/session_store.py` | 세션 파일 생성, 로드, 최근 턴 저장 |
| `org_agent_mvp/openrouter_client.py` | OpenRouter API 호출과 오류 처리 |
| `org_agent_mvp/mock_llm.py` | API 없이 흐름을 재현하는 mock LLM |
| `org_agent_mvp/schemas.py` | `retrieve_memory` tool schema |
| `org_agent_mvp/prompts.py` | 시스템 프롬프트 |

## 한 턴 실행 상세

### 1. CLI 진입

`__main__.py`의 `main()`이 실행 시작점이다.

- `--question`이 있으면 한 번만 질문
- `--question`이 없으면 대화형 세션 시작
- `--mock`이면 `MockLLMClient` 사용
- `--mock`이 없으면 `OpenRouterClient` 사용
- `--verbose`면 실행 이벤트를 터미널에 출력
- `--save-log`면 상세 턴 로그 저장

`build_runtime()`은 config, LLM client, memory store를 묶어 `AgentRuntime`을 만든다.

### 2. 세션 로드

`AgentRuntime.run()`은 먼저 `SessionStore.load_or_create()`를 호출한다.

기존 `--session-id`가 있으면 해당 세션을 읽고, 없으면 새 세션을 만든다. 세션에는 최근 질문과 답변 요약만 저장되며 전체 문서 원문은 누적하지 않는다.

### 3. 질문 분석

실제 OpenRouter 실행에서는 `LLMQueryAnalyzer.analyze()`가 사용자 질문과 최근 세션 턴을 받아 작은 모델에 분석을 요청한다. `--mock` 실행에서는 API 없이 재현할 수 있도록 `RuleBasedQueryAnalyzer.analyze()`를 사용한다.

`QueryPlan`에는 다음 값이 들어간다.

- `intent`
- `can_answer_directly`
- `memory_needed`
- `answer_source`
- `use_session_context`
- `memory_weights`
- `query_rewrites`
- `filters`
- `reason`

LLM 분석기는 JSON schema 기반 응답을 요청하고, 응답을 `QueryPlan`으로 정규화한다. 모델 호출 실패나 JSON 파싱 실패가 생기면 규칙 기반 분석 결과로 fallback한다.

`answer_source`는 이번 질문의 첫 처리 방향을 나타낸다.

| 값 | 의미 |
|---|---|
| `direct_answer` | 조직 문서 검색 없이 일반 답변 가능 |
| `session_only` | 최근 세션 대화 요약만으로 답변 가능 |
| `memory_prefetch` | 조직 메모리 검색 필요 |

예를 들어 “아까 말한거 요약해줘”는 `session_context_answer`, `answer_source=session_only`, `memory_needed=false`로 처리된다. 반면 “아까 말한 일정의 담당자는?”처럼 실제 업무 사실이나 근거가 필요한 질문은 세션을 참고하되 `memory_prefetch`로 이동한다.

여기서 `filters.project`는 `normalize_project_name()`을 거친다. 그래서 `A과제`, `A 과제`, `A-과제`를 같은 프로젝트로 다룰 수 있다.

### 4. Prefetch

`MemoryPrefetcher.prefetch()`는 `QueryPlan.memory_weights`를 보고 전체 top-k를 STM, MTM, LTM에 나눠 배분한다.

`memory_needed=false`인 `direct_answer` 또는 `session_only` 흐름에서는 prefetch가 실행되지 않고 빈 결과를 반환한다. 이때 런타임은 최근 세션 턴만 `[RUNTIME_CONTEXT]`에 포함한다.

각 tier별 검색은 `MemoryStore.retrieve()`로 실행된다. 검색 결과에는 `retrieval_score`가 있고, prefetch 단계에서 `tier_weight`를 더해 `rerank_score`를 만든다.

중복 evidence는 `evidence_id` 기준으로 정리하고, 최종 top-k만 LLM 컨텍스트로 넘긴다.

배분된 top-k는 “최대 검색 슬롯”이다. 해당 tier에서 점수가 0보다 큰 문서가 부족하거나 중복 evidence가 있으면 실제 컨텍스트 후보 수는 배분 합계보다 적을 수 있다.

### 5. Runtime Context 구성

`ContextBuilder.build()`는 다음 정보를 하나의 system message로 만든다.

- query plan
- 최근 세션 턴 최대 4개
- prefetch evidence

결과는 `[RUNTIME_CONTEXT] ... [/RUNTIME_CONTEXT]` 형태다. 이 컨텍스트는 이번 LLM 호출을 위한 선별 자료이며, 이전 턴의 모든 원문을 계속 쌓는 구조가 아니다.

### 6. LLM 호출과 ReAct 루프

`AgentRuntime.run()`은 다음 message 묶음으로 LLM을 호출한다.

```text
system: SYSTEM_PROMPT
system: RUNTIME_CONTEXT
user: 사용자 질문
```

LLM 응답에 `tool_calls`가 없으면 최종 답변으로 종료한다.

`tool_calls`가 있으면 `_execute_tool_call()`이 실행된다. 현재 지원하는 도구는 `retrieve_memory` 하나다.

```json
{
  "tier": "stm | mtm | ltm | all",
  "query": "검색 문장",
  "filters": {
    "project": "A 과제"
  },
  "top_k": 5,
  "reason": "검색 이유"
}
```

tool 결과는 다시 message에 추가되고 LLM을 재호출한다. 한 턴에서 tool call은 `MAX_TOOL_CALLS` 설정값까지만 허용된다.

Query Analyzer LLM은 `QUERY_ANALYZER_MODEL`, ReAct 본체 LLM은 `AGENT_MODEL`을 사용한다.

### 7. 결과 저장

최종 답변이 나오면 `_save_session_turn()`이 세션 파일에 이번 턴 요약을 저장한다.

`--save-log`가 켜져 있으면 `_write_turn_log()`가 상세 JSON 로그를 저장한다.

## MemoryStore 검색 구조

`MemoryStore`는 초기화 시 `memory_seed/stm`, `memory_seed/mtm`, `memory_seed/ltm` 바로 아래의 `.md`, `.json`, `.txt` 파일을 읽는다.

검색 흐름은 다음과 같다.

```text
retrieve()
├─ tier 선택
├─ project/source_type/status/document_types 필터 적용
├─ query 확장
├─ 키워드 점수 계산
├─ 최신성/공식성 보정
└─ evidence card 반환
```

evidence card는 LLM과 로그가 공통으로 사용하는 근거 단위다.

## OpenRouter 오류 처리

`OpenRouterClient.chat()`은 OpenRouter Chat Completions API를 호출한다.

다음 상황은 `RuntimeError`로 정리해서 CLI에 보여준다.

- HTTP 429 같은 OpenRouter HTTP 오류
- 네트워크 연결 오류
- API 응답 안의 `error`
- `choices`가 없는 비정상 응답
- message가 없는 비정상 응답

그래서 실제 모델 호출이 실패해도 터미널에서 원인을 비교적 바로 확인할 수 있다.

## 로그 구조

verbose 출력과 저장 로그는 모두 `AgentRuntime._emit()`을 통해 같은 이벤트 흐름을 따른다.

대표 이벤트는 다음과 같다.

- `turn_start`
- `query_analysis`
- `prefetch_start`
- `prefetch_end`
- `context_built`
- `llm_call_start`
- `llm_decision`
- `llm_call_end`
- `tool_call_start`
- `tool_call_end`
- `final_answer`

`reasoning_steps`는 숨겨진 사고 과정이 아니라 “이번 LLM 응답이 final인지 tool_call인지”를 관찰 가능한 형태로 요약한 기록이다.

세션 전용 질문에서는 verbose 출력에 `answer_source=session_only`와 `메모리 검색 생략`이 표시된다. 문서 근거가 필요한 질문에서는 기존처럼 prefetch 후보와 tool call 흐름이 이어진다.

## RAG로 교체할 위치

나중에 실제 RAG를 붙일 때 가장 먼저 바꿀 곳은 `MemoryStore.retrieve()` 내부다.

현재 런타임은 다음 반환 계약만 맞으면 그대로 동작할 수 있다.

```text
retrieve(tier, query, filters, top_k) -> evidence cards
```

따라서 `query_analyzer.py`, `prefetch.py`, `context_builder.py`, `agent_runtime.py`는 유지하고, `memory_store.py` 내부 검색만 BM25, vector DB, graph RAG, hybrid RAG로 바꾸는 방향이 가장 작게 시작할 수 있는 경로다.
