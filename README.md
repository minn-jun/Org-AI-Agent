# Org AI Agent MVP

조직의 최신 대화, 진행 문서, 공식 기준을 근거로 답하는 세션형 에이전트 프로토타입이다.
Hermes의 ReAct 반복 구조와 prefetch 개념을 참고했으며, 현재는 실제 벡터 DB 대신
STM / MTM / LTM seed 폴더를 RAG 검색기처럼 사용한다.

## Current Flow

```text
사용자 질문
→ 세션의 최근 대화 요약 로드
→ LLM 기반 Query Analyzer
→ answer_source와 STM / MTM / LTM 검색 비율 결정
→ 필요 시 비율 기반 prefetch + 점수 재정렬 + top-k
→ 현재 턴의 컨텍스트 구성
→ OpenRouter 또는 Mock LLM 호출
→ 필요 시 retrieve_memory tool call
→ tool result를 포함해 LLM 재호출
→ 최종 답변 + 출처
→ 세션 캐시와 선택적 턴 로그 저장
```

Query Analyzer는 실제 OpenRouter 실행에서는 가벼운 분석 모델을 호출해 `QueryPlan`
JSON을 만들고, 실패 시 규칙 기반 baseline으로 fallback한다. `--mock` 실행에서는
API 없이 흐름을 재현하기 위해 규칙 기반 분석기를 사용한다.
“아까 말한거 요약”처럼 세션 대화만 필요한 질문은 `session_only`로 분기되어
문서 검색 없이 최근 세션 요약만으로 답변한다.
검색기는 keyword top-k 방식이며 `MemoryStore.retrieve()` 경계를 유지하므로 이후
BM25, vector RAG, hybrid RAG 구현으로 교체할 수 있다.

## Quick Start

```powershell
cd C:\ine_project\중기청_조직지식AI플랫폼\org_agent_mvp
python -m org_agent_mvp --mock --verbose --trace --save-log --question "A 과제의 최근 내부 목표일이 공식 계획과 충돌해?"
```

질문을 이어가는 CLI 세션:

```powershell
python -m org_agent_mvp --mock --verbose
```

대화 중 `exit` 또는 `quit`로 종료하고 `/new`로 새 세션을 시작한다. 기존 세션을
다시 열 때는 `--session-id session-...`를 사용한다.

OpenRouter 사용 시 `.env`의 빈 키 입력란을 채우고 `--mock`을 제거한다.

```env
OPENROUTER_API_KEY=
QUERY_ANALYZER_MODEL=liquid/lfm-2.5-2.6b:free
AGENT_MODEL=nvidia/nemotron-3-super-120b-a12b:free
PREFETCH_TOP_K=8
SESSION_CACHE_TURNS=8
```

## Terminal Events

`--verbose` 실행 시 다음 단계가 순서대로 보인다.

```text
[session]     세션 ID와 이전 턴 수
[analyzer]    질문 의도, 검색 필요 여부, tier 비율
[prefetch]    tier별 후보 수와 rerank 점수 또는 검색 생략
[context]     이번 호출에 포함한 최근 턴과 근거 수
[llm]         모델 호출 시작과 종료
[reasoning]   final_answer 또는 tool_call 판단 요약
[tool]        검색 tier, query, top-k, 결과 출처
[final]       답변 준비 완료
```

`reasoning`은 모델의 숨겨진 사고 과정이 아니라 실제 응답과 tool call에서 확인할 수
있는 판단 결과를 요약한 값이다.

## Session And Logs

- `logs/sessions/*.json`: 최근 8턴의 질문, 답변 요약, 출처 ID를 보관하는 세션 캐시
- `logs/turns/*.json`: `--save-log` 사용 시 한 턴의 분석, prefetch, LLM, tool 실행 기록
- prefetched 문서 원문은 세션에 누적하지 않고 매 질문마다 다시 선별

`logs/`, `.env`, `docs/`는 Git 추적에서 제외한다.

## Memory Tiers

| Tier | 역할 | 예시 |
|---|---|---|
| STM | 최신 대화와 당일 업무 흐름 | 대화 요약, 통화 메모, action item |
| MTM | 진행 중인 프로젝트 지식 | 회의록, 제안서 초안, 일정표, 분석 보고 |
| LTM | 승인된 조직 기준 지식 | 최종 계획서, 규정, 작성 기준 |

seed 데이터의 기준과 연결된 질문은 `memory_seed/README.md`에 정리했다.

## Project Structure

```text
org_agent_mvp/
├─ org_agent_mvp/
│  ├─ __main__.py
│  ├─ agent_runtime.py
│  ├─ query_analyzer.py
│  ├─ prefetch.py
│  ├─ context_builder.py
│  ├─ session_store.py
│  ├─ memory_store.py
│  ├─ mock_llm.py
│  └─ openrouter_client.py
├─ memory_seed/
│  ├─ stm/
│  ├─ mtm/
│  └─ ltm/
├─ tests/
└─ .env.example
```

## Test

```powershell
python -m unittest discover -s tests -v
```

현재 비교 질문의 혼합 계획, 후속 질문의 프로젝트 복원, 비율 기반 prefetch,
mock ReAct 실행과 동일 세션 이어가기를 자동 검증한다.

## Deferred Work

- keyword 검색을 BM25 / vector / hybrid RAG로 교체
- Query Analyzer의 작은 LLM, 규칙 기반 fallback, encoder 방식 비교
- 장기 세션 요약과 일일 STM 적재 작업
- 메모리 승격 후보 선정 및 관리자 검토 시스템
