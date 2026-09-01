# 조직지식 에이전트 동작 흐름

갱신일: 2026-08-27

이 문서는 현재 `org_agent_mvp`가 사용자 질문 한 턴을 어떻게 처리하고, 세션 안에서 다음 질문으로 맥락을 어떻게 넘기는지 설명한다.

## 현재 시스템 한 줄 요약

사용자 질문을 바로 답변 LLM에 보내지 않고, 먼저 가벼운 Query Analyzer LLM이 질문 의도와 메모리 필요성을 분석한 뒤 STM, MTM, LTM에서 후보 근거를 미리 가져오고, 그 근거를 바탕으로 Agent LLM이 답변 또는 추가 도구 호출을 선택하는 구조다.

단, “아까 말한거 요약해줘”처럼 최근 세션 대화 자체만 묻는 질문은 메모리 검색을 생략하고 세션 캐시만으로 답변한다.

## 전체 흐름

```text
사용자 질문 입력
│
├─ 세션 로드
│  └─ 이전 대화 요약, 이전 출처, 이전 tool call 요약 확인
│
├─ Query Analyzer
│  └─ 작은 LLM이 질문 의도, 답변 출처, 검색 필요 여부, STM/MTM/LTM 비율, 과제 필터 결정
│
├─ Prefetch
│  └─ memory_needed=true이면 STM/MTM/LTM에서 후보 근거 검색
│
├─ Rerank
│  └─ 키워드 점수 + tier 비율 보정으로 top-k 근거 선정
│
├─ Runtime Context 구성
│  └─ 최근 세션 요약 + prefetch 근거 + query plan을 LLM 입력에 추가
│
├─ LLM 호출
│  ├─ 근거 충분: 최종 답변 생성
│  └─ 근거 부족: retrieve_memory tool call 요청
│
├─ Tool Call 반복
│  └─ 선택된 tier만 추가 검색하고 결과를 다시 LLM에 전달
│
└─ 결과 저장
   ├─ 최종 답변과 출처 반환
   ├─ 세션 캐시에 최근 턴 저장
   └─ --save-log 사용 시 상세 턴 로그 저장
```

## 세션 단위 대화

현재 시스템은 매 질문마다 새로운 독립 실행을 하지 않고, 세션 파일을 통해 최근 대화 흐름을 이어간다.

세션에는 원문 전체가 아니라 최근 턴의 압축 정보만 저장된다.

```text
logs/sessions/session-YYYYMMDD-HHMMSS-xxxxxx.json
```

저장되는 핵심 정보는 다음과 같다.

- 사용자 질문
- 답변 요약
- 사용한 출처 ID
- query intent
- query analysis 결과
- prefetch 요약
- reasoning decision 요약
- tool call 요약

세션 파일에는 **모든 턴이 누적된다.** 오래된 턴을 지우지 않는다.
컨텍스트에 넣는 것은 그중 최근 구간뿐이라, 대화가 길어져도 컨텍스트는 커지지 않는다.
저장과 주입이 분리되어 있다.

## Query Analyzer 역할

`Query Analyzer`는 질문을 보고 먼저 다음을 결정한다.

- 바로 답변 가능한 질문인지
- 최근 세션 캐시만 보면 되는 질문인지
- 조직 메모리 검색이 필요한 질문인지
- STM, MTM, LTM 중 어느 쪽 비중을 높일지
- `A과제`, `A 과제`, `A-과제`처럼 흔들리는 과제명을 어떤 표준 이름으로 볼지
- 후속 질문일 경우 이전 턴의 프로젝트 맥락을 이어받을지

현재 실제 OpenRouter 실행에서는 `QUERY_ANALYZER_MODEL`에 지정된 작은 LLM이 JSON 형태의 `QueryPlan`을 만든다. 모델 호출 실패, JSON 파싱 실패, 스키마 흔들림이 있으면 규칙 기반 baseline으로 fallback한다. `--mock` 실행에서는 API 없이 흐름을 확인하기 위해 규칙 기반 분석기를 사용한다.

답변을 생성하는 ReAct 본체는 `AGENT_MODEL`을 사용한다. 그래서 쿼리 분석과 최종 답변/도구 호출 판단을 서로 다른 모델로 분리할 수 있다.

현재 `answer_source`는 세 가지로 나뉜다.

| answer_source | 처리 방향 | 예시 |
|---|---|---|
| `direct_answer` | 조직 메모리 검색 없이 답변 | 일반 개념 설명 |
| `session_only` | 최근 세션 턴 요약만으로 답변 | 아까 말한거 요약 |
| `memory_prefetch` | 메모리 후보 검색 후 답변 | 일정, 담당자, 문서 근거 확인 |

```env
QUERY_ANALYZER_MODEL=liquid/lfm-2.5-2.6b:free
AGENT_MODEL=nvidia/nemotron-3-super-120b-a12b:free
```

## 메모리별 의미

| Tier | 의미 | 현재 구현 |
|---|---|---|
| STM | 최신 대화, 당일 결정, 최근 세션성 정보 | `memory_seed/stm` 파일 검색 |
| MTM | 진행 중 회의록, 제안서 초안, 최근 보고서 | `memory_seed/mtm` 파일 검색 |
| LTM | 공식 계획서, 조직 기준, 최종 문서 | `memory_seed/ltm` 파일 검색 |

현재는 실제 벡터 DB가 아니라 seed 폴더 기반 검색이다. 다만 런타임에서는 `retrieve_memory`라는 공통 인터페이스만 바라보므로, 나중에 내부를 vector RAG나 hybrid RAG로 교체할 수 있다.

## Prefetch와 Tool Call의 관계

prefetch는 첫 LLM 호출 전에 “답변 후보 근거”를 미리 모으는 단계다.

Query Analyzer가 `session_only` 또는 `direct_answer`로 판단하면 prefetch는 0건으로 끝난다. 이 경우 `[RUNTIME_CONTEXT]`에는 최근 세션 요약만 들어가며, 모델이 세션 요약만으로 답변한다.

예를 들어 비교 질문이면 STM, MTM, LTM을 모두 어느 정도 검색하고, 공식 기준 질문이면 LTM 비중을 크게 둔다. 이렇게 뽑힌 근거는 `[RUNTIME_CONTEXT]`로 LLM에 들어간다.

LLM이 이 근거만으로 답할 수 있으면 바로 최종 답변을 만든다. 부족하면 `retrieve_memory` tool call을 요청한다.

중요한 점은 tool call이 발생했다고 해서 모든 메모리를 다시 검색하지 않는다는 것이다. 모델이 필요한 tier와 query를 지정하면 그 범위만 추가 검색한다.

## 점수와 Top-K

현재 검색 점수는 키워드 기반이다.

- query token이 문서 본문에 있으면 가산
- 제목에 있으면 추가 가산
- 프로젝트명에 있으면 추가 가산
- `오늘`, `아까`, `최근` 계열 질문이면 문서 날짜 기준으로 최신 문서에 가산

**계층 자체에 주는 보정은 여기에 없다.** 계층 선호는 prefetch의 prior 한 곳에서만 적용한다.
예전에는 계층 신호가 세 곳(쿼리 주입 / 점수 보정 / prior)에서 중복 적용되고 있었다.

prefetch 단계는 계층 비율을 **자리 수가 아니라 점수 배수로** 쓴다.

```text
norm  = retrieval_score / 후보 중 최고점
final = norm * (1 + alpha * tier_weight)
컷    = 0.3 * 최고 final점수
```

가산이 아니라 승산이라, 관련성이 0인 문서는 계층과 무관하게 0으로 남는다.

계층별로 미리 자르지 않고 넓게 모은 뒤 한 번에 순위를 매긴다.
따라서 정답이 한 계층에 몰려 있어도 전부 선택될 수 있고,
쓸 문서가 적으면 8개를 채우지 않고 2~3개만 넣는다.

## 로그와 관찰성

`--verbose`를 사용하면 터미널에서 다음 흐름이 보인다.

```text
[session]     세션 ID와 이전 턴 수
[turn]        사용자 질문
[analyzer]    의도, 검색 필요 여부, tier 비율
[prefetch]    사전 검색 배분과 후보 또는 검색 생략
[context]     LLM 입력 컨텍스트 크기
[llm]         모델 호출 시작과 종료
[reasoning]   final_answer 또는 tool_call 결정 요약
[tool]        retrieve_memory 실행과 결과
[final]       최종 답변 준비
```

`--save-log`를 사용하면 한 턴의 전체 실행 로그가 JSON으로 저장된다.

```text
logs/turns/YYYYMMDD-HHMMSS-xxxxxxxx.json
```

이 로그에는 숨겨진 chain-of-thought가 아니라, 실제 응답과 tool call에서 관찰 가능한 결정 요약이 저장된다.

세션 전용 질문의 verbose 예시는 다음과 같다.

```text
[analyzer] intent=session_context_answer answer_source=session_only memory_needed=False
[prefetch] 메모리 검색 생략 (세션/직접 답변 흐름)
[context] 최근 턴 1개 + 근거 0건
[reasoning #1] decision=final_answer
```

## 현재 구현의 한계

- 실제 vector RAG 미연결
- seed 폴더 기반 키워드 검색
- Query Analyzer는 LLM 기반이지만 실패 시 규칙 기반 fallback 사용
- STM 일일 요약 생성 미구현
- STM에서 MTM, MTM에서 LTM으로 승격하는 관리 시스템 미구현
- 권한 제어는 `permission_scope` 메타데이터 수준

## 다음 개발 방향

1. `MemoryStore.retrieve()` 내부를 BM25 또는 vector 검색으로 교체
2. prefetch 결과와 tool call 결과를 평가할 수 있는 테스트 질문 세트 확장
3. Query Analyzer LLM과 Agent LLM 조합별 품질/비용/속도 비교
4. 세션 종료 또는 하루 종료 시 STM 요약 생성
5. 중요도, 조회수, 반복 참조, 관리자 승인 기반 승격 후보 시스템 설계
6. 권한과 문서 최신성 기준을 검색 점수에 반영
