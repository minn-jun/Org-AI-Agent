# Query Analyzer 강화 개발 구도

갱신일: 2026-08-30

## 배경

현재 `Org-AI-Agent`는 `memory_seed` 기반 키워드 검색을 사용하지만, 사용자가 체감하는 답변 품질은 검색 엔진 자체보다 Query Analyzer가 먼저 결정한다.

Query Analyzer는 사용자 질문을 보고 다음을 정한다.

- 조직 메모리 검색이 필요한 질문인지
- 최근 세션 맥락만으로 답해도 되는지
- STM, MTM, LTM 중 어떤 메모리 비중을 높일지
- 후속 질문에서 이전 프로젝트 맥락을 이어받을지
- 검색 query를 어떻게 재작성할지
- 어떤 필터를 걸어 검색 범위를 좁힐지

따라서 RAG 고도화는 후순위로 두고, 먼저 Query Analyzer의 분류 정확도와 검색 전략 품질을 높이는 방향이 합리적이다.

## 현재 상태 요약

현재 구현은 두 층으로 되어 있다.

- `LLMQueryAnalyzer`: OpenRouter의 작은 모델로 `QueryPlan` JSON 생성
- `RuleBasedQueryAnalyzer`: LLM 실패, JSON 파싱 실패, 스키마 흔들림에 대비한 fallback

현재 `QueryPlan`은 아래 정보를 포함한다.

- `intent`
- `can_answer_directly`
- `memory_needed`
- `answer_source`
- `use_session_context`
- `memory_weights`
- `query_rewrites`
- `filters`
- `reason`

이 구조는 유지하는 편이 좋다. 아직 RAG를 붙이지 않아도 `MemoryStore.retrieve()`와의 계약이 안정적이고, 나중에 BM25/vector/hybrid RAG로 바꿀 때도 Query Analyzer 결과를 그대로 사용할 수 있다.

## 강화 방향

### 1. 평가셋 먼저 만들기

Query Analyzer는 기능 구현보다 평가 기준이 먼저 필요하다. 지금은 테스트가 몇 가지 대표 케이스를 확인하지만, 실제 업무 질문을 충분히 덮지는 못한다.

우선 `tests/fixtures/query_analyzer_cases.jsonl` 같은 평가 파일을 만들고, 각 줄에 다음을 저장한다.

```json
{
  "query": "A 과제 예산 산정 근거가 부족한 부분은?",
  "recent_turns": [],
  "expected": {
    "intent": "working_document_lookup",
    "memory_needed": true,
    "answer_source": "memory_prefetch",
    "required_filters": {
      "project": "A 과제"
    },
    "required_tiers": ["mtm", "ltm"]
  }
}
```

처음에는 30개 정도면 충분하다.

- 일반 개념 질문
- 세션 요약 질문
- 후속 질문
- 일정/담당자 질문
- 예산 질문
- 공식 기준 질문
- 최근 결정과 공식 계획 비교 질문
- 프로젝트명이 생략된 질문
- 프로젝트명이 흔들리는 질문
- 근거/출처 요구 질문

목표는 LLM이 좋은 말을 하게 만드는 것이 아니라, `QueryPlan`이 기대한 방향으로 안정적으로 나오는지 측정하는 것이다.

### 2. 업무 의도 taxonomy 정리

현재 intent는 큰 범주 중심이다.

- `direct_answer`
- `session_context_answer`
- `recent_context_lookup`
- `working_document_lookup`
- `official_knowledge_lookup`
- `organization_memory_lookup`
- `memory_comparison`

당장 intent 종류를 많이 늘리기보다는, 보조 분류를 `filters`나 별도 필드 후보로 정리하는 편이 낫다.

후보 업무 신호:

- 일정: 마감, 내부 검토일, 최종 제출일, 마일스톤
- 담당자: 누가 맡았는지, 남은 action item
- 예산: 단가, 자문비, 운영비, 산정 근거
- 리스크: 지연, 누락, 충돌, 미해결 쟁점
- 문서: 회의록, 제안서, 보고서, 최종 계획서
- 권위: 초안, 검토 완료, 승인, 최종
- 시간: 오늘, 어제, 최근, 이번 주, 이번 달

이 taxonomy를 먼저 문서화하면, Query Analyzer 프롬프트와 rule fallback을 같은 기준으로 맞출 수 있다.

### 3. LLM 분석기 프롬프트 강화

현재 LLM 분석기는 JSON schema에 맞춰 답하도록 되어 있다. 다음 단계에서는 프롬프트에 판단 규칙을 더 명확히 넣는 것이 좋다.

강화할 규칙:

- 일정, 담당자, 예산, 공식 기준, 출처 요청은 일반 질문으로 보지 않는다.
- “아까 말한 거 요약”은 `session_only`지만, “아까 말한 일정의 담당자”는 검색이 필요하다.
- 비교/충돌 질문은 최근 근거와 공식 기준을 함께 보도록 한다.
- 프로젝트명이 현재 질문에 없으면 최근 세션에서 상속한다.
- LLM이 확신이 없으면 `direct_answer`보다 `memory_prefetch`를 선택한다.
- query rewrite에는 원 질문, 세션 맥락, 핵심 업무 키워드를 모두 반영한다.

이 단계에서는 코드 구조를 크게 바꾸지 않고 `_call_llm()`에 들어가는 규칙과 예시를 보강하면 된다.

### 4. RuleBased fallback을 안전망으로 강화

fallback은 단순 백업이 아니라 운영 안정성의 안전망이다. LLM이 실패하거나 이상한 JSON을 내도 업무 질문을 놓치지 않아야 한다.

보강 후보:

- 담당자/후속 조치 신호 감지
- 예산/비용/산정 근거 신호 감지
- 리스크/쟁점/지연 신호 감지
- 공식/승인/최종 기준 신호 감지
- 날짜 표현 감지
- 문서 유형 힌트 생성

단, 이 단계에서 검색 엔진까지 깊게 바꾸지는 않는다. Query Analyzer가 더 좋은 `filters`와 `query_rewrites`를 만들도록 하고, 검색 쪽은 기존 계약 안에서 필요한 최소만 받게 한다.

### 5. LLM 결과 정규화와 방어 로직

작은 모델은 가끔 스키마는 맞추지만 의미를 잘못 낼 수 있다.

예시:

- 예산 질문인데 `direct_answer`
- 공식 기준 질문인데 LTM 비중 0
- 후속 질문인데 세션 맥락 미사용
- project 필터 누락
- query rewrite가 너무 짧거나 원 질문 누락

따라서 `_normalize_plan()`에서 fallback 계획을 기준으로 최소 안전선을 둘 필요가 있다.

권장 방어:

- fallback이 `memory_needed=true`인데 LLM만 `direct_answer`면 `memory_prefetch` 유지
- fallback에 project가 있으면 LLM 결과에 누락되어도 보존
- memory weight 합이 0이면 fallback weight 사용
- query rewrite 첫 항목에는 항상 원 질문 포함
- 허용되지 않는 intent, answer_source는 fallback 값 사용

### 6. 관찰성과 디버깅 강화

Query Analyzer는 “왜 그렇게 판단했는지”를 개발자가 빠르게 봐야 한다.

추가하면 좋은 로그:

- analyzer 모델명
- fallback 사용 여부
- fallback plan과 LLM plan의 차이
- 최종 normalize 이후 plan
- 선택된 query rewrite
- 선택된 filters
- tier weight 분포

이 로그는 사용자 최종 답변에는 노출하지 않고, `--verbose`와 `--save-log`에서만 확인하면 된다.

## 권장 개발 일정

### 1주차: 평가 기준 만들기

- 실제 데모 질문 30개 수집
- intent, answer_source, memory weight 기대값 정의
- 후속 질문용 recent_turns 케이스 작성
- Query Analyzer 평가 테스트 추가

산출물:

- `tests/fixtures/query_analyzer_cases.jsonl`
- Query Analyzer 회귀 테스트
- 현재 analyzer 기준 baseline 점수

### 2주차: LLM 분석기 프롬프트 개선

- 업무 의도 taxonomy 정리
- LLM Query Analyzer 프롬프트에 규칙과 few-shot 예시 추가
- LLM 출력과 fallback 출력 비교 로그 추가
- 작은 모델 후보 2-3개 비교

산출물:

- 개선된 analyzer prompt
- 모델별 정확도/비용/속도 비교표
- 실패 사례 목록

### 3주차: fallback과 정규화 강화

- 담당자, 예산, 리스크, 공식 기준, 날짜 표현 규칙 보강
- `_normalize_plan()` 방어 로직 강화
- project 상속 실패 케이스 보강
- query rewrite 품질 테스트 추가

산출물:

- 안정화된 `RuleBasedQueryAnalyzer`
- LLM 오판 방어 테스트
- 후속 질문 회귀 테스트

### 4주차: 검색 전략 연결 고도화

- document type, status, date range 필터 정책 설계
- prefetch weight와 tool call 전략 조정
- 너무 많은 출처가 붙는 문제 정리
- 데모 질문 기준 답변 근거 수 조절

산출물:

- 검색 전략 정책표
- prefetch/tool call 조정안
- 데모 질문별 trace 샘플

### 5주차 이후: RAG 고도화 착수

- BM25/vector/hybrid RAG는 이 시점부터 검토
- 단, Query Analyzer의 `QueryPlan` 계약은 유지
- `MemoryStore.retrieve()` 내부만 교체하는 방식으로 진행

## 우선순위 결론

지금은 RAG 검색 품질보다 Query Analyzer의 판단 품질을 먼저 끌어올리는 것이 좋다.

특히 발표나 피드백 대응 관점에서는 다음 메시지가 설득력 있다.

> 현재는 RAG 엔진을 성급히 바꾸기보다, 조직 업무 질문을 어떤 의도와 근거 전략으로 해석할지 안정화하는 단계입니다. Query Analyzer 평가셋, 업무 의도 분류, fallback 안전망을 먼저 만들고, 그 다음 `MemoryStore.retrieve()` 내부를 BM25나 vector RAG로 교체할 계획입니다.

