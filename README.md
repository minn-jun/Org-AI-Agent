# Org AI Agent MVP

조직의 최신 대화, 진행 문서, 공식 기준을 근거로 답하는 세션형 에이전트 프로토타입이다.

**구조를 이해하기 위한 실험 환경이며 최종 시스템이 아니다.**
여기서 나온 수치는 이 실험 조건 안에서만 유효하다.

---

## 한 턴의 흐름

```text
사용자 질문
  │
  ├─ 세션 로드                최근 8턴을 컨텍스트 경로로 전달 (저장본은 전체 보관)
  │
  ├─ Query Analyzer           질문 → 검색 계획(QueryPlan)
  │    ├─ RuleBasedQueryAnalyzer   항상 먼저 실행. fallback이자 방어 기준
  │    └─ LLMQueryAnalyzer         실모델 실행 시. 출력은 방어 5종을 통과
  │
  ├─ Prefetch                 근거 선별
  │    1. 계층별로 자르지 않고 넓게 수집 (각 20건)
  │    2. 하나의 풀로 통합 + evidence_id 중복 제거
  │    3. 정규화       norm = raw / 후보최고점
  │    4. 계층 prior   final = norm x (1 + alpha x tier_weight)
  │    5. 상대 컷      final >= 0.3 x 최고점, 계층별 최소 보장, 최대 8건
  │
  ├─ ContextBuilder           [RUNTIME_CONTEXT] 조립 (full / summary / hybrid)
  │
  ├─ LLM 호출 루프 (최대 4회)
  │    ├─ retrieve_memory     새로 검색
  │    ├─ expand_evidence     이미 뽑은 근거의 원문 조회 (재검색 아님)
  │    └─ 최종 답변           근거 출처와 함께
  │
  └─ 저장                     세션 기록(전체 누적) + 턴 로그(--save-log)
```

**LLM 호출은 턴당 2~5회다.** analyzer 1회 + 에이전트 1~4회.
실측 분포는 대부분 2~3회에서 끝난다.

---

## Quick Start

```powershell
cd C:\ine_project\중기청_조직지식AI플랫폼\org_agent_mvp

# API 없이 흐름만 확인
python -m org_agent_mvp --mock --verbose --question "A 과제 예산 검토에서 뭐가 지적됐어?"

# 대화형 세션
python -m org_agent_mvp --mock --verbose
```

대화 중 `exit` 또는 `quit`로 종료하고, `--session-id session-...`으로 기존 세션을 이어간다.

실제 모델을 쓰려면 `.env`에 키를 넣고 `--mock`을 뺀다.

```env
OPENROUTER_API_KEY=
QUERY_ANALYZER_MODEL=openai/gpt-5.6-luna-pro
AGENT_MODEL=openai/gpt-5.6-luna-pro
```

---

## 근거 선별 방식

계층 가중치를 **검색 자리 수가 아니라 점수 배수로** 쓴다.

| | 이전 | 현재 |
|---|---|---|
| 계층 가중치의 역할 | 자리 수 (STM 5 / MTM 2 / LTM 1) | **점수 배수** |
| 자르는 시점 | 계층별로 미리 자름 | 전역 순위 후 한 번 |
| 점수 결합 | `점수 + 가중치 x 5.0` (가산) | `정규화점수 x (1 + a x 가중치)` (승산) |
| 근거 개수 | 항상 8건 고정 | **2~8건 가변** |

**가산에서 승산으로 바꾼 이유**: 가산이면 관련성이 0인 문서도 계층 보너스만으로
점수를 얻는다. 승산이면 `0 x 1.7 = 0`이라 걸러진다.

**계층별 최소 보장(`PREFETCH_TIER_FLOOR`)이 있는 이유**: 문서가 늘면 한 계층이
상위를 독점해 다른 계층의 정답이 밀려난다. 다만 **이미 컷을 통과한 문서 중에서만**
자리를 보장하므로, 옛 쿼터처럼 무관한 문서를 끌어오지 않는다.

파라미터의 값과 근거는 [`docs/parameters-and-rationale.md`](docs/parameters-and-rationale.md)에 정리했다.

---

## 컨텍스트 주입 방식

```powershell
python -m org_agent_mvp --context-mode summary --question "..."
```

| 모드 | 1차 컨텍스트 | 크기(실측) |
|---|---|---|
| `full` (기본) | 근거 원문 전량 | 5,770자 |
| `hybrid` | 상위 2건 원문 + 나머지 요약 | 4,523자 |
| `summary` | 요약만, 원문은 요청 시 | 3,836자 |

`summary`와 `hybrid`에서는 `expand_evidence` 도구가 노출된다.
모델이 요약만으로 판단이 어려우면 특정 근거의 원문을 요청한다.
이 도구는 **재검색이 아니라 턴 안에 이미 들고 있는 카드의 조회**라 비용이 없다.

어느 모드가 나은지는 **원문 확장률**을 재서 정한다. 확장이 잦으면 LLM 호출이
2회가 되어 절감이 사라지기 때문이다. 아직 측정 전이라 기본값은 `full`이다.

---

## 메모리 계층

| 계층 | 역할 | 예시 |
|---|---|---|
| STM | 최신 대화와 당일 업무 흐름 | 회의 요약, 통화 메모, action item |
| MTM | 진행 중인 프로젝트 지식 | 회의록, 제안서 초안, 일정표, 검토 보고 |
| LTM | 승인된 조직 기준 지식 | 최종 계획서, 규정, 작성 기준 |

seed 코퍼스는 **문서 115건**(STM 31 / MTM 59 / LTM 25), 과제 5개와 공통 기준 20건이다.

```powershell
python scripts/build_seed_memory.py            # 코퍼스 재생성
python scripts/build_seed_memory.py --dry-run  # 목록만 확인
```

코퍼스를 스크립트로 두는 이유는 **평가 수치가 코퍼스에 종속**되기 때문이다.
어떻게 만들어졌는지 추적할 수 없으면 수치도 해석할 수 없다.

---

## 평가

측정 축이 둘이고 비용이 다르다.

```powershell
# 근거 선별 품질 - LLM 호출 없음, 결정적, 무료
python eval/run_eval.py                     # 현재 방식
python eval/run_eval.py --mode quota        # 이전 쿼터 방식 baseline
python eval/run_eval.py --alpha 0           # 계층 무시 ablation
python eval/run_eval.py --tier-floor 2      # 계층 보장 조정
python eval/run_eval.py --analyzer llm      # 실제 analyzer 사용 (유료)

# 컨텍스트 주입 방식 - 실제 LLM 필요
python eval/run_context_eval.py --mock      # 흐름 확인
python eval/run_context_eval.py --limit 5   # 실모델
```

정답 라벨은 `tests/fixtures/eval_cases.jsonl`에 있다 (33건 / gold 37개).

`--analyzer rule`(기본)은 analyzer를 고정한 **통제 비교**다.
prefetch 로직만 바꿔가며 볼 때 쓴다. 실운영 수치가 아니다.

### 현재 측정값 (문서 115건 / 케이스 33건)

| 설정 | 정밀도@8 | 재현율 | F1 |
|---|---|---|---|
| quota (이전 방식) | 17.0% | 74.2% | 27.7% |
| prior alpha=0 | 16.8% | 72.6% | 27.3% |
| prior alpha=1 (현재) | 17.1% | 71.0% | 27.6% |
| prior alpha=1, floor=2 | 17.5% | 74.2% | 28.3% |

**모든 조건이 1%p 안쪽으로 붙어 있다.** 이 규모에서는 prefetch 전략이 병목이 아니고,
점수 함수(키워드 매칭)가 병목이라는 뜻이다.

---

## 세션과 로그

| 위치 | 내용 |
|---|---|
| `logs/sessions/*.json` | **전체 턴 누적.** 오래된 턴을 지우지 않는다 |
| `logs/turns/*.json` | `--save-log` 사용 시 한 턴의 분석, prefetch, LLM, 도구 실행 |

컨텍스트에는 최근 4턴만 붙는다(`max_recent_turns`). 저장과 주입은 분리되어 있어서,
20턴짜리 대화의 3번 턴은 파일에는 남아 있지만 컨텍스트로는 올라가지 않는다.
관련 턴을 골라 넣는 캐시 구조는 아직 없다.

---

## 터미널 이벤트

`--verbose` 실행 시 다음이 순서대로 보인다.

```text
[session]     세션 ID와 이전 턴 수
[analyzer]    질문 의도, 검색 필요 여부, 계층 가중치, fallback 여부
[prefetch]    수집 -> 중복제거 -> 컷 -> 선택. 문서별 raw/norm/prior/final 점수
[context]     이번 호출에 포함한 최근 턴과 근거 수
[llm]         호출 시작과 종료, 호출별 토큰
[reasoning]   final_answer 또는 tool_call 판단
[tool]        검색 tier, query, top-k, 결과 출처
[expand]      원문 요청과 반환 (summary/hybrid 모드)
[tokens]      턴 전체 토큰 (analyzer / agent 분리)
[final]       답변 준비 완료
```

`reasoning`은 모델의 숨겨진 사고 과정이 아니라, 실제 응답과 도구 호출에서
확인할 수 있는 판단 결과다.

---

## 프로젝트 구조

```text
org_agent_mvp/
├─ org_agent_mvp/
│  ├─ __main__.py           CLI, verbose 출력
│  ├─ agent_runtime.py      한 턴 실행 제어, 도구 실행, 토큰 집계
│  ├─ query_analyzer.py     의도 분석, 계층 가중치, 필터 추출, 방어 로직
│  ├─ prefetch.py           근거 선별 (정규화, 계층 prior, 상대 컷)
│  ├─ context_builder.py    [RUNTIME_CONTEXT] 조립 (full/summary/hybrid)
│  ├─ memory_store.py       문서 로드, 점수 계산, 근거 카드
│  ├─ session_store.py      세션 기록 (전체 누적)
│  ├─ openrouter_client.py  API 호출, usage 수집
│  ├─ mock_llm.py           API 없이 흐름 재현
│  ├─ schemas.py            retrieve_memory / expand_evidence 도구 정의
│  ├─ prompts.py            시스템 프롬프트
│  └─ normalization.py      과제명 표기 정규화
├─ memory_seed/{stm,mtm,ltm}/
├─ scripts/build_seed_memory.py
├─ eval/
│  ├─ run_eval.py           근거 선별 품질
│  ├─ run_context_eval.py   컨텍스트 주입 방식
│  └─ results/
├─ tests/
│  └─ fixtures/eval_cases.jsonl
└─ docs/
   ├─ parameters-and-rationale.md   모든 수치의 값과 근거
   ├─ code-flow.md
   ├─ operation-flow.md
   └─ operation_guide.md
```

---

## 테스트

```powershell
python -m unittest discover -s tests -v
```

41개. 비교 질문의 계층 혼합, 후속 질문의 프로젝트 상속, 계층 prior 계산,
상대 컷 동작, LLM 오판 방어, 컨텍스트 모드 3종, 원문 확장, 평가셋 유효성을 검증한다.

---

## 알려진 한계

| 항목 | 내용 |
|---|---|
| 점수 함수 | 키워드 매칭. 한국어 조사 처리 안 됨(`예산이` != `예산`) |
| 상수 | `_score()`의 제목 +2.0, 프로젝트 +1.5 등은 근거 없는 임의값 |
| 프로젝트 필터 | 하드 필터라 불일치 시 후보에서 완전 제외 |
| 세션 캐시 | 최근 N턴 고정 주입. 관련 턴 선택 없음 |
| 대화 -> STM | 일일 요약 승격 경로가 구현되지 않음 |
| LLM analyzer | 규칙 기반 대비 우위가 확인되지 않음 |
| B방식 | 원문 확장률 미측정 |
