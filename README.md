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
  │    3. 정규화       PREFETCH_NORMALIZE (기본 global = raw / 후보최고점)
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

---

## Quick Start

저장소에는 실제 과제 자료가 없다. 흐름만 볼 때는 테스트용 가상 저장소를 쓴다.

```powershell
cd org_agent_mvp

# API 없이 흐름만 확인 (가상 과제 A/B)
$env:MEMORY_ROOT = "tests/fixtures/memory"
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

## 메모리와 데이터

| 계층 | 역할 | 예시 |
|---|---|---|
| STM | 최신 대화와 당일 업무 흐름 | 회의 요약, 통화 메모, action item |
| MTM | 진행 중인 프로젝트 지식 | 회의록, 제안서 초안, 일정표, 검토 보고 |
| LTM | 승인된 조직 기준 지식 | 최종 계획서, 규정, 작성 기준 |

두 환경변수로 무엇을 읽을지 정한다.

| 변수 | 의미 | 기본값 |
|---|---|---|
| `MEMORY_ROOT` | `stm/ mtm/ ltm/` 폴더를 가진 시드 폴더 | `memory_seed_20200504` (저장소 미포함) |
| `LTM_CORPUS` | LTM으로 붙일 `chunks.jsonl`. `auto`면 `../datasets/20200504-doc_rag/export/...`에서 찾는다 | 없음 |

`LTM_CORPUS`를 주면 청크 코퍼스를 역색인으로 읽어 LTM에 붙인다(`ltm_corpus.py`).
검색은 청크 단위로 하고 근거 카드는 문서 단위로 접는다.

실제 과제 기반 자료(`memory_seed_20200504/`, 평가셋, 코퍼스 프로파일)는
연구실 문서라 `.gitignore` 대상이다. 없으면 해당 테스트는 건너뛴다.

---

## 근거 선별 방식

계층 가중치를 **검색 자리 수가 아니라 점수 배수로** 쓴다.

| | 이전 (quota) | 현재 (prior) |
|---|---|---|
| 계층 가중치의 역할 | 자리 수 | **점수 배수** |
| 자르는 시점 | 계층별로 미리 자름 | 전역 순위 후 한 번 |
| 점수 결합 | 가산 | `정규화점수 x (1 + a x 가중치)` (승산) |
| 근거 개수 | 항상 8건 고정 | **2~8건 가변** |

**승산인 이유**: 가산이면 관련성이 0인 문서도 계층 보너스만으로 점수를 얻는다.

### 계층 병합 정규화

계층마다 코퍼스 크기가 달라(BM25의 IDF가 N에 따라 달라짐) 점수 눈금이 어긋날 수 있다.

| `PREFETCH_NORMALIZE` | 하는 일 | 비고 |
|---|---|---|
| `global` (기본) | 전역 최고점으로 나눔 | 눈금이 맞을 때 최선 |
| `tier` / `rrf` / `zscore` | 계층 내부 정보만 씀 | 볼 것 없는 계층도 1등이 떠서 LTM이 무너진다 |
| `hybrid` | `global^(1-w) x zscore^w` | 눈금이 어긋날 때(BM25) 재현율이 오른다. `PREFETCH_HYBRID_W`(기본 0.5) |

---

## 검색기 설정

기본값은 전부 0단계(공백 분리 + 빈도 점수)다.

| 변수 | 값 | 필요 패키지 |
|---|---|---|
| `RETRIEVER_TOKENIZER` | `whitespace` \| `morph` | `kiwipiepy` |
| `RETRIEVER_SCORER` | `freq` \| `bm25` | — |
| `RETRIEVER_SCORER_SEED` | STM/MTM만 따로 지정 | — |
| `RETRIEVER_DENSE` | `0` \| `1` | `torch`, `sentence-transformers` |
| `RETRIEVER_DENSE_WEIGHT` | RRF에서 dense 비중 (기본 0.5) | — |

임베딩은 첫 실행에 청크를 인코딩해 `cache/dense/`에 저장하고 이후에는 읽기만 한다.

---

## 컨텍스트 주입 방식

```powershell
python -m org_agent_mvp --context-mode summary --question "..."
```

| 모드 | 1차 컨텍스트 |
|---|---|
| `full` (기본) | 근거 원문 전량 |
| `hybrid` | 상위 2건 원문 + 나머지 요약 |
| `summary` | 요약만, 원문은 요청 시 |

`summary`와 `hybrid`에서는 `expand_evidence` 도구가 노출된다.
이 도구는 **재검색이 아니라 턴 안에 이미 들고 있는 카드의 조회**라 비용이 없다.

---

## 평가

실제 과제 자료가 있는 로컬 환경에서만 돈다.

```powershell
# 근거 선별 품질 - LLM 호출 없음, 결정적, 무료
python eval/run_eval.py                     # 기본: 실코퍼스 평가셋 + LTM_CORPUS=auto
python eval/run_eval.py --mode quota        # 이전 쿼터 방식 baseline
python eval/run_eval.py --alpha 0           # 계층 무시 ablation
python eval/run_eval.py --analyzer llm      # 실제 analyzer 사용 (유료)

# 검색기 단계 비교
python scripts/bench_retriever.py --stage 0 1 2 3 3h
python scripts/bench_ltm_only.py            # 계층 병합 제외, LTM 검색만

# 컨텍스트 주입 방식
python eval/run_context_eval.py --mock      # 흐름 확인
python eval/run_context_eval.py --limit 5   # 실모델
```

`--analyzer rule`(기본)은 analyzer를 고정한 **통제 비교**다. 실운영 수치가 아니다.

`bench_retriever.py` 단계 이름은 숫자가 검색기(0~3)이고 접미사가 변형이다 —
`s` LTM만 BM25 / `h` hybrid 병합.

---

## 세션과 로그

| 위치 | 내용 |
|---|---|
| `logs/sessions/*.json` | **전체 턴 누적.** 오래된 턴을 지우지 않는다 |
| `logs/turns/*.json` | `--save-log` 사용 시 한 턴의 분석, prefetch, LLM, 도구 실행 |

컨텍스트에는 최근 4턴만 붙는다(`max_recent_turns`). 저장과 주입은 분리되어 있다.

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
│  ├─ prefetch.py           근거 선별 (계층 병합 정규화, 계층 prior, 상대 컷)
│  ├─ context_builder.py    [RUNTIME_CONTEXT] 조립 (full/summary/hybrid)
│  ├─ memory_store.py       STM/MTM 파일 로드와 검색, LTM 코퍼스 병합
│  ├─ ltm_corpus.py         청크 코퍼스 역색인, 버전 접기, 문서 단위 근거
│  ├─ tokenizer.py          공백 분리 / 형태소 토큰화
│  ├─ scoring.py            빈도 / BM25 점수, 병합 정규화 선택
│  ├─ dense.py              임베딩 검색과 RRF 융합
│  ├─ session_store.py      세션 기록 (전체 누적)
│  ├─ openrouter_client.py  API 호출, usage 수집
│  ├─ mock_llm.py           API 없이 흐름 재현
│  ├─ schemas.py            retrieve_memory / expand_evidence 도구 정의
│  ├─ prompts.py            시스템 프롬프트
│  └─ normalization.py      과제명 표기 정규화
├─ scripts/
│  ├─ bench_retriever.py    검색기 단계 비교 (에이전트 경로)
│  └─ bench_ltm_only.py     LTM 검색만 측정
├─ eval/
│  ├─ run_eval.py           근거 선별 품질
│  └─ run_context_eval.py   컨텍스트 주입 방식
├─ tests/
│  └─ fixtures/memory/      가상 과제 A/B + 공통 기준 13건 (테스트 전용)
└─ docs/
   ├─ parameters-and-rationale.md   수치의 값과 근거 (09-01 기준)
   └─ code-flow.md                  코드 책임 분리 (09-01 기준)
```

---

## 테스트

```powershell
python -m unittest discover -s tests -v
```

113건. 계층 prior 계산, 상대 컷, 후속 질문의 프로젝트 상속, LLM 오판 방어,
컨텍스트 모드 3종, 원문 확장, 토크나이저·점수 함수·임베딩·LTM 코퍼스 모듈,
병합 정규화, **프로세스를 바꿔도 같은 결과가 나오는지(재현성)**를 검증한다.

실제 과제 평가셋 검사(`test_eval_cases_20200504.py`)는 자료가 없으면 건너뛴다.

---

## 알려진 한계

| 항목 | 내용 |
|---|---|
| 상수 | `_score()`의 제목 +2.0, 프로젝트 +1.5 등은 근거 없는 임의값 |
| 세션 캐시 | 최근 N턴 고정 주입. 관련 턴 선택 없음 |
| 대화 -> STM | 일일 요약 승격 경로가 구현되지 않음 |
| LTM 적재 | 과제 폴더를 정제 없이 넣는다. 승격 규칙 미정 |
| 과제 전용 규칙 | 버전 접기·최종본 폴더 판정이 한 과제 폴더 구조에 맞춰져 있다 |
| 컨텍스트 주입 | LTM 근거가 청크 하나 600자라 summary와 차이가 작다. 원문 확장률 미측정 |
