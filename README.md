# Org AI Agent MVP

조직의 최신 대화, 진행 문서, 공식 기준을 근거로 답하는 세션형 에이전트 프로토타입이다.

**구조를 이해하기 위한 실험 환경이며 최종 시스템이 아니다.**
여기서 나온 수치는 이 실험 조건 안에서만 유효하다.

---

## 한 턴의 흐름

```text
사용자 질문
  │
  ├─ 세션 로드                전체 기록 보관, 최근 8턴 캐시 중 4턴을 컨텍스트에 주입
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

새로 클론한 저장소에는 테스트용 가상 메모리와 Allganize 공개 평가셋이 있다. 실제 과제 자료인 `datasets/20200504-doc_rag/`와 합성 메모리 `memory_seed_20200504/`는 Git에 포함되지 않는다.

```powershell
cd org_agent_mvp

# API 없이 흐름만 확인 (가상 과제 A/B)
$env:MEMORY_ROOT = "tests/fixtures/memory"
$env:LTM_CORPUS = ""
python -m org_agent_mvp --mock --verbose --question "A 과제 예산 검토에서 뭐가 지적됐어?"

# 대화형 세션
python -m org_agent_mvp --mock --verbose
```

대화 중 `exit` 또는 `quit`로 종료하고, `--session-id session-...`으로 기존 세션을 이어간다.

로컬에 실제 과제 자료를 보관하고 있다면 아래처럼 경로를 지정해 실행한다. `LTM_CORPUS=auto`는 `datasets/20200504-doc_rag/export/.../chunks.jsonl`을 찾는다.

```powershell
$env:MEMORY_ROOT = "memory_seed_20200504"
$env:LTM_CORPUS = "auto"
python -m org_agent_mvp --mock --verbose --question "이 과제의 전체 연구개발기간은 언제부터 언제까지야?"
```

`--mock`은 실행 흐름 확인용이며 실제 모델의 답변을 재현하지 않는다.

실제 모델을 쓰려면 `.env`에 키를 넣고 `--mock`을 뺀다.

```env
OPENROUTER_API_KEY=
QUERY_ANALYZER_MODEL=openai/gpt-5.6-luna-pro
AGENT_MODEL=openai/gpt-5.6-luna-pro
OPENROUTER_RETRIES=2          # 연결 끊김·5xx·429 재시도 횟수 (0이면 한 번만 보낸다)
```

---

## 메모리와 데이터

| 계층 | 역할 | 예시 |
|---|---|---|
| STM | 최신 대화와 당일 업무 흐름 | 회의 요약, 통화 메모, action item |
| MTM | 진행 중인 프로젝트 지식 | 회의록, 제안서 초안, 일정표, 검토 보고 |
| LTM | 장기 보관 문서와 기준 지식 | 최종 계획서, 규정, 작성 기준 |

두 환경변수로 무엇을 읽을지 정한다.

| 변수 | 의미 | 기본값 |
|---|---|---|
| `MEMORY_ROOT` | `stm/ mtm/ ltm/` 폴더를 가진 시드 폴더 | `memory_seed_20200504` (로컬 자료, 저장소 제외) |
| `LTM_CORPUS` | LTM으로 붙일 `chunks.jsonl`. `auto`면 `datasets/20200504-doc_rag/export/...`에서 찾는다 | 없음 |

`LTM_CORPUS`를 주면 청크 코퍼스를 역색인으로 읽어 LTM에 붙인다(`ltm_corpus.py`).
검색은 청크 단위로 하고 근거 카드는 문서 단위로 접는다.

### 코퍼스 메타데이터

검색기는 특정 코퍼스의 폴더 이름이나 파일명 습관을 모른다. 문서 성격·버전 묶음·최종본·소속 과제는
**코퍼스 전처리가 채운 필드**로만 읽고, 없으면 그 기능만 꺼진다.

| 필드 | 쓰임 | 없을 때 |
|---|---|---|
| `doc_type` | 필터, analyzer enum | `document` |
| `version_group` / `version_rank` | 같은 문서의 버전을 한 건으로 접고 최신본을 대표로 | 같은 제목끼리만 접힘 |
| `is_final` | 묶음 대표를 고를 때 가장 먼저 봄 | `false` |
| `project` | 과제 필터, 과제명 가산점 | 빈 문자열 |
| `page_nos` (청크) | 청크가 걸친 페이지. 카드 `source_ref.page_nos` | `page_no` |

문서 필드는 청크 `metadata`나 `chunks.jsonl` 옆 `document_meta.jsonl`에 둔다. `LTM_DOC_META=none`이면 옆 파일을 읽지 않는다.
20200504 과제 폴더 전용 규칙(폴더 이름 → 문서 유형, 버전 꼬리표, 최종제출 폴더)은 2026-09-15에
코퍼스 전처리(`datasets/20200504-doc_rag/scripts/enrich_doc_meta.py`)로 옮겼다.

실제 과제 코퍼스는 로컬 `datasets/20200504-doc_rag/`에, 실험용 STM/MTM 시드는 로컬 `memory_seed_20200504/`에 있다. 과제 기반 평가 질문은 저장소의 `tests/fixtures/eval_cases_20200504.jsonl`에 있지만, 채점에는 제외된 두 로컬 폴더가 필요하다. 시드는 실제 업무 대화가 아니라 과제 사실을 바탕으로 만든 합성 자료다.

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

기본값은 **형태소 + BM25(k1 1.2) + 가산점·동의어·임베딩 없음**이다.
공개 평가셋(Allganize 299문항)과 과제 폴더 60건에서 고른 조합이다 — 상위 폴더의
`llm_study/260904-0918/05-allganize-검색-기준선.md`, `07-재정렬기-경량화와-에이전트-적용.md` 참조.
0단계(공백 분리 + 빈도)로 되돌리려면 `RETRIEVER_TOKENIZER=whitespace RETRIEVER_SCORER=freq`.

| 변수 | 값 | 필요 패키지 |
|---|---|---|
| `RETRIEVER_TOKENIZER` | `morph` (기본) \| `whitespace`. `kiwipiepy`가 없으면 자동으로 `whitespace` | `kiwipiepy` |
| `RETRIEVER_SCORER` | `bm25` (기본) \| `freq` \| `bm25plus` (BM25+, δ=1.0) | — |
| `RETRIEVER_SCORER_SEED` | STM/MTM만 따로 지정 (기본은 위 값을 따름 — 다르게 주면 계층 병합이 무너진다) | — |
| `RETRIEVER_BM25_K1` | BM25 k1 (기본 1.2) | — |
| `RETRIEVER_TITLE_BONUS` | `none` (기본) \| `add` (토큰마다 +2.0) \| `mult` (×최대 1.1) | — |
| `QUERY_EXPANSIONS` | 사전 경로 \| `default` (`config/query_expansions.json`) \| `none` (기본) | — |
| `LTM_DOC_META` | 문서 메타데이터 파일 경로 \| `none` (기본 `chunks.jsonl` 옆 `document_meta.jsonl`) | — |
| `RETRIEVER_DENSE` | `0` \| `1` | `torch`, `sentence-transformers` |
| `RETRIEVER_DENSE_WEIGHT` | RRF에서 dense 비중 (기본 0.5) | — |
| `RETRIEVER_RERANK` | `none` (기본) \| cross-encoder 모델 이름. 상위 N개 청크의 **순서만** 다시 매긴다 (점수 값은 유지) | `sentence-transformers` |
| `RETRIEVER_RERANK_TOP_N` | 재정렬할 후보 수 (기본 30) | — |

임베딩은 첫 실행에 청크를 인코딩해 `cache/dense/`에 저장하고 이후에는 읽기만 한다.
재정렬 점수는 `cache/rerank/`에 (모델, 질문, 청크) 단위로 쌓는다.

공개 평가셋(Allganize RAG-Evaluation-Dataset-KO)으로 LTM 검색기만 재는 스크립트는 `scripts/bench_allganize.py`다.
데이터는 `datasets/allganize-rag-eval-ko/ltm/`에 둔다.

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
이 도구는 **재검색이 아니라 턴 안에 이미 들고 있는 카드의 조회**다. 조회 자체에는 검색·API 비용이 없지만, 도구 결과를 모델에 다시 전달하면 모델 호출 토큰은 든다.

---

## 평가

과제 평가 명령은 로컬의 `datasets/20200504-doc_rag/`와 `memory_seed_20200504/`가 있어야 한다. 새로 클론한 저장소에는 과제 평가 질문만 있으므로 해당 결과 수치를 바로 재현할 수 없다. 실모델을 쓰는 옵션은 API 키와 호출 비용이 필요하다.

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

저장소에 포함된 Allganize 공개 평가셋은 과제 자료 없이 별도로 측정할 수 있다.

```powershell
python scripts/bench_allganize.py run --split test --label current
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
│  ├─ bench_ltm_only.py     LTM 검색만 측정
│  └─ bench_allganize.py    공개 평가셋의 페이지/문서 검색 측정
├─ datasets/
│  ├─ 20200504-doc_rag/     로컬 전용: 실제 과제 문서의 파싱 결과, 청크, 메타데이터, 전처리 스크립트
│  └─ allganize-rag-eval-ko/  공개 평가셋 원본 PDF, 라벨, 청크, 검증·실험 자료
├─ memory_seed_20200504/   로컬 전용: 과제 사실 기반 합성 STM/MTM 시드 및 LTM 자료
├─ config/                 검색어 확장 설정
├─ eval/
│  ├─ run_eval.py           근거 선별 품질
│  └─ run_context_eval.py   컨텍스트 주입 방식
├─ tests/
│  └─ fixtures/memory/      가상 과제 A/B + 공통 기준 13건 (테스트 전용)
├─ cache/                  임베딩·재정렬 캐시 (Git 제외)
├─ logs/                   로컬 세션·턴 기록 (Git 제외)
└─ docs/
   ├─ parameters-and-rationale.md   수치의 값과 근거 (09-01 기준)
   └─ code-flow.md                  코드 책임 분리 (09-01 기준)
```

---

## 테스트

```powershell
python -m unittest discover -s tests -v
```

로컬 과제 자료가 있을 때 확인한 **139건**이다. 계층 prior 계산, 상대 컷, 후속 질문의 프로젝트 상속, LLM 오판 방어,
컨텍스트 모드 3종, 원문 확장, 토크나이저·점수 함수·임베딩·LTM 코퍼스 모듈,
병합 정규화, **프로세스를 바꿔도 같은 결과가 나오는지(재현성)**를 검증한다.

실제 과제 평가셋 검사(`test_eval_cases_20200504.py`) 중 자료가 필요한 검사는 해당 로컬 폴더가 없으면 건너뛴다.

---

## 알려진 한계

| 항목 | 내용 |
|---|---|
| 상수 | `_score()`의 프로젝트 +1.5 등은 근거 없는 임의값 (제목 가산점은 2026-09-16부터 기본 꺼짐) |
| 세션 캐시 | 최근 N턴 고정 주입. 관련 턴 선택 없음 |
| 대화 -> STM | 일일 요약 승격 경로가 구현되지 않음 |
| LTM 적재 | 개인정보 제외·마스킹과 버전 메타데이터 정리는 했지만, 승인·승격 규칙을 거친 조직 지식 저장소는 아님 |
| 과제 전용 규칙 | 버전 접기·최종본 폴더 판정이 한 과제 폴더 구조에 맞춰져 있다 |
| 컨텍스트 주입 | LTM 근거가 짧아 summary와 비용 차이가 작을 수 있다. 소수 질문에서만 답변·토큰을 확인했다 |

---

## 2026-09-04~09-17 개발 현황

### 구현한 범위

1. **실제 과제 코퍼스 구축**: 원본 문서를 파싱하고 개인정보 문서 146건을 제외했으며, 식별번호 1,129건을 마스킹했다. 버전 중복을 정리한 검색 대상은 **698문서·26,031청크**다. 파일명·폴더에 의존하던 문서 유형과 버전 판정은 `datasets/20200504-doc_rag/scripts/enrich_doc_meta.py`에서 메타데이터로 만든다.
2. **에이전트 한 턴 연결**: 세션 로드 → 질문 분석 → STM/MTM/LTM 검색과 근거 선별 → 컨텍스트 조립 → 모델의 재검색·원문 확장 도구 호출 → 출처를 단 답변 → 세션·턴 로그 저장까지 실행된다. `--mock`과 실제 OpenRouter 호출 경로가 모두 있다.
3. **검색 기본값 개선**: 형태소 토큰화와 BM25(k1=1.2)를 기본으로 적용했다. 계층별 후보를 전역 점수로 합치고, 출처가 같은 STM/MTM/LTM 카드를 한 장으로 접는다. 제목 가산점·동의어 확장·dense 검색·재정렬은 기본값에서 꺼 두었다.
4. **공개 평가셋 연결**: Allganize 한국어 RAG 평가셋의 PDF **64개**를 **2,960청크**로 변환했다. 300질문 중 라벨이 없는 1개를 제외한 **299개**를 원본 파일명·페이지 라벨로 채점한다.

### 검증 결과

| 범위 | 결과 | 해석 |
|---|---|---|
| 단위·통합 테스트 | **139건 통과** | 로컬 과제 자료가 있는 환경에서 확인한 기록 |
| 자체 과제 60문항, 에이전트 근거 선별 | MRR **0.426 → 0.589**, 재현율 **52.1% → 63.6%** | LTM은 실문서, STM/MTM은 합성 시드. 답변 정확도 지표가 아님 |
| Allganize test 149문항, LTM 검색 | 페이지 MRR@10 **0.516 → 0.823** | dev 150문항에서 설정 선택 후 test 측정한 당시 결과 |
| Allganize 현재 코드 재실행 | 페이지 MRR@10 **0.8201** | 이전 0.8234와 q_274 한 문항의 순위 차이. 실행 코드 버전이 달라 원인 단정 불가 |
| 계층 병합 방해자 실험 | 답 없는 STM/MTM 문서 698건 추가 시 Hit@8 **97.0% → 95.3%** | 공개 평가셋에 합성 방해자를 얹은 스트레스 검사 |
| 동일 출처 중복 제거 | 심은 60건 중 **59건**을 접음; 나머지 239문항 MRR **0.807 → 0.901** | 같은 출처가 근거 카드 자리를 중복 점유하는 문제를 줄임 |

추가 실험에서 `bge-reranker-v2-m3`는 Allganize 전체 299문항 페이지 MRR을 **0.814 → 0.849**로 올렸지만 CPU에서 질문당 약 **58.5초**가 걸려 기본값으로 적용하지 않았다. 이미지·슬라이드 질문은 텍스트 청크 검색만으로 약하고, 실제 조직 대화의 STM/MTM 자동 적재·승격도 아직 없다. 소수 질문의 실제 모델 호출은 답변과 출처 연결을 확인하는 용도이며, 위 검색 지표를 전체 답변 품질로 해석하면 안 된다.

실제 OpenRouter 호출도 한 질문으로 끝까지 확인했다. “이 과제의 전체 연구개발기간은 언제부터 언제까지야?”에 대해 **2020-07-01~2022-12-31**을 답했고, 회수된 HWP 청크에서 날짜를 대조했다. 이 턴은 초기 근거 8장, 추가 검색 2회, analyzer 1회와 agent 3회 호출에 실제 토큰 **140,491개**를 썼다. 단일 성공 사례이며 평균 비용이나 전체 정확도 측정은 아니다.

재현 절차와 실험별 조건·실패 사례는 로컬 상위 폴더의 `llm_study/260904-0918/99-인수인계서.md` 및 `09-2주-발표-구성안.md`에 정리했다. 이 문서들은 현재 Git 저장소 바깥에 있다. `memory_seed_20200504/`, `datasets/20200504-doc_rag/`, `logs/`, `cache/`는 Git에서 제외되며, `.env`의 API 키도 추적하지 않는다.
