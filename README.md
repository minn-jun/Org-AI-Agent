# Org AI Agent MVP

조직의 대화(STM), 진행 문서(MTM), 장기 보관 문서(LTM)에서 근거를 찾아 출처와 함께 답하는 세션형 에이전트 프로토타입이다. 연구용 구현으로, 실제 조직 지식을 자동으로 승인하거나 STM/MTM을 운영 데이터에서 적재하는 기능은 아직 없다.

## 프로젝트 요약

한 턴은 **세션 로드 → 질문 분석 → 계층별 검색 → 근거 선별 → 답변 생성 → 기록 저장** 순서로 실행된다. 모델은 필요하면 `retrieve_memory`로 추가 검색하거나 `expand_evidence`로 선택된 근거의 원문을 조회할 수 있다. 검색 기본값은 형태소 토큰화와 BM25(k1=1.2)이며, 여러 계층의 후보를 전역 점수로 합치고 같은 출처의 중복 근거를 접는다.

| 위치 | 내용 | 기본 제공 |
|---|---|---|
| `org_agent_mvp/` | 에이전트, 검색기, 세션, OpenRouter 연동 코드 | 예 |
| `tests/fixtures/memory/` | 실행 예시에 쓰는 가상 과제 A/B 메모리 | 예 |
| `datasets/allganize-rag-eval-ko/` | 공개 한국어 RAG 검색 평가셋 | 예 |
| `tests/fixtures/eval_cases_20200504.jsonl` | 자체 과제 평가 질문 60개 | 예 |
| `datasets/20200504-doc_rag/` | 실제 과제 문서 코퍼스·청크 | 로컬 별도 보관 |
| `memory_seed_20200504/` | 과제 사실을 바탕으로 만든 합성 STM/MTM 시드 | 로컬 별도 보관 |

실제 과제 코퍼스는 698문서·26,031청크로 구축했다. 개인정보 문서를 제외하고 식별번호를 마스킹했지만, 승인된 조직 지식만 선별한 코퍼스는 아니다. 공개 평가셋은 PDF 64개에서 만든 2,960청크와 채점 가능한 질문 299개로 구성된다.

## 실행 방법

저장소 루트(`org_agent_mvp`)에서 PowerShell로 실행한다. `--mock`은 API 키 없이 흐름을 확인하는 모드이며 실제 모델의 답변을 재현하지 않는다.

```powershell
$env:MEMORY_ROOT = "tests/fixtures/memory"
$env:LTM_CORPUS = ""
python -m org_agent_mvp --mock --verbose --question "A 과제 예산 검토에서 뭐가 지적됐어?"
```

로컬에 별도 보관한 과제 자료가 있으면 두 폴더를 위 표의 경로에 둔 뒤 실행한다. `LTM_CORPUS=auto`는 `datasets/20200504-doc_rag/export/` 아래의 청크 파일을 찾는다.

```powershell
$env:MEMORY_ROOT = "memory_seed_20200504"
$env:LTM_CORPUS = "auto"
python -m org_agent_mvp --mock --verbose --question "이 과제의 전체 연구개발기간은 언제부터 언제까지야?"
```

실제 모델 호출은 `.env.example`을 참고해 `.env`에 `OPENROUTER_API_KEY`를 설정하고 `--mock`을 빼서 실행한다. API 호출 비용이 발생한다. `--context-mode full|hybrid|summary`로 초기 근거 주입 방식을 바꿀 수 있고, `--save-log`를 주면 턴 기록을 `logs/turns/`에 저장한다.

## 테스트 실행 방법

```powershell
python -m unittest discover -s tests -v
```

로컬 과제 자료가 있는 환경에서 **139건 통과**를 확인했다. 자료가 없으면 해당 자료를 검사하는 일부 테스트는 건너뛴다. 검색 품질 평가는 다음과 같이 실행한다.

```powershell
# 공개 평가셋: 저장소에 포함된 데이터만 사용, LLM 호출 없음
python scripts/bench_allganize.py run --split test --label current

# 자체 60문항: 로컬 과제 코퍼스와 메모리 시드 필요, 기본 설정은 LLM 호출 없음
python eval/run_eval.py
```

`eval/run_eval.py --analyzer llm`과 실제 모델을 쓰는 컨텍스트 평가는 API 키와 호출 비용이 필요하다.

## 지표

| 평가 대상 | 결과 | 범위 |
|---|---|---|
| 자체 과제 60문항 근거 선별 | MRR **0.426 → 0.589**, 재현율 **52.1% → 63.6%** | 실문서 LTM + 합성 STM/MTM |
| Allganize test 149문항 LTM 검색 | 페이지 MRR@10 **0.516 → 0.823** | dev 150문항에서 설정을 고른 뒤 측정한 당시 결과 |
| Allganize 현재 코드 재실행 | 페이지 MRR@10 **0.8201** | 과거 0.8234와 한 문항의 순위가 다름 |
| 계층 병합 스트레스 검사 | 답 없는 STM/MTM 문서 698건 추가 시 Hit@8 **97.0% → 95.3%** | 공개 평가셋에 합성 방해 문서를 추가 |
| 재정렬기 실험 | 전체 299문항 페이지 MRR **0.814 → 0.849** | bge-m3 사용 시 CPU 질문당 약 58.5초; 기본값에서는 비활성화 |

MRR과 Hit@k는 **정답 근거를 검색한 순위**를 측정한다. 답변 정확도 지표는 아니다. 이미지·슬라이드 근거는 텍스트 검색에서 상대적으로 약하며, 실제 모델 호출은 소수 질문에서만 확인했다.
