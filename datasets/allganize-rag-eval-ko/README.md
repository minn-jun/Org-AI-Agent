# Allganize RAG-Evaluation-Dataset-KO — 로컬 사본

받은 날: 2026-09-14 · 정리 · 라벨 기준 확정: 2026-09-15
용도: LTM 검색기 일반화 확인 ([03-전제-재검토와-다음-방향.md](../../llm_study/260904-0917/03-전제-재검토와-다음-방향.md) 3~4절)
청크 검증 결과: [04-allganize-청크-검증.md](../../llm_study/260904-0917/04-allganize-청크-검증.md)

---

## 1. 폴더 구성

```
allganize-rag-eval-ko/
├─ README.md                        이 문서
├─ original/                        Allganize 원본 그대로 (손대지 않음)
│  ├─ rag_evaluation_result.csv     ← 원본 정답 라벨 (질문 · 정답 문장 · 정답 파일 · 정답 페이지 · 근거 유형
│  │                                   + 기존 RAG 시스템들의 답변과 O/X — 라벨 아님)
│  ├─ documents.csv                 원본 문서 목록 (기관 게시판 링크 · 페이지 수)
│  ├─ DATA_CARD.md                  원본 데이터 카드
│  └─ official_pdfs/                기관 게시판에서 직접 받은 원본 PDF (복제본이 다른 판본인 문서만, 4-1)
├─ pdfs/<domain>/<file_name>        원문 PDF 64개 (복제본 페이지를 합친 것 + 원본 교체 1개)
├─ labels/                          작업용 골드 라벨  <- 평가에 쓰는 것
│  ├─ questions.jsonl               질문 300개 (채점 299). 원본 라벨 기준, pdfs/ 페이지로 맞춘 값
│  ├─ documents.jsonl               문서 메타 (pid, 페이지 수, 글자층 없는 페이지, source)
│  └─ label_check.json              정답 페이지 ↔ PDF 원문 대조 결과
├─ chunks/chunks.sqlite3            doc_rag로 뽑은 최종 청크 2,960개 (page_nos 포함)
├─ ltm/                             LTM 검색기 평가용 (scripts/export_ltm_jsonl.py)
│  ├─ chunks.jsonl                  청크를 org_agent_mvp LtmCorpus 형식으로 (문서 메타데이터 없음)
│  ├─ cases_dev.jsonl / cases_test.jsonl   채점 299문항 150 / 149 (도메인 × 근거 유형 × 글자층 층화, 시드 20260915)
│  ├─ split.json                    분할 기준과 칸별 개수
│  └─ rerank_pairs.jsonl            재정렬 후보 쌍 8,970개 (scripts/rerank_pairs.py dump, 06 문서)
├─ checks/
│  ├─ chunk_check.json              청크 ↔ 원문 · 골드 라벨 대조 결과 (최종)
│  └─ merge_report.json             PDF 합치기 · 원본 목록 대조 결과 (09-14, 교체 전)
├─ scripts/                         아래 6절
├─ logs/                            최종 빌드 · 대조 · 비교 로그 (09-15)
└─ _raw/                            datalama 복제본 tar 64개 (449MB). pdfs/를 다시 만들 때만 필요
```

`labels/questions.jsonl` 필드: `qid`, `domain`, `question`, `target_answer`, `file_name`, `pid`,
`target_page_no`(pdfs/ 기준, int), `original_target_page_no`(원본 값), `context_type`,
`label_fix`(페이지를 옮긴 이유), `label_note`(옮기지 않았지만 알아둘 점), `exclude`(채점 제외 이유), `gold_page_has_text`

`chunks/chunks.sqlite3`: `documents`(문서별 상태 · 청크 수 · OCR 표기)와 `chunks`(`text`, `headings`, `page_no`, `page_nos`) 테이블.

---

## 2. 출처

| 무엇 | 어디서 | 라이선스 |
|---|---|---|
| 질문 · 정답 · 문서 목록 | `huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO` | MIT |
| PDF 원문 | `huggingface.co/datasets/datalama/RAG-Evaluation-Dataset-KO` (페이지별로 쪼갠 PDF를 tar 64개로 묶은 복제본) | MIT (복제본 표기) |
| 원본 교체 PDF | 기획재정부 게시판 (`documents.csv`의 url) | 공공 문서 |

allganize 원본의 `documents.csv`는 PDF 주소가 아니라 **44개 기관 게시판 페이지 링크**라 자동으로 받을 수 없다.
그래서 복제본의 페이지 PDF를 순서대로 합쳐 원래 파일로 되돌렸다.

> PDF 자체는 각 기관(한국은행, 금융위원회, 법원, 부처 등)의 공개 문서다. 로컬 연구용으로만 쓰고 재배포하지 않는다.

---

## 3. 규모

| 도메인 | 문서 | 페이지 | 질문 |
|---|---|---|---|
| finance | 10 | 301 | 60 |
| public | 12 | 257 | 60 |
| medical | 20 | 302 | 60 |
| law | 12 | 291 | 60 (채점 59) |
| commerce | 10 | 242 | 60 |
| **합계** | **64** | **1,393** | **300 (채점 299)** |

- 근거 유형: paragraph 148 · image 57 · table 50 · text 45 (text는 medical에만 있다. 원본 카드의 "문단 193"은 paragraph + text)
- **정답이 없는 문서 6개**(질문이 안 걸림) — 검색 방해 문서로 그대로 둔다
- public 페이지는 재정동향을 61쪽 복제본에서 4쪽 원본으로 바꿔 314 → 257쪽이 됐다

### 글자층이 없는 페이지

1,393쪽 중 **155쪽**에 글자층이 없다(글자 30자 미만). 문서 22개에 걸쳐 있고 4개는 전 페이지가 이미지다.
**채점 질문 중 정답 페이지가 여기에 걸린 것이 21건**(paragraph 9 · image 10 · text 2)이다. 결과를 볼 때 `gold_page_has_text`로 나눠서 본다.
청크는 OCR로 이 중 134쪽을 채웠다(나머지 21쪽은 백지 · 간지).

> 검사 스크립트(`check_gold_pages.py`)는 공백을 뺀 글자 수로 세서 22건으로 잡는다. 경계에 걸린 1건 차이다.

---

## 4. 정답 라벨 기준

**원칙 (2026-09-15 확정): Allganize 원본 라벨을 따른다.**
원본이 가리키는 내용을 우리 PDF의 같은 위치로 옮기는 것만 하고, 내용 판단으로 페이지를 바꾸지 않는다.
원본 라벨이 의심스러우면 `label_note`에만 남기고, 원본에 정답 페이지가 없으면 채점에서 뺀다(`exclude`).
청크 단위 골드는 따로 없다 — 청크의 `page_nos`에 정답 페이지가 있으면 정답 청크로 본다.

### 4-1. 원본 목록과 페이지 수가 다른 문서

기관 게시판에서 원본 PDF를 직접 받아 복제본과 페이지별로 대조했다(앙골라 · 재정동향).

| 문서 | 원본 목록 | 받은 원본 | 복제본 | 처리 |
|---|---|---|---|---|
| `외교부-2024년 앙골라개황(저).pdf` | 48 | 47 | 47 (원본과 전 페이지 동일) | 4-2 |
| `(240411보도자료) 재정동향 4월호.pdf` | 4 | 4 (보도자료) | 61 (같은 이름의 전체 보고서, 다른 문서) | **pdfs/를 받은 원본으로 교체**, 라벨 그대로 |
| `PRP008.pdf` | 9 | — | 6 | 정답 3건 모두 원본 페이지에 답 문구 있음. 그대로 |
| `[3.22.목.석간]질병관리본부_...9대_생활수칙_발표.pdf` | 3 | — | 12 | 정답 없는 문서. 그대로 |

### 4-2. 페이지를 옮긴 것 — 앙골라개황 13건 (q_68~70, q_78, q_109~117)

원본 라벨 13건이 모두 정답 내용이 있는 페이지보다 **정확히 한 쪽 뒤**를 가리킨다.
(q_68 다이아몬드 생산량 980 → 25쪽, q_116 교역규모 4억 7,471만 불 → 36쪽, q_69 무역수지 223·118 → 37쪽 등, 원문에서 직접 확인)
Allganize가 쓴 48쪽 판본에 라벨 페이지(최소 21쪽)보다 앞에 한 쪽이 더 있었던 것으로 보고 **−1쪽으로 옮겼다**.
배포 파일은 PDF 한 쪽에 인쇄 두 쪽이 들어 있는 펼침면이다.

### 4-3. 원본 그대로 두고 메모만 남긴 것

| qid | 원본 라벨 | 메모 |
|---|---|---|
| q_60 | 4 | 재정동향을 원본 4쪽 PDF로 교체해 라벨 그대로 맞음. 답 수치 전부 4쪽에 있음 |
| q_133 | 21 | 21쪽은 요점 요약, 20쪽에 답 문구가 더 많음 (원문 포함률 21쪽 0.17 · 20쪽 0.61) — **결과 해석 시 표시** |
| q_230 | 2 | 답 수치(59,113,647주, 13.21)가 2쪽에 없고 4쪽에 있음 (2쪽 0.00 · 4쪽 0.29) — **결과 해석 시 표시** |
| q_210, q_211 | 4 | 글자 대조로는 다른 페이지가 높지만 4쪽 도면이 정답이 맞음(렌더 확인). 이미지 질문은 글자 대조가 틀릴 수 있다 |
| q_75 | — | 원본 정답 파일(`..._업무계획.pdf`)이 문서 목록에 없다. 복제본이 `..._보도자료.pdf`에 연결했고 그대로 따른다 |

> q_133 · q_230은 09-14에 글자 대조로 페이지를 바꿨다가 09-15에 원본 기준으로 되돌렸다.

### 4-4. 채점에서 뺀 것

| qid | 이유 |
|---|---|
| q_184 | 원본 정답 페이지가 공란 |

### 4-5. 검사 결과 (`labels/label_check.json`)

답(`target_answer`)의 숫자와 단어가 정답 페이지 본문에 몇 % 들어 있는지 세고, 같은 문서의 다른 페이지가 0.25 이상 더 높으면 의심으로 분류한다.

| 결과 | 건수 |
|---|---|
| 정상 | 273 |
| 정답 페이지에 글자층 없음 (검사 불가) | 22 |
| 의심 | 4 (q_133 · q_230 원본 유지, q_210 · q_211 렌더로 정답 확인) |
| 정답 페이지 없음 (제외) | 1 (q_184) |

> 검사는 거친 휴리스틱이다. 답이 원문을 풀어 쓰면 정상이어도 포함률이 낮다. 라벨을 바꾸는 근거로 쓰지 않는다.

---

## 5. 청크 (`chunks/`) 와 대조 (`checks/chunk_check.json`)

`RAG/doc_rag`로 PDF 64개를 파싱했다(docling + EasyOCR, 도표 안 글자 포함). 대조 방법과 오류 · 수정 기록은 04 문서에 있다.

| 지표 | 결과 |
|---|---|
| 청크 | 2,960 (페이지 없는 청크 0) |
| 원문 대비 단어 회수율 (문서 / 페이지 기준) | 98.6% / 96.4% |
| 정답 페이지 청크 판정 (299문항) ok / recovered / unreachable | 278 / 15 / 6 (lost · page_shift · no_chunk 0) |

> 이 판정은 **정답 페이지 청크들을 합친 텍스트**에 정답 문장 단어가 원문 페이지만큼 있는지 본 것이다.
> 질문 하나 ↔ 청크 하나 매칭이나 의미상 정답 여부는 확인하지 않았다.

---

## 6. 재생성

```bash
cd org_agent_mvp/datasets/allganize-rag-eval-ko
export PYTHONIOENCODING=utf-8                 # Windows

# 원본 -> PDF · 라벨
python scripts/merge_allganize.py             # _raw/*.tar (+ original/official_pdfs/ 우선) -> pdfs/, labels/{documents,questions}.jsonl, checks/merge_report.json
python scripts/apply_label_fixes.py           # 4절 기준 반영 (여러 번 돌려도 같다)
python scripts/check_gold_pages.py            # 정답 페이지 ↔ 원문 대조 -> labels/label_check.json

# 청크 (RAG/config/doc_rag.yaml: pdfs/ -> chunks/). 한글 경로 때문에 ASCII junction venv로, 메모리 때문에 4개씩 반복
cd ../../../RAG && C:/ine_rag_venv/Scripts/python.exe -m doc_rag.build_index --max-new 4
cd ../org_agent_mvp/datasets/allganize-rag-eval-ko
python scripts/check_chunks.py                # -> checks/chunk_check.json (exclude 질문은 건너뜀)
python scripts/compare_checks.py <이전.json> checks/chunk_check.json   # 코드 수정 후 회귀 확인

# LTM 검색기 평가 (결과: org_agent_mvp/logs/allganize/, 문서: llm_study/260904-0918/05-allganize-검색-기준선.md)
python scripts/export_ltm_jsonl.py            # chunks/ + labels/ -> ltm/chunks.jsonl, ltm/cases_{dev,test}.jsonl
cd ../..
python scripts/bench_allganize.py run --split dev --label <이름> --set RETRIEVER_SCORER=bm25 ...
python scripts/bench_allganize.py compare logs/allganize/dev_<A>.json logs/allganize/dev_<B>.json
```

- `export_ltm_jsonl.py`는 `ltm/chunks.jsonl`을 다시 쓴다. 파일 수정 시각이 바뀌면 임베딩 캐시가 무효가 되어 첫 임베딩 실행에서 약 400초 인코딩한다.

- `merge_allganize.py`는 `labels/questions.jsonl`을 원본 값으로 다시 쓰므로 반드시 `apply_label_fixes.py`를 이어서 돌린다.
- `merge_allganize.py`는 PDF 64개를 모두 다시 쓴다. 수정 시간이 바뀌어 doc_rag가 전부 재파싱(약 1시간)하므로, PDF 1개만 바꿀 때는 그 파일만 덮어쓰고 `documents.jsonl`의 해당 행을 고친다.
