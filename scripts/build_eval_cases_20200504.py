"""20200504 과제(실코퍼스 + 시드 STM/MTM)용 평가셋을 만든다.

기존 `tests/fixtures/eval_cases.jsonl` 33건은 `memory_seed`의 합성 문서를 가리킨다.
그건 회귀 테스트로 그대로 두고, 실코퍼스용은 여기서 따로 만든다.

## gold 라벨을 어떻게 정했나

**전부 원문에서 확인했다.** 질문을 먼저 만들고 문서를 찾은 게 아니라,
코퍼스에서 사실을 찾아 그게 든 문서를 확인한 뒤 질문을 붙였다.

    이월 금액 15,955천원  ->  20211230_이월 승인 요청서_가천대학교  (본문 확인)
    정부출연금 443,750    ->  연차보고서_가천대_v1                (본문 확인)
    총사업비 1,346,270    ->  01_경제성 분석 보고서                (본문 확인)

## 계열 확장

LTM gold는 **계열 전체로 넓힌다.**

버전 접기가 어느 버전을 대표로 올릴지는 규칙에 달려 있고, 규칙은 바뀔 수 있다.
`_v3`을 정답으로 박아 두면 대표가 `_v17`로 바뀌는 순간 라벨이 깨진다.
같은 계열 안에서는 어느 문서가 와도 맞은 것으로 본다.

    python scripts/build_eval_cases_20200504.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from org_agent_mvp.config import AppConfig          # noqa: E402
from org_agent_mvp.ltm_corpus import LtmCorpus      # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "eval_cases_20200504.jsonl"
SEED = ROOT / "memory_seed_20200504"

PROJECT = "스마트글래스 과제"

# ---------------------------------------------------------------------------
# 케이스 정의
#
#   gold 는 (tier, 식별자) 목록이다.
#     stm / mtm -> 시드 파일 이름(확장자 제외)
#     ltm       -> 실코퍼스 doc_id. 계열 전체로 자동 확장된다.
# ---------------------------------------------------------------------------

CASES: list[dict] = [

    # ===================================================== STM 단독 (당일 흐름)
    dict(id="stm-today-actions", query="오늘 정한 조치가 뭐야?", route="memory",
         gold=[("stm", "2021-12-30-daily-summary")],
         note="당일 대화 요약", core=True),
    dict(id="stm-carryover-submit", query="이월 승인 요청서 오늘 제출했어?", route="memory",
         gold=[("stm", "2021-12-30-carryover-submission-check")],
         note="제출 점검 목록"),
    dict(id="stm-director-latest", query="총괄책임자가 마지막으로 지시한 게 뭐야?", route="memory",
         gold=[("stm", "2021-12-30-director-note")],
         note="12-30 지시 메모"),
    dict(id="stm-pct-call", query="피씨티랑 통화해서 뭘 요청했어?", route="memory",
         gold=[("stm", "2021-12-29-pct-call-note")],
         note="측정 원본 요청"),
    dict(id="stm-kiat-call", query="전문기관에 문의한 답변이 뭐였어?", route="memory",
         gold=[("stm", "2021-12-28-kiat-guidance-note")],
         note="세목 변경·이월 절차 문의"),
    dict(id="stm-consulting-call", query="디자인 컨설팅 업체랑 통화한 내용 알려줘", route="memory",
         gold=[("stm", "2021-12-21-design-consulting-call")],
         note="대금 지급 기한 확인"),
    dict(id="stm-patent-cost", query="특허 용역비 정산하고 나서 뭐가 문제였어?", route="memory",
         gold=[("stm", "2021-12-26-patent-cost-note")],
         note="연구활동비 잔액 마이너스"),
    dict(id="stm-severance", query="퇴직급여충당금 처리하고 남은 확인 사항은?", route="memory",
         gold=[("stm", "2021-12-29-severance-note")],
         note="이월 잔액 포함 여부"),
    dict(id="stm-pilot-call", query="시범 서비스 협력 기관이 뭐라고 했어?", route="memory",
         gold=[("stm", "2021-12-28-pilot-service-call")],
         note="학사 일정 제약"),
    dict(id="stm-evidence-request", query="성능 근거 자료로 뭘 요청했지?", route="memory",
         gold=[("stm", "2021-12-29-performance-evidence-request")],
         note="측정 원본 항목"),

    # ===================================================== MTM 단독 (진행 지식)
    dict(id="mtm-perf-gap", query="1차년도 성능지표에서 목표 못 채운 항목은?", route="memory",
         gold=[("mtm", "2021-12-23-performance-target-gap-review"),
               ("mtm", "2021-12-08-test-evaluation-result-1")],
         note="영상 출력 지연, 착용 중량", core=True),
    dict(id="mtm-external-rejected", query="외부연구원 참여가 왜 안 됐어?", route="memory",
         gold=[("mtm", "2021-12-01-external-researcher-rejected")],
         note="소속 형태·참여율 요건 미충족", core=True),
    dict(id="mtm-new-hire", query="신규 채용은 언제 공고할 계획이야?", route="memory",
         gold=[("mtm", "2021-12-22-new-hire-plan")],
         note="2022년 1월"),
    dict(id="mtm-budget-item", query="세목 변경을 왜 하려는 거야?", route="memory",
         gold=[("mtm", "2021-12-17-budget-item-change-draft"),
               ("mtm", "2021-12-14-budget-execution-review")],
         note="인건비 잔액, 연구활동비 부족"),
    dict(id="mtm-hmd-design", query="HMD 기구 디자인에서 지적된 문제가 뭐야?", route="memory",
         gold=[("mtm", "2021-10-26-hmd-design-review")],
         note="무게중심, 발열"),
    dict(id="mtm-vision-types", query="시기능 훈련 대상 시각 이상 유형이 몇 가지야?", route="memory",
         gold=[("mtm", "2021-11-01-lab-weekly-meeting"),
               ("mtm", "2021-11-19-vision-type-ui-draft")],
         note="터널시야·황반원공·색각이상·원근시 4종"),
    dict(id="mtm-patent-avoid", query="선행특허 조사에서 회피 설계가 필요한 게 뭐였어?", route="memory",
         gold=[("mtm", "2021-11-26-patent-survey-interim"),
               ("mtm", "2021-12-02-patent-survey-final-review")],
         note="RFID 태그, 다중 안테나"),
    dict(id="mtm-report-outline", query="연차보고서 목차가 어떻게 정해졌어?", route="memory",
         gold=[("mtm", "2021-12-07-annual-report-outline")],
         note="4장 구성"),
    dict(id="mtm-consulting-spec", query="마이크로디스플레이 모듈 컨설팅 계약 조건이 뭐야?", route="memory",
         gold=[("mtm", "2021-11-15-design-consulting-kickoff"),
               ("mtm", "2021-12-13-design-consulting-delivery")],
         note="450만원, 12-13 종료, 12-31 지급"),
    dict(id="mtm-pilot-plan", query="시범 서비스 참여자를 어떻게 모집할 계획이야?", route="memory",
         gold=[("mtm", "2021-11-24-pilot-service-recruit-plan")],
         note="맹학교·연합회 경로"),
    dict(id="mtm-weekly-risk", query="12월 주간 보고에 올라온 리스크가 뭐야?", route="memory",
         gold=[("mtm", "weekly-report-2021-12-w3"),
               ("mtm", "weekly-report-2021-12-w4")],
         note="성능지표 미달, 연구활동비 부족"),
    dict(id="mtm-burndown", query="연말 예산 소진 계획에 뭐가 잡혀 있어?", route="memory",
         gold=[("mtm", "2021-11-29-budget-burndown-plan")],
         note="컨설팅비·용역비·시제품비"),

    # ===================================================== LTM 단독 (실코퍼스)
    dict(id="ltm-carryover-amount", query="연구비 이월 신청 금액과 사유가 뭐야?", route="memory",
         gold=[("ltm", "0dc2d19573495b7d"), ("ltm", "a622b13778a1ec1d")],
         note="15,955천원 / 채용 지연", core=True),
    dict(id="ltm-gov-fund-y1", query="2단계 1차년도 정부출연금이 얼마야?", route="memory",
         gold=[("ltm", "7a94bcd5334cae7e"), ("ltm", "03180780cd0005bd"),
               ("ltm", "5561792a12179d85")],
         note="443,750천원", core=True),
    dict(id="ltm-total-cost", query="총사업비와 순현재가치가 얼마로 산출됐어?", route="memory",
         gold=[("ltm", "78e3bd43bc9498f5"), ("ltm", "0d4d12a78d413f07"),
               ("ltm", "6fe384624d028ad9")],
         note="1,346,270천원 / NPV 12,355,171", core=True),
    dict(id="ltm-mou-count", query="사업화 협력 MOU를 몇 곳과 체결했어?", route="memory",
         gold=[("ltm", "026b791291776b14"), ("ltm", "bc0e58f92d4beecf")],
         note="신규 13곳"),
    dict(id="ltm-project-period", query="이 과제의 전체 연구개발기간이 언제까지야?", route="memory",
         gold=[("ltm", "f5670d7078685fd0"), ("ltm", "9df4026bd6a378f1")],
         note="2020.07.01 ~ 2022.12.31"),
    dict(id="ltm-project-number", query="연구개발과제번호가 뭐야?", route="memory",
         gold=[("ltm", "9df4026bd6a378f1"), ("ltm", "f5670d7078685fd0")],
         note="20012260"),
    dict(id="ltm-patent-contract", query="특허조사 용역 계약금액과 보고서 제출 기한이 어떻게 돼?", route="memory",
         gold=[("ltm", "2e9d9eb04b03717b"), ("ltm", "1a94f3eca0876d14")],
         note="450만원 / 종료일부터 7일 이내", core=True),
    dict(id="ltm-cert-agency", query="공인인증 시험을 어느 기관에 의뢰하기로 했어?", route="memory",
         gold=[("ltm", "b35c6c4630bdc743"), ("ltm", "adc17d0ad5e32e2e")],
         note="㈜코스텍"),
    dict(id="ltm-reliability-test", query="스마트글래스 신뢰성 시험 모델명이 뭐야?", route="memory",
         gold=[("ltm", "2b371307dbc913a5"), ("ltm", "c9da16960f56a43e")],
         note="SM-LowV-001"),
    dict(id="ltm-conference", query="학술대회 논문을 어디에 발표했어?", route="memory",
         gold=[("ltm", "f77247c230d4d911"), ("ltm", "f90853ab08d75c2f")],
         note="ICTC 2021 / 한국통신학회"),
    dict(id="ltm-webrtc", query="영상 송수신에 어떤 프로토콜을 쓰기로 했어?", route="memory",
         gold=[("ltm", "5afd808b9f7652f9"), ("ltm", "fdc1ed83d3c16758")],
         note="WebRTC"),
    dict(id="ltm-indirect-rate", query="간접비 계상 기준은 어디에 나와 있어?", route="memory",
         gold=[("ltm", "3680eabe690e1893"), ("ltm", "a9afa33968d464da")],
         note="기관별 간접비 계상기준 고시"),
    dict(id="ltm-ip-guideline", query="직무발명을 신고하려면 어떤 서류가 필요해?", route="memory",
         gold=[("ltm", "15fd567162a4b21e"), ("ltm", "7317f1b4b63dbd12")],
         note="발명신고서·양도증·내용설명서·선행기술조사서"),
    dict(id="ltm-agreement-change", query="협약 변경 요청서를 언제 냈어?", route="memory",
         gold=[("ltm", "8ac09d3bb27bab40"), ("ltm", "59b423d873117d13")],
         note="2021-12-28"),
    dict(id="ltm-in-kind", query="현물부담 확인서에 적힌 현물 금액이 얼마야?", route="memory",
         gold=[("ltm", "f5670d7078685fd0"), ("ltm", "e915731c7032a765")],
         note="10,800천원"),
    dict(id="ltm-expense-rule", query="사업비를 어떻게 산정하고 정산하는지 기준이 뭐야?", route="memory",
         gold=[("ltm", "e3f530ddbc1512c0")],
         note="산업기술혁신사업 사업비 산정·관리·정산 요령", core=True),
    dict(id="ltm-common-rule", query="산업기술혁신사업 공통 운영요령을 보고 싶어", route="memory",
         gold=[("ltm", "1eea04e7156669fd")],
         note="공통 운영요령"),
    dict(id="ltm-pilot-rounds", query="시범 서비스를 몇 차까지 진행했어?", route="memory",
         gold=[("ltm", "f77247c230d4d911"), ("ltm", "f90853ab08d75c2f")],
         note="3차"),

    # ===================================================== 계층 혼합
    dict(id="mix-carryover-why", query="이월을 신청하게 된 경위를 처음부터 설명해줘", route="memory",
         gold=[("mtm", "2021-12-01-external-researcher-rejected"),
               ("mtm", "2021-12-21-carryover-plan-draft"),
               ("ltm", "0dc2d19573495b7d")],
         note="외부연구원 무산 -> 신규 채용 -> 이월. MTM+LTM", core=True),
    dict(id="mix-report-deadline", query="연차보고서 내부 목표일이 공식 제출 기한이랑 달라?", route="memory",
         gold=[("mtm", "2021-11-29-lab-weekly-meeting"),
               ("mtm", "2021-12-27-annual-report-draft-v2"),
               ("stm", "2021-12-30-director-note")],
         note="내부 12-27 vs 협약 기준. 확인 필요로 남겨둔 지점", core=True),
    dict(id="mix-carryover-limit", query="이월 금액이 1차년도 정부출연금 대비 허용 범위 안이야?", route="memory",
         gold=[("mtm", "2021-12-21-carryover-plan-draft"),
               ("ltm", "0dc2d19573495b7d"), ("ltm", "7a94bcd5334cae7e")],
         note="MTM 계획 + LTM 확정 금액", core=True),
    dict(id="mix-budget-change-rule", query="인건비를 연구활동비로 옮기는 게 허용돼?", route="memory",
         gold=[("mtm", "2021-12-17-budget-item-change-draft"),
               ("stm", "2021-12-28-kiat-guidance-note"),
               ("ltm", "e3f530ddbc1512c0")],
         note="STM 통화 답변 + LTM 제도 문서", core=True),
    dict(id="mix-perf-target", query="성능지표 미달 판단이 공식 목표치 기준이랑 맞아?", route="memory",
         gold=[("mtm", "2021-12-23-performance-target-gap-review"),
               ("ltm", "026b791291776b14")],
         note="MTM 내부 측정 + LTM 목표 성능지표"),
    dict(id="mix-consulting-payment", query="디자인 컨설팅 대금을 연내에 못 주면 어떻게 돼?", route="memory",
         gold=[("stm", "2021-12-21-design-consulting-call"),
               ("mtm", "2021-12-13-design-consulting-delivery"),
               ("ltm", "e3f530ddbc1512c0")],
         note="STM 통화 + MTM 계약 + LTM 정산 기준"),
    dict(id="mix-patent-disclosure", query="특허 출원 전에 학술대회에서 발표해도 괜찮아?", route="memory",
         gold=[("stm", "2021-12-27-patent-application-note"),
               ("mtm", "2021-11-17-conference-presentation-prep"),
               ("ltm", "15fd567162a4b21e")],
         note="STM/MTM 우려 + LTM 지식재산권 지침", core=True),
    dict(id="mix-mou-carryover", query="1단계에서 맺은 MOU가 2단계에도 이어지고 있어?", route="memory",
         gold=[("mtm", "2021-10-08-mou-followup"),
               ("mtm", "2021-12-15-commercialization-status-review"),
               ("ltm", "026b791291776b14")],
         note="MTM 이행 + LTM 체결 현황"),
    dict(id="mix-remote-module", query="원격 교육 모듈을 2차년도로 미뤄도 계획서랑 안 어긋나?", route="memory",
         gold=[("mtm", "2021-11-12-training-service-spec-draft"),
               ("mtm", "2021-11-15-lab-weekly-meeting")],
         note="연차별 개발 내용 대조 필요"),
    dict(id="mix-test-condition", query="시험평가 측정 조건이 계획서 평가 방법이랑 같아?", route="memory",
         gold=[("stm", "2021-12-23-test-evaluation-call"),
               ("mtm", "2021-11-10-test-evaluation-plan")],
         note="STM 합의 + MTM 정의"),
    dict(id="mix-severance-carryover", query="퇴직급여충당금이 이월 대상 잔액에 들어가?", route="memory",
         gold=[("stm", "2021-12-29-severance-note"),
               ("mtm", "2021-12-29-severance-provision-calc"),
               ("ltm", "3680eabe690e1893")],
         note="STM/MTM 미확정 + LTM 계상 기준"),
    dict(id="mix-cert-plan", query="제품 인증을 언제 어디서 받을 계획이야?", route="memory",
         gold=[("mtm", "2021-11-05-pct-hw-progress"),
               ("ltm", "b35c6c4630bdc743")],
         note="MTM 미착수 + LTM 코스텍"),
    dict(id="mix-pilot-schedule", query="시범 서비스 일정이 계획대로 가고 있어?", route="memory",
         gold=[("mtm", "2021-10-18-pilot-service-partner-meeting"),
               ("stm", "2021-12-28-pilot-service-call")],
         note="MTM 협의 + STM 학사 일정 제약"),
    dict(id="mix-org-role", query="가천대랑 피씨티가 각각 뭘 맡고 있어?", route="memory",
         gold=[("mtm", "2021-09-01-phase2-kickoff"),
               ("mtm", "2021-11-05-pct-hw-progress")],
         note="역할 분담"),

    # =========================================== 과제명을 부르는 질의 (project 필터)
    dict(id="proj-budget", query="스마트글래스 과제 예산 집행 상황이 어때?", route="memory",
         project=PROJECT,
         gold=[("mtm", "2021-12-14-budget-execution-review"),
               ("mtm", "2021-11-29-budget-burndown-plan")],
         note="과제명이 질의에 있다. analyzer가 project 필터를 잡아야 한다", core=True),
    dict(id="proj-risk", query="스마트글래스 과제 리스크가 뭐야?", route="memory",
         project=PROJECT,
         gold=[("mtm", "weekly-report-2021-12-w3"),
               ("mtm", "weekly-report-2021-12-w4")],
         note="과제명 + 리스크"),
    dict(id="proj-schedule", query="스마트글래스 과제 연차보고서 제출 일정 알려줘", route="memory",
         project=PROJECT,
         gold=[("mtm", "2021-11-29-lab-weekly-meeting"),
               ("mtm", "2021-12-27-annual-report-draft-v2")],
         note="과제명 + 일정"),
    dict(id="proj-today", query="오늘 스마트글래스 과제 회의에서 정한 게 뭐야?", route="memory",
         project=PROJECT,
         gold=[("stm", "2021-12-30-daily-summary"),
               ("stm", "2021-12-29-lowvision-meeting-summary")],
         note="과제명 + 최신성"),

    # ===================================================== 라우팅 (검색 불필요)
    dict(id="direct-concept-rag", query="RAG라는 게 일반적으로 무슨 개념이야?", route="direct",
         gold=[], note="일반 개념 질문. 조직 문서 검색 불필요"),
    dict(id="direct-brainstorm", query="아이디어 차원에서 브레인스토밍 좀 하자", route="direct",
         gold=[], note="검색 대상 없음"),
]


def build() -> int:
    cfg = AppConfig.load()
    if not cfg.ltm_corpus_path:
        print("LTM_CORPUS가 필요하다. LTM_CORPUS=auto 로 실행할 것.", file=sys.stderr)
        return 2
    corpus = LtmCorpus(cfg.ltm_corpus_path)

    # 계열 -> 문서 목록
    family_members: dict[str, list[str]] = defaultdict(list)
    for d in corpus.documents.values():
        family_members[d.family or d.title].append(d.doc_id)

    seed_ids = {
        tier: {p.stem for p in (SEED / tier).glob("*") if p.suffix in {".md", ".json"}}
        for tier in ("stm", "mtm")
    }

    rows, problems = [], []
    for case in CASES:
        gold: list[str] = []
        for tier, ident in case["gold"]:
            if tier == "ltm":
                doc = corpus.documents.get(ident)
                if doc is None:
                    problems.append(f"{case['id']}: 없는 doc_id {ident}")
                    continue
                # 계열 전체를 정답으로 넓힌다. 어느 버전이 대표로 와도 맞도록.
                for member in family_members[doc.family or doc.title]:
                    gold.append(f"ev_ltm_{member}")
            else:
                if ident not in seed_ids[tier]:
                    problems.append(f"{case['id']}: 없는 {tier} 파일 {ident}")
                    continue
                gold.append(f"ev_{tier}_{ident}")
        rows.append({
            "id": case["id"],
            "query": case["query"],
            "recent_turns": case.get("recent_turns", []),
            "expected": {
                "route": case["route"],
                # analyzer가 질의에서 뽑아내야 할 값이다. 코퍼스 라벨이 아니다.
                # 질의가 과제명을 부르지 않으면 None이 정답이다.
                "project": case.get("project"),
                "gold_evidence_ids": sorted(dict.fromkeys(gold)),
            },
            "note": case.get("note", ""),
            "core": bool(case.get("core", False)),
        })

    if problems:
        print("gold 라벨 오류:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_tier: Counter[str] = Counter()
    for case in CASES:
        tiers = {t for t, _ in case["gold"]}
        by_tier["+".join(sorted(tiers)) or "검색없음"] += 1
    n_gold = sum(len(r["expected"]["gold_evidence_ids"]) for r in rows)
    print(f"작성: {OUT}")
    print(f"  케이스 {len(rows)}건 / gold 근거 {n_gold}건 "
          f"(계열 확장 포함) / core {sum(1 for r in rows if r['core'])}건")
    for k, v in sorted(by_tier.items()):
        print(f"    {k:12s} {v:2d}건")
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
