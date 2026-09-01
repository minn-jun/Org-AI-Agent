"""seed 메모리 생성기.

memory_seed/ 를 통째로 다시 만든다. 손으로 파일을 늘리지 않고 스크립트로 두는 이유는
세 가지다.

1. 재현 가능해야 한다. 평가 수치는 코퍼스에 종속되므로, 코퍼스가 어떻게 만들어졌는지
   추적할 수 없으면 수치도 해석할 수 없다.
2. 설계 의도를 코드로 남긴다. "왜 이 문서가 있는가"가 주석과 자료구조에 드러난다.
3. 규모를 쉽게 바꾼다. 문서가 적으면 검색기 차이가 드러나지 않는다.

    python scripts/build_seed_memory.py            # memory_seed 재생성
    python scripts/build_seed_memory.py --dry-run  # 생성될 목록만 출력

## 코퍼스 설계 원칙

- **계층 의미를 지킨다**
  STM은 휘발성 최신 맥락, MTM은 수정 중인 산출물, LTM은 승인된 기준이다.
  같은 사실이 세 계층에 다른 값으로 존재할 수 있어야 "충돌 감지" 질문이 성립한다.

- **방해 문서를 일부러 넣는다**
  정답만 있는 코퍼스에서는 정밀도가 항상 높게 나온다. 제목과 주제가 비슷하지만
  답이 아닌 문서(다른 과제의 같은 유형, 같은 과제의 옛 버전)를 넣어야
  정밀도가 의미를 갖는다.

- **과제 수를 늘린다**
  프로젝트 필터가 실제로 일을 하는지 보려면 경쟁 과제가 있어야 한다.

- **의도적 충돌 쌍을 심는다**
  MTM의 내부 목표일과 LTM의 공식 마일스톤이 어긋나게 만든다.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED_ROOT = PROJECT_ROOT / "memory_seed"


# ------------------------------------------------------------------- 조사 처리
# 생성기가 이름과 날짜를 문장에 끼워 넣으므로 조사를 자동으로 골라야 한다.
# "박주임가", "한연구원가" 같은 오류가 실제로 나왔다.


def _has_batchim(word: str) -> bool | None:
    """마지막 글자에 받침이 있는지. 판단할 수 없으면 None."""
    if not word:
        return None
    ch = word[-1]
    if "가" <= ch <= "힣":
        return (ord(ch) - 0xAC00) % 28 != 0
    if ch.isdigit():
        # 숫자를 한국어로 읽었을 때 받침이 있는 것: 영 일 삼 육 칠 팔
        return ch in "013678"
    return None


def _josa(word: str, with_batchim: str, without_batchim: str) -> str:
    has = _has_batchim(word)
    return f"{word}{without_batchim if has is False or has is None else with_batchim}"


def ga(word: str) -> str:
    return _josa(word, "이", "가")


def eun(word: str) -> str:
    return _josa(word, "은", "는")


def eul(word: str) -> str:
    return _josa(word, "을", "를")


@dataclass(frozen=True)
class Project:
    key: str
    name: str
    topic: str
    lead: str
    members: tuple[str, ...]
    kickoff: str
    official_review: str      # LTM 공식 계획서의 내부검토 마일스톤
    official_due: str         # LTM 공식 계획서의 최종 제출일
    internal_target: str      # MTM 수정 일정표의 내부 목표일 (공식과 어긋남)
    risks: tuple[str, ...]
    budget_notes: tuple[str, ...] = field(default_factory=tuple)


PROJECTS: tuple[Project, ...] = (
    Project(
        key="a", name="A 과제", topic="저시력자 보조 스마트글래스",
        lead="이과장", members=("김대리", "박주임"),
        kickoff="2026-06-03", official_review="2026-08-23", official_due="2026-08-30",
        internal_target="2026-08-21",
        risks=("일정표 확정 지연", "예산 산정 근거 부족", "기관 협의 일정 미확정"),
        budget_notes=("외부 자문비 근거 부족", "운영비 단가 기준 불명확"),
    ),
    Project(
        key="b", name="B 과제", topic="다기관 의료데이터 분석",
        lead="최선임", members=("한연구원", "박주임"),
        kickoff="2026-06-10", official_review="2026-08-19", official_due="2026-09-05",
        internal_target="2026-08-26",
        risks=("데이터 제공 범위 확정 지연", "기관별 누락값 처리 기준 상이"),
        budget_notes=("기관별 데이터 이용료 편차",),
    ),
    Project(
        key="c", name="C 과제", topic="스마트팩토리 설비 예지보전",
        lead="윤책임", members=("오연구원", "김대리"),
        kickoff="2026-06-17", official_review="2026-08-25", official_due="2026-09-11",
        internal_target="2026-08-28",
        risks=("설비 로그 수집 권한 지연", "현장 실증 일정 변경"),
        budget_notes=("센서 장비 단가 상승", "현장 출장비 증가"),
    ),
    Project(
        key="d", name="D 과제", topic="고령자 돌봄 로봇 실증",
        lead="정팀장", members=("한연구원", "오연구원"),
        kickoff="2026-05-27", official_review="2026-08-14", official_due="2026-08-28",
        internal_target="2026-08-18",
        risks=("실증 기관 참여자 모집 부진", "안전성 검토 절차 지연"),
        budget_notes=("실증 참여자 사례비 기준 필요",),
    ),
    Project(
        key="e", name="E 과제", topic="재난안전 영상분석",
        lead="최선임", members=("오연구원",),
        kickoff="2026-07-01", official_review="2026-09-02", official_due="2026-09-20",
        internal_target="2026-08-31",
        risks=("영상 데이터 라벨링 인력 부족",),
    ),
)

# ---------------------------------------------------------------- LTM 기준 문서
# 조직 전반에 적용되는 승인 문서. 프로젝트 필터가 "공통"이라 모든 질문의 후보에 오른다.
# 그래서 방해 문서 역할도 겸한다.
STANDARDS: tuple[tuple[str, str, str, str, str, tuple[str, ...]], ...] = (
    ("budget-estimation-standard", "standard", "예산 산정 기준", "2026-06-20",
     "예산 산정은 기존 단가표, 유사 사업 사례, 산식 근거를 함께 제시해야 하며 외부 자문비와 운영비는 출처를 명시해야 한다.",
     ("인건비와 운영비는 기존 단가표를 우선 적용한다.",
      "외부 자문비는 유사 사업 사례와 단가표를 함께 비교해 산정한다.",
      "모든 예산 항목에는 산식, 단가, 수량, 출처를 명시한다.",
      "출처가 없는 항목은 보완 대상으로 표시한다.")),
    ("proposal-writing-standard", "standard", "제안서 작성 기준", "2026-06-01",
     "제안서는 추진 배경, 목표, 수행 체계, 일정, 예산 산정 근거, 기대효과를 포함해야 하며 일정과 예산에는 근거 문서를 표시해야 한다.",
     ("필수 항목은 추진 배경, 목표, 수행 체계, 일정, 예산 산정 근거, 기대효과다.",
      "일정과 예산에는 근거 문서 이름을 함께 적는다.",
      "초안은 내부 검토를 거친 뒤 수정본으로 승격한다.")),
    ("org-reporting-guideline", "guideline", "보고서 작성 지침", "2026-05-15",
     "보고서는 결정사항, 미해결 쟁점, 다음 action item, 출처 문서를 구분해 작성한다.",
     ("결정사항과 논의 중 사항을 반드시 구분해 적는다.",
      "미해결 쟁점에는 담당자와 기한을 함께 적는다.",
      "인용한 문서는 출처 항목에 모아 적는다.")),
    ("document-authority-standard", "standard", "문서 권위 판단 기준", "2026-07-15",
     "동일 주제의 문서가 충돌하면 승인된 최종 문서, 검토 완료 문서, 진행 중 초안, 대화 메모 순으로 권위를 판단하며 최신 개정 관계를 함께 확인한다.",
     ("권위 순서는 승인 최종본, 검토 완료본, 진행 중 초안, 대화 메모 순이다.",
      "같은 등급이면 최신 개정본을 우선한다.",
      "충돌이 확인되면 상위 문서 기준을 따르고 하위 문서에 변경 사유를 남긴다.")),
    ("knowledge-promotion-policy", "policy", "지식 승격 정책", "2026-07-01",
     "MTM 문서는 확정성, 재사용성, 대표성, 최신성, 권한성, 출처성이 확인되면 담당자 승인 후 LTM으로 승격할 수 있다.",
     ("승격 판단 요소는 확정성, 재사용성, 대표성, 최신성, 권한성, 출처성이다.",
      "승격은 담당자 승인 후에만 이루어진다.",
      "승격된 문서는 원본 MTM 문서에 승격 사실을 표시한다.")),
    ("data-handling-guideline", "guideline", "데이터 취급 지침", "2026-07-20",
     "여러 기관의 데이터를 분석할 때 공통 전처리 기준, 누락값 처리 방식, 분석 버전, 검토자를 기록한 뒤 결과를 공유해야 한다.",
     ("기관별 데이터는 공통 전처리 기준을 적용한 뒤 비교한다.",
      "누락값 처리 방식은 분석 보고서에 명시한다.",
      "분석 버전과 검토자를 함께 기록한다.")),
    ("access-control-standard", "standard", "문서 접근 권한 기준", "2026-06-05",
     "문서는 공개 범위에 따라 전사, 부서, 과제팀, 담당자 한정으로 구분하며 과제팀 문서는 외부 공유 전 책임자 승인이 필요하다.",
     ("공개 범위는 전사, 부서, 과제팀, 담당자 한정 네 단계다.",
      "과제팀 이상 등급 문서의 외부 공유는 책임자 승인이 필요하다.",
      "권한 등급은 문서 생성 시 지정하고 변경 이력을 남긴다.")),
    ("external-sharing-standard", "standard", "외부 공유 기준", "2026-06-25",
     "외부 기관에 자료를 제공할 때는 제공 목적, 범위, 보관 기간, 파기 방법을 문서로 합의한 뒤 전달한다.",
     ("제공 목적과 범위를 문서로 합의한 뒤 전달한다.",
      "보관 기간과 파기 방법을 명시한다.",
      "제공 이력은 과제 산출물 목록에 기록한다.")),
    ("schedule-change-procedure", "procedure", "일정 변경 절차", "2026-06-12",
     "공식 마일스톤을 변경하려면 변경 사유, 영향 범위, 대안 일정을 정리해 책임자 승인을 받아야 하며 내부 목표일 조정은 팀 내 공유로 갈음한다.",
     ("공식 마일스톤 변경은 책임자 승인을 받는다.",
      "변경 사유, 영향 범위, 대안 일정을 함께 제출한다.",
      "내부 목표일 조정은 팀 내 공유로 갈음하되 공식 일정과의 차이를 표시한다.")),
    ("meeting-operation-guideline", "guideline", "회의 운영 지침", "2026-05-20",
     "회의는 안건과 목표를 사전 공유하고 종료 시 결정사항과 action item을 담당자, 기한과 함께 정리한다.",
     ("안건과 목표를 회의 전에 공유한다.",
      "종료 시 결정사항과 action item을 정리한다.",
      "action item에는 담당자와 기한을 반드시 적는다.")),
    ("deliverable-management-standard", "standard", "산출물 관리 기준", "2026-06-28",
     "과제 산출물은 유형, 버전, 담당자, 승인 상태를 표기하고 최종본은 별도 목록으로 관리한다.",
     ("산출물에는 유형, 버전, 담당자, 승인 상태를 표기한다.",
      "최종본은 별도 목록으로 관리한다.",
      "폐기된 버전은 목록에서 제외하되 이력은 남긴다.")),
    ("risk-escalation-procedure", "procedure", "이슈 보고 절차", "2026-07-08",
     "일정 지연이 예상되거나 예산 초과가 발생하면 발견 즉시 과제책임자에게 보고하고 주간 보고에 리스크로 등재한다.",
     ("지연이나 초과가 예상되면 즉시 과제책임자에게 보고한다.",
      "주간 보고에 리스크 항목으로 등재한다.",
      "해소 시 종결 사유를 함께 기록한다.")),
    ("contract-management-standard", "standard", "용역 계약 관리 기준", "2026-06-15",
     "외주 용역은 과업 범위, 산출물, 검수 기준, 대가 지급 조건을 계약서에 명시하고 검수 결과를 문서로 남긴다.",
     ("과업 범위와 산출물을 계약서에 명시한다.",
      "검수 기준과 대가 지급 조건을 함께 정한다.",
      "검수 결과는 문서로 남긴다.")),
    ("staffing-guideline", "guideline", "인력 투입 지침", "2026-05-25",
     "과제별 참여율은 월 단위로 관리하며 동일 인력의 총 참여율이 100퍼센트를 넘지 않도록 조정한다.",
     ("참여율은 월 단위로 관리한다.",
      "동일 인력의 총 참여율은 100퍼센트를 넘을 수 없다.",
      "변경 시 과제책임자 간 협의를 거친다.")),
    ("quality-review-guideline", "guideline", "품질 검토 지침", "2026-07-11",
     "산출물은 작성자 외 1인 이상의 검토를 거치며 검토 의견과 반영 여부를 기록한다.",
     ("작성자 외 1인 이상이 검토한다.",
      "검토 의견과 반영 여부를 기록한다.",
      "미반영 의견은 사유를 남긴다.")),
    ("equipment-management-guideline", "guideline", "연구장비 관리 지침", "2026-06-08",
     "연구장비는 대장에 등록하고 사용 전후 점검 기록을 남기며 고장 발생 시 즉시 관리자에게 통보한다.",
     ("장비는 대장에 등록한다.",
      "사용 전후 점검 기록을 남긴다.",
      "고장 발생 시 즉시 관리자에게 통보한다.")),
    ("travel-expense-guideline", "guideline", "출장비 처리 지침", "2026-05-18",
     "출장은 사전 신청 후 승인받고 복귀 후 5일 이내에 결과 보고와 증빙을 제출한다.",
     ("출장은 사전 신청과 승인이 필요하다.",
      "복귀 후 5일 이내 결과 보고와 증빙을 제출한다.",
      "증빙이 없는 항목은 정산 대상에서 제외한다.")),
    ("audit-response-procedure", "procedure", "감사 대응 절차", "2026-07-25",
     "감사 요청 자료는 원본과 사본을 구분해 제출하고 제출 목록과 담당자를 기록으로 남긴다.",
     ("원본과 사본을 구분해 제출한다.",
      "제출 목록과 담당자를 기록으로 남긴다.",
      "추가 요청은 과제책임자를 경유한다.")),
    ("training-guideline", "guideline", "연구원 교육 지침", "2026-05-11",
     "신규 참여 연구원은 과제 착수 후 2주 이내에 보안 교육과 데이터 취급 교육을 이수한다.",
     ("착수 후 2주 이내 보안 교육을 이수한다.",
      "데이터 취급 교육을 함께 이수한다.",
      "이수 여부는 과제책임자가 확인한다.")),
    ("version-naming-standard", "standard", "문서 버전 표기 기준", "2026-06-02",
     "문서 버전은 초안, 검토본, 승인본 순으로 표기하며 파일명에 날짜와 버전을 함께 적는다.",
     ("버전은 초안, 검토본, 승인본 순으로 표기한다.",
      "파일명에 날짜와 버전을 함께 적는다.",
      "승인본 이후 변경은 새 버전으로 만든다.")),
)


def frontmatter(meta: dict[str, str]) -> str:
    lines = ["---"]
    for key, value in meta.items():
        lines.append(f"{key}: {value}")
    lines.append("---")
    return "\n".join(lines)


def md_doc(meta: dict[str, str], title: str, sections: list[tuple[str, list[str]]]) -> str:
    body = [frontmatter(meta), "", f"# {title}", ""]
    for heading, bullets in sections:
        body.append(f"## {heading}")
        body.append("")
        body.extend(f"- {item}" for item in bullets)
        body.append("")
    return "\n".join(body)


def build_ltm() -> dict[str, str]:
    docs: dict[str, str] = {}

    for slug, source_type, title, date, summary, rules in STANDARDS:
        docs[f"{slug}.md"] = md_doc(
            {
                "memory_tier": "ltm", "source_type": source_type, "project": "공통",
                "title": title, "date": date, "permission_scope": "company_internal",
                "status": "approved", "summary": summary,
            },
            title,
            [("적용 원칙", list(rules)),
             ("예외 처리", ["예외가 필요하면 사유와 승인자를 문서에 남긴다."])],
        )

    for project in PROJECTS:
        title = f"{project.name} 최종 승인 계획서"
        docs[f"{project.key}-project-final-plan.md"] = md_doc(
            {
                "memory_tier": "ltm", "source_type": "final_plan", "project": project.name,
                "title": title, "date": project.kickoff,
                "permission_scope": "company_internal", "status": "approved",
                "summary": (
                    f"{project.name} 최종 승인 계획서의 공식 마일스톤은 "
                    f"내부 검토 완료({project.official_review})와 최종 제출({project.official_due}) 두 가지다."
                ),
            },
            title,
            [
                ("과제 개요", [f"주제는 {project.topic}이다.",
                              f"과제책임자는 {project.lead}.",
                              f"참여 연구원은 {', '.join(project.members)}이다."]),
                ("공식 마일스톤", [f"착수 {project.kickoff}",
                                  f"내부 검토 완료 {project.official_review}",
                                  f"최종 제출 {project.official_due}"]),
                ("변경 원칙", ["공식 마일스톤 변경은 일정 변경 절차에 따라 승인을 받는다."]),
            ],
        )
    return docs


def build_mtm() -> dict[str, str]:
    docs: dict[str, str] = {}
    for project in PROJECTS:
        key, name = project.key, project.name

        # 킥오프 회의록
        docs[f"{project.kickoff}-{key}-project-kickoff.md"] = md_doc(
            {"memory_tier": "mtm", "source_type": "meeting_note", "project": name,
             "title": f"{project.kickoff} {name} 킥오프 회의록", "date": project.kickoff,
             "permission_scope": "project_team", "status": "reviewed",
             "summary": f"{name} 킥오프에서 과제 범위, 담당자 역할, 초기 일정을 정리했다."},
            f"{project.kickoff} {name} 킥오프 회의록",
            [("결정사항", [f"과제 주제를 {project.topic}으로 확정했다.",
                          f"과제책임자는 {project.lead}이 맡는다.",
                          f"참여 연구원은 {', '.join(project.members)}이다."]),
             ("다음 일정", [f"내부 검토 목표는 {project.internal_target}이다."])],
        )

        # 정기 회의록 (방해 문서 겸용. 주제는 비슷하지만 답이 아닌 경우가 많다)
        for offset, date in enumerate(("2026-07-08", "2026-07-22", "2026-08-05")):
            docs[f"{date}-{key}-project-meeting.md"] = md_doc(
                {"memory_tier": "mtm", "source_type": "meeting_note", "project": name,
                 "title": f"{date} {name} 정기 회의록", "date": date,
                 "permission_scope": "project_team", "status": "reviewed",
                 "summary": f"{date} {name} 정기 회의에서 진행 현황과 다음 작업을 공유했다."},
                f"{date} {name} 정기 회의록",
                [("논의", [f"{project.topic} 진행 현황을 공유했다.",
                          f"{project.risks[offset % len(project.risks)]} 관련 대응을 논의했다."]),
                 ("다음 작업", [f"{ga(project.members[0])} 다음 회의까지 진행 자료를 준비한다."])],
            )

        # 제안서 초안과 수정본 (같은 과제의 옛 버전 = 대표적인 방해 문서)
        docs[f"2026-08-10-{key}-project-proposal-draft.md"] = md_doc(
            {"memory_tier": "mtm", "source_type": "proposal_draft", "project": name,
             "title": f"{name} 제안서 초안", "date": "2026-08-10",
             "permission_scope": "project_team", "status": "draft",
             "summary": f"{name} 제안서 초안에는 내부 검토 완료일을 {project.internal_target} 기준으로 기재했다."},
            f"{name} 제안서 초안",
            [("구성", ["추진 배경, 목표, 수행 체계, 일정, 예산 산정 근거를 담았다."]),
             ("미완 항목", ["예산 산정 근거와 기대효과가 아직 비어 있다."])],
        )
        docs[f"2026-08-23-{key}-project-proposal-v2.md"] = md_doc(
            {"memory_tier": "mtm", "source_type": "proposal_draft", "project": name,
             "title": f"{name} 제안서 수정본", "date": "2026-08-23",
             "permission_scope": "project_team", "status": "in_review",
             "summary": (f"{name} 제안서 수정본은 내부 검토 마일스톤을 완료했고 "
                         f"최종 제출({project.official_due})을 남겨두고 있다.")},
            f"{name} 제안서 수정본",
            [("반영 내용", ["초안에서 지적된 예산 근거와 기대효과를 채웠다."]),
             ("남은 일정", [f"최종 제출 {project.official_due}"])],
        )

        # 예산 검토
        docs[f"2026-08-18-{key}-project-budget-review.md"] = md_doc(
            {"memory_tier": "mtm", "source_type": "budget_review", "project": name,
             "title": f"{name} 예산 검토 메모", "date": "2026-08-18",
             "permission_scope": "project_team", "status": "draft",
             "summary": (f"{name} 예산 검토에서 확인된 보완 항목은 다음과 같다. " +
                         (" / ".join(project.budget_notes) if project.budget_notes
                          else "항목별 산정 근거 보완"))},
            f"{name} 예산 검토 메모",
            [("검토 결과", list(project.budget_notes) or ["항목별 산정 근거를 보완해야 한다."]),
             ("보완 필요", ["항목별 출처 표시", "산식 표 추가"])],
        )

        # 리스크 보고
        docs[f"2026-08-18-{key}-project-risk-report.md"] = md_doc(
            {"memory_tier": "mtm", "source_type": "risk_report", "project": name,
             "title": f"{name} 리스크 보고", "date": "2026-08-18",
             "permission_scope": "project_team", "status": "draft",
             "summary": f"{name}의 주요 리스크는 {project.risks[0]}이다."},
            f"{name} 리스크 보고",
            [("식별된 리스크", list(project.risks)),
             ("대응", [f"{ga(project.lead)} 주간 보고에 리스크로 등재한다."])],
        )

        # 수정 일정표 - LTM 공식 마일스톤과 어긋나게 만든 충돌 쌍
        docs[f"2026-08-19-{key}-project-revised-timeline.md"] = md_doc(
            {"memory_tier": "mtm", "source_type": "revised_timeline", "project": name,
             "title": f"{name} 수정 일정표", "date": "2026-08-19",
             "permission_scope": "project_team", "status": "draft",
             "summary": (f"수정 일정표는 내부 목표 검토일을 {project.internal_target} 기준으로 두고, "
                         f"공식 마일스톤({project.official_review})과 "
                         f"최종 제출일({project.official_due})을 함께 표기했다.")},
            f"{name} 수정 일정표",
            [("내부 목표", [f"내부 검토 목표일 {project.internal_target}"]),
             ("공식 기준", [f"공식 내부 검토 완료 {project.official_review}",
                           f"공식 최종 제출 {project.official_due}"]),
             ("차이", ["내부 목표일이 공식 마일스톤보다 앞서 있어 진행 상황을 함께 확인해야 한다."])],
        )

        # 기관 협의 안건
        docs[f"2026-08-19-{key}-project-institution-agenda.md"] = md_doc(
            {"memory_tier": "mtm", "source_type": "meeting_agenda", "project": name,
             "title": f"{name} 기관 협의 안건", "date": "2026-08-19",
             "permission_scope": "project_team", "status": "draft",
             "summary": f"{name} 기관 협의 안건 초안에는 과제 범위, 데이터 제공 가능성, 예산 항목 검토가 포함되었다."},
            f"{name} 기관 협의 안건",
            [("안건", ["과제 범위 확인", "데이터 제공 가능성", "예산 항목 검토"]),
             ("준비 자료", ["제안서 수정본", "수정 일정표"])],
        )

        # 진행 분석 보고
        docs[f"2026-08-23-{key}-project-progress-report.md"] = md_doc(
            {"memory_tier": "mtm", "source_type": "analysis_report", "project": name,
             "title": f"{name} 진행 분석 보고", "date": "2026-08-23",
             "permission_scope": "project_team", "status": "in_review",
             "summary": f"{name} 진행 분석에서 현재 공정률과 남은 작업을 정리했다."},
            f"{name} 진행 분석 보고",
            [("현황", [f"{project.topic} 관련 주요 작업이 진행 중이다."]),
             ("남은 작업", [f"최종 제출일({project.official_due}) 전까지 산출물 정리가 필요하다."])],
        )

    # 주간 보고 (여러 과제를 함께 언급해 프로젝트 필터의 방해 요인이 된다)
    for week, date in enumerate(("2026-08-02", "2026-08-09", "2026-08-16", "2026-08-23"), start=1):
        docs[f"weekly-report-2026-08-w{week}.md"] = md_doc(
            {"memory_tier": "mtm", "source_type": "weekly_report", "project": "공통",
             "title": f"2026년 8월 {week}주차 주간 보고", "date": date,
             "permission_scope": "company_internal", "status": "draft",
             "summary": f"8월 {week}주차 보고에서는 과제별 진행 현황과 주요 리스크를 정리했다."},
            f"2026년 8월 {week}주차 주간 보고",
            [("과제별 현황", [f"{p.name}: {p.risks[0]}" for p in PROJECTS]),
             ("공통 사항", ["산출물 목록 정리와 예산 근거 보완이 공통 과제로 남아 있다."])],
        )
    return docs


def build_stm() -> dict[str, str | dict]:
    docs: dict[str, str | dict] = {}
    for project in PROJECTS:
        key, name = project.key, project.name

        docs[f"2026-08-18-{key}-project-action-items.json"] = {
            "memory_tier": "stm", "source_type": "action_items", "project": name,
            "title": f"2026-08-18 {name} action item", "date": "2026-08-18",
            "permission_scope": "project_team",
            "summary": f"{name} 회의 후속 action item 목록이다.",
            "action_items": [
                {"owner": project.members[0], "task": "제안서 초안 수정", "due": "2026-08-21"},
                {"owner": project.lead, "task": "예산 산정 근거 확인", "due": "2026-08-20"},
            ],
        }
        docs[f"2026-08-18-{key}-project-meeting-summary.json"] = {
            "memory_tier": "stm", "source_type": "meeting_summary", "project": name,
            "title": f"2026-08-18 {name} 회의 요약", "date": "2026-08-18",
            "permission_scope": "project_team",
            "summary": (f"오늘 회의에서 {name} 제안서 초안 검토 기한을 2026-08-21로 정했다. "
                        f"{ga(project.members[0])} 제안서 초안 수정, {ga(project.lead)} 예산 산정 근거 확인을 맡기로 했다."),
            "decisions": [f"제안서 초안 검토 기한 2026-08-21",
                          f"담당자 {project.members[0]}, {project.lead}"],
        }
        docs[f"2026-08-19-{key}-project-budget-followup.json"] = {
            "memory_tier": "stm", "source_type": "action_items", "project": name,
            "title": f"2026-08-19 {name} 예산 후속 조치", "date": "2026-08-19",
            "permission_scope": "project_team",
            "summary": (f"{name} 예산 관련 후속 조치로 {ga(project.lead)} 기존 단가표와 최근 공고 기준을 비교하고, "
                        f"{ga(project.members[-1])} 외부 자문비 산정 근거를 정리하기로 했다."),
            "action_items": [
                {"owner": project.lead, "task": "기존 단가표와 최근 공고 기준 비교", "due": "2026-08-20"},
                {"owner": project.members[-1], "task": "외부 자문비 산정 근거 정리", "due": "2026-08-21"},
            ],
        }
        docs[f"2026-08-24-{key}-project-daily-summary.json"] = {
            "memory_tier": "stm", "source_type": "daily_conversation_summary", "project": name,
            "title": f"2026-08-24 {name} 대화 요약", "date": "2026-08-24",
            "permission_scope": "project_team", "status": "active",
            "summary": f"오늘 {name} 대화에서는 수정 제안서 검토, 기관 협의 자료 준비, 예산 근거 확인 순서로 진행하기로 했다.",
            "topics": ["수정 제안서 검토", "기관 협의 자료 준비", "예산 근거 확인"],
        }
        docs[f"2026-08-19-{key}-project-institution-call.md"] = md_doc(
            {"memory_tier": "stm", "source_type": "call_note", "project": name,
             "title": f"2026-08-19 {name} 기관 협의 통화 메모", "date": "2026-08-19",
             "permission_scope": "project_team", "status": "active",
             "summary": f"{name} 기관 협의 통화에서 8월 25일 오후 2시에 실무 협의를 진행하는 안이 제안되었다."},
            f"2026-08-19 {name} 기관 협의 통화 메모",
            [("통화 내용", ["8월 25일 오후 2시 실무 협의 제안",
                           "기관 측 참석자는 담당 팀장과 실무자 2명"]),
             ("확인 필요", ["내부 일정 확인 후 회신 예정"])],
        )
        docs[f"2026-08-19-{key}-project-director-note.md"] = md_doc(
            {"memory_tier": "stm", "source_type": "manager_note", "project": name,
             "title": f"2026-08-19 {name} 팀장 지시 메모", "date": "2026-08-19",
             "permission_scope": "project_team", "status": "active",
             "summary": f"팀장은 {name} 초안에서 일정 변경 사유와 공식 계획서 기준일을 함께 표시하라고 요청했다."},
            f"2026-08-19 {name} 팀장 지시 메모",
            [("지시 사항", ["일정 변경 사유를 문서에 남길 것",
                           "공식 계획서 기준일을 함께 표시할 것"])],
        )
    docs["today-conversation-snippet.md"] = md_doc(
        {"memory_tier": "stm", "source_type": "conversation_snippet", "project": "공통",
         "title": "2026-08-18 대화 스니펫", "date": "2026-08-18",
         "permission_scope": "project_team", "status": "archived",
         "summary": "2026년 8월 18일 대화에서는 제안서 검토 기한, 담당자, 예산 근거 확인이 주로 논의되었다."},
        "2026-08-18 대화 스니펫",
        [("논의", ["제안서 검토 기한", "담당자 배정", "예산 근거 확인"])],
    )
    return docs


def write(docs: dict, tier_dir: Path, dry_run: bool) -> int:
    for filename, content in docs.items():
        if dry_run:
            print(f"  {tier_dir.name}/{filename}")
            continue
        path = tier_dir / filename
        if isinstance(content, dict):
            path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            path.write_text(content, encoding="utf-8")
    return len(docs)


def main() -> int:
    parser = argparse.ArgumentParser(description="seed 메모리 재생성")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--root", type=Path, default=SEED_ROOT)
    args = parser.parse_args()

    tiers = {"stm": build_stm(), "mtm": build_mtm(), "ltm": build_ltm()}
    if not args.dry_run:
        for tier in tiers:
            tier_dir = args.root / tier
            if tier_dir.exists():
                shutil.rmtree(tier_dir)
            tier_dir.mkdir(parents=True)

    total = 0
    for tier, docs in tiers.items():
        count = write(docs, args.root / tier, args.dry_run)
        total += count
        print(f"{tier.upper()}: {count}건")
    print(f"합계: {total}건 (과제 {len(PROJECTS)}개, 공통 기준 {len(STANDARDS)}건)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
