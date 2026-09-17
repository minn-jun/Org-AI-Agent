"""1회용: memory_store.py 동의어 사전을 설정 파일로 빼고, 검색기 테스트를 메타데이터 기반으로 바꾼다."""
from pathlib import Path

APP = Path(r"C:\ine_project\중기청_조직지식AI플랫폼\org_agent_mvp")

# ------------------------------------------------------------------ memory_store.py
p = APP / "org_agent_mvp" / "memory_store.py"
s = p.read_text(encoding="utf-8")
start = s.index("# 키는 부분 문자열로 매칭한다.")
end = s.index("#: retrieve()가 한 번에 돌려줄 수 있는 최대 건수.")
new_block = '''#: 질의 확장 동의어 사전. 조직마다 업무 용어가 달라 코드가 아니라 설정 파일에 둔다.
#:
#:   QUERY_EXPANSIONS=<경로>   다른 사전 파일을 쓴다
#:   QUERY_EXPANSIONS=none     확장하지 않는다
#:   (미지정)                  config/query_expansions.json. 파일이 없으면 확장하지 않는다
#:
#: 키는 부분 문자열로 매칭한다. 한국어는 조사가 붙어 토큰이 달라지므로
#: ("예산이" != "예산") 정확 일치로 조회하면 확장이 거의 동작하지 않는다.
DEFAULT_EXPANSIONS_PATH = Path(__file__).resolve().parents[1] / "config" / "query_expansions.json"
_EXPANSIONS_CACHE: dict[str, dict[str, str]] = {}


def query_expansions() -> dict[str, str]:
    raw = os.environ.get("QUERY_EXPANSIONS", "").strip()
    if raw not in _EXPANSIONS_CACHE:
        if raw.lower() in {"none", "off", "0"}:
            table: dict[str, str] = {}
        else:
            target = Path(raw) if raw else DEFAULT_EXPANSIONS_PATH
            table = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
        _EXPANSIONS_CACHE[raw] = table
    return _EXPANSIONS_CACHE[raw]


'''
s = s[:start] + new_block + s[end:]
reps = [
    ("import json\nimport re\n", "import json\nimport os\nimport re\n"),
    ("expansions = [value for key, value in QUERY_EXPANSIONS.items() if key in lowered]",
     "expansions = [value for key, value in query_expansions().items() if key in lowered]"),
    ("            projects.add(self.ltm_corpus.project)\n",
     "            projects.update(self.ltm_corpus.projects())\n"),
    ("확장 결과에는 같은 토큰이 여러 번 들어온다. QUERY_EXPANSIONS의 값이",
     "확장 결과에는 같은 토큰이 여러 번 들어온다. 동의어 사전의 값이"),
]
for a, b in reps:
    assert s.count(a) == 1, a
    s = s.replace(a, b)
p.write_text(s, encoding="utf-8")
print("memory_store.py ok")

# ------------------------------------------------------------------ tests
t = APP / "tests" / "test_retriever_modules.py"
s = t.read_text(encoding="utf-8")
fs = s.index("class FamilyKeyTests(unittest.TestCase):")
fe = s.index("# ------------------------------------------------------------- ltm_corpus 동작")
s = s[:fs] + s[fe:]

NEW_SOURCE_TYPE_TESTS = '''    def test_source_types_come_from_document_meta(self) -> None:
        corpus = self.load([
            chunk("a", "협약서", "본문", doc_type="agreement"),
            chunk("b", "지시서", "본문", doc_type="patent_document"),
        ])
        self.assertEqual(corpus.source_types(), ["agreement", "patent_document"])

    def test_retriever_knows_no_corpus_rules(self) -> None:
        """메타데이터가 없으면 폴더 이름·파일명 꼬리표로 추정하지 않는다.

        2026-09-15에 20200504 과제 폴더 전용 규칙을 코퍼스 전처리로 옮겼다.
        폴더가 "04_협약"이어도 doc_type이 없으면 기본값이고, "_v3"/"_v9"도 접지 않는다.
        """
        corpus = self.load([
            chunk("v3", "차단계 사업계획서_v3", "총사업비 1,346,270천원", source_path="1단계/04_협약/a.hwp"),
            chunk("v9", "차단계 사업계획서_v9", "총사업비 1,346,270천원 변경", source_path="06_최종제출/b.hwp"),
        ])
        self.assertEqual(corpus.source_types(), [DEFAULT_DOC_TYPE])
        self.assertEqual(corpus.projects(), [])
        self.assertEqual(len(corpus.search(["총사업비"], top_k=5)), 2)

    def test_source_type_is_never_a_file_format(self) -> None:
        """source_type이 pdf/hwp가 되면 analyzer enum이 형식 이름으로 오염된다."""
        corpus = self.load([chunk("a", "문서", "본문", source_path="a/b.pdf")])
        self.assertNotIn(corpus.source_types()[0], {"pdf", "hwp", "pptx"})

    def test_final_document_beats_higher_version_as_representative(self) -> None:
        """is_final(실제 제출본)은 파일명 버전 번호보다 강한 신호다."""
        corpus = self.load([
            chunk("v4", "신청용 계획서_v4", "총사업비 내용", version_group="신청용 계획서", version_rank=[0, 4, 0]),
            chunk("sub", "신청용 계획서", "총사업비 내용 제출", version_group="신청용 계획서", is_final=True),
        ])
        _, card = corpus.search(["총사업비"], top_k=5)[0]
        self.assertEqual(card["source_ref"]["document_id"], "sub")

    def test_document_meta_file_overrides_chunk_metadata(self) -> None:
        """문서 필드는 chunks.jsonl 옆 document_meta.jsonl에 둘 수 있고, 청크 metadata보다 우선한다."""
        write_corpus(self.path, [chunk("a", "협약서", "총사업비", doc_type="agreement", project="A 과제")])
        (self.path.parent / "document_meta.jsonl").write_text(
            json.dumps({"doc_id": "a", "doc_type": "official_report", "project": "B 과제"}, ensure_ascii=False) + "\\n",
            encoding="utf-8",
        )
        corpus = LtmCorpus(self.path)
        self.assertEqual(corpus.source_types(), ["official_report"])
        self.assertEqual(corpus.projects(), ["B 과제"])
        _, card = corpus.search(["총사업비"], top_k=1)[0]
        self.assertEqual(card["project"], "B 과제")

    def test_project_filter_uses_each_documents_project(self) -> None:
        corpus = self.load([
            chunk("a", "계획서 가", "총사업비 내용", project="A 과제"),
            chunk("b", "계획서 나", "총사업비 내용 추가", project="B 과제"),
        ])
        results = corpus.search(["총사업비"], filters={"project": "A 과제"}, filter_penalty=0.0, top_k=5)
        self.assertEqual([card["source_ref"]["document_id"] for _, card in results], ["a"])
'''

OLD_SOURCE_TYPE_TEST = '''    def test_source_types_come_from_folders(self) -> None:
        corpus = self.load([
            chunk("a", "협약서", "본문", source_path="1단계/04_협약/협약서.hwp"),
            chunk("b", "지시서", "본문", source_path="2단계/07_특허/지시서.hwp"),
        ])
        self.assertEqual(corpus.source_types(), ["agreement", "patent_document"])
'''

GROUP = 'version_group="차단계 사업계획서"'
reps = [
    ("from org_agent_mvp.ltm_corpus import (\n    LtmCorpus,\n    classify,\n    family_key,\n    version_rank,\n)\n",
     "from org_agent_mvp.ltm_corpus import DEFAULT_DOC_TYPE, LtmCorpus\n"),
    ('def chunk(source_id: str, title: str, text: str, index: int = 1,\n'
     '          source_path: str = "1단계/04_협약/문서.hwp") -> dict:\n    return {',
     'def chunk(source_id: str, title: str, text: str, index: int = 1,\n'
     '          source_path: str = "1단계/04_협약/문서.hwp", **doc_meta) -> dict:\n'
     '    """청크 한 줄. `doc_meta`는 코퍼스 전처리가 채우는 문서 필드(doc_type, version_group ...)다."""\n'
     '    return {'),
    ('            "stage": "1단계",\n        },\n        "text": text,',
     '            "stage": "1단계",\n            **doc_meta,\n        },\n        "text": text,'),
    ('    def test_versions_are_folded_into_one_card(self) -> None:\n        corpus = self.load([\n'
     '            chunk("v3", "차단계 사업계획서_v3", "총사업비 1,346,270천원"),\n'
     '            chunk("v9", "차단계 사업계획서_v9", "총사업비 1,346,270천원"),\n        ])',
     '    def test_versions_are_folded_into_one_card(self) -> None:\n        corpus = self.load([\n'
     f'            chunk("v3", "차단계 사업계획서_v3", "총사업비 1,346,270천원", {GROUP}),\n'
     f'            chunk("v9", "차단계 사업계획서_v9", "총사업비 1,346,270천원", {GROUP}),\n        ])'),
    ('        """gold가 어느 버전을 가리키든 맞출 수 있어야 평가가 성립한다."""\n        corpus = self.load([\n'
     '            chunk("v3", "차단계 사업계획서_v3", "총사업비 1,346,270천원"),\n'
     '            chunk("v9", "차단계 사업계획서_v9", "총사업비 1,346,270천원"),\n        ])',
     '        """gold가 어느 버전을 가리키든 맞출 수 있어야 평가가 성립한다."""\n        corpus = self.load([\n'
     f'            chunk("v3", "차단계 사업계획서_v3", "총사업비 1,346,270천원", {GROUP}),\n'
     f'            chunk("v9", "차단계 사업계획서_v9", "총사업비 1,346,270천원", {GROUP}),\n        ])'),
    ('            chunk("v3", "차단계 사업계획서_v3", "총사업비 총사업비 총사업비"),\n'
     '            chunk("v9", "차단계 사업계획서_v9", "총사업비 한 번"),\n',
     f'            chunk("v3", "차단계 사업계획서_v3", "총사업비 총사업비 총사업비", {GROUP}, version_rank=[0, 3, 0]),\n'
     f'            chunk("v9", "차단계 사업계획서_v9", "총사업비 한 번", {GROUP}, version_rank=[0, 9, 0]),\n'),
    (OLD_SOURCE_TYPE_TEST, NEW_SOURCE_TYPE_TESTS),
]
for a, b in reps:
    assert s.count(a) == 1, a[:90]
    s = s.replace(a, b)
t.write_text(s, encoding="utf-8")
print("test_retriever_modules.py ok")
