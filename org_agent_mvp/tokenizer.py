"""검색용 토큰화. 단계별로 갈아 끼울 수 있게 한 곳에 모았다.

질의와 문서가 **같은 방식으로** 잘려야 매칭이 된다. 그래서 STM/MTM(memory_store)과
LTM(ltm_corpus)이 이 모듈 하나를 공유한다. 한쪽만 바꾸면 조용히 검색이 망가진다.

    RETRIEVER_TOKENIZER=whitespace   공백·문자종류 분리 (기본, 0단계)
    RETRIEVER_TOKENIZER=morph        형태소 분석 (1단계~)

## 왜 형태소가 필요한가

0단계 토큰화는 정규식 하나다.

    TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣_]+")

한국어는 조사가 붙어 표면형이 달라진다. 실측한 실패 사례다.

    질의 토큰   ['인건비를', '연구활동비로', ...]
    문서 토큰   ['인건비',   '연구활동비',   ...]
                     ^ 안 맞는다

형태소 분석기는 `인건비를` -> `인건비` + `를`로 쪼갠다.
조사·어미를 버리고 내용어만 남기면 질의와 문서가 같은 자리에서 만난다.
"""

from __future__ import annotations

import os
import re
from typing import Callable, Iterable

#: 0단계 토큰 패턴. 숫자·영문·한글·밑줄이 이어지는 덩어리를 하나로 본다.
TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣_]+")

#: 형태소 태그 중 검색에 남길 것.
#:
#:   NNG 일반명사   NNP 고유명사   NNB 의존명사(차년도의 "차", "년도")
#:   SL  외국어     SN  숫자        SH  한자
#:   VV  동사어간   VA  형용사어간  XR  어근
#:   XSN 명사파생접미사("사업화"의 "화")
#:
#: 조사(J*), 어미(E*), 기호(S* 나머지), 관형사(MM)는 버린다.
#: 내용을 담지 않으면서 거의 모든 문서에 나와 점수만 흐린다.
CONTENT_TAGS = frozenset({
    "NNG", "NNP", "NNB", "SL", "SN", "SH", "VV", "VA", "XR", "XSN",
})

#: 태그를 통과해도 이 목록에 있으면 버린다.
#: 의존명사 중 내용이 없는 것들이다 — "수 있다", "것이다"의 그 "수", "것".
STOP_FORMS = frozenset({
    "수", "것", "등", "때", "바", "점", "중", "위", "내", "외", "간",
    "및", "또는", "관련", "대한", "위한", "하", "되", "있", "없",
})


def whitespace_tokenize(text: str) -> list[str]:
    """0단계. 문자 종류가 바뀌는 지점에서만 자른다."""
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


# --------------------------------------------------------------------- 형태소

_kiwi = None


def _get_kiwi():
    """Kiwi 인스턴스를 한 번만 만든다. 생성이 1초쯤 걸려서 매번 만들면 안 된다."""
    global _kiwi
    if _kiwi is None:
        from kiwipiepy import Kiwi

        _kiwi = Kiwi()
    return _kiwi


def morph_tokenize(text: str) -> list[str]:
    """1단계. 형태소로 쪼개고 내용어만 남긴다.

    복합어는 갈라진다 — `연구활동비` -> `연구` + `활동비`.
    문서도 같은 방식으로 갈라지므로 매칭에는 문제가 없고,
    오히려 `연구활동비 사용 범위`와 `연구 활동비`가 만나게 된다.
    """
    if not text:
        return []
    out: list[str] = []
    for token in _get_kiwi().tokenize(text):
        if token.tag not in CONTENT_TAGS:
            continue
        form = token.form.lower()
        if len(form) < 1 or form in STOP_FORMS:
            continue
        out.append(form)
    return out


def morph_tokenize_many(texts: Iterable[str]) -> list[list[str]]:
    """여러 문서를 한 번에 자른다. 청크 26,000개를 개별 호출하면 느리다."""
    texts = list(texts)
    if not texts:
        return []
    kiwi = _get_kiwi()
    results: list[list[str]] = []
    for parsed in kiwi.tokenize(texts):
        out = []
        for token in parsed:
            if token.tag not in CONTENT_TAGS:
                continue
            form = token.form.lower()
            if form in STOP_FORMS:
                continue
            out.append(form)
        results.append(out)
    return results


# ----------------------------------------------------------------- 선택 창구

TOKENIZERS: dict[str, Callable[[str], list[str]]] = {
    "whitespace": whitespace_tokenize,
    "morph": morph_tokenize,
}

DEFAULT_TOKENIZER = "whitespace"


def tokenizer_name() -> str:
    name = os.environ.get("RETRIEVER_TOKENIZER", DEFAULT_TOKENIZER).strip().lower()
    return name if name in TOKENIZERS else DEFAULT_TOKENIZER


def tokenize(text: str) -> list[str]:
    """설정된 방식으로 자른다. 질의와 문서 모두 이걸 통과해야 한다."""
    return TOKENIZERS[tokenizer_name()](text)


def tokenize_many(texts: Iterable[str]) -> list[list[str]]:
    """일괄 처리. 형태소일 때만 배치 이득이 있다."""
    if tokenizer_name() == "morph":
        return morph_tokenize_many(texts)
    return [whitespace_tokenize(t) for t in texts]


def available() -> dict[str, bool]:
    """각 방식이 지금 환경에서 쓸 수 있는지."""
    try:
        import kiwipiepy  # noqa: F401

        morph_ok = True
    except ImportError:
        morph_ok = False
    return {"whitespace": True, "morph": morph_ok}
