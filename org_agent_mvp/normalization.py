from __future__ import annotations

import re


PROJECT_NAME_RE = re.compile(
    r"^\s*([A-Za-z0-9가-힣]+)\s*[-_]?\s*(과제|프로젝트|사업|과업)\s*$",
    re.IGNORECASE,
)


def normalize_project_name(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value).strip())
    match = PROJECT_NAME_RE.match(text)
    if not match:
        return text
    prefix = match.group(1)
    suffix = match.group(2)
    if re.fullmatch(r"[A-Za-z]", prefix):
        prefix = prefix.upper()
    return f"{prefix} {suffix}"


def normalized_project_key(value: str) -> str:
    normalized = normalize_project_name(value).lower()
    return re.sub(r"[\s_-]+", "", normalized)
