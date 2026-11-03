"""RAG 문서 청크 다시 만들기 — 원본 변환본(txt·Docling 마크다운)을 정리해 제목·표 단위로 자른다.

  .venv\\Scripts\\python.exe scripts/build_rag_chunks.py            # → data/rag_source/rag_chunks_v2.jsonl
  .venv\\Scripts\\python.exe scripts/build_rag_chunks.py --size 600

예전 청크(RAG_자료/rag_chunks.jsonl, 500자/겹침 100)와 달라진 점
- 고객 응대 가이드 : 페이지 머리말(PART n, 장 제목)·쪽 번호·목차를 빼고, 줄바꿈으로 끊긴 문장을 잇는다
- 안내방송 매뉴얼   : 오타를 고친 최신 txt 로 다시 자른다 (예전 청크엔 OCR 오타 8곳)
- 서울역 혼잡도·철도이용자 보호기준 : 한글에서 PDF 로 저장 → Docling(scripts/convert_docs.py) 마크다운. 표를 통째로 둔다
- 열차사고 체크리스트 : 순서도 그림을 Gemma 비전으로 옮긴 마크다운
- 청크 앞에 "제목 > 소제목" 을 한 번만 붙인다. 큰 표는 머리글 줄을 반복하며 나눈다

한 줄 형식은 예전과 같다: {"question" == "answer" == 청크, "source", "chunk_id"} (sts_server/rag.py load_docs)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "rag_source"
TXT = SRC / "RAG_자료" / "txt_변환"
MD = SRC / "docling"

DOCS = [  # (source 이름, 파일, 형식)
    ("[대용량]학습용_코레일 고객 응대 가이드", TXT / "[대용량]학습용_코레일 고객 응대 가이드.txt", "guide"),
    ("학습용_고속 및 일반철도 역 분야 안내방송 매뉴얼(2025)", TXT / "학습용_고속 및 일반철도 역 분야 안내방송 매뉴얼(2025).txt", "pages"),
    ("학습용_서울역 혼잡도 관리운영 프로세스", MD / "학습용_서울역 혼잡도 관리운영 프로세스.md", "md"),
    ("철도이용자 보호 기준 및 안내방송", MD / "철도이용자 보호 기준 및 안내방송.md", "md"),
    ("[대용량]열차사고 등 이례사항 대비 체크리스트", MD / "[대용량]열차사고 등 이례사항 대비 체크리스트.md", "md"),
]

# 고객 응대 가이드의 페이지 머리말 (매 쪽 위에 찍힌 PART 제목)
GUIDE_HEADERS = {"KORAIL CUSTOMER SERVICE GUIDE", "신뢰를 잇는 우리의 첫모습", "배려를 잇는 우리의 진심",
                 "편의를 잇는 역 안내", "친근함을 잇는 열차 승무", "신뢰를 잇는", "우리의 첫 모습", "배려를 잇는",
                 "우리의 진심", "편의를 잇는", "역 안내", "친근함을 잇는", "열차 승무"}
_PART = re.compile(r"^PART\s*\d\.?$")
_PAGE_NO = re.compile(r"^\d{1,3}$")
_SECTION = re.compile(r"^(\d{1,2})\.?\s+(\S.{0,30})$")       # "4. 인사"·"3 기본 자세 및 행동" 같은 장 제목
_BULLET = re.compile(r"^([•■□◦○▶①-⑳ㅇ※\-–·]|\d+\)|\d+\.\s)")
_ITEM = re.compile(r"^[➀-➉]\s*\S")                           # "➃ 열차 오승(오탑승) 고객 응대" 같은 상황별 항목
_ANNOUNCE = re.compile(r"^\d{1,2}(?:~\d{1,2})?\.\s")          # 안내방송 매뉴얼의 "35. 협박 전화를 접수한 경우 …"
MIN_CHARS = 40   # 이보다 짧은 조각(쪽 끝 자투리)은 버린다
_END = re.compile(r"[.?!。”\"'’)]$|다$|요$|음$|함$|것$")


def clean_guide(text: str) -> list[tuple[str, str]]:
    """고객 응대 가이드 txt → [(장 제목[ > 상황별 항목], 본문)]. 목차·머리말·쪽 번호를 빼고 끊긴 줄을 잇는다.

    상황별 항목(➀ ➁ …)마다 새 구간으로 나눈다 — 한 청크에 여러 상황이 섞이면 임베딩이 흐려져 검색 점수가 크게 떨어진다.
    """
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip() == "단정한 용모와 옷차림의 중요성")
    sections, chapter, title, buf = [], "1. 용모 및 옷차림", "1. 용모 및 옷차림", []
    for raw in lines[start:]:
        line = raw.strip()
        if not line or line in GUIDE_HEADERS or _PART.match(line) or _PAGE_NO.match(line):
            if not line and buf and buf[-1] != "":
                buf.append("")
            continue
        m = _SECTION.match(line)
        is_chapter = m and len(line) <= 20 and not _END.search(line)
        if is_chapter or (_ITEM.match(line) and len(line) <= 45):
            if any(b.strip() for b in buf):
                sections.append((title, buf))
            if is_chapter:
                chapter = title = line
            else:
                title = f"{chapter} > {line}"
            buf = [] if is_chapter else [line]
            continue
        # 들여쓴 줄, 또는 앞 줄이 문장 중간에서 끝났고 이 줄이 새 항목이 아니면 앞 줄에 잇는다
        if buf and buf[-1] and (raw[:1].isspace() and not _BULLET.match(line)
                                or (not _END.search(buf[-1]) and not _BULLET.match(line) and len(buf[-1]) > 25)):
            buf[-1] = f"{buf[-1]} {line}"
        else:
            buf.append(line)
    sections.append((title, buf))
    return [(t, "\n".join(b).strip()) for t, b in sections]


def split_pages(text: str) -> list[tuple[str, str]]:
    """'[원문 n페이지]' 로 나뉜 txt → [(쪽 첫 줄 또는 번호 항목 첫머리, 본문)]. 번호 붙은 방송 문안마다 나눈다."""
    out = []
    for page in re.split(r"^\[원문 \d+페이지\]\s*$", text, flags=re.M):
        page = page.strip()
        if not page:
            continue
        title, buf = page.splitlines()[0].strip()[:40], []
        for line in page.splitlines():
            if _ANNOUNCE.match(line.strip()) and "\n".join(buf).strip():
                out.append((title, "\n".join(buf).strip()))
                title, buf = line.strip()[:40], []
            buf.append(line)
        if "\n".join(buf).strip():
            out.append((title, "\n".join(buf).strip()))
    return out


# 칸·제목 맨 앞에서 "보 통"·"□ 개 요" 처럼 글자마다 띄운 낱말 (가운데는 "및 홈" 같은 진짜 띄어쓰기가 있어 건드리지 않는다)
_SPACED = re.compile(r"^([^가-힣]*)([가-힣](?: [가-힣])+)(?![가-힣])")


def _join(m: re.Match) -> str:
    return m[1] + m[2].replace(" ", "")


def tidy_markdown(text: str) -> str:
    """Docling 표의 칸 채움 공백·긴 구분선을 걷어내고, 표·제목에서 글자마다 띄운 낱말을 붙인다.

    채움 공백 때문에 표 한 줄이 170자를 넘어 500자 청크에 한 줄씩만 들어가고, "보 통" 은 "보통" 으로 찾히지 않았다.
    """
    out = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("|"):
            cells = [re.sub(r"\s+", " ", c).strip() for c in s.strip("|").split("|")]
            if all(re.fullmatch(r":?-+:?", c) for c in cells if c):
                line = "|" + "|".join("---" for _ in cells) + "|"
            else:
                line = "| " + " | ".join(_SPACED.sub(_join, c) for c in cells) + " |"
        elif s.startswith("#"):
            line = _SPACED.sub(_join, line)
        out.append(line)
    return "\n".join(out)


def split_markdown(text: str) -> list[tuple[str, str]]:
    """Docling 마크다운 → [(제목 경로, 본문)]. '##'·'###' 제목에서 나눈다."""
    text = tidy_markdown(re.sub(r"<!--.*?-->", "", text, flags=re.S))
    out, path, buf = [], [], []
    for line in text.splitlines():
        m = re.match(r"^(#{2,4})\s+(.*)$", line)
        if m:
            if "\n".join(buf).strip():
                out.append((" > ".join(path), "\n".join(buf).strip()))
            level = len(m[1]) - 2
            path = path[:level] + [m[2].strip()]
            buf = []
        else:
            buf.append(line)
    if "\n".join(buf).strip():
        out.append((" > ".join(path), "\n".join(buf).strip()))
    return out


def blocks(body: str) -> list[str]:
    """본문을 자를 수 있는 덩어리로: 표는 통째로, 나머지는 빈 줄로 나뉜 문단."""
    out, cur, in_table = [], [], False
    for line in body.splitlines():
        is_row = line.lstrip().startswith("|")
        if is_row != in_table or (not line.strip() and not in_table):
            if cur:
                out.append("\n".join(cur).strip())
            cur, in_table = [], is_row
        if line.strip():
            cur.append(line)
    if cur:
        out.append("\n".join(cur).strip())
    return [b for b in out if b]


def split_table(table: str, size: int) -> list[str]:
    """큰 표는 머리글 두 줄(제목·구분선)을 반복하며 행 단위로 나눈다."""
    rows = table.splitlines()
    head, body = rows[:2], rows[2:]
    parts, cur = [], []
    for r in body:
        if cur and len("\n".join(head + cur + [r])) > size:
            parts.append("\n".join(head + cur))
            cur = []
        cur.append(r)
    if cur:
        parts.append("\n".join(head + cur))
    return parts


def chunk(title: str, body: str, size: int, overlap: int) -> list[str]:
    """제목 + 덩어리들을 size 자 안팎으로 묶는다. 이어지는 청크엔 앞 덩어리 끝 overlap 자를 겹친다."""
    pieces = []
    for b in blocks(body):
        if b.startswith("|") and len(b) > size:
            pieces.extend(split_table(b, size))
        elif len(b) > size * 1.5:   # 긴 문단은 줄 단위로
            pieces.extend(l for l in b.splitlines() if l.strip())
        else:
            pieces.append(b)
    out, cur = [], []
    for p in pieces:
        if cur and len("\n".join(cur + [p])) > size:
            out.append("\n".join(cur))
            tail = cur[-1][-overlap:] if not cur[-1].startswith("|") else ""
            cur = [tail] if tail and overlap else []
        cur.append(p)
    if cur:
        out.append("\n".join(cur))
    head = f"[{title}]\n" if title else ""
    return [head + c.strip() for c in out if len(c.strip()) >= MIN_CHARS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=800, help="청크 목표 글자 수")
    ap.add_argument("--overlap", type=int, default=150)
    ap.add_argument("--out", type=Path, default=SRC / "rag_chunks_v2.jsonl")
    a = ap.parse_args()

    rows, stats = [], Counter()
    for source, path, kind in DOCS:
        text = path.read_text(encoding="utf-8")
        sections = clean_guide(text) if kind == "guide" else split_pages(text) if kind == "pages" else split_markdown(text)
        for title, body in sections:
            for c in chunk(title, body, a.size, a.overlap):
                rows.append({"question": c, "answer": c, "source": source, "chunk_id": f"{source}#{stats[source]}"})
                stats[source] += 1
    with a.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lens = sorted(len(r["answer"]) for r in rows)
    print(f"청크 {len(rows)}개 · 길이 중앙 {lens[len(lens) // 2]}자 (최소 {lens[0]}, 최대 {lens[-1]}) → {a.out}")
    for s, n in stats.items():
        print(f"  {n:4d}  {s}")


if __name__ == "__main__":
    main()
