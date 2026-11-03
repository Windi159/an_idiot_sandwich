"""RAG 문서 읽기·질문 정규화·글자 매칭 검색 (임베딩 모델 없이)."""

from __future__ import annotations

import json

import pytest

from multimodel.tts.textnorm import sino
from sts_server.rag import JsonlRetriever, load_docs, normalize_query


@pytest.fixture
def jsonl_files(tmp_path):
    qa = tmp_path / "qa.jsonl"
    qa.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in [
        {"question": "PSD 501 코드가 뭐야?", "answer": "501번은 하부 가이드레일 이물질 끼임이야[cite: 12]."},
        {"question": "PSD 505 코드가 뭐야?", "answer": "505번은 PSC 시스템 오류야."},
    ]), encoding="utf-8")
    text = "임산부 고객은 균형 유지가 어려우므로 안전과 편의를 최우선으로 응대합니다."
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in [
        {"question": text, "answer": text, "source": "[대용량]학습용_코레일 고객 응대 가이드", "chunk_id": "a_0001"},
        {"question": "#848 00:11 13 32 보통 #70 00:21 6 222", "answer": "#848 00:11 13 32 보통 #70 00:21 6 222",
         "source": "표", "chunk_id": "b_0001"},
    ]), encoding="utf-8")
    return f"{qa};{chunks}"


def test_load_docs(jsonl_files):
    docs = load_docs(jsonl_files)
    assert docs[0] == "Q: PSD 501 코드가 뭐야?\nA: 501번은 하부 가이드레일 이물질 끼임이야."   # cite 꼬리표 제거
    assert docs[2].startswith("(코레일 고객 응대 가이드)\n임산부 고객은")                     # 청크는 출처 + 본문 한 번
    assert len(docs) == 3                                                                   # 숫자만 남은 표 청크는 건너뜀


@pytest.mark.parametrize("query, expected", [
    ("스크린도어 오백일 코드 떴어", "PSD 501 코드 떴어"),
    ("안전문 오백이십칠번", "PSD 527번"),
    ("엘베 고장", "엘리베이터 고장"),
    ("백화점 가는 길", "백화점 가는 길"),
    ("일단 이야기해 봐", "일단 이야기해 봐"),
])
def test_normalize_query(query, expected):
    assert normalize_query(query) == expected


def test_normalize_query_round_trip():
    assert all(normalize_query(sino(n)) == str(n) for n in range(200, 1000))


def test_jsonl_search(jsonl_files):
    r = JsonlRetriever(jsonl_files)
    assert r.search("스크린도어 오백오 떴어", k=1)[0].startswith("Q: PSD 505")
    assert "임산부" in r.search("임산부 응대", k=1)[0]
