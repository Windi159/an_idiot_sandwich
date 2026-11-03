"""A(임베딩 미세 조정) 학습 데이터 — 검색 문서마다 로컬 Gemma 로 "이 문서로 답할 수 있는 질문"을 만든다.

  # 기본 Gemma llama-server(127.0.0.1:8081)가 떠 있어야 한다. 외부 API 는 쓰지 않는다.
  .venv\\Scripts\\python.exe finetune/gen_embed_data.py            # → finetune/data/embed_pairs.jsonl

- 문서 목록은 서버와 똑같이 sts_server.rag.load_docs(STS_RAG_JSONL) 로 읽는다 (문서 번호가 서버와 같다).
- 문서 하나에 질문 4개: 정식 질문 / 역무원이 글라스에 말하듯 반말 / 짧은 키워드 / 상황 설명형.
- 팀 QA 테스트 세트(평가용)와 비슷한 질문은 버린다 — 시험 문제가 학습에 새면 평가가 부풀려진다.
"""

from __future__ import annotations

import d_drive  # noqa: F401 - 캐시·임시 파일을 D 드라이브로 (맨 먼저)
import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sts_server.config import Settings  # noqa: E402
from sts_server.rag import load_docs  # noqa: E402

TESTSET = ROOT / "data" / "qa_testset" / "4.QA 테스트 세트" / "통합" / "한국철도역무특화 LLM 평가용 테스트세트V1.2.xlsx"
OUT = Path(__file__).resolve().parent / "data" / "embed_pairs.jsonl"

PROMPT = """다음은 코레일 역무원용 업무 문서의 한 부분이다.

<문서>
{doc}
</문서>

역무원이 스마트글라스 음성 비서에게 물어볼 법한 질문 중, **이 문서 내용만으로 답할 수 있는 질문** 4개를 만들어라.
1. 정식 질문 (예: "~은 어떻게 처리하나요?")
2. 동료에게 말하듯 짧은 반말 질문 (예: "~하면 어떻게 해?")
3. 핵심 단어 위주의 짧은 질문 (예: "~ 절차 알려줘")
4. 현장 상황을 설명하며 묻는 질문 (예: "지금 ~한 상황인데 뭐부터 해야 돼?")

규칙:
- 문서에 있는 고유한 내용(코드 번호, 절차 이름, 기준, 대상)을 질문에 담아 다른 문서와 구별되게 한다.
- 문서 문장을 그대로 베끼지 말고 사람이 실제로 말하듯 바꿔 쓴다.
- 목차·표지처럼 답할 내용이 없으면 빈 배열 [] 을 낸다.
- JSON 배열(문자열 4개)만 출력한다."""


def bigrams(s: str) -> set[str]:
    s = "".join(s.split())
    return {s[i:i + 2] for i in range(len(s) - 1)}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def test_questions() -> list[set[str]]:
    import pandas as pd   # .venv 에 있음
    try:
        df = pd.read_excel(TESTSET)
    except ImportError:
        sys.exit("openpyxl 이 필요합니다: uv pip install openpyxl")
    return [bigrams(str(q)) for q in df["question"].dropna()]


def ask(client: httpx.Client, url: str, doc: str) -> list[str]:
    r = client.post(f"{url}/v1/chat/completions", json={
        "messages": [{"role": "user", "content": PROMPT.format(doc=doc[:1500])}],
        "temperature": 0.7, "max_tokens": 400, "chat_template_kwargs": {"enable_thinking": False}})
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    m = re.search(r"\[.*\]", text, re.S)
    try:
        qs = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        qs = re.findall(r'"([^"]{6,120})"', text)
    return [q.strip() for q in qs if isinstance(q, str) and 6 <= len(q.strip()) <= 150]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-test-sim", type=float, default=0.5,
                    help="테스트 세트 질문과 바이그램 자카드 유사도가 이보다 크면 버린다")
    a = ap.parse_args()

    docs = load_docs(Settings.from_env().rag_jsonl)
    tests = test_questions()
    print(f"문서 {len(docs)}개 · 테스트 질문 {len(tests)}개 (이와 비슷한 생성 질문은 버림)")
    client = httpx.Client(timeout=180)
    pairs, dropped = [], 0
    with ThreadPoolExecutor(a.workers) as pool:
        for i, qs in enumerate(pool.map(lambda d: ask(client, a.url, d), docs)):
            for q in qs:
                if max((jaccard(bigrams(q), t) for t in tests), default=0) > a.max_test_sim:
                    dropped += 1
                    continue
                pairs.append({"query": q, "doc_id": i})
            if i % 50 == 0:
                print(f"  {i}/{len(docs)}  지금까지 {len(pairs)}쌍 (테스트와 비슷해 버림 {dropped})  예: {qs[:1]}")
    # 앱 Q/A 문서는 원래 질문도 그 문서의 질문으로 쓴다 (테스트 세트와는 별개 자료)
    for i, d in enumerate(docs):
        if d.startswith("Q: "):
            q = d[3:].split("\nA:", 1)[0].strip()
            if max((jaccard(bigrams(q), t) for t in tests), default=0) <= a.max_test_sim:
                pairs.append({"query": q, "doc_id": i})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"저장 {OUT}: {len(pairs)}쌍 · 질문이 생긴 문서 {len({p['doc_id'] for p in pairs})}/{len(docs)} · "
          f"테스트와 비슷해 버림 {dropped}")


if __name__ == "__main__":
    main()
