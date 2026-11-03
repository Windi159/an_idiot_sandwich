"""B(Gemma QLoRA) 학습 데이터 — 서버와 똑같은 프롬프트·검색 결과로 "모범 답"을 만든다 (RAFT 방식).

  # 기본 Gemma llama-server(127.0.0.1:8081)가 떠 있어야 한다. 외부 API 는 쓰지 않는다.
  .venv\\Scripts\\python.exe finetune/gen_llm_data.py --embedding finetune/models/kr-sbert-ft   # → finetune/data/llm_sft.jsonl

만드는 법
1. 질문: embed_pairs.jsonl(테스트 세트와 비슷한 질문은 이미 빠짐)에서 문서마다 최대 2개
2. 모범 답: 기본 Gemma 에게 **정답 문서 하나만** 주고 서버 QA 프롬프트 규칙대로(존댓말·짧게·방송 문안은 그대로) 답하게 한다.
   정답 문서만 보므로 실제 서버(검색 문서 3개)보다 정확하다. 답 글자가 문서와 너무 안 겹치면(지어냄) 버린다.
3. 학습 입력: 서버가 실제로 하는 것처럼 그 질문으로 검색한 문서 3개를 붙인다.
   - 정답 문서가 검색에 있으면 그대로 → "여러 문서 중 맞는 걸 골라 쓰기"를 배운다
   - 없으면 70% 는 끝 문서를 정답 문서로 바꾸고, 30% 는 그대로 두고 답을 "모르는 경우 문구"로 → 지어내지 않기를 배운다
"""

from __future__ import annotations

import d_drive  # noqa: F401 - 캐시·임시 파일을 D 드라이브로 (맨 먼저)
import argparse
import json
import random
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from sts_server import prompts, rag  # noqa: E402
from sts_server.config import Settings  # noqa: E402

FALLBACK = "안내데스크 또는 전광판에서 확인하실 수 있습니다."
GUIDE = ("\n\n(모범 답안 작성 지침) 위 규칙을 지키되, 질문에 답하는 데 필요한 참고자료의 핵심 항목(절차·기준·대상·숫자)은 "
         "빠뜨리지 말고 담으세요. 참고자료에 없는 내용은 덧붙이지 마세요.")


def bigrams(s: str) -> set[str]:
    s = "".join(s.split())
    return {s[i:i + 2] for i in range(len(s) - 1)}


def answer(client: httpx.Client, url: str, q: str, doc: str) -> str:
    r = client.post(f"{url}/v1/chat/completions", json={
        "messages": [{"role": "system", "content": prompts.QA_SYSTEM + GUIDE},
                     {"role": "user", "content": prompts.qa_user_text(q, [doc], from_audio=False)}],
        "temperature": 0.3, "max_tokens": 400, "chat_template_kwargs": {"enable_thinking": False}})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--embedding", default=None, help="학습 입력의 검색에 쓸 임베딩 (B 를 A 와 함께 쓸 거면 A 모델)")
    ap.add_argument("--per-doc", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rnd = random.Random(a.seed)

    s = Settings.from_env()
    if a.embedding:
        s.rag_embedding_model = str(Path(a.embedding).resolve())
    retriever = rag.HybridRetriever(s.rag_jsonl, s)
    docs = retriever.docs
    by_doc = defaultdict(list)
    for p in map(json.loads, (HERE / "data" / "embed_pairs.jsonl").read_text(encoding="utf-8").splitlines()):
        by_doc[p["doc_id"]].append(p["query"])
    items = [(q, d) for d, qs in by_doc.items() for q in rnd.sample(qs, min(a.per_doc, len(qs)))]
    print(f"질문 {len(items)}개 (문서 {len(by_doc)}개) · 검색 임베딩 {s.rag_embedding_model}")

    client = httpx.Client(timeout=180)
    with ThreadPoolExecutor(a.workers) as pool:
        answers = list(pool.map(lambda it: answer(client, a.url, it[0], docs[it[1]]), items))

    out, stats = [], defaultdict(int)
    for (q, gold), ans in zip(items, answers):
        ans = re.sub(r"\s+", " ", ans).strip()
        if not ans:
            stats["빈 답"] += 1
            continue
        grounded = len(bigrams(ans) & bigrams(docs[gold])) / max(1, len(bigrams(ans)))
        if ans != FALLBACK and grounded < 0.35:
            stats["문서와 안 겹쳐 버림"] += 1
            continue
        ctx = retriever.search(q, s.rag_top_k)
        target = ans
        if docs[gold] in ctx:
            stats["정답 문서가 검색됨"] += 1
        elif rnd.random() < 0.7 or not ctx:
            ctx = (ctx[:s.rag_top_k - 1] + [docs[gold]])
            rnd.shuffle(ctx)
            stats["정답 문서를 끼워 넣음"] += 1
        else:
            target = FALLBACK
            stats["정답 없음 → 모름 문구"] += 1
        out.append({"messages": [
            {"role": "system", "content": prompts.QA_SYSTEM},
            {"role": "user", "content": prompts.qa_user_text(q, ctx, from_audio=False)},
            {"role": "assistant", "content": target}], "doc_id": gold, "question": q})
    path = HERE / "data" / "llm_sft.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"저장 {path}: {len(out)}개 · {dict(stats)}")
    for row in out[:3]:
        print(" 예)", row["question"], "→", row["messages"][-1]["content"][:120])


if __name__ == "__main__":
    main()
