"""A. 검색용 임베딩(KR-SBERT) 미세 조정 — 질문과 그 질문의 정답 문서가 가깝게 학습한다.

  .venv\\Scripts\\python.exe finetune/train_embed.py                 # → finetune/models/kr-sbert-ft
  .venv\\Scripts\\python.exe finetune/retrieval_eval.py --model finetune/models/kr-sbert-ft

- 데이터: finetune/data/embed_pairs.jsonl (gen_embed_data.py, 테스트 세트와 비슷한 질문은 이미 빠짐)
- 서버(sts_server.rag.HybridRetriever)와 똑같이 평균 풀링 + 정규화, 문서는 256 토큰까지 본다.
- 손실: MultipleNegativesRanking — 배치 안의 다른 문서들과, 질문마다 하나씩 고른 "헷갈리는 오답 문서"를 밀어낸다.
  같은 배치에 같은 문서가 두 번 들어가면 정답끼리 서로 밀어내므로 배치마다 문서가 겹치지 않게 묶는다.
- 학습 질문 10% 를 떼어 두고 epoch 마다 적중@3 을 재서 가장 좋은 epoch 를 저장한다.
- 캐시·임시 파일은 d_drive 로 전부 D 드라이브.
"""

from __future__ import annotations

import d_drive  # noqa: F401 - 캐시·임시 파일을 D 드라이브로 (맨 먼저)
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from sts_server.config import Settings  # noqa: E402
from sts_server.rag import EMBED_MAX_TOKENS, load_docs, normalize_query  # noqa: E402


def bigrams(s: str) -> set[str]:
    s = "".join(s.split())
    return {s[i:i + 2] for i in range(len(s) - 1)}


def embed(model, tok, texts: list[str], max_len: int, device: str, batch: int = 64, grad: bool = False):
    outs = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            hidden = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        outs.append(F.normalize(emb.float(), dim=-1))
    return torch.cat(outs)


@torch.no_grad()
def hit_at(model, tok, queries, doc_ids, docs, device, k=3) -> float:
    model.eval()
    d = embed(model, tok, docs, EMBED_MAX_TOKENS, device)
    q = embed(model, tok, [normalize_query(x) for x in queries], 64, device)
    top = (q @ d.T).topk(k, dim=1).indices.cpu()
    return sum(int(t in row.tolist()) for t, row in zip(doc_ids, top)) / len(doc_ids)


def batches(pairs: list[dict], size: int, rnd: random.Random) -> list[list[dict]]:
    """문서가 겹치지 않게 배치를 묶는다."""
    pool = pairs[:]
    rnd.shuffle(pool)
    out = []
    while pool:
        batch, used, rest = [], set(), []
        for p in pool:
            if len(batch) < size and p["doc_id"] not in used and p["neg"] not in used:
                batch.append(p)
                used.update((p["doc_id"], p["neg"]))
            else:
                rest.append(p)
        out.append(batch)
        pool = rest
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=HERE / "data" / "embed_pairs.jsonl")
    ap.add_argument("--out", type=Path, default=HERE / "models" / "kr-sbert-ft")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--scale", type=float, default=20.0, help="코사인 점수에 곱하는 온도의 역수")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rnd = random.Random(a.seed)
    s = Settings.from_env()
    docs = load_docs(s.rag_jsonl)
    pairs = [json.loads(l) for l in a.pairs.read_text(encoding="utf-8").splitlines() if l.strip()]
    rnd.shuffle(pairs)
    n_val = max(50, len(pairs) // 10)
    val, train = pairs[:n_val], pairs[n_val:]
    print(f"문서 {len(docs)} · 학습 {len(train)}쌍 · 검증 {len(val)}쌍 · 기본 모델 {s.rag_embedding_model}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(s.rag_embedding_model)
    model = AutoModel.from_pretrained(s.rag_embedding_model).to(device)

    # 헷갈리는 오답: 기본 모델이 2~10위로 올린 문서 중 정답 문서와 내용이 겹치지 않는 것 (겹치면 사실상 정답일 수 있다)
    with torch.no_grad():
        model.eval()
        d_emb = embed(model, tok, docs, EMBED_MAX_TOKENS, device)
        q_emb = embed(model, tok, [normalize_query(p["query"]) for p in train], 64, device)
        top = (q_emb @ d_emb.T).topk(11, dim=1).indices.cpu().tolist()
    grams = [bigrams(d) for d in docs]
    for p, cand in zip(train, top):
        pos = grams[p["doc_id"]]
        negs = [c for c in cand if c != p["doc_id"] and len(grams[c] & pos) / max(1, len(grams[c] | pos)) < 0.3]
        p["neg"] = negs[0] if negs else rnd.randrange(len(docs))

    base_hit = hit_at(model, tok, [p["query"] for p in val], [p["doc_id"] for p in val], docs, device)
    print(f"학습 전 검증 적중@3: {base_hit:.3f}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * math.ceil(len(train) / a.batch) + a.epochs * 5
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / (0.1 * steps)) * max(0.0, (steps - i) / steps))
    best, step = base_hit, 0
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, losses = time.perf_counter(), []
        for b in batches(train, a.batch, rnd):
            q = embed(model, tok, [normalize_query(p["query"]) for p in b], 64, device, batch=len(b))
            d = embed(model, tok, [docs[p["doc_id"]] for p in b] + [docs[p["neg"]] for p in b],
                      EMBED_MAX_TOKENS, device, batch=2 * len(b))
            logits = a.scale * q @ d.T                       # 정답 = 같은 줄의 문서 (0..B-1), 나머지는 오답
            loss = F.cross_entropy(logits, torch.arange(len(b), device=device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            losses.append(loss.item()); step += 1
        hit = hit_at(model, tok, [p["query"] for p in val], [p["doc_id"] for p in val], docs, device)
        print(f"epoch {ep}: 손실 {sum(losses) / len(losses):.3f} · 검증 적중@3 {hit:.3f} · {time.perf_counter() - t0:.0f}초")
        if hit > best:
            best = hit
            a.out.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(a.out)
            tok.save_pretrained(a.out)
            (a.out / "train_info.json").write_text(json.dumps(
                {"base": s.rag_embedding_model, "epoch": ep, "val_hit@3": hit, "val_hit@3_before": base_hit,
                 "train_pairs": len(train), "args": {k: str(v) for k, v in vars(a).items()}},
                ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"  → 저장 {a.out}")
    print(f"끝: 검증 적중@3 {base_hit:.3f} → {best:.3f}")


if __name__ == "__main__":
    main()
