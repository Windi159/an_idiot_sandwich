"""검색만 빠르게 평가 (LLM 없이 몇십 초) — 팀 QA 테스트 세트 질문으로 서버와 같은 하이브리드 검색을 돌린다.

  .venv\\Scripts\\python.exe finetune/retrieval_eval.py                                  # 지금 임베딩 모델
  .venv\\Scripts\\python.exe finetune/retrieval_eval.py --model finetune/models/kr-sbert-ft  # 미세 조정한 모델

지표
  덮음@k  : 정답 문장의 글자 바이그램 중 상위 k개 문서에 들어 있는 비율 (RAGAS 검색 재현율의 값싼 근사)
  적중@k  : "정답 문서"(정답과 가장 많이 겹치는 문서, 겹침 0.35 이상일 때만)가 상위 k개에 있는 비율
  MRR     : 정답 문서 순위의 역수 평균 (상위 10위까지)
  빈 결과 : 점수 문턱(STS_RAG_MIN_SCORE, --min-score)에 걸려 문서가 하나도 안 나온 질문 수
  문턱 분리: 실제 질문 1등 점수와 잡담 1등 점수 — 미세 조정하면 점수 분포가 바뀌어 문턱을 다시 잡아야 할 수 있다
"""

from __future__ import annotations

import d_drive  # noqa: F401 - 캐시·임시 파일을 D 드라이브로 (맨 먼저)
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sts_server import rag  # noqa: E402
from sts_server.config import Settings  # noqa: E402

TESTSET = ROOT / "data" / "qa_testset" / "4.QA 테스트 세트" / "통합" / "한국철도역무특화 LLM 평가용 테스트세트V1.2.xlsx"
# 답하면 안 되는 말 — 1등 점수가 문턱 아래로 떨어져야 한다
NOISE = ["어제 축구 봤어? 진짜 재밌더라", "아 오늘 퇴근하고 뭐 하지", "네 알겠습니다 좋은 하루 보내세요",
         "응 그러니까 그게 말이야", "점심 뭐 먹을까", "AI가 마우스를 안 쳐들어서 내가 아파", "잘 되고 있니?",
         "오늘 날씨 진짜 좋다", "배고프다", "아 졸려", "그거 어디 뒀더라", "내일 몇 시 출근이야?"]


def bigrams(s: str) -> set[str]:
    s = "".join(s.split())
    return {s[i:i + 2] for i in range(len(s) - 1)}


def coverage(ref: set[str], docs: list[str]) -> float:
    if not ref:
        return 0.0
    got = set().union(*(bigrams(d) for d in docs)) if docs else set()
    return len(ref & got) / len(ref)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="임베딩 모델 (이름 또는 폴더). 기본: 설정값")
    ap.add_argument("--k", type=int, default=None, help="검색 문서 수 (기본: 설정 STS_RAG_TOP_K)")
    ap.add_argument("--name", default=None, help="결과 파일 이름")
    ap.add_argument("--min-score", type=float, default=None, help="점수 문턱 (기본: 설정 STS_RAG_MIN_SCORE)")
    ap.add_argument("--reranker", default=None, help="리랭커 (기본: 설정 STS_RAG_RERANKER, 'none' 이면 끔)")
    ap.add_argument("--candidates", type=int, default=None, help="리랭커 후보 수 (기본: 설정 STS_RAG_RERANK_CANDIDATES)")
    a = ap.parse_args()

    import pandas as pd
    s = Settings.from_env()
    if a.model:
        s.rag_embedding_model = str(Path(a.model).resolve()) if Path(a.model).exists() else a.model
    k = a.k or s.rag_top_k
    if a.min_score is not None:
        s.rag_min_score = a.min_score
    if a.reranker is not None:
        s.rag_reranker = "" if a.reranker == "none" else a.reranker
    if a.candidates:
        s.rag_rerank_candidates = a.candidates
    r = rag.HybridRetriever(s.rag_jsonl, s)
    doc_grams = [bigrams(d) for d in r.docs]
    df = pd.read_excel(TESTSET).dropna(subset=["question", "answer"])

    rows = []
    for q, ans, src in zip(df["question"], df["answer"], df["source"]):
        ref = bigrams(str(ans))
        scored = r.ranked(str(q))   # 리랭커가 있으면 그 순서
        ranked = [i for _, i in scored]
        kept = [i for sc, i in scored[:k] if sc >= r.min_score]
        overlap = [len(ref & g) / len(ref) if ref else 0 for g in doc_grams]
        gold = max(range(len(r.docs)), key=overlap.__getitem__)
        labeled = overlap[gold] >= 0.35
        rank = ranked.index(gold) + 1 if labeled else None
        rows.append({"source": src, "question": q, "top1_score": max(sc for sc, _ in scored), "empty": not kept,
                     "cover@k": coverage(ref, [r.docs[i] for i in kept]),
                     "cover@5": coverage(ref, [r.docs[i] for i in ranked[:5]]),
                     "labeled": labeled, "rank": rank})

    lab = [x for x in rows if x["labeled"]]
    noise_top = [r.scored(n)[0][0] for n in NOISE]
    summary = {
        "model": s.rag_embedding_model, "k": k, "n": len(rows), "labeled": len(lab),
        f"덮음@{k}": round(statistics.mean(x["cover@k"] for x in rows), 3),
        "덮음@5": round(statistics.mean(x["cover@5"] for x in rows), 3),
        "적중@1": round(sum(x["rank"] == 1 for x in lab) / len(lab), 3),
        f"적중@{k}": round(sum(x["rank"] <= k for x in lab) / len(lab), 3),
        "적중@5": round(sum(x["rank"] <= 5 for x in lab) / len(lab), 3),
        "MRR@10": round(statistics.mean(1 / x["rank"] if x["rank"] <= 10 else 0 for x in lab), 3),
        "빈 결과": sum(x["empty"] for x in rows),
        "질문 1등 점수(하위 10%)": round(sorted(x["top1_score"] for x in rows)[len(rows) // 10], 3),
        "잡담 1등 점수(최고)": round(max(noise_top), 3),
        "문턱": r.min_score,
    }
    by_src = {}
    for src in sorted({x["source"] for x in rows}):
        part = [x for x in rows if x["source"] == src]
        lp = [x for x in part if x["labeled"]]
        by_src[src] = {"n": len(part), f"덮음@{k}": round(statistics.mean(x["cover@k"] for x in part), 3),
                       f"적중@{k}": round(sum(x["rank"] <= k for x in lp) / len(lp), 3) if lp else None}

    print(json.dumps(summary, ensure_ascii=False, indent=1))
    for src, v in by_src.items():
        print(f"  {src:40s} {v}")
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    name = a.name or Path(s.rag_embedding_model).name
    (out / f"retrieval_{name}.json").write_text(json.dumps({"summary": summary, "by_source": by_src, "rows": rows},
                                                           ensure_ascii=False, indent=1), encoding="utf-8")
    print("저장:", out / f"retrieval_{name}.json")


if __name__ == "__main__":
    main()
