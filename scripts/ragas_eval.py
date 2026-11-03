# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "ragas>=0.2.15,<0.4",
#     # ragas 0.3 은 langchain_community 0.4 에서 없어진 모듈을 불러온다 — langchain 0.3 계열로 묶는다
#     "langchain-core>=0.3,<1",
#     "langchain-community>=0.3,<0.4",
#     "langchain-openai>=0.2,<1",
#     "langchain-huggingface>=0.1,<1",
#     "sentence-transformers>=3.0",
#     "httpx>=0.27",
#     "pandas>=2.0",
#     "openpyxl>=3.1",
# ]
# ///
"""STS 서버 RAGAS 자동 평가기 — 서버 API 를 앱처럼 불러 받은 답변을 RAGAS 로 채점한다.

  # 서버(.venv)와 섞이지 않게 따로 만든 환경에서 돈다 (uv 가 위 의존성으로 자동 생성, 캐시는 D 드라이브)
  $env:UV_CACHE_DIR = "D:/uv-cache"; uv run scripts/ragas_eval.py
  uv run scripts/ragas_eval.py --text 60                 # 글 질문을 60개로 (출처 문서별로 고르게)
  uv run scripts/ragas_eval.py --text -1 --no-voice      # 테스트 세트 253문항 전부 (채점 1시간 이상)
  uv run scripts/ragas_eval.py --text -1 --no-score      # 답만 모으기 → 나중에 --answers <answers.jsonl> 로 채점
  uv run scripts/ragas_eval.py --judge-url https://api.openai.com/v1 --judge-model gpt-4o-mini  # 채점 모델 바꾸기 (키: RAGAS_JUDGE_KEY)

평가 데이터
  음성 : data/eval/manifest.jsonl 의 qa 문항(make_eval_set.py 로 만든 wav + 정답) → POST /api/ask (debug=1)
         받아쓰기 → 분류 → RAG → 답변 전 과정을 본다. 받아쓰기가 틀리면 검색·답이 같이 틀어진다.
  글   : 팀 QA 테스트 세트(기본: data/qa_testset/…/통합/한국철도역무특화 LLM 평가용 테스트세트V1.2.xlsx, 253문항,
         출처 문서 7종)에서 출처별로 고르게 N개 → POST /chat (debug=true). --testset 으로 다른 xlsx/csv/jsonl 도 된다
         (question, answer[, source] 열). 출처 문서는 모두 서버 RAG 문서에 들어 있다.

지표 (0~1, 높을수록 좋음)
  faithfulness       충실도: 답변의 주장이 검색 문서로 뒷받침되는 비율 (지어낸 말이 적을수록 높다)
  answer_relevancy   답변 관련성: 답변이 질문에 맞는 답인지
  context_precision  검색 정밀도: 검색 문서 중 정답에 쓸모 있는 문서가 위에 있는지
  context_recall     검색 재현율: 정답의 내용이 검색 문서에 들어 있는지
  factual_correctness 정답성: 답변 주장과 정답 주장이 겹치는 정도 (F1)

채점 모델(judge) 기본값은 이 PC 의 llama-server(지금 서버가 쓰는 Gemma)다. 같은 모델이 자기 답을 채점하므로
점수가 후할 수 있다 — 버전 간 비교용으로 쓰고, 절대 점수가 필요하면 --judge-url 로 더 큰 모델을 쓴다.

결과는 data/ragas/<시각>/ 에 results.csv(문항별), summary.json, report.md 로 남는다.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import httpx

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sts_server.config import Settings  # noqa: E402 - 표준 라이브러리만 쓰는 설정 모듈

DEFAULT_TESTSET = (ROOT / "data" / "qa_testset" / "4.QA 테스트 세트" / "통합"
                   / "한국철도역무특화 LLM 평가용 테스트세트V1.2.xlsx")
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "factual_correctness"]
LABEL = {"faithfulness": "충실도", "answer_relevancy": "답변 관련성", "context_precision": "검색 정밀도",
         "context_recall": "검색 재현율", "factual_correctness": "정답성"}


# ------------------------------------------------------------------ 1. 서버에 묻기


def post(client: httpx.Client, url: str, path: str, tries: int = 4, **kw) -> httpx.Response:
    """서버가 잠깐 다시 켜지는 중이면(연결 실패·503) 다시 뜰 때까지 기다렸다가 다시 묻는다."""
    for attempt in range(tries):
        try:
            r = client.post(url + path, **kw)
            if r.status_code < 500:
                return r.raise_for_status()
        except httpx.TransportError:
            if attempt == tries - 1:
                raise
        print(f"    서버 응답 없음 — 다시 켜질 때까지 기다림 ({attempt + 1}/{tries})")
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            try:
                if client.get(url + "/health", timeout=5).status_code == 200:
                    break
            except httpx.TransportError:
                pass
            time.sleep(5)
    return r.raise_for_status()


def ask_voice(client: httpx.Client, url: str, limit: int | None) -> list[dict]:
    items = [json.loads(l) for l in (ROOT / "data" / "eval" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
             if l.strip()]
    items = [it for it in items if it["intent"] == "qa"][:limit]
    rows = []
    for it in items:
        wav = ROOT / "data" / "eval" / it["file"]
        t0 = time.perf_counter()
        j = post(client, url, "/api/ask", files={"file": (wav.name, wav.read_bytes(), "audio/wav")},
                 data={"language": "ko", "task": "transcribe", "debug": "1"}).json()
        if "contexts" not in j:
            sys.exit("서버가 contexts 를 주지 않습니다 — 서버를 최신 코드로 다시 켜야 합니다 (debug 옵션은 2026-10-27 추가).")
        rows.append({"set": "음성", "id": it["file"], "source": "음성 평가셋", "user_input": it["text"],
                     "heard": j.get("transcript", ""), "intent": j.get("intent"), "response": j.get("answer", ""),
                     "retrieved_contexts": j["contexts"], "reference": it["reference"],
                     "latency_s": round(time.perf_counter() - t0, 3)})
        print(f"  음성 {it['file']:12s} {rows[-1]['latency_s']:4.1f}s  [{j.get('intent')}] {j.get('transcript', '')}")
    return rows


def load_testset(path: Path) -> list[dict]:
    """question, answer[, source] 열이 있는 xlsx/csv/jsonl."""
    import pandas as pd
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path, encoding="utf-8-sig")
    else:
        df = pd.read_json(path, lines=True)
    df = df.dropna(subset=["question", "answer"])
    sources = df["source"] if "source" in df.columns else [None] * len(df)
    return [{"question": str(q).strip(), "answer": str(a).strip(),
             "source": (str(s).strip() if isinstance(s, str) and s.strip() else "(출처 없음)")}
            for q, a, s in zip(df["question"], df["answer"], sources)]


def pick(pairs: list[dict], n: int, seed: int) -> list[dict]:
    """출처 문서별로 고르게 n개 (각 출처 최소 1개, 나머지는 문항 수 비율대로). n<0 이면 전부."""
    if n < 0 or n >= len(pairs):
        return pairs
    rnd = random.Random(seed)
    groups: dict[str, list[dict]] = {}
    if n < len({p["source"] for p in pairs}):   # 출처 수보다 적게 뽑으면 고르게 나눌 수 없다 — 그냥 무작위
        return rnd.sample(pairs, n)
    for p in pairs:
        groups.setdefault(p["source"], []).append(p)
    quota = {s: 1 for s in groups} if n >= len(groups) else {}
    rest = n - sum(quota.values())
    for s, g in groups.items():
        quota[s] = quota.get(s, 0) + round(rest * len(g) / len(pairs))
    picked = [p for s, g in groups.items() for p in rnd.sample(g, min(quota.get(s, 0), len(g)))]
    rnd.shuffle(picked)
    return picked[:n]


def ask_text(client: httpx.Client, url: str, n: int, seed: int, path: Path) -> list[dict]:
    pairs = load_testset(path)
    chosen = pick(pairs, n, seed)
    print(f"  테스트 세트 {path.name}: {len(pairs)}문항 중 {len(chosen)}개")
    rows = []
    for i, p in enumerate(chosen):
        t0 = time.perf_counter()
        lines = [json.loads(l) for l in post(client, url, "/chat", json={"text": p["question"], "debug": True})
                 .text.splitlines() if l.strip()]
        done = next((l for l in lines if l.get("done")), {})
        intent = lines[0].get("type") if lines else None
        if intent == "qa" and "contexts" not in done:
            sys.exit("서버가 contexts 를 주지 않습니다 — 서버를 최신 코드로 다시 켜야 합니다.")
        rows.append({"set": "글", "id": f"text_{i:03d}", "source": p["source"], "user_input": p["question"],
                     "heard": p["question"], "intent": intent, "response": done.get("answer", ""),
                     "retrieved_contexts": done.get("contexts", []), "reference": p["answer"],
                     "latency_s": round(time.perf_counter() - t0, 3)})
        print(f"  글   {rows[-1]['id']:12s} {rows[-1]['latency_s']:4.1f}s  [{intent}] {p['question']}")
    return rows


# ------------------------------------------------------------------ 2. RAGAS 채점


def judge(args):
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    extra = {}
    if "127.0.0.1" in args.judge_url or "localhost" in args.judge_url:
        # llama-server 의 Gemma 4 는 생각(thinking) 모드를 꺼야 JSON 만 깔끔하게 낸다
        extra["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    llm = ChatOpenAI(base_url=args.judge_url, model=args.judge_model, temperature=0, max_tokens=2048, timeout=180,
                     api_key=os.environ.get("RAGAS_JUDGE_KEY", "none"), **extra)
    emb = HuggingFaceEmbeddings(model_name=args.embedding_model)
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(emb)


def score(rows: list[dict], args) -> list[dict]:
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.metrics import (FactualCorrectness, Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall,
                               ResponseRelevancy)

    llm, emb = judge(args)
    relevancy = ResponseRelevancy(strictness=1)
    # 답변 관련성은 "답변을 보고 질문을 거꾸로 만들어" 원래 질문과 임베딩으로 비교한다. 프롬프트가 영어라 영어 질문이
    # 나오면 한국어 질문과 비교돼 점수가 크게 깎인다 — 답변과 같은 언어로 만들게 한다.
    relevancy.question_generation.instruction += (
        " The generated question MUST be written in the same language as the given answer (e.g. Korean).")
    metrics = {"faithfulness": Faithfulness(), "answer_relevancy": relevancy,
               "context_precision": LLMContextPrecisionWithReference(), "context_recall": LLMContextRecall(),
               "factual_correctness": FactualCorrectness()}
    chosen = [metrics[m] for m in args.metrics]
    columns = {m: metrics[m].name for m in args.metrics}   # RAGAS 결과 열 이름 (예: llm_context_precision_with_reference)
    # 답이 없거나(분류 실패) 검색 문서가 없으면 RAGAS 지표를 계산할 수 없다 — 따로 센다
    gradable = [r for r in rows if r["response"] and r["retrieved_contexts"]]
    if gradable:
        dataset = EvaluationDataset.from_list([{k: r[k] for k in ("user_input", "response", "retrieved_contexts",
                                                                  "reference")} for r in gradable])
        result = evaluate(dataset, metrics=chosen, llm=llm, embeddings=emb, show_progress=True,
                          run_config=RunConfig(max_workers=args.workers, timeout=300, max_retries=3))
        df = result.to_pandas()
        for r, (_, s) in zip(gradable, df.iterrows()):
            for m in args.metrics:
                col = next((c for c in df.columns if c.startswith(columns[m])), None)
                v = s[col] if col else None
                r[m] = None if v is None or v != v else round(float(v), 3)   # NaN → None (채점 실패)
    return rows


# ------------------------------------------------------------------ 3. 보고서


def mean(vals):
    vals = [v for v in vals if v is not None]
    return round(statistics.mean(vals), 3) if vals else None


def summarize(rows: list[dict], metrics: list[str]) -> dict:
    """묶음별(음성 / 글 / 글의 출처 문서별 / 전체) 평균."""
    groups = [("음성", lambda r: r["set"] == "음성"), ("글", lambda r: r["set"] == "글")]
    groups += [(f"글 · {s}", lambda r, s=s: r["set"] == "글" and r["source"] == s)
               for s in sorted({r["source"] for r in rows if r["set"] == "글"})]
    groups.append(("전체", lambda r: True))
    out = {}
    for name, keep in groups:
        part = [r for r in rows if keep(r)]
        if not part:
            continue
        out[name] = {"n": len(part), "answered": sum(bool(r["response"]) for r in part),
                     "no_context": sum(bool(r["response"]) and not r["retrieved_contexts"] for r in part),
                     # 채점 대상인데 지표 하나라도 못 매긴 문항 (심판 모델 JSON 오류·시간 초과 등)
                     "failed": sum(bool(r["response"] and r["retrieved_contexts"])
                                   and any(r.get(m) is None for m in metrics) for r in part),
                     "latency_s": round(statistics.median(r["latency_s"] for r in part), 2),
                     **{m: mean(r.get(m) for r in part) for m in metrics}}
    return out


def fmt(v):
    return "-" if v is None else f"{v:.2f}"


def report(rows: list[dict], summary: dict, args, info: dict, out: Path) -> str:
    lines = [f"# STS 서버 RAGAS 평가 ({time.strftime('%Y-%m-%d %H:%M')})", "",
             f"- 서버: {args.server} · 모델: {info.get('model')} · RAG: {info.get('rag_backend')}",
             f"- 채점 모델: {args.judge_model} @ {args.judge_url} · 임베딩: {args.embedding_model}",
             f"- 테스트 세트: {args.testset.name if args.text else '(안 씀)'}", "",
             "| 묶음 | 문항 | 답함 | 문서 없음 | 채점 실패 | 응답(중앙값) | "
             + " | ".join(LABEL[m] for m in args.metrics) + " |",
             "|---|---|---|---|---|---|" + "---|" * len(args.metrics)]
    for name, s in summary.items():
        lines.append(f"| {name} | {s['n']} | {s['answered']} | {s['no_context']} | {s['failed']} | {s['latency_s']}초 | "
                     + " | ".join(fmt(s[m]) for m in args.metrics) + " |")
    key = args.metrics[-1]
    lines += ["", "- 답함: 서버가 답변을 낸 문항 (나머지는 잡담·위협으로 분류돼 답이 없었음 — RAGAS 점수에서 빠짐)",
              "- 문서 없음: 답은 했지만 RAG 가 문서를 못 찾은 문항 (RAGAS 점수에서 빠짐)",
              "- 채점 실패: 심판 모델이 지표를 못 매긴 문항 (그 지표 평균에서만 빠짐)",
              "", f"## 문항별 ({LABEL[key]} 낮은 순)", "",
              "| 묶음 | 문항 | 출처 | 질문 | 들은 말 | 분류 | " + " | ".join(LABEL[m] for m in args.metrics) + " |",
              "|---|---|---|---|---|---|" + "---|" * len(args.metrics)]
    for r in sorted(rows, key=lambda r: (r.get(key) is not None, r.get(key) or 0)):
        lines.append(f"| {r['set']} | {r['id']} | {r['source']} | {r['user_input']} | {r['heard']} | {r['intent']} | "
                     + " | ".join(fmt(r.get(m)) for m in args.metrics) + " |")
    text = "\n".join(lines) + "\n"
    (out / "report.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description="STS 서버 RAGAS 자동 평가")
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--text", type=int, default=30, help="글 질문 개수 (0 이면 안 함, -1 이면 테스트 세트 전부)")
    ap.add_argument("--testset", type=Path, default=DEFAULT_TESTSET,
                    help="글 질문 테스트 세트 (question, answer[, source] 열의 xlsx/csv/jsonl)")
    ap.add_argument("--no-voice", action="store_true", help="음성 질문은 건너뛴다")
    ap.add_argument("--voice-limit", type=int, default=None, help="음성 질문 최대 개수")
    ap.add_argument("--seed", type=int, default=0, help="글 질문 뽑기 시드 (같으면 매번 같은 질문)")
    ap.add_argument("--judge-url", default="http://127.0.0.1:8081/v1", help="채점 모델 OpenAI 호환 주소")
    ap.add_argument("--judge-model", default="local-gemma")
    ap.add_argument("--embedding-model", default="snunlp/KR-SBERT-V40K-klueNLI-augSTS")
    ap.add_argument("--metrics", nargs="+", default=METRICS, choices=METRICS)
    ap.add_argument("--workers", type=int, default=2, help="동시에 채점할 개수 (llama-server 슬롯 수 이하)")
    ap.add_argument("--answers", type=Path, default=None,
                    help="이전 실행의 answers.jsonl — 서버에 다시 묻지 않고 채점만 한다 (채점이 끊겼을 때)")
    ap.add_argument("--no-score", action="store_true",
                    help="답변만 모아 answers.jsonl 로 저장하고 끝낸다 (채점 모델을 바꿔 --answers 로 따로 채점할 때)")
    args = ap.parse_args()
    args.server = args.server.rstrip("/")
    Settings.from_env()   # HF_HOME 을 models/hf-cache(D 드라이브)로 맞춘다

    out = ROOT / "data" / "ragas" / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    if args.answers:
        saved = [json.loads(l) for l in args.answers.read_text(encoding="utf-8").splitlines() if l.strip()]
        info, rows = saved[0]["info"], saved[1:]
        print(f"# 저장된 답변 {args.answers} ({len(rows)}문항) · {info.get('model')}")
    else:
        client = httpx.Client(timeout=120)
        info = client.get(f"{args.server}/info").raise_for_status().json()
        print(f"# 서버 {args.server} · {info.get('model')} · RAG {info.get('rag_backend')}\n\n1) 서버에 질문")
        rows = ([] if args.no_voice else ask_voice(client, args.server, args.voice_limit))
        rows += ask_text(client, args.server, args.text, args.seed, args.testset) if args.text else []
        # 채점 전에 저장 — 채점이 끊겨도 --answers 로 채점만 다시 할 수 있다
        with (out / "answers.jsonl").open("w", encoding="utf-8") as f:
            for row in [{"info": info}, *rows]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if not rows:
        sys.exit("평가할 문항이 없습니다.")
    if args.no_score:
        print(f"\n답변만 저장함: {out / 'answers.jsonl'}")
        return

    print(f"\n2) RAGAS 채점 ({len(rows)}문항, 채점 모델 {args.judge_model}) — 답변은 {out / 'answers.jsonl'} 에 저장됨")
    t0 = time.perf_counter()
    score(rows, args)
    print(f"   채점 {time.perf_counter() - t0:.0f}초")

    import pandas as pd
    pd.DataFrame(rows).to_csv(out / "results.csv", index=False, encoding="utf-8-sig")   # 엑셀에서 한글이 안 깨지게
    summary = summarize(rows, args.metrics)
    (out / "summary.json").write_text(json.dumps({"summary": summary, "server": info,
                                                  "args": {k: str(v) for k, v in vars(args).items()}},
                                                 ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n" + report(rows, summary, args, info, out).split("## 문항별")[0])
    print(f"결과: {out}")


if __name__ == "__main__":
    main()
