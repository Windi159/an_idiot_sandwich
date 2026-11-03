"""모델·양자화 비교 — data/eval 평가셋(make_eval_set.py)으로 받아쓰기·분류·답변·속도를 잰다.

  uv run python scripts/eval_quant.py --model qwen3-omni-30b --quant fp4
  uv run python scripts/eval_quant.py --summary          # 저장된 결과들을 표로

측정 (음성 → 서버 파이프라인과 같은 단계, TTS 제외)
  받아쓰기 정확도 : 원문과 글자 단위 일치율 (공백·문장부호 무시)
  분류 정확도     : qa / ignore / danger 가 맞은 비율 (위협 단어 가드 적용 후)
  답변 유사도     : RAG 정답 답변과의 글자 단위 유사도 (qa 로 맞게 분류된 것만, 대략적인 지표)
  이해 시간       : 음성 → 받아쓰기+분류 JSON
  답변 첫 토큰 / 생성 속도(자/s)
llama-server 가 없으면 해당 모델로 띄웠다가 끝나면 끈다. 결과는 data/eval/results_<model>_<quant>.json
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sts_server.config import Settings  # noqa: E402
from sts_server.engines.llamacpp import LlamaCppEngine  # noqa: E402
from sts_server.pipeline import STSPipeline  # noqa: E402
from sts_server.rag import create_retriever  # noqa: E402

EVAL = ROOT / "data" / "eval"


def norm(s: str) -> str:
    return re.sub(r"[\s.,!?'\"~·…]", "", s).lower()


def sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def vram_mb() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return -1


def run(model: str, quant: str) -> dict:
    s = Settings.from_env()
    s.model, s.quant = model, quant
    s.rag_backend = "jsonl"
    s.rag_jsonl = s.rag_jsonl or r"D:\post_sh-t\app\src\main\assets\train_data_standardized.jsonl"
    eng = LlamaCppEngine(s)
    t0 = time.perf_counter()
    eng.load()
    load_s = time.perf_counter() - t0
    vram = vram_mb()
    pipe = STSPipeline(s, eng, create_retriever(s))
    items = [json.loads(l) for l in (EVAL / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"# {eng.model_id}  (준비 {load_s:.1f}s, VRAM {vram} MB, CPU expert 레이어 {eng.n_cpu_moe})")

    # 워밍업
    pipe.understand(EVAL / items[0]["file"])

    rows = []
    try:
        for it in items:
            wav = EVAL / it["file"]
            t0 = time.perf_counter()
            transcript, intent = pipe.understand(wav)
            und = time.perf_counter() - t0
            row = {"file": it["file"], "expect": it["intent"], "got": intent, "transcript": transcript,
                   "asr": sim(it["text"], transcript), "understand_s": und}
            if it["intent"] == "qa" and intent == "qa":
                t0 = time.perf_counter()
                first, answer = None, ""
                for piece in eng.stream_text(pipe._qa_messages(transcript, wav), max_new_tokens=s.max_answer_tokens,
                                             temperature=0.0):
                    first = first or time.perf_counter() - t0
                    answer += piece
                total = time.perf_counter() - t0
                row.update(answer=answer.strip(), first_token_s=first or total,
                           chars_per_s=len(answer) / max(total - (first or 0), 1e-6),
                           answer_sim=sim(it["reference"], answer))
            mark = "✅" if intent == it["intent"] else "❌"
            print(f"  {it['file']:14s} {mark} {it['intent']:>6s}→{intent:<6s} 받아쓰기 {row['asr']:4.0%}  "
                  f"이해 {und:.2f}s  '{transcript}'")
            rows.append(row)
    finally:
        eng.close()

    qa_ans = [r for r in rows if "answer" in r]
    summary = {
        "model": model, "quant": quant, "model_id": eng.model_id, "vram_mb": vram, "n_cpu_moe": eng.n_cpu_moe,
        "n": len(rows),
        "asr": statistics.mean(r["asr"] for r in rows),
        "intent_acc": sum(r["expect"] == r["got"] for r in rows) / len(rows),
        "understand_s": statistics.median(r["understand_s"] for r in rows),
        "first_token_s": statistics.median(r["first_token_s"] for r in qa_ans) if qa_ans else None,
        "chars_per_s": statistics.median(r["chars_per_s"] for r in qa_ans) if qa_ans else None,
        "answer_sim": statistics.mean(r["answer_sim"] for r in qa_ans) if qa_ans else None,
    }
    out = EVAL / f"results_{model}_{quant or 'default'}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print_summary([summary])
    return summary


def print_summary(sums: list[dict]) -> None:
    print(f"\n{'모델/양자화':28s} {'VRAM':>7s} {'받아쓰기':>7s} {'분류':>6s} {'답변유사':>7s} {'이해':>6s} {'첫토큰':>6s} {'자/s':>6s}")
    for s in sums:
        f = lambda v, fmt: (fmt.format(v) if v is not None else "-")  # noqa: E731
        print(f"{s['model'] + ' ' + s['quant']:28s} {s['vram_mb']:>5d}MB {s['asr']:>7.1%} {s['intent_acc']:>6.0%} "
              f"{f(s['answer_sim'], '{:>7.1%}')} {s['understand_s']:>5.2f}s {f(s['first_token_s'], '{:>5.2f}s')} "
              f"{f(s['chars_per_s'], '{:>6.0f}')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--quant", default="")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        sums = [json.loads(p.read_text(encoding="utf-8"))["summary"] for p in sorted(EVAL.glob("results_*.json"))]
        print_summary(sums)
        return
    run(a.model, a.quant)


if __name__ == "__main__":
    main()
