"""/sts 지연 측정: 요청 시작부터 각 이벤트가 클라이언트에 도착하기까지의 시간.

  uv run python scripts/bench_sts.py data/samples/*.wav --repeat 3 [--label 개선전]

측정 항목 (중앙값)
  분류      첫 줄({"type": ...}) 도착
  첫문장    첫 {"tts": ...} 도착
  첫음성    첫 audio 데이터 도착  ← 사용자가 소리를 듣기 시작할 수 있는 시점
  완료      {"done": true} 도착
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

sys.stdout.reconfigure(encoding="utf-8")


def run_once(client: httpx.Client, wav: Path, data: dict) -> dict:
    t0 = time.perf_counter()
    marks: dict[str, float] = {}
    audio_bytes = 0
    with client.stream("POST", "/sts", files={"file": (wav.name, wav.read_bytes(), "audio/wav")}, data=data) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.strip():
                continue
            ev = json.loads(line)
            now = time.perf_counter() - t0
            if "type" in ev:
                marks.setdefault("분류", now)
                if ev["type"] != "qa":
                    marks.setdefault("완료", now)
            if ev.get("tts"):
                marks.setdefault("첫문장", now)
            if ev.get("audio"):
                marks.setdefault("첫음성", now)
                audio_bytes += len(ev["audio"]) * 3 // 4
            if ev.get("done"):
                marks["완료"] = now
    marks["audio_kb"] = audio_bytes / 1024
    return marks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--label", default="")
    ap.add_argument("--form", action="append", default=[], help="추가 form 필드 key=value")
    a = ap.parse_args()
    data = dict(kv.split("=", 1) for kv in a.form)
    client = httpx.Client(base_url=a.url, timeout=600)
    info = client.get("/info").json()
    print(f"# {a.label}  model={info.get('model')}  tts={info.get('speech_output')}  form={data}")

    # 첫 요청은 워밍업 (모델 캐시, TTS 연결 등)
    run_once(client, Path(a.wavs[0]), data)

    keys = ["분류", "첫문장", "첫음성", "완료"]
    print(f"{'샘플':14s} " + " ".join(f"{k:>7s}" for k in keys) + "   (중앙값 s, 범위)")
    allm: dict[str, list[float]] = {k: [] for k in keys}
    for wav in map(Path, a.wavs):
        runs = [run_once(client, wav, data) for _ in range(a.repeat)]
        cells = []
        for k in keys:
            vals = [r[k] for r in runs if k in r]
            allm[k] += vals
            cells.append(f"{statistics.median(vals):7.2f}" if vals else f"{'-':>7s}")
        spread = [r.get("첫음성") for r in runs if "첫음성" in r]
        rng = f"  첫음성 {min(spread):.2f}~{max(spread):.2f}" if spread else ""
        print(f"{wav.stem:14s} " + " ".join(cells) + rng)
    print(f"{'전체':14s} " + " ".join(f"{statistics.median(v):7.2f}" if v else f"{'-':>7s}" for v in allm.values()))


if __name__ == "__main__":
    main()
