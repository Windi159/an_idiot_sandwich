"""llama.cpp 모델에 직접 음성을 넣어 받아쓰기/이해/답변 품질과 속도를 확인하는 점검 스크립트.

사용: uv run python scripts/probe_llama.py [--model gemma4-12b] [--quant q8] data/samples/*.wav
llama-server 가 없으면 해당 모델로 띄웠다가 끝나면 끈다.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from sts_server import prompts
from sts_server.config import Settings
from sts_server.engines import Message
from sts_server.engines.llamacpp import LlamaCppEngine
from sts_server.models import PRESETS
from sts_server.text import parse_understand

sys.stdout.reconfigure(encoding="utf-8")
ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=list(PRESETS))
ap.add_argument("--quant")
ap.add_argument("wavs", nargs="+")
args = ap.parse_args()

s = Settings.from_env()
if args.model:
    s.model = args.model
if args.quant:
    s.quant = args.quant
eng = LlamaCppEngine(s)
t0 = time.perf_counter()
eng.load()
print(f"모델: {eng.model_id} (준비 {time.perf_counter() - t0:.1f}s)")

try:
    for wav in args.wavs:
        print(f"\n=== {Path(wav).name}")
        t0 = time.perf_counter()
        raw = eng.generate_text([Message("system", prompts.TRANSCRIBE_SYSTEM.format(language="한국어")),
                                 Message("user", prompts.TRANSCRIBE_USER, audio_path=wav)], 128, 0.0)
        print(f"[받아쓰기 {time.perf_counter() - t0:.2f}s] {raw}")

        t0 = time.perf_counter()
        raw = eng.generate_text([Message("system", prompts.UNDERSTAND_SYSTEM),
                                 Message("user", prompts.UNDERSTAND_USER, audio_path=wav)], 200, 0.0)
        print(f"[이해 {time.perf_counter() - t0:.2f}s] {parse_understand(raw)}  raw={raw!r}")

        t0 = time.perf_counter()
        first = None
        out = ""
        for piece in eng.stream_text([Message("system", prompts.QA_SYSTEM),
                                      Message("user", prompts.qa_user_text(None, [], True), audio_path=wav)],
                                     256, 0.7):
            first = first or time.perf_counter() - t0
            out += piece
        total = time.perf_counter() - t0
        print(f"[답변 첫토큰 {first:.2f}s / 전체 {total:.2f}s, ~{len(out) / max(total - first, 1e-6):.0f}자/s] {out}")
finally:
    eng.close()
