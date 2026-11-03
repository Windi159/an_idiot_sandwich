"""MoE expert 를 CPU 로 내릴 레이어 수(--n-cpu-moe) 찾기 — VRAM 에 들어가는 가장 작은 값이 가장 빠르다.

  uv run python scripts/tune_cpu_moe.py --model qwen3-omni-30b --quant fp4 --try 4 6 8 10
각 값으로 llama-server 를 띄워 음성 질문 1개를 처리한 뒤 VRAM 사용량을 잰다.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sts_server import prompts  # noqa: E402
from sts_server.config import Settings  # noqa: E402
from sts_server.engines import Message  # noqa: E402
from sts_server.engines.llamacpp import LlamaCppEngine  # noqa: E402

TOTAL_MB = 16303  # RTX 5070 Ti


def vram() -> tuple[int, int]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().split(",")
    return int(out[0]), int(out[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-omni-30b")
    ap.add_argument("--quant", default="fp4")
    ap.add_argument("--try", dest="values", type=int, nargs="+", default=[4, 6, 8, 10])
    ap.add_argument("--wav", default=str(ROOT / "data" / "eval" / "07_qa.wav"))
    a = ap.parse_args()
    base_used, total = vram()
    print(f"기준 VRAM {base_used}/{total} MB")
    for n in a.values:
        s = Settings.from_env()
        s.model, s.quant, s.llama_n_cpu_moe = a.model, a.quant, n
        eng = LlamaCppEngine(s)
        try:
            eng.load()
            msgs = [Message("system", prompts.UNDERSTAND_SYSTEM),
                    Message("user", prompts.UNDERSTAND_USER, audio_path=a.wav)]
            t0 = time.perf_counter()
            out = eng.generate_text(msgs, 200, 0.0)
            el = time.perf_counter() - t0
            used, _ = vram()
            margin = total - used
            ok = "✅" if margin >= 400 else "⚠️ 여유 부족"
            print(f"n_cpu_moe={n:2d}: VRAM {used} MB (여유 {margin} MB) {ok}  이해 {el:.2f}s  {out[:60]!r}")
        except Exception as e:  # noqa: BLE001
            print(f"n_cpu_moe={n:2d}: 실패 — {e}")
        finally:
            eng.close()
            time.sleep(3)


if __name__ == "__main__":
    main()
