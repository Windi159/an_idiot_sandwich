"""서버 엔드포인트를 앱과 같은 형식으로 호출해 보는 클라이언트.

  uv run python scripts/try_sts.py sts  data/samples/q_psd501.wav      # 음성 → 답변(텍스트+음성)
  uv run python scripts/try_sts.py chat "PSD 502 코드가 뭐야?"           # 텍스트 → NDJSON
  uv run python scripts/try_sts.py stt  data/samples/q_door.wav         # /api/transcribe
옵션: --url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import httpx

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path(__file__).resolve().parent.parent / "outputs"


def show_stream(resp: httpx.Response, t0: float, save_prefix: str) -> None:
    n_audio = 0
    for line in resp.iter_lines():
        if not line.strip():
            continue
        ev = json.loads(line)
        audio = ev.pop("audio", None)
        stamp = f"{time.perf_counter() - t0:5.2f}s"
        if audio:
            n_audio += 1
            OUT.mkdir(exist_ok=True)
            path = OUT / f"{save_prefix}_{n_audio:02d}.{ev.get('format', 'wav')}"
            path.write_bytes(base64.b64decode(audio))
            ev["audio"] = f"<{path.name}>"
        print(stamp, json.dumps(ev, ensure_ascii=False))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["sts", "chat", "stt"])
    p.add_argument("arg")
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--no-speech", action="store_true")
    a = p.parse_args()
    client = httpx.Client(base_url=a.url, timeout=600)
    t0 = time.perf_counter()

    if a.mode == "chat":
        with client.stream("POST", "/chat", json={"text": a.arg}) as r:
            show_stream(r, t0, "chat")
    elif a.mode == "stt":
        r = client.post("/api/transcribe", files={"file": (Path(a.arg).name, Path(a.arg).read_bytes(), "audio/wav")},
                        data={"language": "ko", "task": "transcribe"})
        print(f"{time.perf_counter() - t0:5.2f}s", r.status_code, r.json())
    else:
        files = {"file": (Path(a.arg).name, Path(a.arg).read_bytes(), "audio/wav")}
        with client.stream("POST", "/sts", files=files, data={"speech": str(not a.no_speech).lower()}) as r:
            show_stream(r, t0, f"sts_{Path(a.arg).stem}")


if __name__ == "__main__":
    main()
