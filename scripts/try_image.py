"""폰 앱 ImageTransferController.deliver 와 똑같은 방식으로 /api/image 를 불러본다.

  uv run python scripts/try_image.py 사진.jpg [더.jpg ...] [--question "이거 뭐야?"] [--url https://desktop-horuhm4.snowfinch-leaffish.ts.net]
"""

from __future__ import annotations

import argparse
import base64
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from try_app_protocol import fetch_speech  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--question", default="")
    ap.add_argument("--url", default="https://desktop-horuhm4.snowfinch-leaffish.ts.net")
    ap.add_argument("--path", default="/api/image")
    ap.add_argument("--speech", action="store_true", help="서버 음성도 받아 본다")
    a = ap.parse_args()
    client = httpx.Client(timeout=60)
    for img in map(Path, a.images):
        body = {"image": base64.b64encode(img.read_bytes()).decode(), "source": "글라스", "userId": "시험"}
        if a.speech:
            body["speech"] = True
        if a.question:
            body["question"] = a.question
        t0 = time.perf_counter()
        r = client.post(a.url.rstrip("/") + a.path, json=body)
        el = time.perf_counter() - t0
        if r.status_code != 200:
            print(f"{img.name}: HTTP {r.status_code} {r.text[:200]}")
            continue
        answer = (r.json().get("answer") or r.json().get("text") or "").strip()
        print(f"{img.name}  {el:4.1f}s (서버 {r.json().get('elapsed')}s)\n    답변: {answer or '(빈 답변)'}")
        if r.json().get("audio_id"):
            fetch_speech(client, a.url, r.json()["audio_id"], img.stem, t0)


if __name__ == "__main__":
    main()
