"""post_sh-t main 앱과 똑같은 방식으로 질의응답 서버를 불러본다 (VoiceServerProtocol.uploadRequest / response).

  uv run python scripts/try_app_protocol.py data/eval/*.wav [--url https://desktop-horuhm4.snowfinch-leaffish.ts.net] [--path /api/transcribe]
  --speech : 서버 음성도 요청해서 audio_id 로 받아 본다 (data/try_speech/ 에 저장)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx

sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--url", default="https://desktop-horuhm4.snowfinch-leaffish.ts.net")
    ap.add_argument("--path", default="/api/transcribe")
    ap.add_argument("--speech", action="store_true")
    a = ap.parse_args()
    client = httpx.Client(timeout=60)
    for wav in map(Path, a.wavs):
        t0 = time.perf_counter()
        r = client.post(a.url.rstrip("/") + a.path,
                        files={"file": ("query.wav", wav.read_bytes(), "audio/wav")},
                        data={"language": "ko", "task": "transcribe", **({"speech": "1"} if a.speech else {})})
        el = time.perf_counter() - t0
        if r.status_code != 200:
            print(f"{wav.name}: HTTP {r.status_code} {r.text[:200]}")
            continue
        j = r.json()
        # 앱과 같은 규칙: answer 우선, 없으면 text
        answer = (j.get("answer") or j.get("text") or "").strip()
        spoken = f"폰 TTS: {answer}" if answer else "(빈 답변 → 앱은 말하지 않음)"
        print(f"{wav.name}  {el:4.1f}s  [{j.get('intent')}{' ⚠위협' if j.get('threat') else ''}]  "
              f"Q: {j.get('transcript', '')}\n    {spoken}")
        if j.get("audio_id"):
            fetch_speech(client, a.url, j["audio_id"], wav.stem, t0)


def fetch_speech(client: httpx.Client, url: str, audio_id: str, name: str, t0: float) -> None:
    """앱 AnswerSpeaker 와 같은 방식으로 답변 음성을 받는다. t0 은 질문을 보낸 시각."""
    import soundfile as sf
    r = client.get(f"{url.rstrip('/')}/api/audio/{audio_id}")
    el = time.perf_counter() - t0
    if r.status_code != 200:
        print(f"    서버 음성: HTTP {r.status_code}")
        return
    out = Path(__file__).resolve().parent.parent / "data" / "try_speech" / f"{name}.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(r.content)
    info = sf.info(str(out))
    print(f"    서버 음성: 질문부터 {el:.1f}s 만에 받음 · {info.duration:.1f}초 길이 · {info.samplerate}Hz "
          f"{info.subtype} · {len(r.content) // 1024}KB → {out}")


if __name__ == "__main__":
    main()
