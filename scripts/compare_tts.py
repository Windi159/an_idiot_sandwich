"""로컬 TTS 비교: 같은 답변 문장을 엔진별로 합성해 속도·발음 정확도를 재고 샘플 파일을 남긴다.

  uv run python scripts/compare_tts.py [--engines supertonic edge] [--voices supertonic=F1,M1]

- 속도: 문장별 합성 시간과 RTF(합성 시간 / 음성 길이, 1 미만이면 실시간보다 빠름)
- 발음: 합성한 음성을 STS 서버의 /api/transcribe 로 다시 받아써서 원문과 글자 단위로 비교 (서버가 떠 있을 때만)
- 샘플: outputs/tts_compare/<엔진>_<음성>_<번호>.wav|mp3 — 직접 들어보고 고르기
"""

from __future__ import annotations

import argparse
import difflib
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from multimodel import get_tts  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "outputs" / "tts_compare"
SENTENCES = [
    "501번은 하부 가이드레일 이물질 끼임이야.",
    "승객 안전부터 확인하고 발판 사이를 점검해봐.",
    "승객이 선로로 접근하지 못하게 즉시 해당 구역 통제부터 해.",
    "관제실에 3번 승강장 PSD 비상문 개방이라고 바로 보고해.",
    "알겠어, 잠깐만 기다려줘.",
]


def norm(s: str) -> str:
    return re.sub(r"[\s.,!?'\"~·]", "", s)


def char_acc(ref: str, hyp: str) -> float:
    return difflib.SequenceMatcher(None, norm(ref), norm(hyp)).ratio()


def vram_mb() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", nargs="+", default=["supertonic", "edge"])
    ap.add_argument("--voices", default="supertonic=F1,M1;edge=ko-KR-InJoonNeural")
    ap.add_argument("--stt-url", default="http://127.0.0.1:8000")
    a = ap.parse_args()
    voices = {k: v.split(",") for k, v in (item.split("=") for item in a.voices.split(";"))}
    OUT.mkdir(parents=True, exist_ok=True)

    stt = httpx.Client(base_url=a.stt_url, timeout=60)
    try:
        stt_ok = stt.get("/health").status_code == 200
    except httpx.HTTPError:
        stt_ok = False
    if not stt_ok:
        print("(STS 서버가 없어 발음 확인은 건너뜀)")

    for name in a.engines:
        tts = get_tts(name)
        before = vram_mb()
        t0 = time.perf_counter()
        tts.synthesize("준비.", OUT / f"_warmup{tts.extension}")  # 모델 로드·워커 기동
        print(f"\n## {name}  (준비 {time.perf_counter() - t0:.1f}s, VRAM {before}→{vram_mb()} MB)")
        for voice in voices.get(name, [None]):
            rtfs, accs = [], []
            for i, sent in enumerate(SENTENCES, 1):
                path = OUT / f"{name}_{voice}_{i}{tts.extension}"
                t0 = time.perf_counter()
                tts.synthesize(sent, path, voice=voice)
                el = time.perf_counter() - t0
                dur = sf.info(str(path)).duration if path.suffix == ".wav" else float("nan")
                rtf = el / dur if dur == dur and dur > 0 else float("nan")
                rtfs.append(rtf)
                heard = ""
                if stt_ok:
                    r = stt.post("/api/transcribe", files={"file": (path.name, path.read_bytes())},
                                 data={"language": "ko"})
                    heard = r.json().get("text", "")
                    accs.append(char_acc(sent, heard))
                print(f"  [{voice}] {el:5.2f}s / 음성 {dur:4.1f}s  RTF {rtf:4.2f}  "
                      + (f"정확도 {accs[-1]:.0%}  '{heard}'" if heard else sent))
            avg_rtf = sum(x for x in rtfs if x == x) / max(1, sum(1 for x in rtfs if x == x))
            avg_acc = f"{sum(accs) / len(accs):.0%}" if accs else "-"
            print(f"  → [{voice}] 평균 RTF {avg_rtf:.2f}, 평균 발음 정확도 {avg_acc}")
        if hasattr(tts, "close"):
            tts.close()
    print(f"\n샘플: {OUT}")


if __name__ == "__main__":
    main()
