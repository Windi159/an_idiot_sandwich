"""팀 QA 테스트 세트(글 253문항)의 질문을 TTS 로 읽어 음성 테스트 세트 만들기 → data/eval_testset/manifest.jsonl + wav

목소리는 Supertonic F1 하나 (서버 기본 음성).
manifest 한 줄은 data/eval/manifest.jsonl 과 같은 모양(text, intent, reference, file, voice)에
source(출처 문서)와 row(테스트 세트에서 몇 번째 문항인지)를 더한 것.
※ 합성 음성이라 받아쓰기 정확도는 실제 사람 목소리·잡음 환경보다 좋게 나온다.

  SUPERTONIC_DEVICE=cpu uv run python scripts/make_testset_audio.py   # 서버가 GPU 를 쓰는 중이면 CPU 로
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from multimodel import get_tts  # noqa: E402
from ragas_eval import DEFAULT_TESTSET, load_testset  # noqa: E402
from sts_server.audio import normalize_audio  # noqa: E402

OUT = ROOT / "data" / "eval_testset"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", type=Path, default=DEFAULT_TESTSET)
    ap.add_argument("--voice", default="F1", help="Supertonic 음성 (F1~F5, M1~M5)")
    ap.add_argument("--limit", type=int, default=None, help="앞에서부터 이 개수만 (시험용)")
    a = ap.parse_args()

    pairs = load_testset(a.testset)[: a.limit]
    OUT.mkdir(parents=True, exist_ok=True)
    eng = get_tts("supertonic")
    for old in OUT.glob("*.wav"):   # 예전에 만든 음성은 지운다
        old.unlink()
    with (OUT / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for i, p in enumerate(pairs):
            tmp = OUT / f"_tmp{eng.extension}"
            eng.synthesize(p["question"], tmp, voice=a.voice)
            dst = OUT / f"{i:03d}_qa.wav"
            shutil.move(normalize_audio(tmp.read_bytes(), out_dir=OUT), dst)
            tmp.unlink()
            it = {"text": p["question"], "intent": "qa", "reference": p["answer"], "source": p["source"],
                  "row": i, "file": dst.name, "voice": f"supertonic:{a.voice}"}
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
            print(f"{dst.name}  [{it['voice']}]  {p['question']}", flush=True)
    print(f"\n{len(pairs)}개 → {OUT / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
