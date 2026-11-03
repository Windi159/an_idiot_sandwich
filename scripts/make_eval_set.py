"""모델·양자화 비교용 음성 평가셋 만들기 → data/eval/manifest.jsonl + wav

- qa     : 앱 QA 데이터(train_data_standardized.jsonl)에서 고른 질문 (정답 답변 포함)
- ignore : 비서에게 하는 말이 아닌 잡담
- danger : 직원을 향한 위협·폭언
음성은 TTS 4종(Edge 남/여, Supertonic 남/여)을 돌아가며 써서 목소리를 섞는다.
※ 실제 사람 목소리·잡음 환경이 아니라서 받아쓰기 정확도는 실제보다 좋게 나온다.

  uv run python scripts/make_eval_set.py [--n-qa 14]
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from multimodel import get_tts  # noqa: E402
from sts_server.audio import normalize_audio  # noqa: E402

QA_JSONL = Path(r"D:\post_sh-t\app\src\main\assets\train_data_standardized.jsonl")
OUT = ROOT / "data" / "eval"

# 프롬프트(prompts.INTENT_RULES)의 예시 문장과 겹치지 않게 할 것 — 겹치면 점수가 부풀려진다
IGNORE = [
    "아 오늘 퇴근하고 뭐 하지.",
    "네 알겠습니다, 좋은 하루 보내세요.",
    "어제 축구 봤어? 진짜 재밌더라.",
    "음 그러니까 그게 말이야.",
]
DANGER = [
    "야 너 거기 서. 오늘 너 죽었어 진짜.",
    "너 같은 건 좀 맞아야 정신 차리지 이 새끼야.",
    "한 번만 더 그러면 때려버린다.",
]
VOICES = [("edge", "ko-KR-InJoonNeural"), ("edge", "ko-KR-SunHiNeural"), ("supertonic", "M1"), ("supertonic", "F1")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-qa", type=int, default=14)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    rows = [json.loads(l) for l in QA_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
    random.Random(a.seed).shuffle(rows)
    # 너무 긴 질문은 빼고 고른다 (음성 질문은 보통 짧다)
    qa = [r for r in rows if len(r["question"]) <= 40][: a.n_qa]

    items = [{"text": r["question"], "intent": "qa", "reference": r["answer"]} for r in qa]
    items += [{"text": t, "intent": "ignore"} for t in IGNORE]
    items += [{"text": t, "intent": "danger"} for t in DANGER]

    OUT.mkdir(parents=True, exist_ok=True)
    engines = {}
    with (OUT / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for i, it in enumerate(items):
            eng_name, voice = VOICES[i % len(VOICES)]
            eng = engines.setdefault(eng_name, get_tts(eng_name))
            tmp = OUT / f"_tmp{eng.extension}"
            eng.synthesize(it["text"], tmp, voice=voice)
            wav = normalize_audio(tmp.read_bytes())
            dst = OUT / f"{i:02d}_{it['intent']}.wav"
            shutil.move(wav, dst)
            tmp.unlink()
            it.update(file=dst.name, voice=f"{eng_name}:{voice}")
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
            print(f"{dst.name}  [{it['voice']}]  {it['text']}")
    print(f"\n{len(items)}개 → {OUT / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
