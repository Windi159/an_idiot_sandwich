"""안내방송 빈칸을 소리 낼 때만 바꿔 읽는지 (화면 문장은 원문 그대로)."""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf

from multimodel.tts.textnorm import normalize_ko
from sts_server.config import Settings
from sts_server.pipeline import STSPipeline
from sts_server.text import speak_placeholders


def test_blanks_become_natural_speech():
    text = ("우리 역에서 00시 00분에 장소로 가는 (열차명칭), (열차번호) 열차의 타는 곳이 (사유)로 "
            "00번에서 00번으로 변경됐습니다.")
    spoken = speak_placeholders(text)
    assert spoken == ("우리 역에서 몇 시 몇 분에 장소로 가는 열차명칭, 열차번호 열차의 타는 곳이 사유로 "
                      "몇 번에서 몇 번으로 변경됐습니다.")
    # 숫자 읽기(normalize_ko)를 거쳐도 "영 시 영 분"이 나오지 않는다
    assert "영" not in normalize_ko(spoken)
    assert speak_placeholders("○○역 방면 열차가 약 00분 늦어지고 있습니다.") == "해당 역 방면 열차가 약 몇 분 늦어지고 있습니다."


def test_real_numbers_and_abbreviations_are_untouched():
    for text in ("3번 출구", "2000년", "100번 버스", "010-1234-5678", "10:00 출발", "0.00초", "1,000원",
                 "(주)코레일", "(KTX)", "(1) 항목", "PSD 501 코드"):
        assert speak_placeholders(text) == text, text


class _RecordingTTS:
    extension = ".wav"
    max_concurrency = 1

    def __init__(self):
        self.heard: list[str] = []

    def stream(self, text, voice=None, rate=0, lang=None):
        self.heard.append(text)
        buf = io.BytesIO()
        sf.write(buf, np.zeros(160, dtype=np.float32), 16000, subtype="PCM_16", format="WAV")
        yield buf.getvalue()


def test_only_speech_gets_placeholders_rewritten():
    pipeline = STSPipeline(Settings(), engine=None, retriever=None)
    tts = _RecordingTTS()
    pipeline._tts = tts
    text = "00시 00분 (열차명칭) 열차가 출발합니다."
    pipeline._fallback_tts(text)
    assert tts.heard == ["몇 시 몇 분 열차명칭 열차가 출발합니다."]
    # 한국어가 아닌 음성(번역 등)에는 쓰지 않는다
    pipeline._fallback_tts("Train 00 departs.", lang="en")
    assert tts.heard[-1] == "Train 00 departs."


class _FakeEngine:
    supports_speech_output = False

    def __init__(self, text: str):
        self.text = text

    def stream_text(self, messages, max_new_tokens=256, temperature=0.7):
        yield from self.text


def test_streaming_shows_original_text_but_speaks_rewritten():
    from concurrent.futures import ThreadPoolExecutor

    answer_text = "우리 역에서 00시 00분에 (열차명칭) 열차가 출발합니다. 타는 곳 00번에서 기다려 주시기 바랍니다."
    pipeline = STSPipeline(Settings(), engine=_FakeEngine(answer_text), retriever=None)
    tts = _RecordingTTS()
    pipeline._tts, pipeline._tts_pool = tts, ThreadPoolExecutor(max_workers=1)
    answer: list[str] = []
    events = list(pipeline._fallback_speech([], answer, mode="chunk"))
    shown = [e["tts"] for e in events if "tts" in e]
    assert shown == ["우리 역에서 00시 00분에 (열차명칭) 열차가 출발합니다.", "타는 곳 00번에서 기다려 주시기 바랍니다."]
    assert tts.heard == ["우리 역에서 몇 시 몇 분에 열차명칭 열차가 출발합니다.", "타는 곳 몇 번에서 기다려 주시기 바랍니다."]
    assert sum(1 for e in events if e.get("audio_end")) == 2
