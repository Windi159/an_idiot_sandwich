"""GPU·모델 없이 API 와 앱 연동을 점검하기 위한 가짜 엔진. 프롬프트 종류를 보고 그럴듯한 응답을 낸다."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator

import numpy as np

from sts_server.config import Settings
from sts_server.engines.base import Message, OmniEngine, SpeechResult


class MockEngine(OmniEngine):
    name = "mock"
    supports_speech_output = True

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _reply(self, messages: list[Message]) -> str:
        system = next((m.text or "" for m in messages if m.role == "system"), "")
        user = messages[-1]
        heard = user.text or ""
        if "[MODE:TRANSCRIBE]" in system:
            return "PSD 501 코드가 뭐야?"
        if "[MODE:UNDERSTAND]" in system:
            return json.dumps({"transcript": "PSD 501 코드가 뭐야?", "intent": "qa"}, ensure_ascii=False)
        if "[MODE:CLASSIFY]" in system:
            return "danger" if any(w in heard for w in ("죽여", "씨발", "때릴")) else "qa"
        if "[MODE:TRANSLATE]" in system:
            return "What is the PSD 501 code?"
        if "[MODE:IMAGE]" in system:
            return "모의 사진 설명이야. 승강장에 특이 사항은 안 보여."
        return "모의 응답이야. 실제 모델을 연결하면 여기에 답변이 나와. 서버와 앱 연동은 정상이야."

    def stream_text(self, messages: list[Message], max_new_tokens: int = 256,
                    temperature: float = 0.7) -> Iterator[str]:
        for ch in self._reply(messages):
            time.sleep(0.002)
            yield ch

    def generate_speech(self, messages: list[Message], max_new_tokens: int = 256, on_text=None) -> SpeechResult:
        text = ""
        for piece in self.stream_text(messages, max_new_tokens):
            text += piece
            if on_text:
                on_text(piece)
        sr = 24000
        t = np.linspace(0, 0.5, sr // 2, endpoint=False)
        tone = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        return SpeechResult(text=text, audio=tone, sample_rate=sr)
