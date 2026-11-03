from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass
class Message:
    role: str                      # system | user | assistant
    text: str | None = None
    audio_path: str | None = None  # 16 kHz mono WAV (audio.normalize_audio 로 만든 파일)
    image_path: str | None = None  # JPEG/PNG/WebP (audio.save_image 로 만든 파일)


@dataclass
class SpeechResult:
    text: str
    audio: np.ndarray | None       # float32 mono
    sample_rate: int = 24000


class OmniEngine(ABC):
    """음성/텍스트를 입력받아 텍스트(그리고 가능하면 음성)를 내는 멀티모달 모델."""

    name = "base"
    #: 모델이 직접 음성을 생성할 수 있는지 (False 면 서버가 fallback TTS 로 읽는다)
    supports_speech_output = False

    def load(self) -> None:
        """무거운 초기화. 서버 시작 시 한 번 호출."""

    @property
    def model_id(self) -> str:
        return self.name

    @abstractmethod
    def stream_text(self, messages: list[Message], max_new_tokens: int = 256,
                    temperature: float = 0.7) -> Iterator[str]:
        """텍스트 응답을 토큰 조각 단위로 흘려보낸다."""

    def generate_text(self, messages: list[Message], max_new_tokens: int = 256,
                      temperature: float = 0.0) -> str:
        return "".join(self.stream_text(messages, max_new_tokens, temperature)).strip()

    def generate_speech(self, messages: list[Message], max_new_tokens: int = 256,
                        on_text: callable | None = None) -> SpeechResult:
        """텍스트 + 음성을 한 번에 생성. on_text 로 텍스트 조각을 먼저 받아볼 수 있다."""
        raise NotImplementedError(f"{self.name} 엔진은 음성 출력을 지원하지 않습니다.")

    def close(self) -> None:
        pass
