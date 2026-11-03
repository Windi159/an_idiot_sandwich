from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Transcription:
    text: str
    language: str | None = None
    duration: float = 0.0
    segments: list[Segment] = field(default_factory=list)


class STTEngine(ABC):
    """음성 파일 → 텍스트."""

    name: str = "base"

    @abstractmethod
    def transcribe(self, audio_path: str | Path, language: str | None = None) -> Transcription:
        """language=None 이면 자동 감지."""
