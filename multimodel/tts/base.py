from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Voice:
    id: str
    language: str
    gender: str = ""
    description: str = ""


class TTSEngine(ABC):
    """텍스트 → 음성 파일."""

    name: str = "base"
    #: 출력 파일 기본 확장자
    extension: str = ".wav"
    default_voice: str | None = None
    #: 문장 여러 개를 동시에 합성해도 되는 수. 로컬 엔진은 동시에 돌리면 서로 느려져 1.
    max_concurrency: int = 4

    @abstractmethod
    def synthesize(
        self,
        text: str,
        output_path: str | Path,
        voice: str | None = None,
        rate: int = 0,
    ) -> Path:
        """rate 는 -50 ~ +50 (%) 범위의 속도 조절값."""

    @abstractmethod
    def list_voices(self, language: str | None = None) -> list[Voice]:
        """language 는 'ko', 'ko-KR' 처럼 접두어로 필터링."""

    def stream(self, text: str, voice: str | None = None, rate: int = 0) -> Iterator[bytes]:
        """합성된 오디오(extension 형식)를 조각 단위로 내보낸다.

        기본 구현은 파일로 다 만든 뒤 한 조각으로 내보낸다. 진짜 스트리밍이 되는 엔진은 오버라이드한다.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            yield self.synthesize(text, Path(d) / f"tts{self.extension}", voice=voice, rate=rate).read_bytes()
