"""엔진 레지스트리.

새 엔진을 추가하려면 클래스를 구현하고 아래 딕셔너리에 "모듈:클래스" 경로를 한 줄 추가하면 된다.
무거운 의존성(torch, ctranslate2 등)은 실제로 엔진을 쓸 때만 import 된다.
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Any

from multimodel.stt.base import STTEngine
from multimodel.tts.base import TTSEngine

STT_ENGINES: dict[str, str] = {
    "whisper": "multimodel.stt.whisper:WhisperSTT",
}

TTS_ENGINES: dict[str, str] = {
    "edge": "multimodel.tts.edge:EdgeTTS",              # 온라인 (Microsoft)
    "windows": "multimodel.tts.windows:WindowsTTS",     # 로컬, Windows 내장 음성
    "supertonic": "multimodel.tts.supertonic:SupertonicTTS",  # 로컬, CPU (Supertone)
}

DEFAULT_STT = "whisper"
DEFAULT_TTS = "edge"


def _load_class(path: str) -> type:
    module_name, class_name = path.split(":")
    return getattr(importlib.import_module(module_name), class_name)


def _resolve(table: dict[str, str], name: str, kind: str) -> type:
    if name not in table:
        raise ValueError(f"알 수 없는 {kind} 엔진: {name!r} (사용 가능: {', '.join(table)})")
    return _load_class(table[name])


@lru_cache(maxsize=8)
def _cached(kind: str, name: str, frozen_kwargs: tuple[tuple[str, Any], ...]):
    table = STT_ENGINES if kind == "STT" else TTS_ENGINES
    return _resolve(table, name, kind)(**dict(frozen_kwargs))


def get_stt(name: str = DEFAULT_STT, **kwargs: Any) -> STTEngine:
    """같은 설정이면 모델을 다시 로드하지 않도록 인스턴스를 캐시한다."""
    return _cached("STT", name, tuple(sorted(kwargs.items())))


def get_tts(name: str = DEFAULT_TTS, **kwargs: Any) -> TTSEngine:
    return _cached("TTS", name, tuple(sorted(kwargs.items())))


def list_stt_engines() -> list[str]:
    return list(STT_ENGINES)


def list_tts_engines() -> list[str]:
    return list(TTS_ENGINES)
