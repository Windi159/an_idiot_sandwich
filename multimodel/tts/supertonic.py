"""Supertonic 3 (Supertone) — 로컬 ONNX TTS. 31개 언어(한국어 포함), 프리셋 음성 F1~F5 / M1~M5.

onnxruntime-gpu 가 깔려 있으면 GPU(CUDA)로 돈다 (두 문장 CPU ~1.0s → GPU ~0.23s, VRAM ~1GB).
SUPERTONIC_DEVICE=cpu 면 GPU 가 있어도 CPU 로.
"""

from __future__ import annotations

import io
import os
import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import soundfile as sf

from multimodel.tts.base import TTSEngine, Voice
from multimodel.tts.textnorm import normalize_ko

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(os.environ.get("SUPERTONIC_MODEL_DIR", ROOT / "models" / "supertonic-3"))
DEVICE = os.environ.get("SUPERTONIC_DEVICE", "auto")   # auto | cpu


def _use_cuda() -> bool:
    """supertonic 은 CPU provider 로 고정돼 있다 — CUDA 를 쓸 수 있으면 provider 목록을 바꿔 둔다."""
    import onnxruntime as ort
    if DEVICE == "cpu" or "CUDAExecutionProvider" not in ort.get_available_providers():
        return False
    try:
        # torch 를 먼저 올려 torch 에 든 cuDNN(9.19)을 같이 쓰게 한다. pip 의 nvidia-cudnn-cu12 9.27 은
        # onnxruntime-gpu 1.26 과 RTX 50xx 에서 Conv 가 CUDNN_BACKEND_API_FAILED 로 실패한다 (2026-10-29 확인)
        import torch  # noqa: F401
    except ImportError:
        pass
    ort.preload_dlls()
    import supertonic.loader
    supertonic.loader.DEFAULT_ONNX_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return True


class SupertonicTTS(TTSEngine):
    name = "supertonic"
    extension = ".wav"
    default_voice = "F1"  # 직접 들어보고 고른 음성. 발음 정확도 99% (M1 100%, 나머지는 85~95%)
    voices = [f"{g}{i}" for g in "FM" for i in range(1, 6)]
    max_concurrency = 1   # CPU 를 나눠 쓰면 첫 문장이 늦어진다. 실시간의 10배 속도라 순서대로 해도 충분
    # stream(lang=...) 으로 고를 수 있는 언어 (supertonic.config.SUPPORTED_LANGUAGES — 중국어는 없다)
    languages = ("en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi", "fr", "hi", "hr", "hu",
                 "id", "it", "lt", "lv", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi")

    def __init__(self, steps: int = 8, lang: str = "ko") -> None:
        self.steps = steps          # 합성 단계 수: 높을수록 품질↑ 속도↓ (기본 8)
        self.lang = lang
        self.device = "cpu"         # 처음 불러올 때 정해진다 (cuda | cpu)
        self._tts = None
        self._styles: dict = {}
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._tts is None:
                self.device = "cuda" if _use_cuda() else "cpu"
                from supertonic import TTS

                # 처음 한 번 Hugging Face(Supertone/supertonic-3)에서 models/supertonic-3 으로 받는다
                self._tts = TTS(model="supertonic-3", model_dir=MODEL_DIR, auto_download=True)
            return self._tts

    def _style(self, voice: str):
        tts = self._load()
        if voice not in self._styles:
            self._styles[voice] = tts.get_voice_style(voice_name=voice)
        return self._styles[voice]

    def _wav(self, text: str, voice: str | None, rate: int, lang: str | None = None) -> tuple[np.ndarray, int]:
        if not text.strip():
            raise ValueError("텍스트가 비어 있습니다.")
        tts = self._load()
        lang = lang or self.lang
        if lang == "ko":
            text = normalize_ko(text)  # Supertonic 은 숫자를 못 읽는다 ("501번" → "오백일 번")
        speed = 1.05 * (1 + rate / 100)
        wav, _ = tts.synthesize(text, voice_style=self._style(voice or self.default_voice),
                                total_steps=self.steps, speed=speed, lang=lang)
        return np.asarray(wav, dtype=np.float32).reshape(-1), tts.sample_rate

    def synthesize(self, text: str, output_path: str | Path, voice: str | None = None, rate: int = 0,
                   lang: str | None = None) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        wav, sr = self._wav(text, voice, rate, lang)
        sf.write(output_path, wav, sr, subtype="PCM_16")
        return output_path

    def stream(self, text: str, voice: str | None = None, rate: int = 0, lang: str | None = None) -> Iterator[bytes]:
        """lang 을 주면 이 문장만 그 언어로 읽는다 (기본은 만들 때 정한 self.lang)."""
        # 문장 하나가 수십 ms 에 끝나서 조각으로 나눌 이득이 없다 — 파일 없이 메모리에서 바로
        wav, sr = self._wav(text, voice, rate, lang)
        buf = io.BytesIO()
        sf.write(buf, wav, sr, subtype="PCM_16", format="WAV")
        yield buf.getvalue()

    def list_voices(self, language: str | None = None) -> list[Voice]:
        # 31개 언어 모두 같은 프리셋 음성을 쓴다 — 언어 필터 없음
        return [Voice(id=v, language="multi", gender="Female" if v[0] == "F" else "Male") for v in self.voices]
