"""faster-whisper 기반 STT (CTranslate2)."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from multimodel.stt.base import Segment, STTEngine, Transcription

log = logging.getLogger(__name__)

_dll_dirs_added = False


def _add_nvidia_dll_dirs() -> None:
    """pip 로 설치한 nvidia-cublas/cudnn 의 DLL 경로를 등록 (Windows, CUDA Toolkit 미설치 환경용)."""
    global _dll_dirs_added
    if _dll_dirs_added or sys.platform != "win32":
        return
    _dll_dirs_added = True
    for site in sys.path:
        nvidia_root = Path(site) / "nvidia"
        if not nvidia_root.is_dir():
            continue
        for bin_dir in nvidia_root.glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            # ctranslate2 는 LoadLibrary 로 cuDNN 을 찾기 때문에 PATH 에도 추가
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"


class WhisperSTT(STTEngine):
    name = "whisper"

    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        device: str = "auto",
        compute_type: str = "auto",
        beam_size: int = 5,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self._model = None
        self.active_device: str | None = None

    def _load(self):
        if self._model is not None:
            return self._model
        _add_nvidia_dll_dirs()
        from faster_whisper import WhisperModel

        candidates: list[tuple[str, str]]
        if self.device == "auto":
            candidates = [("cuda", "float16"), ("cpu", "int8")]
        else:
            default_ct = "float16" if self.device == "cuda" else "int8"
            candidates = [(self.device, default_ct)]
        if self.compute_type != "auto":
            candidates = [(d, self.compute_type) for d, _ in candidates]

        last_err: Exception | None = None
        for device, compute_type in candidates:
            try:
                model = WhisperModel(self.model_size, device=device, compute_type=compute_type)
                # 실제 커널이 동작하는지 1초짜리 무음으로 확인 (DLL 누락은 여기서 터진다)
                import numpy as np

                list(model.transcribe(np.zeros(16000, dtype=np.float32), language="en")[0])
                self._model, self.active_device = model, device
                log.info("Whisper %s 로드 완료 (%s/%s)", self.model_size, device, compute_type)
                return model
            except Exception as e:  # noqa: BLE001 - 다음 후보로 폴백
                log.warning("Whisper %s 로드 실패 (%s): %s", device, compute_type, e)
                last_err = e
        raise RuntimeError(f"Whisper 모델을 로드할 수 없습니다: {last_err}") from last_err

    def transcribe(self, audio_path: str | Path, language: str | None = None) -> Transcription:
        model = self._load()
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=self.beam_size,
            vad_filter=True,
        )
        segments = [Segment(s.start, s.end, s.text.strip()) for s in segments_iter]
        return Transcription(
            text=" ".join(s.text for s in segments).strip(),
            language=info.language,
            duration=info.duration,
            segments=segments,
        )
