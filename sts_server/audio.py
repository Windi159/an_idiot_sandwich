"""오디오 입출력 유틸."""

from __future__ import annotations

import base64
import io
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000


def normalize_audio(src: str | Path | bytes, out_dir: str | Path | None = None) -> Path:
    """어떤 포맷이든(wav/mp3/m4a/webm...) 16 kHz mono 16bit WAV 로 바꿔 임시 파일 경로를 돌려준다.

    Qwen3-Omni 오디오 인코더와 llama.cpp input_audio 모두 이 형식을 가장 안정적으로 받는다.
    """
    from faster_whisper.audio import decode_audio  # PyAV 기반 디코더 + 리샘플러

    if isinstance(src, bytes):
        src = io.BytesIO(src)
    samples = decode_audio(src, sampling_rate=SAMPLE_RATE)
    fd, path = tempfile.mkstemp(suffix=".wav", dir=out_dir)
    with open(fd, "wb") as f:
        sf.write(f, samples, SAMPLE_RATE, subtype="PCM_16", format="WAV")
    return Path(path)


IMAGE_MIME = {".jpg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


def save_image(data: bytes, out_dir: str | Path | None = None) -> Path:
    """사진 바이트를 확장자가 맞는 임시 파일로 저장한다. 내용(매직 바이트)으로 형식을 판별하고, 모르는 형식이면 ValueError."""
    if data.startswith(b"\xff\xd8\xff"):
        suffix = ".jpg"
    elif data.startswith(b"\x89PNG\r\n\x1a\n"):
        suffix = ".png"
    elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        suffix = ".webp"
    else:
        raise ValueError("JPEG/PNG/WebP 사진만 받습니다")
    fd, path = tempfile.mkstemp(suffix=suffix, dir=out_dir)
    with open(fd, "wb") as f:
        f.write(data)
    return Path(path)


def duration_seconds(path: str | Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate


def wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.clip(audio, -1.0, 1.0), sample_rate, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")
