"""Windows 내장 음성(System.Speech) TTS — 오프라인, 추가 패키지 불필요."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from multimodel.tts.base import TTSEngine, Voice

# 텍스트/경로는 환경 변수로 넘겨 PowerShell 이스케이프 문제를 피한다.
_SPEAK_PS = r"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
if ($env:MMT_VOICE) { $s.SelectVoice($env:MMT_VOICE) }
$s.Rate = [int]$env:MMT_RATE
$s.SetOutputToWaveFile($env:MMT_OUT)
$s.Speak($env:MMT_TEXT)
$s.Dispose()
"""

_LIST_PS = r"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.GetInstalledVoices() | Where-Object Enabled | ForEach-Object {
  $i = $_.VoiceInfo
  [pscustomobject]@{ id = $i.Name; language = $i.Culture.Name; gender = "$($i.Gender)"; description = $i.Description }
} | ConvertTo-Json -Compress
"""


def _powershell(script: str, env: dict[str, str] | None = None) -> str:
    import os

    full_env = {**os.environ, **(env or {})}
    result = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command",
         "[Console]::OutputEncoding=[Text.Encoding]::UTF8;" + script],
        capture_output=True, text=True, encoding="utf-8", env=full_env, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Windows TTS 실패: {result.stderr.strip()}")
    return result.stdout


class WindowsTTS(TTSEngine):
    name = "windows"
    extension = ".wav"

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise RuntimeError("windows TTS 엔진은 Windows 에서만 사용할 수 있습니다.")

    def synthesize(self, text: str, output_path: str | Path, voice: str | None = None, rate: int = 0) -> Path:
        if not text.strip():
            raise ValueError("텍스트가 비어 있습니다.")
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if voice is None:
            ko = self.list_voices("ko")
            voice = ko[0].id if ko else ""
        # System.Speech Rate 는 -10 ~ 10
        sapi_rate = max(-10, min(10, round(rate / 5)))
        _powershell(_SPEAK_PS, {
            "MMT_TEXT": text, "MMT_OUT": str(output_path),
            "MMT_VOICE": voice, "MMT_RATE": str(sapi_rate),
        })
        return output_path

    def list_voices(self, language: str | None = None) -> list[Voice]:
        out = _powershell(_LIST_PS).strip()
        if not out:
            return []
        data = json.loads(out)
        if isinstance(data, dict):
            data = [data]
        voices = [Voice(**d) for d in data]
        if language:
            voices = [v for v in voices if v.language.lower().startswith(language.lower())]
        return voices
