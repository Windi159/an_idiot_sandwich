"""Microsoft Edge 뉴럴 TTS (온라인, API 키 불필요)."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path

from multimodel.tts.base import TTSEngine, Voice


def _run(coro):
    # Gradio 워커 스레드처럼 이벤트 루프가 없는 곳에서도, 이미 루프가 도는 곳에서도 동작하도록
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        return pool.submit(asyncio.run, coro).result()


class EdgeTTS(TTSEngine):
    name = "edge"
    extension = ".mp3"
    default_voice = "ko-KR-SunHiNeural"

    def synthesize(self, text: str, output_path: str | Path, voice: str | None = None, rate: int = 0) -> Path:
        import edge_tts

        if not text.strip():
            raise ValueError("텍스트가 비어 있습니다.")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        communicate = edge_tts.Communicate(text, voice or self.default_voice, rate=f"{rate:+d}%")
        _run(communicate.save(str(output_path)))
        return output_path

    def stream(self, text: str, voice: str | None = None, rate: int = 0) -> Iterator[bytes]:
        """MP3 조각을 서버에서 받는 즉시 내보낸다 (첫 조각 ~0.3s). 이벤트 루프가 없는 워커 스레드에서 호출할 것."""
        import edge_tts

        if not text.strip():
            raise ValueError("텍스트가 비어 있습니다.")
        agen = edge_tts.Communicate(text, voice or self.default_voice, rate=f"{rate:+d}%").stream()
        loop = asyncio.new_event_loop()
        try:
            while True:
                try:
                    chunk = loop.run_until_complete(agen.__anext__())
                except StopAsyncIteration:
                    break
                if chunk["type"] == "audio" and chunk["data"]:
                    yield chunk["data"]
        finally:
            loop.run_until_complete(agen.aclose())
            loop.close()

    def list_voices(self, language: str | None = None) -> list[Voice]:
        import edge_tts

        voices = [
            Voice(id=v["ShortName"], language=v["Locale"], gender=v.get("Gender", ""), description=v.get("FriendlyName", ""))
            for v in _run(edge_tts.list_voices())
        ]
        if language:
            voices = [v for v in voices if v.language.lower().startswith(language.lower())]
        return sorted(voices, key=lambda v: v.id)
