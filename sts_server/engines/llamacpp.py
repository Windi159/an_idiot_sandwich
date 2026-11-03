"""llama.cpp(GGUF) 멀티모달 엔진 — 16GB GPU 용. 모델은 sts_server/models.py 프리셋에서 고른다.

llama-server 의 OpenAI 호환 API(/v1/chat/completions)에 음성을 input_audio 로 넣는다.
llama.cpp 는 텍스트 출력만 지원하므로(Qwen3-Omni 의 talker 미지원, Gemma 4 는 원래 텍스트 출력)
음성 출력은 서버의 fallback TTS 가 맡는다.
"""

from __future__ import annotations

import base64
import json
import logging
import shlex
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlparse

import httpx

from sts_server.audio import IMAGE_MIME
from sts_server.config import Settings
from sts_server.engines.base import Message, OmniEngine
from sts_server.models import resolve

log = logging.getLogger(__name__)


class LlamaCppEngine(OmniEngine):
    name = "llamacpp"
    supports_speech_output = False

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.preset, self.quant, files = resolve(settings.model, settings.quant)
        self.model_path = settings.llama_model or files.model
        self.mmproj_path = settings.llama_mmproj or files.mmproj
        default_moe = files.n_cpu_moe if files.n_cpu_moe is not None else self.preset.n_cpu_moe
        self.n_cpu_moe = default_moe if settings.llama_n_cpu_moe < 0 else settings.llama_n_cpu_moe
        self.base_url = settings.llama_url.rstrip("/")
        self._proc: subprocess.Popen | None = None
        self._client = httpx.Client(timeout=httpx.Timeout(600, connect=5))

    @property
    def model_id(self) -> str:
        return f"{self.preset.display} ({self.quant}, {Path(self.model_path).name})"

    # ------------------------------------------------------------------ 서버 기동

    def _healthy(self) -> bool:
        try:
            return self._client.get(f"{self.base_url}/health", timeout=2).status_code == 200
        except httpx.HTTPError:
            return False

    def _running_model(self) -> str | None:
        try:
            return self._client.get(f"{self.base_url}/props", timeout=5).json().get("model_path")
        except (httpx.HTTPError, ValueError):
            return None

    def load(self) -> None:
        if self._healthy():
            running = self._running_model()
            if running and Path(running).name != Path(self.model_path).name:
                raise RuntimeError(
                    f"{self.base_url} 의 llama-server 는 다른 모델({Path(running).name})을 쓰고 있습니다. "
                    f"그 서버를 끄거나 STS_LLAMA_URL 을 다른 포트로 지정하세요.")
            log.info("이미 실행 중인 llama-server 사용: %s", self.base_url)
            return
        if not self.settings.llama_autostart:
            raise RuntimeError(f"llama-server({self.base_url})에 연결할 수 없습니다. STS_LLAMA_AUTOSTART=true 로 자동 실행 가능")

        s = self.settings
        for p in (s.llama_exe, self.model_path, self.mmproj_path):
            if not Path(p).is_file():
                raise FileNotFoundError(f"llama.cpp 실행에 필요한 파일이 없습니다: {p}")
        port = urlparse(self.base_url).port or 8081
        cmd = [
            s.llama_exe, "-m", self.model_path, "--mmproj", self.mmproj_path,
            "--host", "127.0.0.1", "--port", str(port),
            "-c", str(s.llama_ctx), "-ngl", "999",
            "--jinja", "-fa", "on", "--no-webui",
        ]
        if self.n_cpu_moe > 0:
            cmd += ["--n-cpu-moe", str(self.n_cpu_moe)]
        cmd += shlex.split(s.llama_extra_args)
        log_path = Path(s.data_dir) / "llama-server.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log.info("llama-server 시작 [%s]: %s (로그: %s)", self.model_id, " ".join(cmd), log_path)
        self._log_file = open(log_path, "w", encoding="utf-8", errors="replace")
        self._proc = subprocess.Popen(cmd, stdout=self._log_file, stderr=subprocess.STDOUT,
                                      creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        deadline = time.time() + 600
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(f"llama-server 가 종료됨 (code={self._proc.returncode}). 로그: {log_path}")
            if self._healthy():
                log.info("llama-server 준비 완료")
                return
            time.sleep(1)
        raise TimeoutError("llama-server 가 10분 안에 준비되지 않았습니다.")

    def close(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._client.close()

    # ------------------------------------------------------------------ 추론

    @staticmethod
    def _to_openai(messages: list[Message]) -> list[dict]:
        out = []
        for m in messages:
            if m.audio_path or m.image_path:
                content = []
                if m.image_path:
                    data = base64.b64encode(Path(m.image_path).read_bytes()).decode()
                    mime = IMAGE_MIME.get(Path(m.image_path).suffix.lower(), "image/jpeg")
                    content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})
                if m.audio_path:
                    data = base64.b64encode(Path(m.audio_path).read_bytes()).decode()
                    content.append({"type": "input_audio", "input_audio": {"data": data, "format": "wav"}})
                if m.text:
                    content.append({"type": "text", "text": m.text})
                out.append({"role": m.role, "content": content})
            else:
                out.append({"role": m.role, "content": m.text or ""})
        return out

    def stream_text(self, messages: list[Message], max_new_tokens: int = 256,
                    temperature: float = 0.7) -> Iterator[str]:
        body = {
            "messages": self._to_openai(messages),
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": 0.9 if temperature > 0 else 1.0,
            "stream": True,
        }
        if self.preset.template_kwargs:
            body["chat_template_kwargs"] = self.preset.template_kwargs
        with self._client.stream("POST", f"{self.base_url}/v1/chat/completions", json=body) as resp:
            if resp.status_code != 200:
                raise RuntimeError(f"llama-server 오류 {resp.status_code}: {resp.read().decode(errors='replace')[:500]}")
            for line in resp.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                delta = json.loads(payload)["choices"][0].get("delta", {})
                piece = delta.get("content")
                if piece:
                    yield piece
