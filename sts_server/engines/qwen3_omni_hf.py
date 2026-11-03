"""Qwen3-Omni (transformers) — 음성 입력 → 텍스트 + 음성 출력까지 모델 하나로 처리하는 완전한 STS.

BF16 기준 약 80GB VRAM 이 필요하다 (A100/H100 80GB 급, 또는 device_map="auto" 로 여러 GPU).
transformers 5.x 는 MoE expert 를 하나의 텐서로 합쳐 로드하기 때문에 4bit 체크포인트를 써도
expert 가 BF16 으로 풀려 메모리가 크게 줄지 않는다. 16GB GPU 에서는 llamacpp 엔진을 쓴다.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator

import numpy as np

from sts_server.config import Settings
from sts_server.engines.base import Message, OmniEngine, SpeechResult

log = logging.getLogger(__name__)


class Qwen3OmniHF(OmniEngine):
    name = "qwen3_omni_hf"
    supports_speech_output = True

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = None
        self.processor = None
        # GPU 하나에 모델 하나 — 동시 요청은 순서대로 처리
        self._lock = threading.Lock()

    @property
    def model_id(self) -> str:
        return self.settings.hf_model

    def load(self) -> None:
        import torch
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

        log.info("Qwen3-Omni 로드 중: %s", self.settings.hf_model)
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            self.settings.hf_model,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=self.settings.hf_attn,
        )
        self.model.eval()
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(self.settings.hf_model)
        log.info("Qwen3-Omni 로드 완료 (talker=%s)", self.model.has_talker)

    # ------------------------------------------------------------------

    def _inputs(self, messages: list[Message]):
        from qwen_omni_utils import process_mm_info

        conversation = []
        for m in messages:
            content = []
            if m.image_path:
                content.append({"type": "image", "image": m.image_path})
            if m.audio_path:
                content.append({"type": "audio", "audio": m.audio_path})
            if m.text:
                content.append({"type": "text", "text": m.text})
            conversation.append({"role": m.role, "content": content})

        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos,
                                return_tensors="pt", padding=True, use_audio_in_video=False)
        return inputs.to(self.model.device).to(self.model.dtype)

    def _gen_kwargs(self, max_new_tokens: int, temperature: float) -> dict:
        kw = {"thinker_max_new_tokens": max_new_tokens, "use_audio_in_video": False}
        if temperature > 0:
            kw.update(thinker_do_sample=True, thinker_temperature=temperature, thinker_top_p=0.9)
        else:
            kw.update(thinker_do_sample=False)
        return kw

    def _run_streaming(self, messages, max_new_tokens, temperature, return_audio, on_text=None):
        """generate 를 별도 스레드에서 돌리며 텍스트 조각을 흘려보낸다. (조각 iterator, 결과 holder) 반환."""
        from transformers import TextIteratorStreamer

        inputs = self._inputs(messages)
        streamer = TextIteratorStreamer(self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True,
                                        timeout=600)
        holder: dict = {}
        kwargs = self._gen_kwargs(max_new_tokens, temperature)
        if return_audio:
            kwargs.update(return_audio=True, speaker=self.settings.speaker)
        else:
            kwargs.update(return_audio=False)

        def work():
            try:
                holder["result"] = self.model.generate(**inputs, thinker_streamer=streamer, **kwargs)
            except Exception as e:  # noqa: BLE001 - 호출측으로 전달
                holder["error"] = e
                streamer.end()

        t = threading.Thread(target=work, daemon=True)
        t.start()
        return streamer, holder, t

    def stream_text(self, messages: list[Message], max_new_tokens: int = 256,
                    temperature: float = 0.7) -> Iterator[str]:
        with self._lock:
            streamer, holder, t = self._run_streaming(messages, max_new_tokens, temperature, return_audio=False)
            for piece in streamer:
                if piece:
                    yield piece
            t.join()
            if "error" in holder:
                raise holder["error"]

    def generate_speech(self, messages: list[Message], max_new_tokens: int = 256,
                        on_text=None) -> SpeechResult:
        with self._lock:
            streamer, holder, t = self._run_streaming(messages, max_new_tokens, 0.7, return_audio=True)
            pieces = []
            for piece in streamer:
                if piece:
                    pieces.append(piece)
                    if on_text:
                        on_text(piece)
            t.join()
            if "error" in holder:
                raise holder["error"]
            _, wav = holder["result"]
            audio = None if wav is None else wav.reshape(-1).detach().float().cpu().numpy().astype(np.float32)
            return SpeechResult(text="".join(pieces).strip(), audio=audio, sample_rate=24000)
