"""STS 파이프라인: 받아쓰기 · 의도 분류 · RAG · 답변 생성 · 음성 출력.

이벤트(dict)를 yield 하고, app.py 가 이를 NDJSON 한 줄씩으로 내보낸다.
이벤트 형식은 GlassAssist 앱(MainActivity.sendToLLM / streamQaAnswer)과 호환된다:
  {"type": "qa"|"ignore"|"danger", ...}   ← 첫 줄
  {"tts": "문장"}                          ← 문장마다
  {"done": true, "answer": "전체 답변"}     ← 마지막
/sts 는 여기에 transcript, audio(base64) 필드가 추가된다. 앱은 모르는 필드를 무시한다.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

from sts_server import prompts
from sts_server.audio import b64, wav_bytes
from sts_server.config import Settings
from sts_server.engines import Message, OmniEngine
from sts_server.rag import Retriever
from sts_server.text import (SentenceSplitter, clean_for_speech, cut_loop, find_loop, has_threat_word, parse_intent,
                             parse_understand, speak_placeholders)

log = logging.getLogger(__name__)


class STSPipeline:
    def __init__(self, settings: Settings, engine: OmniEngine, retriever: Retriever) -> None:
        self.s = settings
        self.engine = engine
        self.retriever = retriever
        self._whisper = None
        self._tts = None
        self._tts_pool: ThreadPoolExecutor | None = None

    # ------------------------------------------------------------------ 받아쓰기

    def transcribe(self, wav_path: Path, language: str = "ko") -> str:
        if self.s.stt_backend == "whisper":
            if self._whisper is None:
                from multimodel import get_stt
                self._whisper = get_stt("whisper", model_size=self.s.whisper_model)
            return self._whisper.transcribe(wav_path, language=language).text
        msgs = [
            Message("system", prompts.TRANSCRIBE_SYSTEM.format(language=prompts.LANGUAGES.get(language, language))),
            Message("user", prompts.TRANSCRIBE_USER, audio_path=str(wav_path)),
        ]
        return self._generate_listening(msgs, max_new_tokens=256).strip().strip('"')

    # ------------------------------------------------------------------ 번역

    def translate(self, text: str, target: str) -> str:
        """글을 target 언어로 (prompts.LANGUAGES 의 코드). 고유명사는 prompts.TRANSLATE_GLOSSARY 표기로."""
        msgs = [Message("system", prompts.TRANSLATE_SYSTEM.format(target=prompts.LANGUAGES[target])),
                Message("user", prompts.translate_user_text(text, target))]
        return self.engine.generate_text(msgs, max_new_tokens=512, temperature=0.0).strip().strip('"')

    def translate_audio(self, wav_path: Path, source: str, target: str) -> dict:
        """음성 → 받아쓰기 → 번역. 음성을 바로 번역시키면 엉뚱한 문장이 나와서 (2026-10-30 시험) 두 단계로 한다."""
        t0 = time.perf_counter()
        transcript = self.transcribe(wav_path, source)
        transcribe_s = time.perf_counter() - t0
        translation = self.translate(transcript, target) if transcript else ""
        elapsed = time.perf_counter() - t0
        log.info("번역 %s→%s (%.2fs): %r → %r", source, target, elapsed, transcript, translation[:60])
        return {"transcript": transcript, "translation": translation, "elapsed": round(elapsed, 3),
                "transcribe_s": round(transcribe_s, 3), "translate_s": round(elapsed - transcribe_s, 3)}

    def _generate_until_loop(self, msgs: list[Message], max_new_tokens: int,
                             temperature: float) -> tuple[str, bool]:
        """generate_text 와 같지만, 같은 말을 되풀이하는 루프에 빠지면 바로 끊고 (지금까지 나온 글, True)."""
        raw = ""
        with closing(self.engine.stream_text(msgs, max_new_tokens=max_new_tokens, temperature=temperature)) as stream:
            for piece in stream:
                raw += piece
                if find_loop(raw) >= 0:
                    return raw, True
        return raw.strip(), False

    def _generate_listening(self, msgs: list[Message], max_new_tokens: int) -> str:
        """음성을 듣고 받아쓰는 생성. greedy 가 반복 루프("ed ed ed ...")에 빠지면 샘플링으로 한 번 더 듣고,
        그래도 루프면 반복 부분을 잘라낸다."""
        raw, looped = self._generate_until_loop(msgs, max_new_tokens, temperature=0.0)
        if looped:
            log.warning("받아쓰기 반복 루프 → 다시 듣기: %r", raw[:80])
            raw, looped = self._generate_until_loop(msgs, max_new_tokens, temperature=0.3)
        return cut_loop(raw) if looped else raw

    # ------------------------------------------------------------------ 의도 분류

    def _guard(self, intent: str, text: str) -> str:
        if intent == "danger" and not has_threat_word(text):
            log.info("danger 판정이지만 위협 단어 없음 → %s: %r", self.s.danger_unverified_as, text)
            return self.s.danger_unverified_as
        return intent

    def classify(self, text: str) -> str:
        msgs = [Message("system", prompts.CLASSIFY_SYSTEM), Message("user", f'발화: "{text}"')]
        raw = self.engine.generate_text(msgs, max_new_tokens=8, temperature=0.0)
        return self._guard(parse_intent(raw), text)

    def understand(self, wav_path: Path) -> tuple[str, str]:
        """음성 한 번 듣고 받아쓰기 + 분류를 동시에."""
        msgs = [
            Message("system", prompts.UNDERSTAND_SYSTEM),
            Message("user", prompts.UNDERSTAND_USER, audio_path=str(wav_path)),
        ]
        transcript, intent = parse_understand(self._generate_listening(msgs, max_new_tokens=200))
        return transcript, self._guard(intent, transcript)

    # ------------------------------------------------------------------ 답변

    def _qa_messages(self, question: str, wav_path: Path | None, docs_out: list[str] | None = None) -> list[Message]:
        """docs_out 을 주면 검색된 문서를 거기 담는다 (RAGAS 평가용 contexts)."""
        docs = self.retriever.search(question, self.s.rag_top_k) if question else []
        if docs_out is not None:
            docs_out.extend(docs)
        if docs:
            log.info("RAG %d건 검색", len(docs))
        audio = wav_path if self.s.qa_audio else None
        user_text = prompts.qa_user_text(question, docs, from_audio=audio is not None)
        return [
            Message("system", prompts.QA_SYSTEM),
            Message("user", user_text, audio_path=str(audio) if audio else None),
        ]

    def _stream_sentences(self, msgs: list[Message]) -> Iterator[str]:
        splitter = SentenceSplitter(first_soft_split=self.s.first_split_chars)
        for piece in self.engine.stream_text(msgs, max_new_tokens=self.s.max_answer_tokens, temperature=0.7):
            yield from splitter.feed(piece)
        yield from splitter.flush()

    def _get_tts(self):
        if self._tts is None:
            from multimodel import get_tts
            self._tts = get_tts(self.s.fallback_tts)
            # 엔진이 허용하는 만큼만 동시에 합성 (로컬 엔진은 1 — 나눠 쓰면 첫 문장이 늦어진다)
            self._tts_pool = ThreadPoolExecutor(max_workers=self._tts.max_concurrency, thread_name_prefix="tts")
        return self._tts

    @property
    def speech_wav_available(self) -> bool:
        """답변 전체를 PCM16 WAV 한 파일로 줄 수 있는지 (앱은 WAV 만 재생한다 — supertonic/windows 는 되고 edge(mp3)는 안 됨)."""
        try:
            return self._get_tts().extension == ".wav"
        except Exception:  # noqa: BLE001 - TTS 를 못 불러오면 폰 TTS 로 읽게 둔다
            log.exception("TTS 로드 실패")
            return False

    def warm_tts(self) -> None:
        """TTS 모델을 올리고 한 번 합성해 둔다 — 첫 답변 음성이 모델 로드·GPU 예열로 늦어지지 않게."""
        if not self.speech_wav_available:
            return
        t0 = time.perf_counter()
        try:
            self._fallback_tts("안녕하세요. 준비됐습니다.")
        except Exception:  # noqa: BLE001 - 실패해도 첫 요청 때 다시 시도한다
            log.exception("TTS 예열 실패")
            return
        log.info("TTS 예열 %.2fs (%s, %s)", time.perf_counter() - t0, self.s.fallback_tts,
                 getattr(self._tts, "device", "cpu"))

    def speech_lang_available(self, lang: str) -> bool:
        """그 언어로 답변 음성을 줄 수 있는지. 한국어 말고는 언어를 고를 수 있는 TTS(Supertonic 3, 중국어 없음)만 된다."""
        if not self.speech_wav_available:
            return False
        return lang == "ko" or lang in getattr(self._get_tts(), "languages", ())

    def speech_wav(self, text: str, lang: str = "ko") -> bytes:
        """답변 전체를 서버 TTS(기본 Supertonic F1)로 합성한 WAV."""
        t0 = time.perf_counter()
        data, _ = self._fallback_tts(clean_for_speech(text), lang)
        log.info("답변 음성 합성 %.2fs (%d자, %dKB)", time.perf_counter() - t0, len(text), len(data) // 1024)
        return data

    def _fallback_tts(self, text: str, lang: str = "ko") -> tuple[bytes, str]:
        tts = self._get_tts()
        extra = {"lang": lang} if lang != "ko" else {}   # 언어를 고를 수 있는 TTS 만 lang 을 받는다
        if lang == "ko":
            text = speak_placeholders(text)   # 안내방송 빈칸(00시, (열차명칭))을 소리 내기 좋게
        data = b"".join(tts.stream(text, voice=self.s.fallback_voice or None, **extra))
        return data, tts.extension.lstrip(".")

    @staticmethod
    def join_wavs(parts: list[bytes], gap_s: float = 0.12) -> bytes:
        """문장별 WAV 를 문장 사이 짧은 쉼을 넣어 PCM16 WAV 하나로 잇는다."""
        import io

        import numpy as np
        import soundfile as sf
        chunks, sr = [], None
        for data in parts:
            audio, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
            if sr is not None and rate != sr:
                raise ValueError(f"샘플레이트가 다른 조각: {rate} != {sr}")
            sr = rate
            if chunks:
                chunks.append(np.zeros(int(sr * gap_s), dtype=np.float32))
            chunks.append(audio if audio.ndim == 1 else audio.mean(axis=1))
        return wav_bytes(np.concatenate(chunks), sr)

    # ------------------------------------------------------------------ 엔드포인트별 흐름

    def ask(self, wav_path: Path, speech: bool = False) -> dict:
        """음성 질문 → 답변 텍스트 한 번에 (새 GlassAssist 앱용).

        danger 면 답변 없이 threat=True, ignore 나 말소리 없음이면 빈 답변 — 앱은 빈 답변이면 아무 말도 안 한다.
        speech 면 문장이 나오는 대로 서버 TTS 합성을 시작해 두고, 결과의 "speech" 에 전체 WAV 를 기다려 주는 함수를 넣는다.
        """
        t0 = time.perf_counter()
        transcript, intent = self.understand(wav_path)
        understand_s = time.perf_counter() - t0
        answer, parts, docs = "", [], []
        if transcript and intent == "qa":
            synth = speech and self.speech_wav_available
            sentences = []
            for sentence in self._stream_sentences(self._qa_messages(transcript, wav_path, docs)):
                sentences.append(sentence)
                if synth and (spoken := clean_for_speech(sentence)):
                    # LLM 이 다음 문장을 만드는 동안 이 문장을 합성 — 답변이 끝난 뒤 전체를 합성하는 것보다 빠르다
                    parts.append(self._tts_pool.submit(self._fallback_tts, spoken))
            answer = " ".join(sentences)
        log.info("ask %s (%.2fs): %r → %r", intent, time.perf_counter() - t0, transcript, answer[:60])
        elapsed = time.perf_counter() - t0
        result = {"transcript": transcript, "intent": intent if transcript else "ignore",
                  "answer": answer, "threat": intent == "danger", "elapsed": round(elapsed, 3),
                  # 추론 시간 내역: 받아쓰기+분류 / 답변 생성(RAG 포함)
                  "understand_s": round(understand_s, 3), "answer_s": round(elapsed - understand_s, 3),
                  "contexts": docs}
        if parts:
            result["speech"] = lambda: self.join_wavs([f.result()[0] for f in parts])
        return result

    def describe_image(self, image_path: Path, question: str = "") -> dict:
        """사진(+선택 질문) → 답변 텍스트.

        사진만으로는 RAG 검색어가 없어서 두 번 묻는다: 먼저 사진을 읽게 하고, 읽은 내용+질문으로 문서를 찾아
        찾은 게 있으면 문서를 붙여 다시 답하게 한다. ("이 코드 어떻게 대응해?" 처럼 질문에 코드 번호가 없어도 된다)
        """
        t0 = time.perf_counter()
        question = question.strip()

        def run(docs: list[str]) -> str:
            msgs = [Message("system", prompts.IMAGE_SYSTEM),
                    Message("user", prompts.image_user_text(question, docs), image_path=str(image_path))]
            return " ".join(self._stream_sentences(msgs))

        answer = run([])
        read_s = time.perf_counter() - t0
        docs = self.retriever.search(f"{answer} {question}".strip(), self.s.rag_top_k) if answer else []
        if docs:
            answer = run(docs)
        elapsed = time.perf_counter() - t0
        log.info("image (%.2fs, RAG %d건): %r → %r", elapsed, len(docs), question, answer[:60])
        # 추론 시간 내역: 사진 읽기 / 문서 붙여 다시 답하기 (문서가 없으면 0)
        return {"answer": answer, "question": question, "elapsed": round(elapsed, 3),
                "read_s": round(read_s, 3), "answer_s": round(elapsed - read_s, 3)}

    def chat_events(self, text: str, debug: bool = False) -> Iterator[dict]:
        """/chat — 텍스트 질문 (앱이 직접 TTS). debug 면 마지막 줄에 검색된 문서(contexts)도 싣는다."""
        t0 = time.perf_counter()
        intent = self.classify(text)
        log.info("분류 %s (%.2fs): %r", intent, time.perf_counter() - t0, text)
        if intent == "danger":
            yield {"type": "danger", "tts": prompts.DANGER_TTS}
            return
        if intent != "qa":
            yield {"type": intent}
            return
        yield {"type": "qa"}
        answer, docs = [], []
        for sent in self._stream_sentences(self._qa_messages(text, None, docs)):
            answer.append(sent)
            yield {"tts": sent}
        yield {"done": True, "answer": " ".join(answer), **({"contexts": docs} if debug else {})}
        log.info("/chat 완료 %.2fs", time.perf_counter() - t0)

    def sts_events(self, wav_path: Path, speech: bool = True, audio_stream: str = "chunk") -> Iterator[dict]:
        """/sts — 음성 질문 → 분류 → 답변 문장 + 답변 음성. audio_stream: chunk | sentence"""
        t0 = time.perf_counter()
        transcript, intent = self.understand(wav_path)
        log.info("이해 %s (%.2fs): %r", intent, time.perf_counter() - t0, transcript)
        if not transcript:
            yield {"type": "ignore", "transcript": ""}
            return
        if intent == "danger":
            event = {"type": "danger", "transcript": transcript, "tts": prompts.DANGER_TTS}
            if speech:
                data, fmt = self._fallback_tts(prompts.DANGER_TTS)
                event.update(audio=b64(data), format=fmt)
            yield event
            return
        if intent != "qa":
            yield {"type": intent, "transcript": transcript}
            return

        yield {"type": "qa", "transcript": transcript}
        msgs = self._qa_messages(transcript, wav_path)
        answer: list[str] = []

        if speech and self.engine.supports_speech_output:
            # 모델이 직접 말한다: 텍스트는 생성되는 대로 문장 단위로, 음성은 끝에 한 번에
            yield from self._native_speech(msgs, answer)
        elif speech:
            yield from self._fallback_speech(msgs, answer, mode=audio_stream)
        else:
            for sent in self._stream_sentences(msgs):
                answer.append(sent)
                yield {"tts": sent}
        yield {"done": True, "answer": " ".join(answer), "transcript": transcript,
               "elapsed": round(time.perf_counter() - t0, 2)}
        log.info("/sts 완료 %.2fs", time.perf_counter() - t0)

    def _fallback_speech(self, msgs: list[Message], answer: list[str], mode: str = "chunk") -> Iterator[dict]:
        """LLM 문장 생성 · 문장별 TTS(병렬) · 전송을 겹쳐서 돌린다.

        chunk    : {"tts", "seq"} 를 문장이 완성되는 즉시 보내고, 음성은 {"audio", "seq"} 조각으로
                   TTS 가 내놓는 대로 보낸다. 조각은 seq 순서대로만 나가며, 문장 끝에 {"audio_end", "seq"}.
        sentence : 문장 하나의 음성이 다 만들어지면 {"tts", "audio", "seq"} 한 줄로 (예전 형식).
        어느 쪽이든 앞 문장 음성이 준비되는 즉시 내보낸다 — 다음 문장이나 답변 끝을 기다리지 않는다.
        """
        tts = self._get_tts()
        fmt = tts.extension.lstrip(".")
        out: queue.Queue = queue.Queue()
        cond = threading.Condition()
        chunk_queues: list[queue.Queue] = []   # 문장별 오디오 조각 큐 (None = 끝)
        llm_done = [False]

        tts_first: dict[int, float] = {}   # 문장별 TTS 첫 조각까지 걸린 시간 (진단 로그용)

        def synth(i: int, text: str, q: queue.Queue) -> None:
            t_start = time.perf_counter()
            try:
                buf = b""
                first = True
                for piece in tts.stream(speak_placeholders(text), voice=self.s.fallback_voice or None):
                    buf += piece
                    # 첫 조각은 바로, 이후는 너무 잘게 쪼개지지 않게 모아서
                    if first or len(buf) >= 4096:
                        if first:
                            tts_first[i] = time.perf_counter() - t_start
                        q.put(buf)
                        buf, first = b"", False
                if buf:
                    q.put(buf)
            except Exception as e:  # noqa: BLE001 - 순서 관리 스레드가 보고
                q.put(e)
            finally:
                q.put(None)

        def produce() -> None:
            try:
                for i, sent in enumerate(self._stream_sentences(msgs)):
                    q: queue.Queue = queue.Queue()
                    with cond:
                        chunk_queues.append(q)
                        cond.notify_all()
                    out.put(("text", i, sent))
                    self._tts_pool.submit(synth, i, sent, q)
            except Exception as e:  # noqa: BLE001
                out.put(("error", None, e))
            finally:
                with cond:
                    llm_done[0] = True
                    cond.notify_all()

        def sequence() -> None:
            """문장 순서대로 오디오를 꺼내 보낸다. 뒤 문장은 미리 합성돼 큐에 쌓여 있다."""
            i = 0
            while True:
                with cond:
                    while i >= len(chunk_queues) and not llm_done[0]:
                        cond.wait()
                    if i >= len(chunk_queues):
                        break
                    q = chunk_queues[i]
                while (item := q.get()) is not None:
                    if isinstance(item, Exception):
                        log.warning("TTS 실패 (문장 %d): %s", i, item)
                        continue
                    out.put(("audio", i, item))
                out.put(("audio_end", i, None))
                i += 1
            out.put(("end", None, None))

        threading.Thread(target=produce, daemon=True, name="llm-sentences").start()
        threading.Thread(target=sequence, daemon=True, name="tts-sequencer").start()

        texts: dict[int, str] = {}
        audio: dict[int, bytearray] = {}
        while True:
            kind, i, payload = out.get()
            if kind == "end":
                log.info("TTS 첫 조각 지연(문장별): %s",
                         ", ".join(f"{k}:{v:.2f}s" for k, v in sorted(tts_first.items())))
                break
            if kind == "error":
                raise payload
            if kind == "text":
                answer.append(payload)
                texts[i] = payload
                if mode == "chunk":
                    yield {"tts": payload, "seq": i}
            elif kind == "audio":
                if mode == "chunk":
                    yield {"audio": b64(payload), "format": fmt, "seq": i}
                else:
                    audio.setdefault(i, bytearray()).extend(payload)
            elif kind == "audio_end":
                if mode == "chunk":
                    yield {"audio_end": True, "seq": i}
                else:
                    event = {"tts": texts.get(i, ""), "seq": i}
                    if i in audio:
                        event.update(audio=b64(bytes(audio.pop(i))), format=fmt)
                    yield event

    def _native_speech(self, msgs: list[Message], answer: list[str]) -> Iterator[dict]:
        q: queue.Queue = queue.Queue()
        holder: dict = {}

        def work():
            try:
                holder["result"] = self.engine.generate_speech(
                    msgs, max_new_tokens=self.s.max_answer_tokens, on_text=lambda p: q.put(("text", p)))
            except Exception as e:  # noqa: BLE001
                holder["error"] = e
            finally:
                q.put(("end", None))

        threading.Thread(target=work, daemon=True).start()
        splitter = SentenceSplitter()
        while True:
            kind, piece = q.get()
            if kind == "end":
                break
            for sent in splitter.feed(piece):
                answer.append(sent)
                yield {"tts": sent}
        for sent in splitter.flush():
            answer.append(sent)
            yield {"tts": sent}
        if "error" in holder:
            raise holder["error"]
        result = holder["result"]
        if result.audio is not None:
            yield {"audio": b64(wav_bytes(result.audio, result.sample_rate)), "format": "wav",
                   "sample_rate": result.sample_rate}
        if not answer and result.text:
            answer.append(clean_for_speech(result.text))
