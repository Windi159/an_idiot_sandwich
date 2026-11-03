"""FastAPI 서버 — GlassAssist 앱의 LLM 서버 + STT 서버 역할을 멀티모달 모델(Qwen3-Omni / Gemma 4) 하나로.

기존 API (앱 수정 없이 동작)
  GET  /health
  POST /chat             {"text": "..."} → NDJSON
  POST /api/transcribe   multipart(file, language, task)
                         → answer 모드(기본, post_sh-t main 앱): {"text"=답변, "answer", "transcript", "intent", "threat"}
                         → transcript 모드(예전 앱): {"text"=받아쓰기, "elapsed"}
  POST /db/qa            QA 기록 저장
신규 API
  POST /api/ask          /api/transcribe 의 answer 모드와 같음 (설정과 무관)
  POST /api/image        {"image": base64[, "question"]} → {"text"=답변, "answer"} (글라스·갤러리 사진)
  POST /api/translate    {"text", "target"[, "speech"]} → {"text"=번역, "translation", "source_text"}
  POST /api/translate/audio  multipart(file, language=원문 언어, target[, speech]) → 받아쓰기 후 번역
  GET  /api/audio/{id}   위 API 들에 speech=1 을 주면 받는 audio_id 의 답변 음성 (서버 TTS, PCM16 WAV)
  POST /sts              multipart(file[, speech]) → NDJSON (transcript + 분류 + 답변 문장 + 답변 음성)
  GET  /info             현재 엔진·설정 확인
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import re
import secrets
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from sts_server import prompts
from sts_server.audio import duration_seconds, normalize_audio, save_image
from sts_server.config import Settings
from sts_server.engines import create_engine
from sts_server.pipeline import STSPipeline
from sts_server.rag import create_retriever

log = logging.getLogger("sts_server")


class State:
    settings: Settings
    pipeline: STSPipeline | None = None
    error: str | None = None
    loading = True


state = State()


def _load(settings: Settings) -> None:
    try:
        t0 = time.perf_counter()
        engine = create_engine(settings)
        engine.load()
        state.pipeline = STSPipeline(settings, engine, create_retriever(settings))
        # 답변 음성(speech=1)을 처음 요청할 때 TTS 모델 로드로 늦어지지 않게 미리 불러 한 번 합성해 둔다
        threading.Thread(target=state.pipeline.warm_tts, daemon=True).start()
        log.info("엔진 준비 완료: %s (%.1fs)", settings.engine, time.perf_counter() - t0)
    except Exception as e:  # noqa: BLE001 - /health 로 노출
        log.exception("엔진 로드 실패")
        state.error = f"{type(e).__name__}: {e}"
    finally:
        state.loading = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.settings = Settings.from_env()
    Path(state.settings.data_dir).mkdir(parents=True, exist_ok=True)
    # 모델 로드는 몇 분 걸릴 수 있어 백그라운드에서 — 그동안 /health 는 503 "loading"
    threading.Thread(target=_load, args=(state.settings,), daemon=True).start()
    yield
    if state.pipeline:
        state.pipeline.engine.close()


app = FastAPI(title="GlassAssist STS Server", lifespan=lifespan)


def _pipeline() -> STSPipeline:
    if state.pipeline is None:
        raise HTTPException(503, "모델 로딩 중" if state.loading else f"엔진 오류: {state.error}")
    return state.pipeline


_record_lock = threading.Lock()


def _record(endpoint: str, **fields) -> None:
    """요청 하나의 질문·답변을 data/requests.jsonl 에 남긴다. scripts/watch_requests.py 가 실시간으로 보여준다."""
    row = {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "endpoint": endpoint, **fields}
    try:
        with _record_lock, (Path(state.settings.data_dir) / "requests.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        log.exception("요청 기록 실패")


def _ndjson(events, endpoint: str | None = None, **fields):
    received = time.time()

    def gen():
        t0 = time.perf_counter()
        seen = dict(fields)
        try:
            for ev in events:
                if endpoint:
                    if "type" in ev and "intent" not in seen:
                        seen["intent"] = ev["type"]
                    if ev.get("transcript"):
                        seen["transcript"] = ev["transcript"]
                    if ev.get("done"):
                        seen["answer"] = ev.get("answer", "")
                yield json.dumps(ev, ensure_ascii=False) + "\n"
        except Exception as e:  # noqa: BLE001 - 스트림 중간 오류도 한 줄로 알린다
            log.exception("스트리밍 중 오류")
            seen["error"] = f"{type(e).__name__}: {e}"
            yield json.dumps({"error": seen["error"], "done": True, "answer": ""},
                             ensure_ascii=False) + "\n"
        finally:
            if endpoint:
                _record(endpoint, received=_clock(received), elapsed=round(time.perf_counter() - t0, 3),
                        total_s=round(time.time() - received, 3), **seen)
    return StreamingResponse(gen(), media_type="application/x-ndjson")


async def _save_upload(file: UploadFile) -> Path:
    data = await file.read()
    if not data:
        raise HTTPException(400, "빈 오디오 파일")
    try:
        return normalize_audio(data)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"오디오를 읽을 수 없습니다: {e}") from e


# ---------------------------------------------------------------------- 기존 API


@app.get("/health")
def health():
    body = {"status": "ok" if state.pipeline else ("loading" if state.loading else "error"),
            "engine": state.settings.engine}
    if state.error:
        body["error"] = state.error
    return JSONResponse(body, status_code=200 if state.pipeline else 503)


class ChatRequest(BaseModel):
    text: str
    debug: bool = False        # true 면 마지막 줄에 검색된 문서(contexts) — scripts/ragas_eval.py 용


@app.post("/chat")
def chat(req: ChatRequest):
    pipeline = _pipeline()
    text = req.text.strip()
    if not text:
        return _ndjson(iter([{"type": "ignore"}]))
    return _ndjson(pipeline.chat_events(text, req.debug), "/chat", transcript=text)


class SpeechCache:
    """답변 음성을 응답과 동시에 합성 시작해 두고 GET /api/audio/{audio_id} 로 내준다.

    음성을 응답 JSON 에 넣지 않는 이유: 앱은 JSON 을 64KiB 까지만 읽고, 서버가 준 주소는 따라가지 않는다.
    앱은 audio_id(32자리 16진수)만 받아 같은 음성 서버의 고정 경로로 받아 간다.
    """

    TTL = 120  # 초. 앱은 답변을 받자마자 받아 가므로 넉넉하다

    def __init__(self) -> None:
        self._items: dict[str, tuple[float, Future]] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="answer-speech")

    def start(self, make: Callable[[], bytes], on_ready: Callable[[str, bytes], None] | None = None) -> str:
        audio_id = secrets.token_hex(16)

        def job() -> bytes:
            data = make()
            if on_ready:
                on_ready(audio_id, data)
            return data

        future = self._pool.submit(job)
        now = time.monotonic()
        with self._lock:
            self._items = {k: v for k, v in self._items.items() if now - v[0] < self.TTL}
            self._items[audio_id] = (now, future)
        return audio_id

    def get(self, audio_id: str) -> Future | None:
        with self._lock:
            item = self._items.get(audio_id)
        return item[1] if item and time.monotonic() - item[0] < self.TTL else None


speech_cache = SpeechCache()


def _clock(t: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(t)) + f".{int(t % 1 * 1000):03d}"


def _speech_job(speech: bool, answer: str, received: float,
                make: Callable[[], bytes] | None = None) -> Callable[[], str] | None:
    """서버 음성을 줄 수 있으면 "합성을 시작하고 audio_id 를 돌려주는 함수"를, 아니면 None.

    make 가 있으면(문장별로 이미 합성 중) 그걸 쓰고, 없으면 답변 전체를 지금부터 합성한다.
    음성이 다 만들어지면 requests.jsonl 에 "speech" 줄을 남긴다 (요청이 들어온 때부터 음성 준비까지 걸린 시간).
    호출한 쪽이 질문 줄을 먼저 남기고 시작해야 보기 창에서 순서가 맞는다.
    """
    pipeline = state.pipeline
    if not (speech and answer and pipeline):
        return None
    if make is None:
        if not pipeline.speech_wav_available:
            return None
        make = lambda: pipeline.speech_wav(answer)  # noqa: E731
    answered = time.time()

    def ready(audio_id: str, data: bytes) -> None:
        import io

        import soundfile as sf
        now = time.time()
        _record("speech", received=_clock(received), audio_id=audio_id, ready_s=round(now - received, 3),
                after_answer_s=round(now - answered, 3), audio_s=round(sf.info(io.BytesIO(data)).duration, 1),
                kb=len(data) // 1024)

    return lambda: speech_cache.start(make, ready)


def _ask_response(result: dict, language: str, endpoint: str, received: float, speech: bool = False,
                  debug: bool = False) -> dict:
    start_speech = _speech_job(speech, result["answer"], received, result.get("speech"))
    _record(endpoint, received=_clock(received), transcript=result["transcript"], intent=result["intent"],
            threat=result["threat"], answer=result["answer"], elapsed=result["elapsed"],
            understand_s=result.get("understand_s"), answer_s=result.get("answer_s"),
            total_s=round(time.time() - received, 3), server_speech=start_speech is not None)
    audio_id = start_speech() if start_speech else None
    # post_sh-t main 앱(VoiceServerProtocol)은 answer → text 순으로 답변을 읽는다. 팀 서버(server/)처럼 text 에도 답변을 싣는다.
    body = {"text": result["answer"], "answer": result["answer"], "transcript": result["transcript"],
            "intent": result["intent"], "threat": result["threat"], "language": language,
            "elapsed": result["elapsed"]}
    if audio_id:
        body["audio_id"] = audio_id
    if debug:   # RAGAS 평가용: 답변에 쓴 검색 문서와 추론 시간 내역
        body.update(contexts=result.get("contexts", []), understand_s=result.get("understand_s"),
                    answer_s=result.get("answer_s"))
    return body


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = Form("ko"), task: str = Form("transcribe"),
                     speech: bool = Form(False)):
    """STS_TRANSCRIBE_MODE=answer(기본): 답변까지 / transcript: 받아쓰기만 (예전 앱)

    speech=1 이면 답변 음성(서버 TTS)도 만들어 audio_id 를 함께 준다 → GET /api/audio/{audio_id}
    """
    received = time.time()
    pipeline = _pipeline()
    wav = await _save_upload(file)
    try:
        if state.settings.transcribe_mode == "answer":
            return _ask_response(await run_in_threadpool(pipeline.ask, wav, speech), language, "/api/transcribe",
                                 received, speech)
        t0 = time.perf_counter()
        text = await run_in_threadpool(pipeline.transcribe, wav, language)
        elapsed = round(time.perf_counter() - t0, 3)
        log.info("/api/transcribe %.2fs (음성 %.1fs): %r", elapsed, duration_seconds(wav), text)
        _record("/api/transcribe", received=_clock(received), transcript=text, elapsed=elapsed,
                total_s=round(time.time() - received, 3))
        return {"text": text, "language": language, "elapsed": elapsed}
    finally:
        wav.unlink(missing_ok=True)


@app.post("/api/ask")
async def ask(file: UploadFile = File(...), language: str = Form("ko"), task: str = Form("transcribe"),
              speech: bool = Form(False), debug: bool = Form(False)):
    """설정과 관계없이 항상 음성 질문 → 답변 (앱 서버 설정의 API 경로를 /api/ask 로 둘 때)

    debug=1 이면 답변에 쓴 검색 문서(contexts)도 준다 — scripts/ragas_eval.py 용
    """
    received = time.time()
    pipeline = _pipeline()
    wav = await _save_upload(file)
    try:
        return _ask_response(await run_in_threadpool(pipeline.ask, wav, speech), language, "/api/ask", received, speech,
                             debug)
    finally:
        wav.unlink(missing_ok=True)


MAX_IMAGE_BYTES = 15 * 1024 * 1024


class ImageRequest(BaseModel):
    image: str                 # base64 (data:image/...;base64, 접두어도 허용)
    question: str = ""
    source: str = ""           # 갤러리 | 글라스 (기록용)
    userId: str = ""
    speech: bool = False       # true 면 답변 음성도 만들어 audio_id 를 준다


@app.post("/api/image")
async def image(req: ImageRequest):
    """폰 앱 ImageTransferController 가 올리는 사진 → Gemma/Qwen 이 보고 답한다. 응답 형식은 /api/ask 와 같다."""
    received = time.time()
    pipeline = _pipeline()
    raw = req.image.split(",", 1)[1] if req.image.startswith("data:") else req.image
    if len(raw) > MAX_IMAGE_BYTES * 4 // 3 + 4:
        raise HTTPException(413, "사진이 너무 큽니다 (15MB 이하)")
    try:
        data = base64.b64decode(raw, validate=True)
        path = save_image(data)
    except (binascii.Error, ValueError) as e:
        raise HTTPException(400, f"사진을 읽을 수 없습니다: {e}") from e
    try:
        result = await run_in_threadpool(pipeline.describe_image, path, req.question)
    finally:
        path.unlink(missing_ok=True)
    log.info("/api/image %s %dKB (%s)", req.source or "?", len(data) // 1024, req.userId or "?")
    start_speech = _speech_job(req.speech, result["answer"], received)
    _record("/api/image", received=_clock(received), source=req.source, user=req.userId, kb=len(data) // 1024,
            transcript=result["question"], answer=result["answer"], elapsed=result["elapsed"],
            read_s=result.get("read_s"), answer_s=result.get("answer_s"),
            total_s=round(time.time() - received, 3), server_speech=start_speech is not None)
    audio_id = start_speech() if start_speech else None
    body = {"text": result["answer"], "answer": result["answer"], "question": result["question"],
            "elapsed": result["elapsed"]}
    if audio_id:
        body["audio_id"] = audio_id
    return body


def _check_lang(code: str, what: str) -> str:
    code = code.strip().lower()
    if code not in prompts.LANGUAGES:
        raise HTTPException(400, f"{what} 언어 코드를 모릅니다: {code!r} (가능: {', '.join(prompts.LANGUAGES)})")
    return code


def _translate_response(result: dict, source: str, target: str, endpoint: str, received: float,
                        speech: bool) -> dict:
    """번역 결과 → 응답. speech 면 번역문을 target 언어 음성으로 (TTS 가 그 언어를 못 읽으면 audio_id 없음)."""
    pipeline = state.pipeline
    translation = result["translation"]
    make = None
    if speech and translation and pipeline and pipeline.speech_lang_available(target):
        make = lambda: pipeline.speech_wav(translation, target)  # noqa: E731
    start_speech = _speech_job(speech, translation, received, make) if make else None
    _record(endpoint, received=_clock(received), source=source, target=target, transcript=result["transcript"],
            answer=translation, elapsed=result["elapsed"], total_s=round(time.time() - received, 3),
            server_speech=start_speech is not None)
    audio_id = start_speech() if start_speech else None
    body = {"text": translation, "translation": translation, "source_text": result["transcript"],
            "source": source, "target": target, "elapsed": result["elapsed"]}
    if audio_id:
        body["audio_id"] = audio_id
    return body


class TranslateRequest(BaseModel):
    text: str
    target: str = "en"         # prompts.LANGUAGES 의 코드 (en, ja, zh, vi, ko …)
    source: str = ""           # 기록용 (모델이 원문 언어를 알아서 본다)
    speech: bool = False       # true 면 번역문 음성도 만들어 audio_id 를 준다 (중국어는 TTS 가 못 읽어 없음)


@app.post("/api/translate")
async def translate(req: TranslateRequest):
    """글 → target 언어. 역 이름·노선·열차 이름은 prompts.TRANSLATE_GLOSSARY 표기로."""
    received = time.time()
    pipeline = _pipeline()
    target = _check_lang(req.target, "target")
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "번역할 글이 없습니다")
    t0 = time.perf_counter()
    translation = await run_in_threadpool(pipeline.translate, text, target)
    result = {"transcript": text, "translation": translation, "elapsed": round(time.perf_counter() - t0, 3)}
    return _translate_response(result, req.source.strip().lower(), target, "/api/translate", received, req.speech)


@app.post("/api/translate/audio")
async def translate_audio(file: UploadFile = File(...), language: str = Form("ko"), target: str = Form("en"),
                          speech: bool = Form(False)):
    """음성(language) → 받아쓰기 → target 언어. 응답의 source_text 가 받아쓴 원문, transcribe_s/translate_s 가 단계별 시간."""
    received = time.time()
    pipeline = _pipeline()
    source = _check_lang(language, "language")
    target = _check_lang(target, "target")
    wav = await _save_upload(file)
    try:
        result = await run_in_threadpool(pipeline.translate_audio, wav, source, target)
    finally:
        wav.unlink(missing_ok=True)
    body = _translate_response(result, source, target, "/api/translate/audio", received, speech)
    body.update(transcribe_s=result["transcribe_s"], translate_s=result["translate_s"])
    return body


@app.get("/api/audio/{audio_id}")
async def audio(audio_id: str):
    """/api/transcribe·/api/ask·/api/image·/api/translate 가 준 audio_id 의 답변 음성 (PCM16 WAV). 합성 중이면 끝날 때까지 기다린다."""
    future = speech_cache.get(audio_id) if re.fullmatch(r"[0-9a-f]{32}", audio_id) else None
    if future is None:
        raise HTTPException(404, "없거나 만료된 음성")
    try:
        data = await run_in_threadpool(future.result, 30)
    except FutureTimeout as e:
        raise HTTPException(504, "음성 합성 시간 초과") from e
    except Exception as e:  # noqa: BLE001
        log.exception("답변 음성 합성 실패")
        raise HTTPException(500, f"음성 합성 실패: {type(e).__name__}") from e
    return Response(data, media_type="audio/wav")


@app.post("/db/qa")
def db_qa(record: dict):
    # 코레일 DB 연동 전까지는 파일에 쌓아둔다
    path = Path(state.settings.data_dir) / "qa_log.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"status": "ok"}


# ---------------------------------------------------------------------- 신규 API


@app.post("/sts")
async def sts(file: UploadFile = File(...), speech: bool = Form(True), audio_stream: str = Form("chunk")):
    """audio_stream=chunk: 음성을 조각 단위로 즉시 전송 / sentence: 문장 하나씩 완성된 음성으로 전송"""
    pipeline = _pipeline()
    if audio_stream not in ("chunk", "sentence"):
        raise HTTPException(400, "audio_stream 은 chunk 또는 sentence")
    wav = await _save_upload(file)

    def events():
        try:
            yield from pipeline.sts_events(wav, speech=speech, audio_stream=audio_stream)
        finally:
            wav.unlink(missing_ok=True)

    return _ndjson(events(), "/sts")


@app.get("/info")
def info():
    s = state.settings
    p = state.pipeline
    return {
        "engine": s.engine,
        "model": p.engine.model_id if p else s.model,
        "speech_output": "native" if (p and p.engine.supports_speech_output) else f"fallback:{s.fallback_tts}",
        "stt_backend": s.stt_backend,
        "rag_backend": s.rag_backend,
        "status": "ok" if p else ("loading" if state.loading else "error"),
        "error": state.error,
    }
