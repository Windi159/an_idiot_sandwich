"""mock 엔진으로 API 계약(앱 호환성) 점검.  실행: uv run python -m pytest tests -q"""

from __future__ import annotations

import io
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

os.environ["STS_ENGINE"] = "mock"
os.environ["STS_FALLBACK_TTS"] = "windows"  # 네트워크 없이도 돌도록
os.environ["STS_FALLBACK_VOICE"] = ""        # Windows 기본 한국어 음성
os.environ["STS_DATA_DIR"] = tempfile.mkdtemp(prefix="sts-test-")  # 실제 data/requests.jsonl 을 더럽히지 않게
os.environ["STS_RAG_BACKEND"] = "none"      # .env 의 hybrid 는 임베딩 모델을 올려서 느리다 — RAG 는 test_rag.py 에서

from fastapi.testclient import TestClient  # noqa: E402

from sts_server.app import app, state  # noqa: E402
from sts_server.text import SentenceSplitter, parse_understand  # noqa: E402


def _wav() -> bytes:
    buf = io.BytesIO()
    t = np.linspace(0, 1, 16000, endpoint=False)
    sf.write(buf, (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 16000, format="WAV")
    return buf.getvalue()


def _lines(resp) -> list[dict]:
    return [json.loads(l) for l in resp.text.splitlines() if l.strip()]


def _client() -> TestClient:
    c = TestClient(app)
    c.__enter__()
    for _ in range(100):
        if c.get("/health").status_code == 200:
            break
        time.sleep(0.05)
    return c


client = _client()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_chat_qa_ndjson_contract():
    lines = _lines(client.post("/chat", json={"text": "PSD 501 코드가 뭐야?"}))
    assert lines[0] == {"type": "qa"}                     # 앱은 첫 줄로 분기
    assert any("tts" in l for l in lines[1:-1])           # 문장 단위 TTS
    assert lines[-1]["done"] is True and lines[-1]["answer"]


def test_chat_danger_guard():
    # mock 은 "죽여" 가 있으면 danger — 위협 단어 사전에도 있으므로 그대로 유지
    lines = _lines(client.post("/chat", json={"text": "너 죽여버린다"}))
    assert lines[0]["type"] == "danger" and lines[0]["tts"]


def _upload(path="/api/transcribe"):
    return client.post(path, files={"file": ("q.wav", _wav(), "audio/wav")},
                       data={"language": "ko", "task": "transcribe"})


def test_transcribe_answer_mode_matches_main_app():
    # post_sh-t main 앱(VoiceServerProtocol.response)은 answer → text 순으로 답변을, transcript 를 질문으로 읽는다
    body = _upload().json()
    assert body["transcript"] == "PSD 501 코드가 뭐야?"
    assert body["answer"] and body["text"] == body["answer"] != body["transcript"]
    assert body["intent"] == "qa" and body["threat"] is False
    assert _upload("/api/ask").json()["answer"] == body["answer"]


def test_transcribe_transcript_mode_for_old_app(monkeypatch):
    monkeypatch.setattr(state.settings, "transcribe_mode", "transcript")
    body = _upload().json()
    assert body["text"] == "PSD 501 코드가 뭐야?" and "answer" not in body and "elapsed" in body


@pytest.mark.parametrize("intent, threat", [("danger", True), ("ignore", False)])
def test_ask_no_answer_for_danger_and_ignore(monkeypatch, intent, threat):
    monkeypatch.setattr(state.pipeline, "understand", lambda wav: ("야 너 죽여버린다", intent))
    body = _upload("/api/ask").json()
    assert body["answer"] == "" and body["text"] == "" and body["threat"] is threat


def _jpeg_b64() -> str:
    import base64
    return base64.b64encode(b"\xff\xd8\xff\xe0" + b"\x00" * 64).decode()


def test_image_answer_matches_phone_upload():
    # 폰 앱 ImageTransferController.deliver 가 보내는 본문 그대로
    body = client.post("/api/image", json={"image": _jpeg_b64(), "source": "글라스", "userId": "작업자"}).json()
    assert body["answer"] and body["text"] == body["answer"]
    seen = {}
    real = state.pipeline.describe_image
    def spy(path, question=""):
        seen.update(suffix=path.suffix, question=question)
        return real(path, question)
    state.pipeline.describe_image = spy
    try:
        client.post("/api/image", json={"image": "data:image/jpeg;base64," + _jpeg_b64(), "question": "이거 뭐야?"})
    finally:
        state.pipeline.describe_image = real
    assert seen == {"suffix": ".jpg", "question": "이거 뭐야?"}


@pytest.mark.parametrize("image", ["@@@", "aGVsbG8="])   # base64 아님 / 사진 아님
def test_image_rejects_bad_input(image):
    assert client.post("/api/image", json={"image": image}).status_code == 400


def test_answer_speech_by_audio_id():
    # 앱은 JSON 을 64KiB 까지만 읽으므로 음성은 audio_id 로 따로 받아 간다
    plain = _upload().json()
    assert "audio_id" not in plain
    body = client.post("/api/transcribe", files={"file": ("q.wav", _wav(), "audio/wav")},
                       data={"language": "ko", "task": "transcribe", "speech": "1"}).json()
    assert len(body["audio_id"]) == 32 and len(json.dumps(body).encode()) < 65536
    r = client.get(f"/api/audio/{body['audio_id']}")
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav" and r.content[:4] == b"RIFF"
    img = client.post("/api/image", json={"image": _jpeg_b64(), "speech": True}).json()
    assert client.get(f"/api/audio/{img['audio_id']}").content[:4] == b"RIFF"
    assert client.get("/api/audio/" + "0" * 32).status_code == 404
    assert client.get("/api/audio/..%2Fsecret").status_code == 404


def test_no_speech_for_empty_answer(monkeypatch):
    monkeypatch.setattr(state.pipeline, "understand", lambda wav: ("어제 축구 봤어?", "ignore"))
    body = client.post("/api/ask", files={"file": ("q.wav", _wav(), "audio/wav")}, data={"speech": "1"}).json()
    assert body["answer"] == "" and "audio_id" not in body


def test_requests_are_recorded_for_watcher():
    log = Path(state.settings.data_dir) / "requests.jsonl"
    log.unlink(missing_ok=True)
    _upload()
    client.post("/api/image", json={"image": _jpeg_b64(), "source": "글라스"})
    client.post("/chat", json={"text": "PSD 501 코드가 뭐야?"})
    rows = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
    assert [r["endpoint"] for r in rows] == ["/api/transcribe", "/api/image", "/chat"]
    assert rows[0]["transcript"] == "PSD 501 코드가 뭐야?" and rows[0]["answer"] and rows[0]["intent"] == "qa"
    assert rows[1]["source"] == "글라스" and rows[1]["answer"]
    assert rows[2]["intent"] == "qa" and rows[2]["answer"]
    # 보기 창의 입력 시각·추론 시간 내역
    assert all(r["received"] and r["total_s"] >= 0 for r in rows)
    assert rows[0]["understand_s"] >= 0 and rows[0]["answer_s"] >= 0 and rows[1]["read_s"] >= 0


def test_speech_ready_is_recorded_after_its_question():
    log = Path(state.settings.data_dir) / "requests.jsonl"
    log.unlink(missing_ok=True)
    body = client.post("/api/transcribe", files={"file": ("q.wav", _wav(), "audio/wav")},
                       data={"speech": "1"}).json()
    client.get(f"/api/audio/{body['audio_id']}")   # 합성이 끝날 때까지 기다린다
    rows = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
    assert [r["endpoint"] for r in rows] == ["/api/transcribe", "speech"]
    assert rows[0]["server_speech"] is True and rows[1]["received"] == rows[0]["received"]
    assert rows[1]["ready_s"] >= rows[1]["after_answer_s"] >= 0 and rows[1]["audio_s"] > 0


def test_sts_native_speech():
    lines = _lines(client.post("/sts", files={"file": ("q.wav", _wav(), "audio/wav")}))
    assert lines[0]["type"] == "qa" and lines[0]["transcript"]
    assert any("audio" in l for l in lines)
    assert lines[-1]["done"] is True


def test_db_qa():
    assert client.post("/db/qa", json={"question": "q", "answer": "a"}).json() == {"status": "ok"}


def test_sentence_splitter():
    s = SentenceSplitter()
    out = []
    for piece in ["501번은 '하부", " 가이드레일' 이물질이야. 승객", " 안전부터 확인해! 버전 3.5", "는 괜찮아"]:
        out += s.feed(piece)
    out += s.flush()
    assert out == ["501번은 '하부 가이드레일' 이물질이야.", "승객 안전부터 확인해!", "버전 3.5는 괜찮아"]


def test_parse_understand_broken_json():
    assert parse_understand('```json\n{"transcript": "PSD 501 뭐야", "intent": "qa"}\n```') == ("PSD 501 뭐야", "qa")
    assert parse_understand('{"transcript": "야 죽고 싶냐", "intent": "danger"') == ("야 죽고 싶냐", "danger")


# ---------------------------------------------------------------- fallback TTS 스트리밍

@pytest.fixture
def fallback_tts(monkeypatch):
    """mock 엔진이 직접 말하지 못하는 것처럼 바꿔 서버 TTS 경로를 탄다."""
    monkeypatch.setattr(state.pipeline.engine, "supports_speech_output", False)


def test_sts_chunk_stream_order(fallback_tts):
    lines = _lines(client.post("/sts", files={"file": ("q.wav", _wav(), "audio/wav")}))
    assert lines[0]["type"] == "qa"
    body = lines[1:-1]
    texts = {l["seq"]: i for i, l in enumerate(body) if "tts" in l}
    audio_seqs = [l["seq"] for l in body if "audio" in l]
    ends = [l["seq"] for l in body if l.get("audio_end")]
    assert len(texts) >= 2 and audio_seqs, body
    assert audio_seqs == sorted(audio_seqs)                 # 음성 조각은 문장 순서대로
    assert ends == sorted(texts)                             # 모든 문장에 audio_end, 순서대로
    for i, l in enumerate(body):                             # 문장 텍스트가 그 문장의 음성보다 먼저
        if "audio" in l:
            assert texts[l["seq"]] < i
    assert lines[-1]["done"] is True


def test_sts_sentence_mode(fallback_tts):
    lines = _lines(client.post("/sts", files={"file": ("q.wav", _wav(), "audio/wav")},
                               data={"audio_stream": "sentence"}))
    body = lines[1:-1]
    assert body and all("tts" in l and "audio" in l for l in body)   # 문장 + 완성된 음성 한 줄씩
    assert [l["seq"] for l in body] == list(range(len(body)))


def test_first_sentence_soft_split():
    s = SentenceSplitter(first_soft_split=16)
    out = []
    for piece in ["승객이 선로로 접근하지 못하게, 즉시 ", "통제해. 그리고 관제실에 보고하고, 문을 잠가."]:
        out += s.feed(piece)
    out += s.flush()
    # 첫 문장만 쉼표에서 끊고, 이후 문장은 쉼표에서 끊지 않는다
    assert out == ["승객이 선로로 접근하지 못하게,", "즉시 통제해.", "그리고 관제실에 보고하고, 문을 잠가."]


def test_debug_returns_contexts_for_ragas(monkeypatch):
    # scripts/ragas_eval.py 는 서버가 실제로 쓴 검색 문서를 받아 채점한다
    monkeypatch.setattr(state.pipeline.retriever, "search", lambda q, k=3: [f"Q: {q}\nA: 문서 답"])
    body = client.post("/api/ask", files={"file": ("q.wav", _wav(), "audio/wav")}, data={"debug": "1"}).json()
    assert body["contexts"] == ["Q: PSD 501 코드가 뭐야?\nA: 문서 답"] and body["understand_s"] >= 0
    assert "contexts" not in _upload("/api/ask").json()             # 앱 요청에는 싣지 않는다
    done = _lines(client.post("/chat", json={"text": "PSD 501 코드가 뭐야?", "debug": True}))[-1]
    assert done["done"] and done["contexts"] == ["Q: PSD 501 코드가 뭐야?\nA: 문서 답"]
    assert "contexts" not in _lines(client.post("/chat", json={"text": "PSD 501 코드가 뭐야?"}))[-1]


# ---------------------------------------------------------------- 번역

def test_translate_text():
    body = client.post("/api/translate", json={"text": "PSD 501 코드가 뭐야?", "target": "EN"}).json()
    assert body["text"] == body["translation"] == "What is the PSD 501 code?"
    assert body["source_text"] == "PSD 501 코드가 뭐야?" and body["target"] == "en" and "audio_id" not in body


@pytest.mark.parametrize("payload", [{"text": "안녕", "target": "xx"}, {"text": "  ", "target": "en"}])
def test_translate_rejects_bad_input(payload):
    assert client.post("/api/translate", json=payload).status_code == 400


def test_translate_audio_transcribes_then_translates():
    seen = []
    real = state.pipeline.translate
    state.pipeline.translate = lambda text, target: seen.append((text, target)) or real(text, target)
    try:
        body = client.post("/api/translate/audio", files={"file": ("q.wav", _wav(), "audio/wav")},
                           data={"language": "ko", "target": "ja"}).json()
    finally:
        state.pipeline.translate = real
    assert seen == [("PSD 501 코드가 뭐야?", "ja")]                   # 받아쓴 글을 번역한다
    assert body["source_text"] == "PSD 501 코드가 뭐야?" and body["translation"] and body["source"] == "ko"
    assert body["transcribe_s"] >= 0 and body["translate_s"] >= 0


def test_translate_speech_only_in_tts_language():
    # 테스트의 windows TTS 는 언어를 못 고른다 → 한국어로 옮길 때만 음성
    ko = client.post("/api/translate", json={"text": "Where is the exit?", "target": "ko", "speech": True}).json()
    assert client.get(f"/api/audio/{ko['audio_id']}").content[:4] == b"RIFF"
    en = client.post("/api/translate", json={"text": "출구가 어디예요?", "target": "en", "speech": True}).json()
    assert en["translation"] and "audio_id" not in en


def test_translate_glossary_only_matching_terms():
    from sts_server.prompts import translate_user_text
    zh = translate_user_text("오송역에서 경부선 무궁화호로 갈아타세요.", "zh")
    assert "오송역 → 五松站" in zh and "경부선 → 京釜线" in zh and "무궁화호 → 无穷花号" in zh
    assert "서울역" not in zh and zh.endswith("[원문]\n오송역에서 경부선 무궁화호로 갈아타세요.")
    assert translate_user_text("在五松站换乘", "ko").startswith("[용어집]\n五松站 → 오송역")
    assert translate_user_text("출구가 어디예요?", "en") == "[원문]\n출구가 어디예요?"     # 용어 없으면 용어집도 없음
    assert "[용어집]" not in translate_user_text("오송역", "zh-tw")                      # 번체 표기는 아직 없음


def test_supertonic_languages_match_package():
    from supertonic.config import SUPPORTED_LANGUAGES

    from multimodel.tts.supertonic import SupertonicTTS
    assert set(SupertonicTTS.languages) == set(SUPPORTED_LANGUAGES) - {"na"} and "zh" not in SupertonicTTS.languages
