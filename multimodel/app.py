"""Gradio 웹 UI: STT / TTS / 음성 왕복(마이크 → 텍스트 → 음성)."""

from __future__ import annotations

import gradio as gr

from multimodel import registry
from multimodel.cli import default_output

WHISPER_MODELS = ["large-v3-turbo", "large-v3", "medium", "small", "base", "tiny"]
LANGUAGES = {"자동 감지": None, "한국어": "ko", "English": "en", "日本語": "ja", "中文": "zh"}


def _voices_for(engine: str) -> list[str]:
    try:
        return [v.id for v in registry.get_tts(engine).list_voices("ko")] or \
               [v.id for v in registry.get_tts(engine).list_voices()]
    except Exception:  # noqa: BLE001 - 네트워크 없을 때 등
        return []


def stt_fn(audio, model, language):
    if not audio:
        raise gr.Error("음성을 녹음하거나 파일을 올려주세요.")
    engine = registry.get_stt("whisper", model_size=model)
    r = engine.transcribe(audio, language=LANGUAGES[language])
    segs = "\n".join(f"[{s.start:6.2f} → {s.end:6.2f}] {s.text}" for s in r.segments)
    info = f"언어: {r.language} · 길이: {r.duration:.1f}s · 장치: {engine.active_device}"
    return r.text, segs, info


def tts_fn(text, engine_name, voice, rate):
    if not text or not text.strip():
        raise gr.Error("텍스트를 입력해주세요.")
    engine = registry.get_tts(engine_name)
    return str(engine.synthesize(text, default_output(engine.extension), voice=voice or None, rate=int(rate)))


def update_voices(engine_name):
    voices = _voices_for(engine_name)
    default = registry.get_tts(engine_name).default_voice
    return gr.update(choices=voices, value=default if default in voices else (voices[0] if voices else None))


def roundtrip_fn(audio, model, language, engine_name, voice):
    text, _, info = stt_fn(audio, model, language)
    if not text:
        raise gr.Error("인식된 음성이 없습니다.")
    return text, tts_fn(text, engine_name, voice, 0), info


def build_app() -> gr.Blocks:
    tts_engines = registry.list_tts_engines()
    init_voices = _voices_for(registry.DEFAULT_TTS)
    init_voice = registry.get_tts(registry.DEFAULT_TTS).default_voice

    def tts_controls():
        eng = gr.Dropdown(tts_engines, value=registry.DEFAULT_TTS, label="TTS 엔진")
        voice = gr.Dropdown(init_voices, value=init_voice, label="음성", allow_custom_value=True)
        eng.change(update_voices, eng, voice)
        return eng, voice

    def stt_controls():
        model = gr.Dropdown(WHISPER_MODELS, value="large-v3-turbo", label="Whisper 모델")
        lang = gr.Dropdown(list(LANGUAGES), value="한국어", label="언어")
        return model, lang

    with gr.Blocks(title="Multi Model · STT + TTS") as demo:
        gr.Markdown("# 🎙️ Multi Model — STT + TTS")

        with gr.Tab("STT (음성 → 텍스트)"):
            with gr.Row():
                with gr.Column():
                    audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="음성")
                    model, lang = stt_controls()
                    btn = gr.Button("인식", variant="primary")
                with gr.Column():
                    text_out = gr.Textbox(label="인식 결과", lines=6, buttons=["copy"])
                    seg_out = gr.Textbox(label="구간별", lines=6)
                    info_out = gr.Markdown()
            btn.click(stt_fn, [audio, model, lang], [text_out, seg_out, info_out])

        with gr.Tab("TTS (텍스트 → 음성)"):
            with gr.Row():
                with gr.Column():
                    text_in = gr.Textbox(label="텍스트", lines=6, value="안녕하세요. 멀티 모델 음성 합성 테스트입니다.")
                    eng, voice = tts_controls()
                    rate = gr.Slider(-50, 50, value=0, step=5, label="속도 (%)")
                    btn = gr.Button("합성", variant="primary")
                with gr.Column():
                    audio_out = gr.Audio(label="결과", type="filepath", autoplay=True)
            btn.click(tts_fn, [text_in, eng, voice, rate], audio_out)

        with gr.Tab("왕복 (말하기 → 인식 → 다시 읽기)"):
            with gr.Row():
                with gr.Column():
                    mic = gr.Audio(sources=["microphone", "upload"], type="filepath", label="말하기")
                    model, lang = stt_controls()
                    eng, voice = tts_controls()
                    btn = gr.Button("실행", variant="primary")
                with gr.Column():
                    rt_text = gr.Textbox(label="인식된 텍스트", lines=4)
                    rt_audio = gr.Audio(label="다시 읽은 음성", type="filepath", autoplay=True)
                    rt_info = gr.Markdown()
            btn.click(roundtrip_fn, [mic, model, lang, eng, voice], [rt_text, rt_audio, rt_info])

    return demo


if __name__ == "__main__":
    build_app().launch(inbrowser=True)
