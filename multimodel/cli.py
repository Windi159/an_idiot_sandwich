"""명령줄 인터페이스: mmt stt | tts | voices | roundtrip | ui"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from multimodel import registry

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def default_output(engine_ext: str, prefix: str = "tts") -> Path:
    return OUTPUT_DIR / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S_%f}{engine_ext}"


def _stt_kwargs(args) -> dict:
    return {"model_size": args.model, "device": args.device}


def cmd_stt(args) -> None:
    engine = registry.get_stt(args.engine, **_stt_kwargs(args))
    t0 = time.perf_counter()
    result = engine.transcribe(args.audio, language=args.language)
    elapsed = time.perf_counter() - t0
    if args.segments:
        for s in result.segments:
            print(f"[{s.start:6.2f} → {s.end:6.2f}] {s.text}")
    else:
        print(result.text)
    print(f"\n(언어: {result.language}, 길이: {result.duration:.1f}s, 처리: {elapsed:.1f}s, "
          f"장치: {getattr(engine, 'active_device', '?')})", file=sys.stderr)


def cmd_tts(args) -> None:
    engine = registry.get_tts(args.engine)
    text = args.text if args.text != "-" else sys.stdin.read()
    out = Path(args.output) if args.output else default_output(engine.extension)
    path = engine.synthesize(text, out, voice=args.voice, rate=args.rate)
    print(path)


def cmd_voices(args) -> None:
    for v in registry.get_tts(args.engine).list_voices(args.language):
        print(f"{v.id:40s} {v.language:8s} {v.gender:8s} {v.description}")


def cmd_roundtrip(args) -> None:
    """TTS 로 만든 음성을 다시 STT 로 인식해 두 모델을 함께 점검."""
    tts = registry.get_tts(args.engine)
    audio = tts.synthesize(args.text, default_output(tts.extension, "roundtrip"), voice=args.voice)
    stt = registry.get_stt("whisper", **_stt_kwargs(args))
    result = stt.transcribe(audio, language=args.language)
    print(f"원문  : {args.text}")
    print(f"인식  : {result.text}")
    print(f"오디오: {audio}")


def cmd_ui(args) -> None:
    from multimodel.app import build_app

    build_app().launch(server_name=args.host, server_port=args.port, share=args.share,
                       inbrowser=not args.no_browser)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mmt", description="STT + TTS 멀티 모델 툴킷")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    def add_stt_opts(sp):
        sp.add_argument("--model", default="large-v3-turbo",
                        help="Whisper 모델 (tiny/base/small/medium/large-v3/large-v3-turbo)")
        sp.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
        sp.add_argument("-l", "--language", default=None, help="언어 코드 (예: ko). 생략 시 자동 감지")

    sp = sub.add_parser("stt", help="음성 → 텍스트")
    sp.add_argument("audio")
    sp.add_argument("--engine", default=registry.DEFAULT_STT, choices=registry.list_stt_engines())
    sp.add_argument("--segments", action="store_true", help="타임스탬프별 출력")
    add_stt_opts(sp)
    sp.set_defaults(func=cmd_stt)

    sp = sub.add_parser("tts", help="텍스트 → 음성")
    sp.add_argument("text", help="합성할 텍스트 ('-' 이면 stdin)")
    sp.add_argument("-o", "--output")
    sp.add_argument("--engine", default=registry.DEFAULT_TTS, choices=registry.list_tts_engines())
    sp.add_argument("--voice")
    sp.add_argument("--rate", type=int, default=0, help="속도 -50 ~ 50 (%%)")
    sp.set_defaults(func=cmd_tts)

    sp = sub.add_parser("voices", help="TTS 음성 목록")
    sp.add_argument("--engine", default=registry.DEFAULT_TTS, choices=registry.list_tts_engines())
    sp.add_argument("-l", "--language", default="ko")
    sp.set_defaults(func=cmd_voices)

    sp = sub.add_parser("roundtrip", help="TTS → STT 왕복 테스트")
    sp.add_argument("text")
    sp.add_argument("--engine", default=registry.DEFAULT_TTS, choices=registry.list_tts_engines())
    sp.add_argument("--voice")
    add_stt_opts(sp)
    sp.set_defaults(func=cmd_roundtrip)

    sp = sub.add_parser("ui", help="웹 UI 실행 (Gradio)")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=7860)
    sp.add_argument("--share", action="store_true")
    sp.add_argument("--no-browser", action="store_true", help="브라우저 자동 열기 끄기")
    sp.set_defaults(func=cmd_ui)
    return p


def main(argv: list[str] | None = None) -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
