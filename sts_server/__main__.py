"""실행: uv run sts-server  (또는 python -m sts_server)"""

from __future__ import annotations

import argparse
import logging
import os
import sys


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    from sts_server.models import PRESETS

    p = argparse.ArgumentParser(prog="sts-server", description="멀티모달 STS 서버 (GlassAssist 호환)")
    p.add_argument("--engine", help="llamacpp | qwen3_omni_hf | mock (STS_ENGINE 보다 우선)")
    p.add_argument("--model", choices=list(PRESETS), help="llamacpp 엔진의 모델 (STS_MODEL 보다 우선)")
    p.add_argument("--quant", help="양자화 버전, 예: q4 | q8 (STS_QUANT 보다 우선)")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.add_argument("--rag", help="none | jsonl | chroma")
    args = p.parse_args()

    for key, value in (("ENGINE", args.engine), ("MODEL", args.model), ("QUANT", args.quant),
                       ("HOST", args.host), ("PORT", args.port), ("RAG_BACKEND", args.rag)):
        if value is not None:
            os.environ[f"STS_{key}"] = str(value)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import uvicorn

    from sts_server.config import Settings

    s = Settings.from_env()
    uvicorn.run("sts_server.app:app", host=s.host, port=s.port, log_level="info")


if __name__ == "__main__":
    main()
