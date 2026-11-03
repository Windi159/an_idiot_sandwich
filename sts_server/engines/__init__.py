from __future__ import annotations

from sts_server.config import Settings
from sts_server.engines.base import Message, OmniEngine

_ENGINES = {
    "qwen3_omni_hf": "sts_server.engines.qwen3_omni_hf:Qwen3OmniHF",
    "llamacpp": "sts_server.engines.llamacpp:LlamaCppEngine",
    "mock": "sts_server.engines.mock:MockEngine",
}


def create_engine(settings: Settings) -> OmniEngine:
    import importlib

    if settings.engine not in _ENGINES:
        raise ValueError(f"알 수 없는 엔진: {settings.engine!r} (사용 가능: {', '.join(_ENGINES)})")
    module_name, class_name = _ENGINES[settings.engine].split(":")
    return getattr(importlib.import_module(module_name), class_name)(settings)


__all__ = ["Message", "OmniEngine", "create_engine"]
