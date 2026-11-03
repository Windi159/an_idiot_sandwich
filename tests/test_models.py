"""모델 프리셋 · 양자화 선택 · 예전 엔진 이름 호환."""

from __future__ import annotations

import pytest

from sts_server.config import Settings
from sts_server.engines.llamacpp import LlamaCppEngine
from sts_server.models import resolve


def test_gemma_default_is_q4_qat():
    preset, quant, files = resolve("gemma4-12b", "")
    assert quant == "q4"
    assert files.model.endswith("gemma-4-12b-it-qat-q4_0.gguf")
    assert preset.n_cpu_moe == 0


def test_gemma_q8():
    _, quant, files = resolve("gemma4-12b", "Q8")
    assert quant == "q8" and files.model.endswith("gemma-4-12B-it-Q8_0.gguf")
    assert files.mmproj.endswith("mmproj-gemma-4-12B-it-BF16.gguf")


def test_unknown_quant_rejected():
    with pytest.raises(ValueError, match="q8"):
        resolve("qwen3-omni-30b", "q8")


def test_engine_uses_preset_and_overrides(monkeypatch):
    s = Settings(model="gemma4-12b", quant="q8")
    e = LlamaCppEngine(s)
    assert "Gemma 4 12B" in e.model_id and "q8" in e.model_id
    assert e.n_cpu_moe == 0
    s = Settings(model="qwen3-omni-30b", llama_n_cpu_moe=20, llama_model="X:/custom.gguf")
    e = LlamaCppEngine(s)
    assert e.n_cpu_moe == 20 and e.model_path == "X:/custom.gguf"


@pytest.mark.parametrize("old, model", [("qwen3_omni_llamacpp", "qwen3-omni-30b"), ("gemma4_llamacpp", "gemma4-12b")])
def test_old_engine_names(monkeypatch, old, model):
    monkeypatch.setattr("sts_server.config._load_dotenv", lambda path: None)  # 로컬 .env 영향 차단
    for k in ("STS_MODEL", "STS_QUANT"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("STS_ENGINE", old)
    s = Settings.from_env()
    assert (s.engine, s.model) == ("llamacpp", model)


def test_explicit_model_wins_over_old_engine_name(monkeypatch):
    monkeypatch.setattr("sts_server.config._load_dotenv", lambda path: None)
    monkeypatch.setenv("STS_ENGINE", "qwen3_omni_llamacpp")
    monkeypatch.setenv("STS_MODEL", "gemma4-12b")
    s = Settings.from_env()
    assert (s.engine, s.model) == ("llamacpp", "gemma4-12b")
