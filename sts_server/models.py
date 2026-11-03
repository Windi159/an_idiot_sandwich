"""llama.cpp 로 돌리는 멀티모달 모델 프리셋. STS_MODEL / STS_QUANT 로 고른다.

새 모델을 추가하려면 GGUF(+음성 인코더가 든 mmproj)를 models/ 아래 받고 PRESETS 에 한 항목 추가.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sts_server.config import ROOT

MODELS_DIR = ROOT / "models"


@dataclass(frozen=True)
class ModelFiles:
    model: str
    mmproj: str
    #: 이 파일에 맞는 MoE expert CPU 레이어 수. None 이면 프리셋 값
    n_cpu_moe: int | None = None


@dataclass(frozen=True)
class LlamaPreset:
    key: str
    display: str
    quants: dict[str, ModelFiles]
    default_quant: str
    #: MoE expert 를 CPU 로 내릴 레이어 수 (dense 모델은 0)
    n_cpu_moe: int = 0
    #: chat template 에 넘길 옵션 (예: thinking 끄기)
    template_kwargs: dict = field(default_factory=dict)


def _f(folder: str, model: str, mmproj: str, n_cpu_moe: int | None = None) -> ModelFiles:
    return ModelFiles(str(MODELS_DIR / folder / model), str(MODELS_DIR / folder / mmproj), n_cpu_moe)


PRESETS: dict[str, LlamaPreset] = {
    "qwen3-omni-30b": LlamaPreset(
        key="qwen3-omni-30b",
        display="Qwen3-Omni-30B-A3B-Instruct",
        quants={
            # ggml-org 공식 Q4_K_M 17.3GB
            "q4": _f("qwen3-omni-gguf", "Qwen3-Omni-30B-A3B-Instruct-Q4_K_M.gguf",
                     "mmproj-Qwen3-Omni-30B-A3B-Instruct-bf16.gguf"),
            # MXFP4(FP4) 15.9GB — grapeV-ai 변환본 (일본어 위주 imatrix 로 만든 저장소)
            # n_cpu_moe 2·4 는 VRAM 이 넘쳐 공유 메모리로 새면서 오히려 느렸고 6 이 가장 빨랐다 (scripts/tune_cpu_moe.py)
            "fp4": _f("qwen3-omni-gguf", "Qwen3-Omni-30B-A3B-Instruct-MXFP4_MOE-grapeV.gguf",
                      "mmproj-Qwen3-Omni-30B-A3B-Instruct-bf16.gguf", n_cpu_moe=6),
            # MXFP4(FP4) — 공식 BF16 에서 이 PC 의 llama-quantize 로 직접 변환 (scripts/quantize_qwen_fp4.ps1)
            "fp4-own": _f("qwen3-omni-gguf", "Qwen3-Omni-30B-A3B-Instruct-MXFP4_MOE-own.gguf",
                          "mmproj-Qwen3-Omni-30B-A3B-Instruct-bf16.gguf", n_cpu_moe=6),
        },
        default_quant="q4",
        n_cpu_moe=12,       # Q4_K_M 17.3GB → 16GB GPU 에 맞추려고 expert 일부를 CPU 로
    ),
    "gemma4-12b": LlamaPreset(
        key="gemma4-12b",
        display="Gemma 4 12B IT",
        quants={
            # Google 공식 QAT(양자화 인지 학습) 4bit — 품질 손실이 적고 VRAM ~8GB
            "q4": _f("gemma4-12b-gguf", "gemma-4-12b-it-qat-q4_0.gguf", "mmproj-gemma-4-12b-it-qat-q4_0.gguf"),
            # 8bit — 원본에 더 가깝지만 VRAM ~13GB
            "q8": _f("gemma4-12b-gguf", "gemma-4-12B-it-Q8_0.gguf", "mmproj-gemma-4-12B-it-BF16.gguf"),
        },
        default_quant="q4",
        template_kwargs={"enable_thinking": False},  # 음성 비서는 지연이 중요해 사고 과정 출력은 끈다
    ),
}

# 예전 엔진 이름 호환: STS_ENGINE=qwen3_omni_llamacpp → llamacpp + qwen3-omni-30b
ENGINE_ALIASES = {
    "qwen3_omni_llamacpp": "qwen3-omni-30b",
    "gemma4_llamacpp": "gemma4-12b",
}


def resolve(model: str, quant: str) -> tuple[LlamaPreset, str, ModelFiles]:
    if model not in PRESETS:
        raise ValueError(f"알 수 없는 모델: {model!r} (사용 가능: {', '.join(PRESETS)})")
    preset = PRESETS[model]
    quant = (quant or preset.default_quant).lower()
    if quant not in preset.quants:
        raise ValueError(f"{model} 에 {quant!r} 버전이 없습니다 (사용 가능: {', '.join(preset.quants)})")
    return preset, quant, preset.quants[quant]
