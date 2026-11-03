"""서버 설정. 환경 변수(또는 프로젝트 루트의 .env)로 덮어쓴다. 키 목록은 .env.example 참고."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """의존성 없이 KEY=VALUE 형식만 읽는다. 이미 설정된 환경 변수가 우선."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.split(" #", 1)[0]  # 줄 끝 주석 제거 (경로에 '#' 이 있으면 따옴표 없이도 앞에 공백이 없어야 함)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class Settings:
    # --- 엔진 선택: llamacpp | qwen3_omni_hf | mock
    engine: str = "llamacpp"

    # --- llama.cpp 로 돌릴 모델 (sts_server/models.py 의 PRESETS)
    model: str = "gemma4-12b"             # gemma4-12b(기본) | qwen3-omni-30b
    quant: str = ""                       # 비우면 프리셋 기본값. gemma4-12b: q4(기본) | q8

    # --- transformers (전체 STS, 80GB급 GPU)
    hf_model: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    hf_attn: str = "sdpa"                 # flash_attention_2 설치 시 변경
    speaker: str = "Ethan"                # Ethan | Chelsie | Aiden

    # --- llama.cpp 서버 (16GB GPU, 음성 출력은 fallback TTS)
    llama_url: str = "http://127.0.0.1:8081"
    llama_autostart: bool = True
    llama_exe: str = str(ROOT / "runtime" / "llama.cpp" / "llama-server.exe")
    llama_model: str = ""                 # 지정하면 프리셋 대신 이 GGUF 사용
    llama_mmproj: str = ""
    llama_ctx: int = 8192
    llama_n_cpu_moe: int = -1             # -1 이면 프리셋 기본값. MoE expert 를 CPU 에 둘 레이어 수
    llama_extra_args: str = ""

    # --- 엔진이 음성을 직접 못 낼 때 쓰는 TTS (multimodel 패키지의 엔진 이름)
    #   supertonic(로컬 CPU, 기본) | windows(로컬, 음질 낮음) | edge(온라인)
    fallback_tts: str = "supertonic"
    fallback_voice: str = ""              # 비우면 엔진 기본 음성 (supertonic F1, edge SunHi)

    # --- /api/transcribe 가 돌려주는 것
    #   answer     : 받아쓰기 → 분류 → RAG → 답변까지 하고 text 에 "답변"을 싣는다 (post_sh-t main 앱, 기본)
    #   transcript : text 에 "받아쓰기"만 싣는다 (예전 앱: 받아쓴 뒤 /chat 을 따로 부르는 방식)
    transcribe_mode: str = "answer"

    # --- 받아쓰기 백엔드: omni(현재 멀티모달 모델로 받아쓰기) | whisper(faster-whisper)
    stt_backend: str = "omni"
    whisper_model: str = "large-v3-turbo"

    # --- RAG: none | jsonl | hybrid | chroma
    rag_backend: str = "none"
    rag_top_k: int = 3
    rag_jsonl: str = ""                   # JSONL 파일들 (; 로 구분) — 질문/답 쌍 또는 문서 청크
    rag_chroma_dir: str = ""              # 기존 GlassAssist Chroma DB 폴더
    rag_chroma_collection: str = "langchain"
    rag_embedding_model: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"   # 미세 조정본: finetune/models/kr-sbert-ft
    # hybrid: 1등 점수가 이보다 낮으면 관련 문서 없음으로 본다. 임베딩을 바꾸면 점수 분포가 달라져 다시 잡아야 한다
    #   (finetune/retrieval_eval.py). 기본 KR-SBERT 0.5 / 미세 조정본 0.45
    rag_min_score: float = 0.5
    # hybrid: 리랭커(크로스 인코더). 비우면 안 씀. 문턱을 넘은 상위 rag_rerank_candidates 개를 다시 줄 세운다
    rag_reranker: str = ""                # 예: dragonkue/bge-reranker-v2-m3-ko
    rag_rerank_candidates: int = 20
    # 음성 질문의 답변 단계에도 음성을 다시 넣을지. false 면 받아쓴 글만 넣는다 (음성 인코딩·프리필이 한 번 줄어 빠르다)
    qa_audio: bool = True

    # --- 의도 분류 가드: danger 판정이어도 위협 단어가 없으면 이 값으로 내린다
    danger_unverified_as: str = "ignore"

    # --- 생성 길이
    max_answer_tokens: int = 384
    # 첫 문장이 이 글자 수를 넘으면 쉼표에서 먼저 끊어 TTS 를 일찍 시작 (0 = 끄기)
    first_split_chars: int = 16

    # --- 서버
    host: str = "0.0.0.0"
    port: int = 8000
    data_dir: str = str(ROOT / "data")
    hf_home: str = str(ROOT / "models" / "hf-cache")

    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv(ROOT / ".env")
        s = cls()
        for f in fields(cls):
            if f.name == "extra":
                continue
            raw = os.environ.get(f"STS_{f.name.upper()}")
            if raw is None:
                continue
            current = getattr(s, f.name)
            if isinstance(current, bool):
                value = raw.lower() in ("1", "true", "yes", "on")
            elif isinstance(current, int):
                value = int(raw)
            elif isinstance(current, float):
                value = float(raw)
            else:
                value = raw
            setattr(s, f.name, value)
        # 예전 엔진 이름(qwen3_omni_llamacpp 등)은 llamacpp + 해당 모델로 바꿔 읽는다
        from sts_server.models import ENGINE_ALIASES
        if s.engine in ENGINE_ALIASES:
            if "STS_MODEL" not in os.environ:
                s.model = ENGINE_ALIASES[s.engine]
            s.engine = "llamacpp"
        # C 드라이브가 작아서 HF 캐시는 기본적으로 프로젝트 안(D 드라이브)에 둔다
        os.environ.setdefault("HF_HOME", s.hf_home)
        return s
