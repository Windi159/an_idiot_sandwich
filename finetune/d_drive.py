"""미세 조정 스크립트가 맨 먼저 import 한다 — 캐시·임시 파일이 C 드라이브에 쌓이지 않게 전부 D 드라이브로 돌린다.

torch·transformers·triton·matplotlib 등은 import 할 때 이 값을 읽으므로 반드시 다른 import 보다 먼저 불러야 한다.
"""

import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # D:\multi model test
CACHE = ROOT / "cache"

_DIRS = {
    "HF_HOME": ROOT / "models" / "hf-cache",           # 서버 설정(STS_HF_HOME)과 같은 곳
    "TORCH_HOME": CACHE / "torch",
    "TRITON_CACHE_DIR": CACHE / "triton",
    "TORCHINDUCTOR_CACHE_DIR": CACHE / "torchinductor",
    "CUDA_CACHE_PATH": CACHE / "cuda",
    "MPLCONFIGDIR": CACHE / "matplotlib",
    "PIP_CACHE_DIR": CACHE / "pip",
    "UV_CACHE_DIR": Path("D:/uv-cache"),
    "TMP": CACHE / "tmp",
    "TEMP": CACHE / "tmp",
    "TMPDIR": CACHE / "tmp",
}

for key, path in _DIRS.items():
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)                          # 바깥 설정보다 우선 — C 로 새는 걸 막는 게 목적
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
tempfile.tempdir = str(_DIRS["TMP"])

if __name__ == "__main__":
    for key in _DIRS:
        print(f"{key:24s} {os.environ[key]}")
