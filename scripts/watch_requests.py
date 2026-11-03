"""서버에 들어온 질문과 답변을 실시간으로 보여준다 (data/requests.jsonl 을 따라 읽는다).

  uv run python scripts/watch_requests.py            # 지금부터 들어오는 것만
  uv run python scripts/watch_requests.py --last 10  # 최근 10건부터
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
os.system("")  # Windows 콘솔에서 색 코드가 먹게

LOG = Path(__file__).resolve().parent.parent / "data" / "requests.jsonl"

GRAY, CYAN, GREEN, YELLOW, RED, BLUE, BOLD, RESET = ("\033[90m", "\033[96m", "\033[92m", "\033[93m", "\033[91m",
                                                     "\033[94m", "\033[1m", "\033[0m")
KIND = {"/api/image": "사진", "/api/transcribe": "음성", "/api/ask": "음성", "/sts": "음성(STS)", "/chat": "글"}
INTENT = {"qa": "질문", "ignore": "잡담·무시", "danger": "위협"}


def sec(v) -> str:
    return f"{v:.2f}초" if isinstance(v, (int, float)) else "-"


def show_speech(row: dict) -> None:
    """서버 음성(답변 TTS)이 다 만들어졌다는 줄. 해당 질문 바로 뒤에 들어온다."""
    print(f"  {BLUE}🔊 서버 음성 준비{RESET} 입력부터 {BOLD}{sec(row.get('ready_s'))}{RESET}"
          f"  {GRAY}(답변 뒤 +{sec(row.get('after_answer_s'))} · {row.get('audio_s', 0):.1f}초 분량 · {row.get('kb', 0)}KB"
          f" · {row.get('received', '')} 입력분){RESET}\n")


def show(row: dict) -> None:
    if row.get("endpoint") == "speech":
        show_speech(row)
        return
    kind = KIND.get(row.get("endpoint"), row.get("endpoint"))
    if row.get("source"):
        kind += f" · {row['source']}"
    intent = row.get("intent")
    # 입력 시각: 서버가 요청(녹음·사진 업로드)을 다 받은 때. 예전 기록에는 없어서 응답 시각으로 대신한다.
    received = row.get("received") or row.get("at", "")
    head = f"{GRAY}[입력 {received}]{RESET} {BOLD}{kind}{RESET}"
    if intent:
        color = RED if intent == "danger" or row.get("threat") else (GRAY if intent == "ignore" else CYAN)
        head += f"  {color}{INTENT.get(intent, intent)}{RESET}"
    if row.get("threat"):
        head += f"  {RED}⚠ 위협 감지{RESET}"
    print(head)
    question = row.get("transcript") or ("(질문 없이 사진만)" if row.get("endpoint") == "/api/image" else "(말소리 없음)")
    print(f"  {YELLOW}Q{RESET} {question}")
    if row.get("error"):
        print(f"  {RED}오류{RESET} {row['error']}")
    elif "answer" in row:
        print(f"  {GREEN}A{RESET} {row['answer'] or GRAY + '(답변 없음 — 앱은 말하지 않음)' + RESET}")

    # 추론 시간 = 모델이 쓴 시간. 내역은 음성이면 받아쓰기·분류 / 답변, 사진이면 사진 읽기 / 문서 붙여 다시 답하기
    detail = ""
    if row.get("understand_s") is not None:
        detail = f" (받아쓰기·분류 {sec(row['understand_s'])} + 답변 {sec(row.get('answer_s'))})"
    elif row.get("read_s") is not None:
        detail = f" (사진 읽기 {sec(row['read_s'])} + 문서 보고 다시 답 {sec(row.get('answer_s'))})"
    total = f" · 입력부터 응답까지 {sec(row['total_s'])}" if row.get("total_s") is not None else ""
    print(f"  {GRAY}⏱{RESET} 추론 {BOLD}{sec(row.get('elapsed'))}{RESET}{GRAY}{detail}{total}{RESET}")
    if row.get("answer") and row.get("endpoint") in ("/api/transcribe", "/api/ask", "/api/image"):
        if row.get("server_speech"):
            print(f"  {BLUE}🔊 서버 음성 만드는 중...{RESET}")
        else:
            print(f"  {GRAY}🔊 폰 TTS로 읽음 (폰에서 합성 — 서버에서는 시간을 잴 수 없음){RESET}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", type=int, default=0, help="시작할 때 최근 몇 건을 먼저 보여줄지")
    a = ap.parse_args()
    print(f"{BOLD}STS 서버 질문·답변 보기{RESET}  {GRAY}({LOG}){RESET}\n{GRAY}닫으려면 Ctrl+C{RESET}\n")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.touch(exist_ok=True)
    with LOG.open(encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines[-a.last:] if a.last else []:
            show(json.loads(line))
        buf = ""
        while True:
            chunk = f.readline()
            if not chunk:
                time.sleep(0.3)
                continue
            buf += chunk
            if not buf.endswith("\n"):   # 서버가 아직 쓰는 중인 줄
                continue
            try:
                show(json.loads(buf))
            except json.JSONDecodeError:
                pass
            buf = ""


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
