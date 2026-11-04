"""텍스트 후처리: 문장 분리, 음성용 정리, 의도 파싱, 위협 단어 가드."""

from __future__ import annotations

import json
import re

INTENTS = ("qa", "ignore", "danger")

# 오탐을 줄이기 위한 위협·욕설 단어 사전. LLM 이 danger 로 분류해도 이 중 하나가 없으면 내린다.
# 실제 운영 전에는 AIHub 욕설/혐오표현 데이터 등으로 보강하거나 전용 분류 모델로 대체할 것.
THREAT_WORDS = (
    "죽여", "죽일", "죽인다", "죽고 싶", "죽을래", "패버", "때려", "때릴", "칼로", "찔러", "불 질러", "불질러",
    "가만 안", "가만두지", "씨발", "시발", "ㅅㅂ", "개새", "새끼", "병신", "좆", "꺼져", "닥쳐", "미친년", "미친놈",
)

_SENT_END = re.compile(r"(.+?(?:[.!?。…~]+|\n+))(?=\s|$)", re.S)
_MARKDOWN = re.compile(r"[*#`>_|]+|^\s*[-•·]\s+|^\s*\d+\.\s+", re.M)


def clean_for_speech(text: str) -> str:
    text = _MARKDOWN.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


# 안내방송 문안의 빈칸. 화면에는 원문 그대로 두고, 소리로 읽을 때만 자연스러운 말로 바꾼다.
#   "00시 00분" → "몇 시 몇 분", "00번" → "몇 번", "○○역" → "해당 역", "(열차명칭)" → "열차명칭"
# 바꾸지 않으면 TTS 가 "영 시 영 분", "괄호 열차명칭"처럼 읽는다.
_ZERO_BLANK = re.compile(r"(?<![\d.,:])0{2,}(?![\d.,:])\s*")
_CIRCLE_BLANK = re.compile(r"[○◯〇]{2,}\s*")
# 괄호 안이 한글 두 글자 이상일 때만 빈칸으로 본다 — "(주)코레일" 같은 약어는 그대로 둔다
_PAREN_BLANK = re.compile(r"[(（]\s*([가-힣][가-힣 /·]{0,13}[가-힣])\s*[)）]")


def speak_placeholders(text: str) -> str:
    """음성 합성 직전에만 쓴다 (한국어). 숫자 빈칸(00)은 '몇', 기호 빈칸(○○)은 '해당', 괄호 빈칸은 괄호만 뗀다."""
    text = _ZERO_BLANK.sub("몇 ", text)
    text = _CIRCLE_BLANK.sub("해당 ", text)
    text = _PAREN_BLANK.sub(r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


_SOFT_BREAK = re.compile(r"(.+?[,，])(?=\s)", re.S)


class SentenceSplitter:
    """스트리밍 토큰을 받아 완성된 문장만 내보낸다.

    first_soft_split 이상 길어진 첫 문장은 쉼표에서 먼저 끊는다 — 첫 음성을 빨리 내기 위해.
    """

    def __init__(self, min_chars: int = 2, first_soft_split: int = 0) -> None:
        self.buf = ""
        self.min_chars = min_chars
        self.first_soft_split = first_soft_split
        self.emitted = 0

    def _emit(self, raw: str, out: list[str]) -> None:
        sent = clean_for_speech(raw)
        if len(sent) >= self.min_chars:
            out.append(sent)
            self.emitted += 1

    def feed(self, piece: str) -> list[str]:
        self.buf += piece
        out: list[str] = []
        while True:
            m = _SENT_END.match(self.buf)
            # 문장 끝 기호 뒤에 공백(또는 다음 글자)이 와야 확정 — "3.5" 같은 숫자에서 끊기지 않게
            if m and m.end() < len(self.buf):
                self.buf = self.buf[m.end():]
                self._emit(m.group(1), out)
                continue
            if self.first_soft_split and self.emitted == 0 and len(self.buf) >= self.first_soft_split:
                soft = _SOFT_BREAK.match(self.buf)
                if soft and len(soft.group(1)) >= self.first_soft_split // 2:
                    self.buf = self.buf[soft.end():]
                    self._emit(soft.group(1), out)
                    continue
            break
        return out

    def flush(self) -> list[str]:
        out: list[str] = []
        self._emit(self.buf, out)
        self.buf = ""
        return out


def parse_intent(raw: str) -> str:
    raw = raw.strip().lower()
    for label in INTENTS:
        if raw.startswith(label) or re.search(rf"\b{label}\b", raw):
            return label
    return "qa"


def parse_understand(raw: str) -> tuple[str, str]:
    """UNDERSTAND 응답(JSON 한 줄)을 (transcript, intent) 로. 형식이 깨져도 최대한 살린다."""
    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            return str(obj.get("transcript", "")).strip(), parse_intent(str(obj.get("intent", "qa")))
        except json.JSONDecodeError:
            pass
    t = re.search(r'"transcript"\s*:\s*"([^"]*)', raw)
    i = re.search(r'"intent"\s*:\s*"([^"]*)', raw)
    return (t.group(1).strip() if t else raw.strip()), parse_intent(i.group(1) if i else "qa")


# 같은 덩어리(1~20자)가 8번 넘게 연달아 나오면 greedy 디코딩이 반복 루프에 빠진 것 ("ed ed ed ...")
# 단어 처음에서 시작하는 덩어리를 먼저 찾는다 — "god ed ed" 가 "d e" 반복으로 잘리지 않게
_LOOP_AT_WORD = re.compile(r"(?<![^\s\"'])(\S.{0,19}?)(?:\s*\1){7,}", re.S)
_LOOP = re.compile(r"(\S.{0,19}?)(?:\s*\1){7,}", re.S)


def _loop_match(text: str) -> re.Match | None:
    return _LOOP_AT_WORD.search(text) or _LOOP.search(text)


def find_loop(text: str) -> int:
    """반복 루프가 시작되는 위치. 없으면 -1."""
    m = _loop_match(text)
    return m.start() if m else -1


def cut_loop(text: str) -> str:
    """반복 루프를 한 번만 남기고 잘라낸다."""
    m = _loop_match(text)
    return (text[:m.start()] + m.group(1)).strip() if m else text


def has_threat_word(text: str) -> bool:
    return any(w in text for w in THREAT_WORDS)
