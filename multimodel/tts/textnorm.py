"""TTS 전 한국어 텍스트 정규화 — 숫자를 한글 읽기로 바꾼다.

Supertonic 처럼 숫자 읽기를 스스로 못 하는 엔진용. ("501번" → "오백일 번", "3개" → "세 개")
- 기본은 한자어 수사(일, 이, 삼 …). 장애 코드·승강장 번호·시각 등 이 도메인에서 대부분이 이쪽이다.
- 고유어 단위(개, 명, 살 …) 앞의 1~99 는 고유어 수사(한, 두, 세 …).
- "번" 은 횟수("세 번 확인해")와 번호("3번 승강장")가 갈린다 — 뒤에 '째'가 붙거나 동사가 바로 오면 횟수로 본다.
"""

from __future__ import annotations

import re

_DIGIT = "영일이삼사오육칠팔구"
_SMALL = ["", "십", "백", "천"]
_BIG = ["", "만", "억", "조"]

_NATIVE_ONES = ["", "하나", "둘", "셋", "넷", "다섯", "여섯", "일곱", "여덟", "아홉"]
_NATIVE_ONES_ADN = ["", "한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉"]  # 단위 앞 관형형
_NATIVE_TENS = ["", "열", "스물", "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔"]

# 고유어 수사를 쓰는 단위 (1~99 일 때)
_NATIVE_COUNTERS = ("개", "명", "살", "마리", "가지", "시간", "시", "잔", "권", "대", "군데", "사람", "번째", "줄", "칸", "통")
# '번' 뒤에 오면 번호(한자어)로 읽는 말
_NUMBER_NOUNS = ("승강장", "출구", "게이트", "문", "칸", "선", "열차", "역", "홈", "플랫폼", "코드", "호기", "번지")


def sino(n: int) -> str:
    """한자어 수사. 십·백·천·만 앞의 '일'은 생략 (110 → 백십, 10000 → 만)."""
    if n == 0:
        return "영"
    out = []
    group_idx = 0
    while n > 0:
        group = n % 10000
        if group:
            s = []
            for pos in range(3, -1, -1):
                d = (group // 10 ** pos) % 10
                if d == 0:
                    continue
                s.append(("" if d == 1 and pos > 0 else _DIGIT[d]) + _SMALL[pos])
            word = "".join(s)
            if group_idx > 0 and word == "일":
                word = ""  # 일만 → 만
            out.append(word + _BIG[group_idx])
        n //= 10000
        group_idx += 1
    return "".join(reversed(out))


def native(n: int, adnominal: bool = True) -> str:
    """고유어 수사 (1~99). adnominal=True 면 단위 앞 형태 (한, 두, 스무)."""
    tens, ones = divmod(n, 10)
    ten = _NATIVE_TENS[tens]
    if tens == 2 and ones == 0 and adnominal:
        ten = "스무"
    return ten + (_NATIVE_ONES_ADN if adnominal else _NATIVE_ONES)[ones]


def _read_digits(s: str) -> str:
    return " ".join(_DIGIT[int(c)] for c in s)


_NUM = re.compile(r"(?<![A-Za-z0-9])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(\s*)([가-힣%]*)")


def _replace(m: re.Match) -> str:
    int_part, frac, space, after = m.group(1), m.group(2), m.group(3), m.group(4)
    digits = int_part.replace(",", "")
    # 0으로 시작하는 긴 숫자(전화번호 등)는 한 자리씩
    if len(digits) >= 4 and digits.startswith("0") and not frac:
        return _read_digits(digits) + space + after
    n = int(digits)
    if frac:
        reading = f"{sino(n)} 점 {_read_digits(frac).replace(' ', '')}"
        return f"{reading} {after}" if after else f"{reading}{space}"
    if after.startswith("%"):
        return f"{sino(n)} 퍼센트{after[1:]}"
    if 0 < n < 100 and after.startswith(_NATIVE_COUNTERS):
        return f"{native(n)} {after}"
    if after.startswith("번") and not after.startswith("번째") and 0 < n < 100:
        rest = after[1:]
        # "세 번 확인", "두 번만" 처럼 횟수 / "3번 승강장", "3번은" 처럼 번호
        if rest == "" or rest.startswith(("만", "더", "씩", "이나", "정도", "쯤")):
            following = m.string[m.end():].lstrip()
            if not following.startswith(_NUMBER_NOUNS):
                return f"{native(n)} {after}"
    return f"{sino(n)}{' ' if after else space}{after}" if after else f"{sino(n)}{space}"


def normalize_ko(text: str) -> str:
    return _NUM.sub(_replace, text)
