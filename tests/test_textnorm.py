"""TTS 전 숫자 읽기 정규화."""

from __future__ import annotations

import pytest

from multimodel.tts.textnorm import native, normalize_ko, sino


@pytest.mark.parametrize("n, word", [
    (0, "영"), (1, "일"), (10, "십"), (11, "십일"), (110, "백십"), (501, "오백일"),
    (1000, "천"), (2024, "이천이십사"), (10000, "만"), (15000, "만오천"), (120000, "십이만"),
])
def test_sino(n, word):
    assert sino(n) == word


def test_native():
    assert [native(n) for n in (1, 2, 3, 20, 21, 35)] == ["한", "두", "세", "스무", "스물한", "서른다섯"]


@pytest.mark.parametrize("text, expected", [
    ("501번은 하부 가이드레일 이물질 끼임이야.", "오백일 번은 하부 가이드레일 이물질 끼임이야."),
    ("관제실에 3번 승강장 비상문 개방이라고 보고해.", "관제실에 삼 번 승강장 비상문 개방이라고 보고해."),
    ("문을 3번 확인해.", "문을 세 번 확인해."),
    ("2번만 더 눌러봐.", "두 번만 더 눌러봐."),
    ("승객 2명이 다쳤어.", "승객 두 명이 다쳤어."),
    ("3시에 열차가 와.", "세 시에 열차가 와."),
    ("5분 뒤에 도착해.", "오 분 뒤에 도착해."),
    ("배터리가 85%야.", "배터리가 팔십오 퍼센트야."),
    ("전압이 3.5볼트야.", "전압이 삼 점 오 볼트야."),
    ("요금은 1,250원이야.", "요금은 천이백오십 원이야."),
    ("PSD 501 코드", "PSD 오백일 코드"),
    ("2호선", "이 호선"),
])
def test_normalize(text, expected):
    assert normalize_ko(text) == expected


def test_no_numbers_unchanged():
    s = "승객 안전부터 확인하고 발판 사이를 점검해봐."
    assert normalize_ko(s) == s
