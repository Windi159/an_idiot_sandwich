# Multi Model — 멀티모달 STS 서버 (GlassAssist 호환)

GlassAssist 앱(`D:\post_sh-t`)의 **LLM 서버 + STT 서버** 역할을 멀티모달 모델 하나로 처리한다.
음성을 그대로 모델에 넣어 받아쓰기·의도 분류·답변을 만들고, 답변을 음성으로 돌려준다.
모델은 **Gemma 4 12B**(기본)와 **Qwen3-Omni-30B-A3B** 중에서 설정으로 고른다.

```
앱 ──음성──▶ /sts ──▶ 모델: 듣고 받아쓰기 + 분류(qa/ignore/danger)
                         └─ qa ─▶ RAG 검색 ─▶ 모델: 음성 질문 + 문서로 답변 ─▶ 문장 + 음성
```

## 모델 고르기

`.env` 의 `STS_MODEL` / `STS_QUANT`, 또는 실행 옵션 `--model` / `--quant` 로 바꾼다. 바꾼 뒤에는 서버를 다시 시작한다.

```bash
uv run sts-server                                # .env 설정 (기본: gemma4-12b q4)
uv run sts-server --quant q8                     # Gemma 4 12B 8bit
uv run sts-server --model qwen3-omni-30b         # Qwen3-Omni
```

| `STS_MODEL` | `STS_QUANT` | 파일 | VRAM (실측) | `/sts` 전체 | 음성 이해 |
|---|---|---|---|---|---|
| `gemma4-12b` | `q4` (기본) | Google QAT 4bit 6.5GB | **9.6GB** | **~2.9s** | ~0.5s |
| `gemma4-12b` | `q8` | 8bit 11.8GB | 15.0GB | ~3.3–4.3s | ~0.8s |
| `qwen3-omni-30b` | `q4` | Q4_K_M 17.3GB (expert 일부 CPU) | 15.9GB | ~3.9s | ~1.2s |

- 실측: RTX 5070 Ti 16GB, TTS 로 만든 짧은 한국어 질문 3개, RAG(jsonl) 사용, 답변 음성은 Edge TTS (측정 당시).
- 세 가지 모두 음성 입력을 llama.cpp 로 받고 **답변 음성은 fallback TTS** 가 만든다. Gemma 4 는 원래 텍스트만 출력하고,
  Qwen3-Omni 의 음성 생성부(talker)는 llama.cpp 가 지원하지 않는다.
- 새 모델은 `sts_server/models.py` 의 `PRESETS` 에 GGUF + mmproj(음성 인코더 포함) 경로를 추가하면 된다.

## 엔진

| `STS_ENGINE` | 하드웨어 | 음성 입력 | 음성 출력 |
|---|---|---|---|
| `llamacpp` (기본) | 16GB GPU (이 PC, RTX 5070 Ti) | 위 표의 모델 | fallback TTS (아래 "답변 음성(TTS)") |
| `qwen3_omni_hf` | 80GB급 GPU (A100/H100) | Qwen3-Omni | **Qwen3-Omni 가 직접** (talker, 24kHz) |
| `mock` | 없음 | 가짜 | 가짜 — API/앱 연동 점검용 |

예전 이름 `qwen3_omni_llamacpp` 는 `llamacpp` + `qwen3-omni-30b` 로 읽는다.

16GB GPU 에서 Qwen3-Omni talker 를 못 돌리는 이유: transformers 5.x 는 MoE expert 를 하나의 텐서로 합쳐 로드하면서
4bit 체크포인트도 BF16 으로 풀어버려 약 80GB 가 필요하다.

## 답변 음성 (TTS)

모델이 직접 말하지 못하는 엔진(llamacpp)은 `STS_FALLBACK_TTS` 로 답변을 읽는다. **기본은 로컬 Supertonic 3** — 서버 전체가 인터넷 없이 돈다.

| `STS_FALLBACK_TTS` | 실행 | 문장당 합성 | 발음 정확도* | 비고 |
|---|---|---|---|---|
| `supertonic` (기본) | 로컬 CPU, 0.38GB | **0.3–0.5s** (RTF 0.11) | F1 99%, M1 100% | 음성 F1~F5 / M1~M5, 기본 F1(들어보고 선택). 숫자는 `multimodel/tts/textnorm.py` 로 한글 읽기 변환 |
| `windows` | 로컬 | 빠름 | 낮음 | Windows 내장 음성 |
| `edge` | 온라인 (Microsoft) | 0.5–1.6s | 96% | 응답 편차 큼 |

\* 답변 문장 5개를 합성해 Gemma 로 다시 받아쓴 글자 일치율. Gemma 의 받아쓰기 오차가 섞여 있다. `scripts/compare_tts.py` 로 재현,
샘플은 `outputs/tts_compare/`.

- 안내방송 문안의 빈칸은 **소리 낼 때만** 바꿔 읽는다 (`sts_server/text.py`의 `speak_placeholders`, 한국어만).
  `00시 00분` → "몇 시 몇 분", `00번` → "몇 번", `○○역` → "해당 역", `(열차명칭)` → "열차명칭". 화면에 가는 문장은 원문 그대로다.
  바꾸지 않으면 "영 시 영 분", "괄호 열차명칭"처럼 읽힌다.
- Supertonic 은 CPU 를 나눠 쓰면 첫 문장이 늦어져 문장을 하나씩 합성한다(`max_concurrency = 1`). 실시간의 약 10배 속도라 재생이 끊기지 않는다.
- `/sts` 첫 음성: Supertonic **1.45s** (1.26–1.66s) vs Edge 1.35–1.48s (가끔 2s 이상) — 평균은 비슷하고 편차가 작다.
- Qwen3-TTS(0.6B, 화자 Sohee)도 비교했지만 발음은 정확(100%)한 대신 문장당 8–10s(RTF 1.66)로 실시간이 안 돼 제외했다.

## 설치

```bash
uv sync --extra cuda --extra omni
```

- uv 캐시(`D:/uv-cache`)와 HF 캐시(`models/hf-cache`)는 C 드라이브 공간 때문에 D 드라이브로 지정돼 있다.
- llama.cpp 엔진에 필요한 파일 (이미 받아둠):
  - `runtime/llama.cpp/` — llama.cpp b11889 Windows CUDA 13.4 빌드
  - `models/gemma4-12b-gguf/` — Gemma 4 12B: `gemma-4-12b-it-qat-q4_0.gguf`(Google 공식 QAT), `gemma-4-12B-it-Q8_0.gguf`(ggml-org) + 각 mmproj
  - `models/qwen3-omni-gguf/` — `Qwen3-Omni-30B-A3B-Instruct-Q4_K_M.gguf` + `mmproj-...-bf16.gguf`
  - `models/supertonic-3/` — Supertonic 3 (처음 쓸 때 자동으로 받음)
- 기존 Chroma RAG 를 쓰려면 `uv sync --extra cuda --extra omni --extra rag`

## 실행

```bash
copy .env.example .env
uv run sts-server
```

llama-server 는 자동으로 같이 뜬다(`data/llama-server.log`). 포트 8000 — 기존 Whisper 서버와 같은 포트라
`tailscale serve` 설정을 그대로 쓸 수 있다.

## API

### 기존 (앱 수정 없이 동작)

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/health` | 준비되면 200, 로딩 중/오류면 503 |
| POST | `/chat` | `{"text": "..."}` → NDJSON. 첫 줄 `{"type":"qa"}`, 이후 `{"tts":"문장"}`, 끝 `{"done":true,"answer":"..."}` |
| POST | `/api/transcribe` | multipart `file`, `language`, `task`. 기본(`STS_TRANSCRIBE_MODE=answer`, post_sh-t main 앱) → `{"text"=답변, "answer", "transcript", "intent", "threat"}` / `transcript` 모드(예전 앱) → `{"text"=받아쓰기, "elapsed"}` |
| POST | `/db/qa` | QA 기록 → `data/qa_log.jsonl` (코레일 DB 연동 전 임시) |

### 신규

| 메서드 | 경로 | 설명 |
|---|---|---|
| POST | `/api/ask` | `/api/transcribe` 의 answer 모드와 같음 (설정과 무관) |
| GET | `/api/audio/{audio_id}` | `/api/transcribe`·`/api/ask` 에 `speech=1`, `/api/image` 에 `"speech": true` 를 주면 응답에 `audio_id` 가 붙는다. 그 답변 음성(서버 TTS, PCM16 WAV). 음성 질문은 문장이 나오는 대로 합성해 두어 답변 텍스트 뒤 약 1초, 사진은 답변 전체를 합성해 약 1.5초 뒤 받을 수 있다. 2분 뒤 만료. 시험: `try_app_protocol.py --speech` |
| POST | `/api/image` | `{"image": base64[, "question"]}` → `{"text"=답변, "answer"}`. 글라스·갤러리 사진을 보고 답한다. 먼저 사진을 읽고, 읽은 내용+질문으로 RAG 를 찾아 있으면 다시 답한다 (약 2초). 시험: `scripts/try_image.py` |
| POST | `/api/translate` | `{"text", "target"[, "speech"]}` → `{"text"=번역, "translation", "source_text", "target"}`. 언어 코드는 `prompts.LANGUAGES` (ko·en·ja·zh·zh-tw·vi·th·id·ru·es·fr·de). 역 이름·노선·열차 이름은 `prompts.TRANSLATE_GLOSSARY` 표기로 (원문에 나온 것만 프롬프트에 붙인다). 문장 하나 약 0.3~0.6초. `speech` 면 번역문 음성 `audio_id` — Supertonic 3 가 못 읽는 중국어는 없음 |
| POST | `/api/translate/audio` | multipart `file`, `language`(원문 언어, 기본 ko), `target`(기본 en)[, `speech`] → 받아쓰기 후 번역. 응답은 위와 같고 `source_text`=받아쓴 원문, `transcribe_s`·`translate_s`. 음성을 바로 번역시키면 엉뚱한 문장이 나와 두 단계로 한다 (약 0.5~0.8초) |
| POST | `/sts` | multipart `file`(음성, 아무 포맷), `speech`(기본 true), `audio_stream`(chunk \| sentence) → NDJSON |
| GET | `/info` | 현재 엔진·모델·RAG 설정 |

`/sts` 응답 — `audio_stream=chunk` (기본, 지연 최소):

```
{"type": "qa", "transcript": "PSD 501 코드가 뭐야?"}
{"tts": "501번은 하부 가이드레일 이물질 끼임이야.", "seq": 0}    ← 문장 완성 즉시
{"audio": "<base64 mp3 조각>", "format": "mp3", "seq": 0}        ← TTS 가 내놓는 대로 (여러 줄)
{"tts": "승객 안전부터 확인하고 발판 사이를 점검해봐.", "seq": 1}
{"audio": "...", "format": "mp3", "seq": 0}
{"audio_end": true, "seq": 0}                                    ← 0번 문장 음성 끝
{"audio": "...", "format": "mp3", "seq": 1}
{"audio_end": true, "seq": 1}
{"done": true, "answer": "...", "transcript": "...", "elapsed": 2.48}
```

- 오디오 조각은 **seq 순서대로만** 온다. 같은 seq 의 조각을 이어 붙이면 그 문장의 MP3 가 되고, 순서대로 이어서 재생하면 된다.
  뒤 문장은 서버에서 미리 병렬 합성해 두었다가 앞 문장이 끝나면 바로 보낸다.
- 문장 텍스트(`tts`)는 음성보다 먼저, 순서가 섞여서 올 수 있다 (화면 표시용).
- `audio_stream=sentence` 면 예전처럼 문장 하나가 `{"tts", "audio", "format", "seq"}` 한 줄로 온다.
- `qwen3_omni_hf` 엔진은 문장 이벤트에 audio 가 없고, 끝에 `{"audio": ..., "format": "wav", "sample_rate": 24000}` 한 번으로 모델 음성이 온다.
- `ignore` 면 첫 줄만, `danger` 면 첫 줄에 `tts`(+`audio`) 가 함께 온다.
- 형식은 `/chat` 과 같아서 앱의 `streamQaAnswer` 파서가 그대로 읽는다(모르는 필드는 무시).

`/sts` 지연 실측 (Gemma 4 12B q4, Edge TTS, 샘플 2개 × 5회 중앙값, `scripts/bench_sts.py`):

| | 분류 | 첫 문장 | **첫 음성** | 완료 |
|---|---|---|---|---|
| 개선 전 (문장마다 완성 후, 다음 문장이 나와야 전송) | 0.56s | 1.90s | **1.90s** | 3.02s |
| `audio_stream=sentence` (준비되는 즉시 전송) | 0.56s | 1.71s | **1.71s** | 3.28s |
| `audio_stream=chunk` (조각 스트리밍) | 0.52s | 0.98s | **1.35–1.48s** | 3.0–3.2s |

남은 지연의 대부분은 Edge TTS(클라우드) 응답 편차다. 첫 조각이 보통 0.25–0.35s 지만 가끔 0.9–2.25s 까지 튄다
(서버 로그 `TTS 첫 조각 지연(문장별)` 에 매 요청 기록).

## 테스트

```bash
uv run python -m pytest tests -q                              # mock 엔진으로 API 계약 점검
uv run python scripts/try_sts.py sts data/samples/q_psd501.wav # 실제 서버에 음성 질문
uv run python scripts/try_sts.py chat "PSD 502 코드가 뭐야?"
uv run python scripts/try_sts.py stt data/samples/q_door.wav
uv run python scripts/probe_llama.py --model gemma4-12b data/samples/*.wav  # 모델별 품질/속도 점검
```

응답 음성은 `outputs/` 에 저장된다.

### RAGAS 자동 평가

`scripts/ragas_eval.py` 가 서버 API 를 앱처럼 불러(`/api/ask`·`/chat` 에 `debug` 를 줘서 답변에 쓴 검색 문서도 받는다)
RAGAS 로 채점한다. 서버 `.venv` 와 섞이지 않게 uv 가 스크립트 전용 환경을 따로 만든다.

```powershell
$env:UV_CACHE_DIR = "D:/uv-cache"
uv run scripts/ragas_eval.py                          # 음성 평가셋 14문항 + 팀 QA 테스트 세트 30문항 (약 20분)
uv run scripts/ragas_eval.py --text -1 --no-voice     # 테스트 세트 253문항 전부 (약 2시간)
uv run scripts/ragas_eval.py --server http://127.0.0.1:8010 --voice-limit 2 --text 8   # 빠른 점검
```

- 테스트 세트: `data/qa_testset/4.QA 테스트 세트/통합/한국철도역무특화 LLM 평가용 테스트세트V1.2.xlsx`
  (팀 QA, 출처 문서 7종 253문항). 출처 문서별로 고르게 뽑고 출처별 점수도 낸다. `--testset` 으로 바꿀 수 있다.
- 지표: 충실도(faithfulness) · 답변 관련성 · 검색 정밀도 · 검색 재현율 · 정답성(factual correctness)
- 채점 모델: 기본은 이 PC 의 llama-server(Gemma). 자기 답을 자기가 채점해 후할 수 있으니 버전 비교용으로 쓰고,
  `--judge-url`/`--judge-model`(키는 `RAGAS_JUDGE_KEY`)로 더 큰 모델을 쓸 수 있다.
- 결과: `data/ragas/<시각>/` 의 `report.md`(요약·문항별), `results.csv`, `summary.json`
- 서버가 `debug` 를 모르는 예전 코드면 멈추고 다시 켜라고 알려준다.

## RAG

`STS_RAG_BACKEND` 로 선택. 연결 지점은 `sts_server/rag.py` 의 `Retriever.search(query, k)` 하나.

- `none` — 검색 안 함
- `jsonl` — question/answer JSONL. 현재 `.env.example` 은 앱의 `train_data_standardized.jsonl`(338건)을 가리킨다
- `chroma` — 기존 GlassAssist 서버의 KR-SBERT + Chroma DB. `STS_RAG_CHROMA_DIR` 에 폴더 지정

## 구조

```
sts_server/
  app.py                  FastAPI 엔드포인트
  pipeline.py             받아쓰기 · 분류(+위협 단어 가드) · RAG · 답변 · 음성 출력
  prompts.py              시스템 프롬프트 (분류 기준, 답변 말투)
  text.py                 문장 분리, 음성용 정리, JSON 파싱, 위협 단어 사전
  rag.py                  Retriever (none / jsonl / chroma)
  audio.py                업로드 음성 → 16kHz mono WAV
  config.py               STS_* 환경 변수 설정
  models.py               llama.cpp 모델 프리셋 (gemma4-12b q4/q8, qwen3-omni-30b)
  engines/
    llamacpp.py              llama-server 자동 기동 + OpenAI 호환 API (input_audio)
    qwen3_omni_hf.py         transformers, thinker + talker 완전 STS
    mock.py
multimodel/               STT/TTS 툴킷 (fallback TTS, whisper 백엔드로 사용)
  tts/supertonic.py       Supertonic 3 (로컬 CPU)
  tts/textnorm.py         숫자 → 한글 읽기
scripts/                  점검용 클라이언트
tests/                    API 계약 테스트
```

## 남은 일 / 한계

- **앱을 /sts 로 전환**: 지금 앱은 `/api/transcribe` → `/chat` 두 번 호출 + 기기 TTS. `/sts` 한 번으로 바꾸면 왕복이 줄고 서버 음성을 쓸 수 있다 (MainActivity, WhisperSttClient 수정 필요).
- **danger 분류**: 프롬프트 + 위협 단어 사전 수준. 운영 전 전용 데이터로 검증 필요 (앱 쪽 처리도 아직 주석 상태).
- **llama.cpp 오디오 입력은 실험 단계**라고 llama.cpp 가 경고한다. 짧은 질문 위주로는 정확했지만 긴 발화·잡음 환경은 추가 검증 필요.
- `qwen3_omni_hf` 엔진은 80GB GPU 가 없어 이 PC 에서 실행 검증하지 못했다 (mock 으로 이벤트 흐름만 검증).
