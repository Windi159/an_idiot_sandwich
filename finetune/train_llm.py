# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#     "torch>=2.7",
#     "transformers>=5.20",
#     "accelerate>=1.10",
#     "peft>=0.17",
#     "bitsandbytes>=0.46",
# ]
#
# [tool.uv]
# cache-dir = "D:/uv-cache"
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# ///
"""B. Gemma 4 12B QLoRA 미세 조정 — 서버 .venv 와 섞이지 않게 uv 가 이 스크립트 전용 환경을 만든다.

  $env:UV_CACHE_DIR = "D:/uv-cache"; uv run finetune/train_llm.py                 # → finetune/models/gemma4-12b-lora
  uv run finetune/train_llm.py --epochs 1 --limit 20                            # 빠른 점검

- 원본: unsloth/gemma-4-12b-it (Apache-2.0, models/hf-cache 에 받아 둔 것). 4bit(NF4)로 올리고 언어 모델 쪽
  q/k/v/o·gate/up/down 에만 LoRA 를 단다 (음성·이미지 인코더는 그대로 — 서버의 mmproj 를 계속 쓸 수 있다).
- 데이터: finetune/data/llm_sft.jsonl (gen_llm_data.py). 답변 토큰에만 손실을 준다.
- 서버 STS 를 끄고 돌린다 (16GB 를 혼자 쓴다). 캐시·임시 파일은 d_drive 로 전부 D 드라이브.
"""

from __future__ import annotations

import d_drive  # noqa: F401 - 캐시·임시 파일을 D 드라이브로 (맨 먼저)
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
BASE = "unsloth/gemma-4-12b-it"


END_OF_TURN = "<turn|>\n"


def encode(tok, messages: list[dict], max_len: int):
    """프롬프트 부분은 -100 으로 가리고 답(+턴 끝 토큰)에만 손실.

    Gemma 4 는 생각 모드를 끄고 답을 만들 때도 프롬프트 끝에 빈 생각 칸(<|channel>thought\\n<channel|>)을 붙이고,
    지난 대화를 그릴 때는 그 칸을 뺀다. 그래서 "대화 전체"를 그린 것과 프롬프트가 어긋난다 —
    서버(llama-server, enable_thinking=False)가 실제로 보는 모양 그대로 "프롬프트 + 답 + 턴 끝"을 이어 붙인다.
    """
    prompt = tok.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=True,
                                     enable_thinking=False, return_dict=False)
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    answer = tok(messages[-1]["content"] + END_OF_TURN, add_special_tokens=False)["input_ids"]
    full = list(prompt) + answer
    if len(full) > max_len:
        return None
    return full, [-100] * len(prompt) + answer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=HERE / "data" / "llm_sft.jsonl")
    ap.add_argument("--out", type=Path, default=HERE / "models" / "gemma4-12b-lora")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=8, help="기울기 누적 (배치 1 × accum)")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--limit", type=int, default=0, help="점검용으로 앞 N개만")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    rows = [json.loads(l) for l in a.data.read_text(encoding="utf-8").splitlines() if l.strip()]
    if a.limit:
        rows = rows[:a.limit]
    tok = AutoTokenizer.from_pretrained(BASE)
    data = [e for e in (encode(tok, r["messages"], a.max_len) for r in rows) if e]
    random.shuffle(data)
    n_val = max(1, len(data) // 20)
    val, train = data[:n_val], data[n_val:]
    print(f"예시 {len(rows)} → 사용 {len(data)} (학습 {len(train)} · 검증 {len(val)}) · "
          f"평균 길이 {sum(len(x) for x, _ in data) / len(data):.0f} 토큰")

    t0 = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        BASE, device_map={"": 0}, dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_use_double_quant=True,
                                               bnb_4bit_compute_dtype=torch.bfloat16))
    print(f"모델 로드 {time.perf_counter() - t0:.0f}초 · VRAM {torch.cuda.memory_allocated() / 2**30:.1f}GB")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"))
    model.print_trainable_parameters()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    total = a.epochs * math.ceil(len(train) / a.accum)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / max(1, 0.05 * total)) * 0.5 * (1 + math.cos(math.pi * min(i, total) / total)))

    def loss_of(ids, labels):
        x = torch.tensor([ids], device="cuda")
        y = torch.tensor([labels], device="cuda")
        return model(input_ids=x, attention_mask=torch.ones_like(x), labels=y).loss

    @torch.no_grad()
    def val_loss():
        model.eval()
        v = sum(loss_of(*e).item() for e in val) / len(val)
        model.train()
        return v

    print(f"학습 전 검증 손실 {val_loss():.3f} · 총 {total} 스텝")
    model.train()
    best, step = float("inf"), 0
    for ep in range(1, a.epochs + 1):
        random.shuffle(train)
        t0, run = time.perf_counter(), []
        for i, (ids, labels) in enumerate(train, 1):
            loss = loss_of(ids, labels) / a.accum
            loss.backward()
            run.append(loss.item() * a.accum)
            if i % a.accum == 0 or i == len(train):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    el = time.perf_counter() - t0
                    print(f"  ep{ep} 스텝 {step}/{total} 손실 {sum(run[-80:]) / len(run[-80:]):.3f} · "
                          f"{el / i:.1f}초/예시 · 남은 이번 epoch {el / i * (len(train) - i) / 60:.0f}분 · "
                          f"VRAM 최대 {torch.cuda.max_memory_allocated() / 2**30:.1f}GB", flush=True)
        v = val_loss()
        print(f"epoch {ep}: 학습 손실 {sum(run) / len(run):.3f} · 검증 손실 {v:.3f} · {(time.perf_counter() - t0) / 60:.0f}분")
        if v < best:
            best = v
            model.save_pretrained(a.out)
            tok.save_pretrained(a.out)
            (a.out / "train_info.json").write_text(json.dumps(
                {"base": BASE, "epoch": ep, "val_loss": v, "train": len(train), "val": len(val),
                 "args": {k: str(x) for k, x in vars(a).items()}}, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"  → 저장 {a.out}")
    print("끝")


if __name__ == "__main__":
    main()
