# 공식 BF16 GGUF(ggml-org) → MXFP4(FP4) 변환. MoE expert 는 MXFP4, 나머지 레이어는 llama.cpp 기본 규칙대로.
#   powershell -File scripts/quantize_qwen_fp4.ps1
# 끝나면 models/qwen3-omni-gguf/Qwen3-Omni-30B-A3B-Instruct-MXFP4_MOE-own.gguf 가 생긴다 (STS_QUANT=fp4-own).
# 원본 BF16(56.9GB)은 변환 후 지워도 된다.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$dir = Join-Path $root "models\qwen3-omni-gguf"
$src = Join-Path $dir "Qwen3-Omni-30B-A3B-Instruct-bf16.gguf"
$dst = Join-Path $dir "Qwen3-Omni-30B-A3B-Instruct-MXFP4_MOE-own.gguf"
$exe = Join-Path $root "runtime\llama.cpp\llama-quantize.exe"

if (-not (Test-Path $src)) { throw "원본이 없습니다: $src" }
$threads = [Environment]::ProcessorCount
$t0 = Get-Date
& $exe $src $dst MXFP4_MOE $threads
if ($LASTEXITCODE -ne 0) { throw "llama-quantize 실패 (code $LASTEXITCODE)" }
"완료: {0} ({1:N1} GB, {2:N0}초)" -f $dst, ((Get-Item $dst).Length / 1GB), ((Get-Date) - $t0).TotalSeconds
