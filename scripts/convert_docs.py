# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#     "docling>=2.0",
#     "torch>=2.7",
#     "torchvision",
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
# torchvision = { index = "pytorch-cu128" }
# ///
"""RAG 원본 문서(PDF) → 마크다운. 표 구조를 살린다 (Docling). 서버 .venv 와 섞이지 않게 uv 가 따로 환경을 만든다.

  $env:UV_CACHE_DIR = "D:/uv-cache"; uv run scripts/convert_docs.py <PDF 파일 또는 폴더> ...
  → data/rag_source/docling/<파일 이름>.md

- 한글(hwp·hwpx) 원본은 배포용 암호화라 직접 못 읽는다 — 한글에서 PDF 로 저장해 data/rag_source/원본_pdf 에 두고 돌린다.
- OCR 은 끈다 (원본이 글자 PDF 이고, EasyOCR 은 모델을 C 드라이브 사용자 폴더에 받는다).
- 모델·캐시·임시 파일은 finetune/d_drive.py 로 전부 D 드라이브.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "finetune"))
import d_drive  # noqa: E402,F401 - 캐시·임시 파일을 D 드라이브로 (docling 을 불러오기 전에)

import argparse  # noqa: E402
import time  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
OUT = ROOT / "data" / "rag_source" / "docling"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()

    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
    opts.table_structure_options.do_cell_matching = True
    opts.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CPU if a.cpu else AcceleratorDevice.CUDA)
    # 원본 PDF 는 권한 제한(/Encrypt, 열기 암호 없음)이 걸려 있어 기본 docling-parse 가 못 연다 → pdfium 으로 읽는다
    conv = DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=opts, backend=PyPdfiumDocumentBackend)})

    files = [f for p in a.inputs for f in (sorted(p.glob("*.pdf")) if p.is_dir() else [p])]
    a.out.mkdir(parents=True, exist_ok=True)
    for f in files:
        t0 = time.perf_counter()
        doc = conv.convert(f).document
        md = doc.export_to_markdown()
        dst = a.out / f"{f.stem}.md"
        dst.write_text(md, encoding="utf-8")
        print(f"{f.name}: {len(doc.pages)}쪽 · 표 {len(doc.tables)}개 · {len(md):,}자 · "
              f"{time.perf_counter() - t0:.0f}초 → {dst}", flush=True)


if __name__ == "__main__":
    main()
