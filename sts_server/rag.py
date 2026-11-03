"""RAG 검색기. STS_RAG_BACKEND 로 선택한다.

- none   : 검색 안 함 (기본)
- jsonl  : JSONL 파일을 글자 바이그램 TF-IDF 로 검색 (추가 의존성 없음)
- hybrid : jsonl 검색 + KR-SBERT 임베딩 검색을 섞는다 — 말을 바꿔 물어도 찾고, 장애 코드 번호는 정확히 맞춘다
- chroma : 기존 GlassAssist 서버의 KR-SBERT + Chroma DB 를 그대로 연결 (`uv sync --extra rag` 필요)

jsonl·hybrid 는 STS_RAG_JSONL 에 파일 여러 개를 ; 로 이어 줄 수 있다. 한 줄 형식:
  {"question", "answer"}                     질문/답 쌍 (앱 학습 데이터)
  {"question" == "answer", "source"}         문서 청크 (RAG_자료/rag_chunks.jsonl)
  {"text" 또는 "content"}                     그냥 글
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections import Counter
from pathlib import Path

from sts_server.config import Settings

log = logging.getLogger(__name__)

NUMBER_WEIGHT = 3      # 숫자 토큰 하나를 바이그램 몇 개만큼 칠지
LEXICAL_WEIGHT = 0.5   # hybrid: 임베딩 코사인 점수에 더할 바이그램 점수 비중
# hybrid: 이보다 낮으면 관련 없는 문서로 보고 버린다. 실측: 실제 질문의 1등 0.53 이상, 잡음·잘못 받아쓴 질문 0.49 이하
MIN_HYBRID_SCORE = 0.5   # 기본값 — 실제로는 Settings.rag_min_score(STS_RAG_MIN_SCORE)를 쓴다
# KR-SBERT 는 128 토큰으로 학습됐지만 문서 청크(~450자)가 잘리지 않게 늘린다 — 실측상 청크 검색이 조금 낫다
EMBED_MAX_TOKENS = 256
# 리랭커는 질문+문서를 한 번에 읽는다 (bge-reranker-v2-m3 는 8192 까지 되지만 청크는 ~1000자라 512 면 충분)
RERANK_MAX_TOKENS = 512

_CITE = re.compile(r"\s*\[cite:\s*[\d,\s]+\]")
_HANGUL = re.compile(r"[가-힣]")

# 음성 질문은 문서와 다르게 부르는 경우가 많다
_SYNONYMS = {"스크린도어": "PSD", "스크린 도어": "PSD", "안전문": "PSD", "엘베": "엘리베이터"}
_KDIGIT = {"일": 1, "이": 2, "삼": 3, "사": 4, "오": 5, "육": 6, "칠": 7, "팔": 8, "구": 9}
_KNUM = re.compile(r"([이삼사오육칠팔구])백(?:([일이삼사오육칠팔구])?(십))?([일이삼사오육칠팔구])?")


def _knum(m: re.Match) -> str:
    tens = (_KDIGIT[m[2]] if m[2] else 1) if m[3] else 0
    return str(_KDIGIT[m[1]] * 100 + tens * 10 + (_KDIGIT[m[4]] if m[4] else 0))


def normalize_query(query: str) -> str:
    """'스크린도어 오백일 코드' → 'PSD 501 코드' (장애 코드는 세 자리라 '백'이 있는 한글 숫자만 바꾼다)"""
    for word, canon in _SYNONYMS.items():
        query = query.replace(word, canon)
    return _KNUM.sub(_knum, query)


def _source_label(source: str) -> str:
    """'[대용량]학습용_코레일 고객 응대 가이드' → '코레일 고객 응대 가이드'"""
    return re.sub(r"^\[[^\]]*\]|학습용_", "", source).strip()


def load_docs(paths: str) -> list[str]:
    """; 로 이은 JSONL 파일들을 검색 문서 목록으로."""
    docs: list[str] = []
    for path in filter(None, (p.strip() for p in paths.split(";"))):
        skipped = 0
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if "question" in row and row["question"] == row.get("answer"):
                text = row["question"].strip()
                # 표가 깨져 숫자만 남은 청크("#848 00:11 13 32 보통 ...")는 답변에 도움이 안 된다
                if len(_HANGUL.findall(text)) < 0.3 * len("".join(text.split())):
                    skipped += 1
                    continue
                source = _source_label(row.get("source", ""))
                docs.append(f"({source})\n{text}" if source else text)
            elif "question" in row:
                docs.append(f"Q: {row['question']}\nA: {_CITE.sub('', row.get('answer', ''))}")
            else:
                docs.append(str(row.get("text") or row.get("content") or row))
        log.info("RAG 문서 읽음: %s (건너뜀 %d)", path, skipped)
    return docs


class Retriever:
    name = "none"

    def search(self, query: str, k: int = 3) -> list[str]:
        return []


class JsonlRetriever(Retriever):
    name = "jsonl"

    def __init__(self, paths: str) -> None:
        self.docs = load_docs(paths)
        tfs = [self._grams(d) for d in self.docs]
        df = Counter(g for tf in tfs for g in tf)
        n = len(self.docs)
        self.idf = {g: math.log((n + 1) / (c + 1)) + 1 for g, c in df.items()}
        self.vecs = [self._weigh(tf) for tf in tfs]
        log.info("%s RAG: 문서 %d건", self.name, n)

    @staticmethod
    def _grams(text: str) -> Counter:
        s = "".join(text.lower().split())
        grams = Counter(s[i:i + 2] for i in range(len(s) - 1))
        # 장애 코드·승강장 번호는 통째로도 맞춰 본다 — 바이그램만으로는 505 와 525·518 이 비슷하게 보인다
        for num in re.findall(r"\d{2,}", text):
            grams[f"#{num}"] += NUMBER_WEIGHT
        return grams

    def _weigh(self, tf: Counter) -> dict[str, float]:
        v = {g: c * self.idf.get(g, 1.0) for g, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {g: x / norm for g, x in v.items()}

    def lexical_scores(self, query: str) -> list[float]:
        q = self._weigh(self._grams(query))
        return [sum(w * d.get(g, 0.0) for g, w in q.items()) for d in self.vecs]

    def search(self, query: str, k: int = 3) -> list[str]:
        scored = sorted(zip(self.lexical_scores(normalize_query(query)), range(len(self.docs))), reverse=True)
        return [self.docs[i] for score, i in scored[:k] if score > 0.05]


class HybridRetriever(JsonlRetriever):
    name = "hybrid"

    def __init__(self, paths: str, settings: Settings) -> None:
        super().__init__(paths)
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(settings.rag_embedding_model)
        self.model = AutoModel.from_pretrained(settings.rag_embedding_model).to(self.device).eval()
        self.doc_emb = self._cached_doc_embeddings(settings)
        self.min_score = settings.rag_min_score
        log.info("hybrid RAG: 임베딩 %s (%s)", settings.rag_embedding_model, self.device)
        self.reranker = None
        self.rerank_n = settings.rag_rerank_candidates
        if settings.rag_reranker:
            from transformers import AutoModelForSequenceClassification

            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.rerank_tokenizer = AutoTokenizer.from_pretrained(settings.rag_reranker)
            self.reranker = AutoModelForSequenceClassification.from_pretrained(
                settings.rag_reranker, dtype=dtype).to(self.device).eval()
            log.info("hybrid RAG: 리랭커 %s (후보 %d개)", settings.rag_reranker, self.rerank_n)

    def _embed(self, texts: list[str], batch: int = 32):
        """KR-SBERT 문장 임베딩 (mean pooling, 정규화)."""
        torch = self._torch
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), batch):
                enc = self.tokenizer(texts[i:i + batch], padding=True, truncation=True, max_length=EMBED_MAX_TOKENS,
                                     return_tensors="pt").to(self.device)
                hidden = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                out.append(torch.nn.functional.normalize(emb, dim=-1).float().cpu())
        return torch.cat(out)

    def _cached_doc_embeddings(self, settings: Settings):
        """문서 임베딩은 문서·모델이 같으면 data/rag_cache 에 저장해 둔 걸 쓴다."""
        torch = self._torch
        key = hashlib.sha1("\n\0".join([settings.rag_embedding_model, str(EMBED_MAX_TOKENS), *self.docs])
                           .encode()).hexdigest()[:16]
        path = Path(settings.data_dir) / "rag_cache" / f"{key}.pt"
        if path.exists():
            return torch.load(path)
        emb = self._embed(self.docs)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(emb, path)
        return emb

    def scored(self, query: str) -> list[tuple[float, int]]:
        """(점수, 문서 번호) 높은 순. 디버깅·튜닝용으로도 쓴다."""
        query = normalize_query(query)
        cos =(self.doc_emb @ self._embed([query])[0]).tolist()
        lex = self.lexical_scores(query)
        return sorted(((c + LEXICAL_WEIGHT * l, i) for i, (c, l) in enumerate(zip(cos, lex))), reverse=True)

    def rerank_scores(self, query: str, idx: list[int]) -> list[float]:
        """리랭커 점수 (0~1, 질문과 문서가 맞을수록 높다)."""
        torch = self._torch
        with torch.inference_mode():
            enc = self.rerank_tokenizer([[query, self.docs[i]] for i in idx], padding=True, truncation=True,
                                        max_length=RERANK_MAX_TOKENS, return_tensors="pt").to(self.device)
            return torch.sigmoid(self.reranker(**enc).logits.view(-1).float()).tolist()

    def ranked(self, query: str) -> list[tuple[float, int]]:
        """scored 와 같되, 리랭커가 있으면 문턱을 넘은 상위 후보를 리랭커 순서로 앞에 둔다 (점수는 hybrid 점수 그대로)."""
        scored = self.scored(query)
        if self.reranker is None:
            return scored
        n = sum(1 for score, _ in scored[:self.rerank_n] if score >= self.min_score)
        if n < 2:
            return scored
        head = scored[:n]
        rr = self.rerank_scores(normalize_query(query), [i for _, i in head])
        return [h for _, h in sorted(zip(rr, head), key=lambda t: -t[0])] + scored[n:]

    def search(self, query: str, k: int = 3) -> list[str]:
        return [self.docs[i] for score, i in self.ranked(query)[:k] if score >= self.min_score]


class ChromaRetriever(Retriever):
    name = "chroma"

    def __init__(self, settings: Settings) -> None:
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings

        emb = HuggingFaceEmbeddings(model_name=settings.rag_embedding_model)
        self.db = Chroma(persist_directory=settings.rag_chroma_dir, embedding_function=emb,
                         collection_name=settings.rag_chroma_collection)
        log.info("Chroma RAG: %s (collection=%s)", settings.rag_chroma_dir, settings.rag_chroma_collection)

    def search(self, query: str, k: int = 3) -> list[str]:
        return [d.page_content for d in self.db.similarity_search(query, k=k)]


def create_retriever(settings: Settings) -> Retriever:
    backend = settings.rag_backend.lower()
    if backend == "none":
        return Retriever()
    if backend in ("jsonl", "hybrid"):
        if not settings.rag_jsonl:
            raise ValueError("STS_RAG_JSONL 경로를 지정하세요.")
        return JsonlRetriever(settings.rag_jsonl) if backend == "jsonl" else HybridRetriever(settings.rag_jsonl, settings)
    if backend == "chroma":
        if not settings.rag_chroma_dir:
            raise ValueError("STS_RAG_CHROMA_DIR 경로를 지정하세요.")
        return ChromaRetriever(settings)
    raise ValueError(f"알 수 없는 RAG 백엔드: {settings.rag_backend}")
