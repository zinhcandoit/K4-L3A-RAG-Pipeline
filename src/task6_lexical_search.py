"""
Task 6 — Lexical search bằng BM25.

Dùng cùng corpus chunks với Task 5. BM25 phù hợp với từ khóa chính xác, mã tài
liệu và tên riêng. Output phải theo SearchResult và sort score giảm dần.
Hỗ trợ cache nhị phân (.pkl) ra đĩa để khởi động tức thì.
"""

import math
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi


CORPUS: list[dict] = []
_CACHED_BM25 = None
_CACHED_CORPUS_ID = None
_CACHED_CORPUS_LEN = 0

CACHE_PATH = Path(__file__).parent.parent / "data" / "bm25_cache.pkl"


class BM25OkapiWithFloor(BM25Okapi):
    """Kế thừa BM25Okapi với Lucene-style non-negative IDF floor để hoạt động chính xác trên corpus nhỏ."""

    def _calc_idf(self, nd):
        super()._calc_idf(nd)
        for word in self.idf:
            if self.idf[word] <= 0:
                # Chuẩn Lucene: log((N - n + 0.5) / (n + 0.5) + 1.0)
                self.idf[word] = math.log((self.corpus_size - nd[word] + 0.5) / (nd[word] + 0.5) + 1.0)


def _ensure_corpus() -> None:
    """Tự động tải CORPUS và BM25 từ file nhị phân .pkl nếu đã có, ngược lại mới parse và lưu cache."""
    global CORPUS, _CACHED_BM25, _CACHED_CORPUS_ID, _CACHED_CORPUS_LEN
    if CORPUS:
        return

    # 1. Thử nạp tức thì từ file nhị phân .pkl đã lưu
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "rb") as f:
                saved_corpus, saved_bm25 = pickle.load(f)
                if saved_corpus and saved_bm25:
                    CORPUS = saved_corpus
                    _CACHED_BM25 = saved_bm25
                    _CACHED_CORPUS_ID = id(CORPUS)
                    _CACHED_CORPUS_LEN = len(CORPUS)
                    return
        except Exception:
            pass

    # 2. Nếu chưa có file cache, đọc từ standardized documents và chia chunk
    from .task4_chunking_indexing import chunk_documents, load_documents
    docs = load_documents()
    CORPUS = chunk_documents(docs)
    _CACHED_BM25 = build_bm25_index(CORPUS)
    _CACHED_CORPUS_ID = id(CORPUS)
    _CACHED_CORPUS_LEN = len(CORPUS)

    # 3. Lưu ra đĩa (.pkl) để các lần chạy sau đọc lên trong 0.01 giây
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump((CORPUS, _CACHED_BM25), f)
    except Exception:
        pass


def build_bm25_index(corpus: list[dict]) -> BM25Okapi:
    """Tạo BM25 index từ danh sách chunks."""
    tokenized = [item["content"].lower().split() for item in corpus]
    return BM25OkapiWithFloor(tokenized)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về BM25 SearchResult theo score giảm dần với cache index nhị phân."""
    global _CACHED_BM25, _CACHED_CORPUS_ID, _CACHED_CORPUS_LEN
    _ensure_corpus()
    if not CORPUS or top_k <= 0 or not query.strip():
        return []

    # Tự phát hiện nếu CORPUS bị thay thế (ví dụ monkeypatch trong unit test)
    if _CACHED_BM25 is None or len(CORPUS) != _CACHED_CORPUS_LEN or id(CORPUS) != _CACHED_CORPUS_ID:
        _CACHED_BM25 = build_bm25_index(CORPUS)
        _CACHED_CORPUS_ID = id(CORPUS)
        _CACHED_CORPUS_LEN = len(CORPUS)

    bm25 = _CACHED_BM25
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    results = []
    for index in ranked_indices:
        item = CORPUS[index]
        results.append({
            "id": item["id"],
            "content": item["content"],
            "score": float(scores[index]),
            "metadata": item["metadata"],
            "retrieval_method": "bm25",
        })
        if len(results) >= top_k:
            break

    return results


if __name__ == "__main__":
    for result in lexical_search("test query", top_k=3):
        print(result)
