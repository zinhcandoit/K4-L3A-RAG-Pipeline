"""
Task 6 — Lexical search bằng BM25.

Dùng cùng corpus chunks với Task 5. BM25 phù hợp với từ khóa chính xác, mã tài
liệu và tên riêng. Output phải theo SearchResult và sort score giảm dần.
"""

import re

CORPUS: list[dict] = []

_INDEX_CACHE: dict = {}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def load_corpus() -> list[dict]:
    """Nạp CORPUS từ chunks của Task 4 nếu chưa có."""
    if not CORPUS:
        from .task4_chunking_indexing import chunk_documents, load_documents

        CORPUS.extend(chunk_documents(load_documents()))
    return CORPUS


def build_bm25_index(corpus: list[dict]):
    """Tạo BM25 index từ cùng corpus chunks của Task 4."""
    from rank_bm25 import BM25Okapi

    return BM25Okapi([_tokenize(item["content"]) for item in corpus])


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về BM25 SearchResult theo score giảm dần."""
    import numpy as np

    corpus = load_corpus()
    if not corpus:
        return []
    key = (id(corpus), len(corpus))
    if key not in _INDEX_CACHE:
        _INDEX_CACHE.clear()
        _INDEX_CACHE[key] = build_bm25_index(corpus)
    query_tokens = _tokenize(query)
    scores = _INDEX_CACHE[key].get_scores(query_tokens)
    results = []
    for index in np.argsort(scores)[::-1][:top_k]:
        item = corpus[index]
        # BM25Okapi cho IDF = 0 với corpus nhỏ, nên lọc theo token khớp thay vì score > 0.
        if not set(query_tokens) & set(_tokenize(item["content"])):
            continue
        results.append({
            "id": item["id"],
            "content": item["content"],
            "score": float(scores[index]),
            "metadata": item["metadata"],
            "retrieval_method": "bm25",
        })
    return results


if __name__ == "__main__":
    for result in lexical_search("test query", top_k=3):
        print(result)
