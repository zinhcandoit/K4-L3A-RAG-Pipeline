"""
Task 6 — Lexical search bằng BM25.

Dùng cùng corpus chunks với Task 5. BM25 phù hợp với từ khóa chính xác, mã tài
liệu và tên riêng. Output phải theo SearchResult và sort score giảm dần.
"""


CORPUS: list[dict] = []
_CACHED_BM25 = None
_CACHED_CORPUS_LEN = 0


def build_bm25_index(corpus: list[dict]):
    """Tạo BM25 index từ cùng corpus chunks của Task 4."""
    import math
    from rank_bm25 import BM25Okapi

    class BM25Lucene(BM25Okapi):
        def _calc_idf(self, nd):
            for word, freq in nd.items():
                self.idf[word] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    tokenized = [item["content"].lower().split() for item in corpus]
    return BM25Lucene(tokenized)


def _ensure_corpus() -> list[dict]:
    global CORPUS
    if not CORPUS:
        from .task4_chunking_indexing import load_documents, chunk_documents
        CORPUS = chunk_documents(load_documents())
    return CORPUS


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về BM25 SearchResult theo score giảm dần."""
    global _CACHED_BM25, _CACHED_CORPUS_LEN

    if not query.strip() or top_k <= 0:
        return []

    corpus = _ensure_corpus()
    if not corpus:
        return []

    # Cache BM25 index nếu corpus không đổi
    if _CACHED_BM25 is None or _CACHED_CORPUS_LEN != len(corpus):
        _CACHED_BM25 = build_bm25_index(corpus)
        _CACHED_CORPUS_LEN = len(corpus)

    tokenized_query = query.lower().split()
    if not tokenized_query:
        return []

    scores = _CACHED_BM25.get_scores(tokenized_query)
    import numpy as np
    indices = np.argsort(scores)[::-1]

    results: list[dict] = []
    seen_ids: set[str] = set()

    for idx in indices:
        if len(results) >= top_k:
            break
        score = float(scores[idx])
        if score <= 0.0:
            continue
        item = corpus[idx]
        item_id = item["id"]
        if item_id in seen_ids:
            continue
        seen_ids.add(item_id)

        meta = dict(item["metadata"])
        if meta.get("url") == "":
            meta["url"] = None
        if "chunk_index" in meta:
            meta["chunk_index"] = int(meta["chunk_index"])

        results.append({
            "id": item_id,
            "content": item["content"],
            "score": score,
            "metadata": meta,
            "retrieval_method": "bm25",
        })

    return results


if __name__ == "__main__":
    for res in lexical_search("hộ kinh doanh", top_k=3):
        print(res)
