"""
Task 6 — Lexical search bằng BM25.

Dùng cùng corpus chunks với Task 5. BM25 phù hợp với từ khóa chính xác, mã tài
liệu và tên riêng — ví dụ "68/2026/NĐ-CP" hay "thuế khoán" — là chỗ dense
retrieval hay trượt. Output phải theo SearchResult và sort score giảm dần.
"""

import re


CORPUS: list[dict] = []

# Giữ chữ số và ký tự tiếng Việt; tách theo ranh giới không phải chữ/số.
TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)

_bm25_cache: dict[tuple[int, int], object] = {}


def tokenize(text: str) -> list[str]:
    """Tách token thống nhất cho cả corpus lẫn query."""
    return TOKEN_PATTERN.findall(text.lower())


def ensure_corpus() -> list[dict]:
    """Nạp CORPUS từ cùng nguồn chunks của Task 4 nếu chưa có."""
    global CORPUS
    if not CORPUS:
        from .task4_chunking_indexing import chunk_documents, load_documents

        CORPUS = chunk_documents(load_documents())
    return CORPUS


def build_bm25_index(corpus: list[dict]):
    """Tạo BM25 index từ cùng corpus chunks của Task 4."""
    from rank_bm25 import BM25Okapi

    if not corpus:
        raise ValueError("Corpus rỗng; chạy task4 hoặc kiểm tra data/standardized/")
    return BM25Okapi([tokenize(item["content"]) for item in corpus])


def _cached_index(corpus: list[dict]):
    """Cache theo (id, độ dài) để không build lại index mỗi lần gọi."""
    key = (id(corpus), len(corpus))
    index = _bm25_cache.get(key)
    if index is None:
        index = build_bm25_index(corpus)
        _bm25_cache[key] = index
    return index


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về BM25 SearchResult theo score giảm dần."""
    if not query or not query.strip() or top_k <= 0:
        return []

    corpus = CORPUS if CORPUS else ensure_corpus()
    tokens = tokenize(query)
    if not tokens:
        return []

    scores = _cached_index(corpus).get_scores(tokens)
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)

    # Corpus rất nhỏ làm IDF của BM25Okapi bằng 0 (log(1.5) - log(1.5)) nên mọi
    # score đều bằng 0. Bỏ hết thì mất luôn thứ hạng, vì vậy chỉ lọc score dương
    # khi thực sự có; nếu không có cái nào dương thì giữ nguyên thứ hạng thô.
    positives = [index for index in order if scores[index] > 0]
    candidates = positives if positives else order

    results = []
    seen: set[str] = set()
    for index in candidates:
        item = corpus[index]
        if item["id"] in seen:
            continue
        seen.add(item["id"])
        results.append(
            {
                "id": item["id"],
                "content": item["content"],
                "score": float(scores[index]),
                "metadata": item["metadata"],
                "retrieval_method": "bm25",
            }
        )
        if len(results) == top_k:
            break

    return results


if __name__ == "__main__":
    for result in lexical_search("thuế khoán hộ kinh doanh", top_k=3):
        print(f"{result['score']:.4f}  {result['id']}")
        print(f"   {result['content'][:120]}...")
