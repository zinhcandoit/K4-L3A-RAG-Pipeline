"""
Task 9 — Retrieval pipeline hoàn chỉnh.

Luồng xử lý:
    1. Chạy semantic_search và lexical_search.
    2. Fuse hai danh sách bằng RRF đúng một lần.
    3. Lấy best cosine score gốc từ dense results.
    4. Nếu score dưới threshold, thử PageIndex fallback.
    5. Nếu fallback lỗi, trả hybrid results thay vì crash.

Không so sánh threshold với RRF score vì hai thang đo khác nhau: RRF chỉ là
tổng nghịch đảo thứ hạng (~0.03 cho hạng 1), còn threshold hiệu chỉnh trên
cosine similarity (0..1). Lẫn hai cái là luôn rơi fallback.
"""

import os

from dotenv import load_dotenv

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


load_dotenv()

SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD") or 0.3)
DEFAULT_TOP_K = 5


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """Trả về hybrid hoặc pageindex SearchResult."""
    dense = semantic_search(query, top_k=top_k * 2)

    if use_reranking:
        sparse = lexical_search(query, top_k=top_k * 2)
        # RRF chạy đúng một lần trong toàn pipeline
        hybrid = rerank_rrf([dense, sparse], top_k=top_k)
    else:
        # Dense-only: không gọi BM25, nếu không baseline A/B phải trả chi phí
        # lexical mà nó không hề dùng, làm số latency so sánh vô nghĩa.
        hybrid = dense[:top_k]

    # Quyết định fallback dựa trên cosine score gốc, không phải RRF score
    best_dense_score = dense[0]["score"] if dense else 0.0

    if best_dense_score < score_threshold:
        try:
            fallback = pageindex_search(query, top_k=top_k)
            if fallback:
                return fallback[:top_k]
        except Exception as error:
            # Provider ngoài lỗi thì vẫn trả hybrid; giao diện không được crash
            print(f"[retrieve] PageIndex fallback lỗi, dùng hybrid: {error}")

    return hybrid[:top_k]


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "Hộ kinh doanh nộp thuế theo phương pháp nào từ 2026?"
    print(f"Query: {question}\n")
    for item in retrieve(question, top_k=5):
        print(f"[{item['retrieval_method']}] {item['score']:.4f}  {item['metadata']['source']}")
        print(f"   {item['content'][:140]}...\n")
