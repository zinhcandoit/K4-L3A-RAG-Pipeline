"""
Task 7 — Reciprocal Rank Fusion & Reranking.

RRF gộp nhiều bảng xếp hạng mà không cộng trực tiếp cosine score với BM25
score. Công thức: RRF(d) = sum(1 / (k + rank)), rank bắt đầu từ 1.

Lưu ý: RRF score chỉ phản ánh thứ hạng, không dùng để quyết định fallback.

Đồng thời hỗ trợ reranker tiên tiến BAAI/bge-reranker-v2-m3 qua hàm `rerank_bge`.
"""

import os
from dotenv import load_dotenv

load_dotenv()

_bge_reranker = None
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")


def get_bge_reranker():
    """Khởi tạo lazy mô hình CrossEncoder BAAI/bge-reranker-v2-m3."""
    global _bge_reranker
    if _bge_reranker is None:
        from sentence_transformers import CrossEncoder
        _bge_reranker = CrossEncoder(RERANKER_MODEL)
    return _bge_reranker


def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
) -> list[dict]:
    """Fuse nhiều ranked lists bằng Reciprocal Rank Fusion và trả hybrid SearchResult."""
    if top_k <= 0 or not ranked_lists:
        return []

    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            item_id = item["id"]
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
            if item_id not in items:
                items[item_id] = item

    ranked_ids = sorted(scores.keys(), key=lambda item_id: scores[item_id], reverse=True)
    results = []
    for item_id in ranked_ids[:top_k]:
        result = items[item_id].copy()
        result["score"] = float(scores[item_id])
        result["retrieval_method"] = "hybrid"
        results.append(result)

    return results


def rerank_bge(query: str, items: list[dict], top_k: int = 5) -> list[dict]:
    """Rerank các chunk ứng viên bằng mô hình BAAI/bge-reranker-v2-m3."""
    if not items or top_k <= 0 or not query.strip():
        return []

    # Loại bỏ duplicate item id trước khi rerank
    unique_items = []
    seen_ids = set()
    for item in items:
        if item["id"] not in seen_ids:
            seen_ids.add(item["id"])
            unique_items.append(item)

    model = get_bge_reranker()
    pairs = [[query, item["content"]] for item in unique_items]
    raw_scores = model.predict(pairs)

    scored_results = []
    for item, score in zip(unique_items, raw_scores):
        res = item.copy()
        res["score"] = float(score)
        res["retrieval_method"] = "hybrid"
        scored_results.append(res)

    scored_results.sort(key=lambda x: x["score"], reverse=True)
    return scored_results[:top_k]


if __name__ == "__main__":
    print("Task 7 reranking module ready.")
