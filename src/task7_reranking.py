"""
Task 7 — Reciprocal Rank Fusion & Jina Reranker.

RRF gộp nhiều bảng xếp hạng mà không cộng trực tiếp cosine score với BM25
score. Công thức: RRF(d) = sum(1 / (k + rank)), rank bắt đầu từ 1.

Lưu ý: RRF score chỉ phản ánh thứ hạng, không dùng để quyết định fallback.

Đồng thời hỗ trợ Jina AI Reranker API (jina-reranker-v2-base-multilingual)
siêu nhanh, hỗ trợ tiếng Việt cực mạnh qua Cloud API.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY", "")
JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
JINA_MODEL = "jina-reranker-v2-base-multilingual"


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


def rerank_jina(query: str, items: list[dict], top_k: int = 5) -> list[dict]:
    """Rerank các chunk ứng viên bằng Jina Reranker Cloud API (nhanh, nhẹ, chính xác)."""
    if not items or top_k <= 0 or not query.strip():
        return []

    # Loại bỏ duplicate item id trước khi gửi API
    unique_items = []
    seen_ids = set()
    for item in items:
        if item["id"] not in seen_ids:
            seen_ids.add(item["id"])
            unique_items.append(item)

    api_key = os.getenv("JINA_API_KEY", JINA_API_KEY)
    if not api_key:
        print("[Warning] JINA_API_KEY chưa được cấu hình. Fallback về RRF/thứ hạng gốc.")
        return unique_items[:top_k]

    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "model": JINA_MODEL,
            "query": query,
            "documents": [item["content"] for item in unique_items],
            "top_n": min(top_k, len(unique_items)),
        }
        response = requests.post(JINA_RERANK_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()

        results = []
        for res_item in data.get("results", []):
            original_idx = res_item["index"]
            item = unique_items[original_idx].copy()
            item["score"] = float(res_item.get("relevance_score", 0.0))
            item["retrieval_method"] = "hybrid"
            results.append(item)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    except Exception as e:
        print(f"[Warning] Lỗi khi gọi Jina Reranker API ({e}). Giữ nguyên thứ tự ứng viên.")
        return unique_items[:top_k]


if __name__ == "__main__":
    print("Task 7 reranking module ready.")
