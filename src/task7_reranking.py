"""
Task 7 — Reciprocal Rank Fusion.

RRF gộp nhiều bảng xếp hạng mà không cộng trực tiếp cosine score với BM25
score. Công thức: RRF(d) = sum(1 / (k + rank)), rank bắt đầu từ 1.

Lưu ý: RRF score chỉ phản ánh thứ hạng, không dùng để quyết định fallback.
Task 9 đọc cosine score gốc từ dense list cho quyết định đó.

-> Dùng Jina hoặc self host hoặc bất cứ công cụ nào bạn quen
"""


def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
) -> list[dict]:
    """Fuse nhiều ranked lists và trả hybrid SearchResult."""
    if top_k <= 0:
        return []

    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        if not ranked_list:
            continue
        for rank, item in enumerate(ranked_list, 1):
            item_id = item["id"]
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
            # Giữ bản gặp đầu tiên: list đứng trước trong ranked_lists được ưu tiên
            items.setdefault(item_id, item)

    # Sort theo score giảm dần; hòa điểm thì giữ thứ tự ID cho kết quả ổn định
    ranked_ids = sorted(scores, key=lambda item_id: (-scores[item_id], item_id))

    results = []
    for item_id in ranked_ids[:top_k]:
        # copy trước khi ghi đè để không mutate ranked list đầu vào
        result = dict(items[item_id])
        result["score"] = scores[item_id]
        result["retrieval_method"] = "hybrid"
        results.append(result)

    return results


if __name__ == "__main__":
    print("Implement rerank_rrf, then run contract tests.")
