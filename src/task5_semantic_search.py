"""
Task 5 — Semantic search.

Embed query bằng chính hàm của Task 4, query ChromaDB và đổi cosine distance
thành similarity. Output phải theo SearchResult, sort giảm dần và không quá top_k.
"""

from .task4_chunking_indexing import embed_texts, get_collection


def _restore_metadata(metadata: dict) -> dict:
    """Chroma lưu url rỗng thay cho None; trả lại đúng contract."""
    restored = dict(metadata)
    if restored.get("url") == "":
        restored["url"] = None
    return restored


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về dense SearchResult theo score giảm dần."""
    if not query or not query.strip() or top_k <= 0:
        return []

    query_vector = embed_texts([query])[0]
    response = get_collection().query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    if not response.get("ids") or not response["ids"][0]:
        return []

    results = []
    seen: set[str] = set()
    for item_id, content, metadata, distance in zip(
        response["ids"][0],
        response["documents"][0],
        response["metadatas"][0],
        response["distances"][0],
    ):
        if item_id in seen:
            continue
        seen.add(item_id)
        results.append(
            {
                "id": item_id,
                "content": content,
                # cosine distance -> similarity; clamp để score không âm
                "score": max(0.0, 1.0 - float(distance)),
                "metadata": _restore_metadata(metadata),
                "retrieval_method": "dense",
            }
        )

    return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]


if __name__ == "__main__":
    for result in semantic_search("hộ kinh doanh nộp thuế theo phương pháp nào", top_k=3):
        print(f"{result['score']:.4f}  {result['id']}")
        print(f"   {result['content'][:120]}...")
