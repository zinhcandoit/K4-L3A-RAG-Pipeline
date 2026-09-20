"""
Task 5 — Semantic search.

Embed query bằng chính hàm của Task 4, query ChromaDB và đổi cosine distance
thành similarity. Output phải theo SearchResult, sort giảm dần và không quá top_k.
"""

from .task4_chunking_indexing import embed_texts, get_collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về dense SearchResult theo score giảm dần."""
    if not query.strip() or top_k <= 0:
        return []

    collection = get_collection()
    query_vectors = embed_texts([query])
    if not query_vectors or not query_vectors[0]:
        return []
    query_vector = query_vectors[0]

    response = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    if not response or not response.get("ids") or not response["ids"][0]:
        return []

    results: list[dict] = []
    seen_ids: set[str] = set()

    ids = response["ids"][0]
    docs = response["documents"][0]
    metas = response["metadatas"][0]
    distances = response["distances"][0]

    for item_id, content, metadata, distance in zip(ids, docs, metas, distances):
        if item_id in seen_ids:
            continue
        seen_ids.add(item_id)

        meta = dict(metadata)
        if meta.get("url") == "":
            meta["url"] = None
        if "chunk_index" in meta:
            meta["chunk_index"] = int(meta["chunk_index"])

        score = float(max(0.0, 1.0 - float(distance)))
        results.append({
            "id": item_id,
            "content": content,
            "score": score,
            "metadata": meta,
            "retrieval_method": "dense",
        })

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    for res in semantic_search("thuế hộ kinh doanh", top_k=3):
        print(res)
