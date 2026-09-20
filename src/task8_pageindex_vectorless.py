"""
Task 8 — PageIndex vectorless fallback.

Luồng: upload PDF gốc trong data/landing/legal/ -> cache doc_id vào
data/pageindex_docs.json -> submit_query từng doc -> parse node thành SearchResult.

PageIndex là dịch vụ ngoài nên mọi lỗi (thiếu key, timeout, doc chưa index xong,
response đổi shape) đều phải nuốt tại đây hoặc ở Task 9; giao diện không được crash.
Không có PAGEINDEX_API_KEY thì trả [] để pipeline dùng thẳng hybrid result.

LƯU Ý: nhánh gọi API thật chưa chạy được vì nhóm chưa có key. Parser bên dưới
đọc nhiều shape response khác nhau thay vì đoán cứng một tên field; khi có key
phải chạy lại và đối chiếu response thật trước khi tin kết quả.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
LANDING_LEGAL_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
CACHE_PATH = Path(__file__).parent.parent / "data" / "pageindex_docs.json"

REQUEST_TIMEOUT = 30
MAX_DOCS = 5  # giới hạn số doc query mỗi lần để fallback không kéo dài


def _client():
    from pageindex import PageIndexClient

    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _extract_doc_id(response: object) -> str:
    if isinstance(response, dict):
        for key in ("doc_id", "document_id", "id"):
            value = response.get(key)
            if isinstance(value, str) and value:
                return value
        data = response.get("data")
        if isinstance(data, dict):
            return _extract_doc_id(data)
    return ""


def upload_documents() -> dict:
    """Upload tài liệu và lưu document IDs để tái sử dụng."""
    if not PAGEINDEX_API_KEY:
        print("[pageindex] Chưa có PAGEINDEX_API_KEY, bỏ qua upload.")
        return {}

    client = _client()
    cache = _load_cache()

    for path in sorted(LANDING_LEGAL_DIR.glob("*.pdf")):
        if path.name in cache:
            print(f"  skip (đã upload): {path.name}")
            continue
        try:
            doc_id = _extract_doc_id(client.submit_document(file_path=str(path)))
            if not doc_id:
                print(f"  FAILED: {path.name} — response không có doc_id")
                continue
            cache[path.name] = doc_id
            print(f"  uploaded: {path.name} -> {doc_id}")
        except Exception as error:
            print(f"  FAILED: {path.name} — {error}")

    _save_cache(cache)
    return cache


def _iter_nodes(payload: object):
    """Nhặt các node kết quả bất kể response bọc ở tầng nào."""
    if isinstance(payload, dict):
        for key in ("retrieved_nodes", "nodes", "results", "sources", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                yield from (node for node in value if isinstance(node, dict))
                return
            if isinstance(value, dict):
                yield from _iter_nodes(value)
                return
    elif isinstance(payload, list):
        yield from (node for node in payload if isinstance(node, dict))


def _node_text(node: dict) -> str:
    for key in ("text", "content", "node_text", "summary", "page_text"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Trả về pageindex SearchResult."""
    if not PAGEINDEX_API_KEY or not query.strip() or top_k <= 0:
        return []

    cache = _load_cache()
    if not cache:
        return []

    client = _client()
    collected: list[dict] = []

    for source, doc_id in list(cache.items())[:MAX_DOCS]:
        try:
            if not client.is_retrieval_ready(doc_id):
                continue
            response = client.submit_query(doc_id=doc_id, query=query)
            retrieval_id = response.get("retrieval_id") if isinstance(response, dict) else None
            if retrieval_id:
                response = client.get_retrieval(retrieval_id)

            for node in _iter_nodes(response):
                text = _node_text(node)
                if not text:
                    continue
                collected.append(
                    {
                        "source": source,
                        "doc_id": doc_id,
                        "node_id": str(node.get("node_id") or node.get("id") or len(collected)),
                        "text": text,
                        "raw_score": node.get("score"),
                    }
                )
        except Exception as error:
            print(f"[pageindex] Lỗi khi query {source}: {error}")
            continue

    if not collected:
        return []

    # API không đảm bảo có score; gán score giảm dần theo rank để giữ contract
    results = []
    seen: set[str] = set()
    for rank, item in enumerate(collected[:top_k], 1):
        item_id = f"pageindex::{item['doc_id']}::{item['node_id']}"
        if item_id in seen:
            continue
        seen.add(item_id)
        results.append(
            {
                "id": item_id,
                "content": item["text"],
                "score": 1.0 / rank,
                "metadata": {
                    "source": item["source"],
                    "title": Path(item["source"]).stem,
                    "doc_type": "legal",
                    "url": None,
                    "chunk_index": rank - 1,
                },
                "retrieval_method": "pageindex",
            }
        )

    return results


if __name__ == "__main__":
    upload_documents()
