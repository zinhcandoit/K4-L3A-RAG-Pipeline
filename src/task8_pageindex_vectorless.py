"""
Task 8 — PageIndex vectorless fallback.

Hướng dẫn:
    1. Đọc PAGEINDEX_API_KEY từ .env.
    2. Upload tài liệu ở định dạng PageIndex hỗ trợ.
    3. Cache document IDs để không upload lại.
    4. Parse kết quả thành SearchResult có method pageindex.

PageIndex là dịch vụ ngoài: cần timeout và xử lý lỗi để pipeline không crash.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CACHE_DIR = Path(__file__).parent.parent / "chroma_db"
PAGEINDEX_DOCS_CACHE = CACHE_DIR / "pageindex_docs.json"


def upload_documents() -> None:
    """Upload tài liệu và lưu document IDs để tái sử dụng."""
    api_key = os.getenv("PAGEINDEX_API_KEY", PAGEINDEX_API_KEY)
    if not api_key:
        print("PAGEINDEX_API_KEY not set. Skipping upload.")
        return

    try:
        from pageindex import PageIndexClient
        client = PageIndexClient(api_key=api_key)
        
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        doc_map = {}
        if PAGEINDEX_DOCS_CACHE.exists():
            try:
                doc_map = json.loads(PAGEINDEX_DOCS_CACHE.read_text(encoding="utf-8"))
            except Exception:
                doc_map = {}

        for path in STANDARDIZED_DIR.rglob("*.md"):
            if path.name.startswith("."):
                continue
            rel_id = path.relative_to(STANDARDIZED_DIR).as_posix()
            if rel_id in doc_map:
                continue

            try:
                res = client.submit_document(file=str(path))
                doc_id = res.get("id") or res.get("document_id") or str(res)
                doc_map[rel_id] = doc_id
            except Exception as e:
                print(f"Could not upload {rel_id}: {e}")

        PAGEINDEX_DOCS_CACHE.write_text(json.dumps(doc_map, ensure_ascii=False), encoding="utf-8")
        print(f"Uploaded and cached {len(doc_map)} documents with PageIndex.")
    except Exception as e:
        print(f"PageIndex client error: {e}")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Trả về pageindex SearchResult."""
    if not query.strip() or top_k <= 0:
        return []

    api_key = os.getenv("PAGEINDEX_API_KEY", PAGEINDEX_API_KEY)
    if not api_key:
        return []

    try:
        from pageindex import PageIndexClient
        client = PageIndexClient(api_key=api_key)

        response = client.submit_query(query=query)
        nodes = response.get("nodes") or response.get("results") or []

        results: list[dict] = []
        for index, node in enumerate(nodes[:top_k]):
            content = node.get("content") or node.get("text") or str(node)
            source = node.get("source") or "pageindex_source.md"
            title = node.get("title") or Path(source).stem
            score = float(node.get("score", max(0.1, 1.0 - index * 0.1)))

            results.append({
                "id": f"pageindex::{index}",
                "content": content,
                "score": score,
                "metadata": {
                    "source": source,
                    "title": title,
                    "doc_type": "legal",
                    "url": None,
                    "chunk_index": index,
                },
                "retrieval_method": "pageindex",
            })
        return results
    except Exception as e:
        raise RuntimeError(f"PageIndex provider error: {e}") from e


if __name__ == "__main__":
    upload_documents()
