"""
Task 8 — PageIndex vectorless fallback.

Hướng dẫn:
    1. Đọc PAGEINDEX_API_KEY từ .env.
    2. Upload tài liệu ở định dạng PageIndex hỗ trợ.
    3. Cache document IDs để không upload lại.
    4. Parse kết quả thành SearchResult có method pageindex.

PageIndex là dịch vụ ngoài: cần timeout và xử lý lỗi để pipeline không crash.
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CACHE_PATH = Path(__file__).parent.parent / "data" / "pageindex_docs.json"
FONT_PATHS = [
    r"C:\Windows\Fonts\arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
POLL_TIMEOUT = 60
POLL_INTERVAL = 2


def _client():
    from pageindex import PageIndexClient

    if not PAGEINDEX_API_KEY:
        raise RuntimeError("PAGEINDEX_API_KEY is not set")
    return PageIndexClient(PAGEINDEX_API_KEY)


def _load_cache() -> dict[str, str]:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _markdown_to_pdf(path: Path, out_path: Path) -> None:
    from fpdf import FPDF

    font = next((f for f in FONT_PATHS if Path(f).exists()), None)
    pdf = FPDF()
    pdf.add_page()
    if font:
        pdf.add_font("body", "", font)
        pdf.set_font("body", size=11)
    else:
        pdf.set_font("Helvetica", size=11)
    for line in path.read_text(encoding="utf-8").splitlines():
        pdf.multi_cell(0, 6, line or " ", new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(out_path))


def upload_documents() -> None:
    """Upload tài liệu và lưu document IDs để tái sử dụng."""
    import tempfile

    client = _client()
    cache = _load_cache()
    with tempfile.TemporaryDirectory() as tmp:
        for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
            key = path.relative_to(STANDARDIZED_DIR).as_posix()
            if key in cache:
                continue
            pdf_path = Path(tmp) / f"{path.stem}.pdf"
            _markdown_to_pdf(path, pdf_path)
            cache[key] = client.submit_document(str(pdf_path))["doc_id"]
            CACHE_PATH.write_text(
                json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    print(f"PageIndex documents: {len(cache)}")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Trả về pageindex SearchResult."""
    cache = _load_cache()
    if not cache:
        return []
    client = _client()
    results = []
    for key, doc_id in cache.items():
        retrieval_id = client.submit_query(doc_id, query)["retrieval_id"]
        deadline = time.time() + POLL_TIMEOUT
        while True:
            response = client.get_retrieval(retrieval_id)
            if response.get("status") == "completed":
                break
            if response.get("status") == "failed" or time.time() > deadline:
                response = {}
                break
            time.sleep(POLL_INTERVAL)
        path = STANDARDIZED_DIR / key
        for index, node in enumerate(response.get("retrieved_nodes") or []):
            contents = node.get("relevant_contents") or []
            text = "\n".join(
                c.get("relevant_content", "") if isinstance(c, dict) else str(c)
                for c in contents
            ).strip()
            if not text:
                continue
            results.append({
                "id": f"{key}::pageindex-{index}",
                "content": text,
                "score": 0.0,
                "metadata": {
                    "source": path.name,
                    "title": node.get("title") or path.stem,
                    "doc_type": "legal" if "legal" in path.parts else "news",
                    "url": None,
                    "chunk_index": index,
                },
                "retrieval_method": "pageindex",
            })
    results = results[:top_k]
    for rank, item in enumerate(results):
        item["score"] = 1.0 / (rank + 1)
    return results


if __name__ == "__main__":
    upload_documents()
