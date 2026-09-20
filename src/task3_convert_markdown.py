"""
Task 3 — Chuẩn hóa dữ liệu sang Markdown.

landing/legal/*.pdf   -> standardized/legal/*.md   (MarkItDown, fallback pypdf)
landing/news/*.json   -> standardized/news/*.md    (metadata header + content)

Kèm bước đối chiếu: số hiệu văn bản khai trong manifest phải xuất hiện trong phần
text trích ra. Đây là chỗ bắt lỗi "tên file một đằng, nội dung một nẻo" — test
acceptance không bắt được lỗi này.

Chạy:
    python -m src.task3_convert_markdown
    python -m src.task3_convert_markdown --force   # convert lại toàn bộ
"""

import json
import re
import sys
from pathlib import Path


LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"

LEGAL_SUFFIXES = {".pdf", ".doc", ".docx"}
MIN_OUTPUT_CHARS = 200  # acceptance test yêu cầu mỗi file >= 200 ký tự


def _load_manifest() -> dict:
    """Map filename -> metadata do Task 1 ghi lại."""
    path = LANDING_DIR / "legal" / "manifest.json"
    if not path.exists():
        return {}
    return {item["file"]: item for item in json.loads(path.read_text(encoding="utf-8"))}


def _extract_text(path: Path) -> str:
    """Trích text từ PDF/DOCX. Ưu tiên MarkItDown, fallback pypdf."""
    try:
        from markitdown import MarkItDown

        return MarkItDown().convert(str(path)).text_content
    except ImportError:
        pass

    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as error:
            raise RuntimeError(
                "Cần `markitdown[pdf]` hoặc `pypdf`. Chạy: uv sync"
            ) from error
        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)

    raise RuntimeError(f"Không có bộ chuyển đổi cho {path.suffix}; chạy `uv sync`")


def _fetch_fulltext(url: str) -> str:
    """Lấy toàn văn dạng HTML và tái dùng parser của Task 2."""
    from src.task2_crawl_news import fetch, parse_article

    return parse_article(fetch(url), url)["content_markdown"]


def _clean(text: str) -> str:
    """Gom khoảng trắng thừa do PDF xuống dòng theo layout."""
    text = text.replace("\r\n", "\n").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _verify_doc_no(text: str, doc_no: str) -> bool:
    """Số hiệu khai báo có thật sự nằm trong nội dung không."""
    if not doc_no:
        return True
    # PDF hay chèn khoảng trắng lạ giữa các ký tự số hiệu
    pattern = r"\s*".join(re.escape(ch) for ch in doc_no if not ch.isspace())
    return re.search(pattern, text, re.I) is not None


def convert_legal_docs(force: bool = False) -> list[str]:
    """Convert PDF/DOCX vào standardized/legal."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()
    warnings = []

    sources = sorted(p for p in legal_dir.iterdir() if p.suffix.lower() in LEGAL_SUFFIXES)
    if not sources:
        raise FileNotFoundError(
            f"Không có PDF/DOC/DOCX trong {legal_dir}. Chạy task1 trước."
        )

    for path in sources:
        target = output_dir / f"{path.stem}.md"
        if target.exists() and not force and target.stat().st_mtime >= path.stat().st_mtime:
            print(f"  skip (đã mới): {target.name}")
            continue

        meta = manifest.get(path.name, {})
        doc_no = meta.get("doc_no", "")

        body = _clean(_extract_text(path))
        source_note = f"`data/landing/legal/{path.name}` (PDF ký số)"

        if len(body) < MIN_OUTPUT_CHARS:
            # PDF ký số của Chính phủ là bản scan -> lấy text từ trang toàn văn HTML.
            fulltext_url = meta.get("fulltext_page", "")
            if not fulltext_url:
                warnings.append(
                    f"{path.name}: PDF là bản scan ({len(body)} ký tự) và chưa khai "
                    f"'fulltext' trong task1 -> bỏ qua"
                )
                continue
            try:
                body = _clean(_fetch_fulltext(fulltext_url))
            except Exception as error:
                warnings.append(f"{path.name}: lấy toàn văn thất bại — {error}")
                continue
            source_note += f"; text lấy từ {fulltext_url}"

        if len(body) < MIN_OUTPUT_CHARS:
            warnings.append(f"{path.name}: nội dung quá ngắn ({len(body)} ký tự)")
            continue
        if not _verify_doc_no(body, doc_no):
            warnings.append(f"{path.name}: KHÔNG tìm thấy số hiệu '{doc_no}' trong nội dung")

        header = f"# {meta.get('title', path.stem)}\n\n"
        if doc_no:
            header += f"**Số hiệu:** {doc_no}\n\n"
        if meta.get("source_page"):
            header += f"**Nguồn:** {meta['source_page']}\n\n"
        header += f"**File gốc:** {source_note}\n\n---\n\n"

        target.write_text(header + body, encoding="utf-8")
        print(f"  saved: {target.name}  ({len(body)} ký tự)")

    return warnings


def convert_news_articles(force: bool = False) -> list[str]:
    """Convert JSON vào standardized/news."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings = []

    sources = sorted(news_dir.glob("*.json"))
    if not sources:
        raise FileNotFoundError(f"Không có JSON trong {news_dir}. Chạy task2 trước.")

    for path in sources:
        target = output_dir / f"{path.stem}.md"
        if target.exists() and not force and target.stat().st_mtime >= path.stat().st_mtime:
            print(f"  skip (đã mới): {target.name}")
            continue

        data = json.loads(path.read_text(encoding="utf-8"))
        missing = [k for k in ("url", "title", "date_crawled", "content_markdown") if not str(data.get(k, "")).strip()]
        if missing:
            warnings.append(f"{path.name}: thiếu metadata {missing}")
            continue

        body = _clean(data["content_markdown"])
        if len(body) < MIN_OUTPUT_CHARS:
            warnings.append(f"{path.name}: nội dung quá ngắn ({len(body)} ký tự)")
            continue

        header = (
            f"# {data['title']}\n\n"
            f"**Source:** {data['url']}\n\n"
            f"**Crawled:** {data['date_crawled']}\n\n"
        )
        if data.get("date_published"):
            header += f"**Published:** {data['date_published']}\n\n"
        header += "---\n\n"

        target.write_text(header + body, encoding="utf-8")
        print(f"  saved: {target.name}  ({len(body)} ký tự)")

    return warnings


def convert_all(force: bool = False) -> None:
    """Convert toàn bộ dữ liệu landing."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Legal:")
    warnings = convert_legal_docs(force)
    print("News:")
    warnings += convert_news_articles(force)

    print(f"\nSaved Markdown to: {OUTPUT_DIR}")
    if warnings:
        print("\nCẦN KIỂM TRA LẠI BẰNG TAY:")
        for warning in warnings:
            print(f"  ! {warning}")


if __name__ == "__main__":
    convert_all(force="--force" in sys.argv)
