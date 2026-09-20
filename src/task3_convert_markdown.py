"""
Task 3 — Chuẩn hóa dữ liệu sang Markdown.

Hướng dẫn:
    1. Dùng MarkItDown để convert PDF/DOCX.
    2. Đọc JSON và giữ metadata ở đầu file Markdown.
    3. Giữ cấu trúc thư mục legal/ và news/.
    4. Không tạo file rỗng hoặc file trùng khi chạy lại.

Cài đặt:
    Dependency MarkItDown đã được khai báo trong pyproject.toml.
    
-> Hoặc dùng công cụ nào bạn quen khác Markitdown
"""

from pathlib import Path


LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _clean_text(text: str) -> str:
    """Chuan hoa van ban: gop dong trong lien tiep, xoa khoang trang thua."""
    import re

    # Xoa khoang trang/tab thua o cuoi moi dong
    cleaned = re.sub(r"[ \t]+\n", "\n", text)
    # Gop 2+ dong trong lien tiep thanh 1 dong duy nhat
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    return cleaned.strip() + "\n"


def convert_legal_docs() -> None:
    """Doc .md tu landing/legal, clean va ghi vao standardized/legal."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in sorted(legal_dir.iterdir()):
        if path.is_file() and not path.name.startswith(".") and path.suffix.lower() == ".md":
            text = path.read_text(encoding="utf-8")
            cleaned = _clean_text(text)
            dest = output_dir / path.name
            dest.write_text(cleaned, encoding="utf-8")
            saved = len(text) - len(cleaned)
            tag = f" (-{saved:,}b)" if saved > 0 else ""
            print(f"  [OK] legal: {path.name}{tag}")


def convert_news_articles() -> None:
    """Convert JSON crawl results thanh Markdown voi metadata header, da clean."""
    import json

    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in sorted(news_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        header = (
            f"# {data['title']}\n"
            f"**Source:** {data['url']}\n"
            f"**Crawled:** {data['date_crawled']}\n---\n"
        )
        raw = header + data["content_markdown"]
        cleaned = _clean_text(raw)
        (output_dir / f"{path.stem}.md").write_text(cleaned, encoding="utf-8")
        print(f"  [OK] news: {path.stem}.md")


def convert_all() -> None:
    """Convert toàn bộ dữ liệu landing."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    convert_legal_docs()
    convert_news_articles()
    print(f"Saved Markdown to: {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()
