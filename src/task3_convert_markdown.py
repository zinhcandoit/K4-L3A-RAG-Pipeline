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
import sys

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_legal_docs() -> None:
    """Convert PDF/DOCX từ data/landing/legal vào data/standardized/legal siêu tốc."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import time
    import pypdfium2 as pdfium

    pdf_files = [p for p in sorted(legal_dir.iterdir()) if not p.name.startswith(".") and p.suffix.lower() in {".pdf", ".doc", ".docx"}]
    print(f"\n--- Bắt đầu convert {len(pdf_files)} tài liệu pháp lý sang Markdown ---")
    
    count = 0
    for idx, path in enumerate(pdf_files, 1):
        start_time = time.time()
        print(f"[{idx}/{len(pdf_files)}] Đang xử lý: {path.name} ({path.stat().st_size / (1024*1024):.1f} MB)...", end=" ", flush=True)
        
        extracted_text = ""
        # 1. Thử dùng pypdfium2 siêu tốc (Google PDFium engine)
        if path.suffix.lower() == ".pdf":
            try:
                pdf = pdfium.PdfDocument(str(path))
                page_texts = []
                for p in pdf:
                    tpage = p.get_textpage()
                    t = tpage.get_text_range()
                    if t.strip():
                        page_texts.append(t.strip())
                extracted_text = "\n\n".join(page_texts)
            except Exception as e:
                extracted_text = ""

        # 2. Nếu pypdfium2 không ra text hoặc là file doc/docx, fallback sang MarkItDown
        if len(extracted_text.strip()) < 50:
            try:
                from markitdown import MarkItDown
                converter = MarkItDown()
                result = converter.convert(str(path))
                if result and result.text_content:
                    extracted_text = result.text_content
            except Exception as e:
                pass

        elapsed = time.time() - start_time
        if extracted_text.strip():
            (output_dir / f"{path.stem}.md").write_text(extracted_text, encoding="utf-8")
            count += 1
            print(f"XONG ({len(extracted_text)} ký tự trong {elapsed:.2f}s)")
        else:
            print(f"CẢNH BÁO: File không chứa text layer (PDF dạng ảnh quét/vector outline).")

    print(f"--- Hoàn tất convert {count}/{len(pdf_files)} file sang {output_dir} ---\n")


def convert_news_articles() -> None:
    """Convert JSON từ data/landing/news vào data/standardized/news."""
    import json
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(news_dir.glob("*.json")):
        if path.name.startswith("."):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            header = (
                f"# {data.get('title', path.stem)}\n\n"
                f"**Source:** {data.get('url', '')}\n\n"
                f"**Crawled:** {data.get('date_crawled', '')}\n\n---\n\n"
            )
            (output_dir / f"{path.stem}.md").write_text(
                header + data.get("content_markdown", ""), encoding="utf-8"
            )
            count += 1
        except Exception as e:
            print(f"Lỗi khi convert {path.name}: {e}")
    print(f"Đã convert {count} bài báo sang Markdown.")


def convert_all() -> None:
    """Convert toàn bộ dữ liệu landing."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    convert_legal_docs()
    convert_news_articles()
    print(f"Saved Markdown to: {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()
