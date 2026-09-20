"""
Task 1 — Thu thập tài liệu chính sách/quy định.

Hướng dẫn:
    1. Chọn chủ đề của nhóm.
    2. Tìm tối thiểu 3 tài liệu PDF/DOC/DOCX từ nguồn công khai.
    3. Lưu file gốc vào data/landing/legal/ (định dạng: .doc, .docx, .pdf; không dùng .md ở tầng landing).
    4. Đặt tên không dấu và thể hiện đúng nội dung.
"""

from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
ALLOWED_EXTENSIONS = {".pdf", ".doc", ".docx"}


def setup_directory() -> None:
    """Tạo thư mục lưu tài liệu gốc."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {DATA_DIR}")


def download_documents() -> None:
    """Kiểm tra tài liệu pháp luật đã thu thập.

    Chỉ chấp nhận các định dạng gốc: .doc, .docx, .pdf trong data/landing/legal/.
    """
    legal_files = [
        p for p in DATA_DIR.iterdir()
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in ALLOWED_EXTENSIONS
    ]
    if len(legal_files) < 3:
        raise RuntimeError(
            f"Cần ít nhất 3 tài liệu trong {DATA_DIR}, hiện có {len(legal_files)}"
        )
    for f in sorted(legal_files):
        print(f"  [OK] {f.name} ({f.stat().st_size:,} bytes)")


if __name__ == "__main__":
    setup_directory()
    download_documents()
