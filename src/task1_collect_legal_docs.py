"""
Task 1 — Thu thập tài liệu chính sách/quy định.

Hướng dẫn:
    1. Chọn chủ đề của nhóm.
    2. Tìm tối thiểu 3 tài liệu PDF/DOCX từ nguồn công khai.
    3. Lưu file gốc vào data/landing/legal/.
    4. Đặt tên không dấu và thể hiện đúng nội dung.

Ví dụ tài liệu: học phí, học bổng, ký túc xá, quy trình đăng ký.
Nếu website chặn crawler, hãy chọn nguồn công khai khác; không vượt WAF.
"""

from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"


def setup_directory() -> None:
    """Tạo thư mục lưu tài liệu gốc."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {DATA_DIR}")


def download_documents() -> None:
    """Kiểm tra tài liệu pháp luật đã thu thập.

    Các file .md đã được convert từ PDF qua ILovePDF/nguồn công khai
    và đặt sẵn trong data/landing/legal/.
    """
    legal_files = [
        p for p in DATA_DIR.iterdir()
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() == ".md"
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

